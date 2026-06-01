"""tests/test_calibration.py - Phase 3 calibration pipeline gates.

11 tests covering CAL-01..04, the variant hard-fail boundary, and the end-to-end
quantize_static integration smoke. Synthetic NSNet2 + DatasetDict fixtures only -
no real checkpoint load (DET-02 sidestep), no real VBD download.

Test runner: uv run pytest tests/test_calibration.py -xvs
Phase 1 conftest fixture provides determinism (CUBLAS env, manual_seed(0), cudnn flags).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict

from common.env import AttrDict
from nsnet2.model import NSNet2
from nsnet2.streaming import NSNet2Streaming
from nsnet2.calibration import VBDCalibrationReader


# ----------------------------------------------------------------------------
# Helper builders (used across multiple tests)
# ----------------------------------------------------------------------------

# Tiny dimensions: ~50 ONNX nodes, completes integration smoke in ~2s.
_H_TINY_BASE = {
    "n_fft": 64,
    "hop_size": 16,
    "win_size": 64,
    "compress_factor": 0.3,
    "sampling_rate": 16000,
    "seed": 1234,
    "hidden_dim": 16,
    "fc_hidden_dim": 16,
    "num_gru_layers": 2,
    "linear": {"kind": "linear"},
    "gru": {"kind": "gru"},
}


def _make_split(audios, prefix, sr=16000):
    """Build a Dataset.from_dict-compatible payload mirroring HF VBD schema."""
    return {
        "id":    [f"{prefix}_{i}" for i in range(len(audios))],
        "clean": [{"path": f"{prefix}_{i}_c", "array": a, "sampling_rate": sr} for i, a in enumerate(audios)],
        "noisy": [{"path": f"{prefix}_{i}_n", "array": a, "sampling_rate": sr} for i, a in enumerate(audios)],
    }


def _make_dataset_dict(train_audios, test_audios) -> DatasetDict:
    return DatasetDict({
        "train": Dataset.from_dict(_make_split(train_audios, "tr")),
        "test":  Dataset.from_dict(_make_split(test_audios,  "te")),
    })


def _make_h(extra_calibration: dict | None = None) -> AttrDict:
    """Build a complete h (no h.calibration unless requested) - D-07 backward compat target."""
    cfg = dict(_H_TINY_BASE)
    if extra_calibration is not None:
        cfg["calibration"] = AttrDict(extra_calibration)
    return AttrDict(cfg)


def _make_tiny_streaming() -> NSNet2Streaming:
    torch.manual_seed(0)
    h = AttrDict(dict(_H_TINY_BASE))
    base = NSNet2(h)
    base.eval()
    return NSNet2Streaming(base).eval()


def _random_audio(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n_samples).astype(np.float32)


def _redirect_hf_cache(monkeypatch, tmp_path: Path):
    """Point datasets_config.HF_DATASETS_CACHE at tmp_path so vbd-hashes/ writes are isolated per test."""
    from datasets import config as datasets_config
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", str(tmp_path))
    # calibration.py imports `datasets_config` directly; patch that reference too.
    import nsnet2.calibration as calibration_mod
    monkeypatch.setattr(calibration_mod.datasets_config, "HF_DATASETS_CACHE", str(tmp_path))


# ----------------------------------------------------------------------------
# CAL-01 tests (smoke + shape/dtype + count) -- D-01, D-02, D-03, D-04, D-05
# ----------------------------------------------------------------------------

def test_get_next_smoke(monkeypatch, tmp_path):
    """CAL-01 smoke: reader constructs, get_next() yields dicts, eventually returns None."""
    _redirect_hf_cache(monkeypatch, tmp_path)
    streaming = _make_tiny_streaming()
    audios_train = [_random_audio(2048, seed=i) for i in range(4)]
    audios_test  = [_random_audio(2048, seed=100 + i) for i in range(2)]
    ds = _make_dataset_dict(audios_train, audios_test)
    h = _make_h({"num_utterances": 2, "seed": 0, "frames_per_utterance": None})

    reader = VBDCalibrationReader(streaming, h, ds)

    first = reader.get_next()
    assert isinstance(first, dict)
    assert set(first.keys()) == {"frame_in", "states_in"}

    # Drain
    n = 1
    while True:
        item = reader.get_next()
        if item is None:
            break
        n += 1
    assert n > 0

    # Past exhaustion stays None
    assert reader.get_next() is None


def test_get_next_shape_and_dtype(monkeypatch, tmp_path):
    """CAL-01: frame_in (1, F) float32, states_in (L, 1, H) float32. Sourced from train split."""
    _redirect_hf_cache(monkeypatch, tmp_path)
    streaming = _make_tiny_streaming()
    L = streaming.num_layers
    H = streaming.hidden_size
    F = _H_TINY_BASE["n_fft"] // 2 + 1   # 33

    audios_train = [_random_audio(2048, seed=i) for i in range(4)]
    audios_test  = [_random_audio(2048, seed=100 + i) for i in range(2)]
    ds = _make_dataset_dict(audios_train, audios_test)
    h = _make_h({"num_utterances": 2, "seed": 0, "frames_per_utterance": None})

    reader = VBDCalibrationReader(streaming, h, ds)
    item = reader.get_next()
    assert item is not None

    frame = item["frame_in"]
    states = item["states_in"]

    # Shape contracts
    assert frame.shape == (1, F), f"expected (1, {F}), got {frame.shape}"
    assert states.shape == (L, 1, H), f"expected ({L}, 1, {H}), got {states.shape}"

    # Dtype contracts (ORT enforces strict float32 - Phase 2 finding)
    assert frame.dtype == np.float32
    assert states.dtype == np.float32

    # First-frame state MUST be all-zeros (D-04 zero h0 invariant at frame 0 of utt 0)
    assert np.all(states == 0.0), "first frame must have zero h0 (D-04)"


def test_yields_expected_count(monkeypatch, tmp_path):
    """CAL-01: total frame count == sum-over-utts of T_frames per utterance."""
    _redirect_hf_cache(monkeypatch, tmp_path)
    streaming = _make_tiny_streaming()

    n_samples = 2048
    audios_train = [_random_audio(n_samples, seed=i) for i in range(4)]
    audios_test  = [_random_audio(n_samples, seed=100 + i) for i in range(2)]
    ds = _make_dataset_dict(audios_train, audios_test)
    num_utts = 3
    h = _make_h({"num_utterances": num_utts, "seed": 0, "frames_per_utterance": None})

    reader = VBDCalibrationReader(streaming, h, ds)
    frames = []
    while True:
        item = reader.get_next()
        if item is None:
            break
        frames.append(item)

    # Expected per-utt frame count: torch.stft with n_fft=64, hop=16, win=64, center=True
    # T_frames = floor(N / hop) + 1 = floor(2048/16) + 1 = 129
    expected_per_utt = (n_samples // _H_TINY_BASE["hop_size"]) + 1
    assert len(frames) == num_utts * expected_per_utt, (
        f"expected {num_utts}*{expected_per_utt}={num_utts * expected_per_utt} frames, got {len(frames)}"
    )


# ----------------------------------------------------------------------------
# CAL-02 tests (disjoint assertion + hash cache) -- D-09, D-10, D-11
# ----------------------------------------------------------------------------

def test_disjoint_assertion_fires_on_overlap(monkeypatch, tmp_path):
    """CAL-02 / D-11: gate-firing - overlapping audio in train+test triggers AssertionError."""
    _redirect_hf_cache(monkeypatch, tmp_path)
    streaming = _make_tiny_streaming()

    shared = _random_audio(2048, seed=42)   # SAME audio in both splits
    other_train = _random_audio(2048, seed=99)
    other_test  = _random_audio(2048, seed=77)
    ds = _make_dataset_dict([shared, other_train], [shared, other_test])
    h = _make_h({"num_utterances": 2, "seed": 0, "frames_per_utterance": None})

    with pytest.raises(AssertionError, match="Calibration set leakage"):
        VBDCalibrationReader(streaming, h, ds)


def test_hash_cache_reused_on_second_construct(monkeypatch, tmp_path):
    """CAL-02 / D-10: on-disk hash cache exists after first construct and is read on second."""
    _redirect_hf_cache(monkeypatch, tmp_path)
    streaming = _make_tiny_streaming()
    audios_train = [_random_audio(2048, seed=i) for i in range(4)]
    audios_test  = [_random_audio(2048, seed=100 + i) for i in range(2)]
    ds = _make_dataset_dict(audios_train, audios_test)
    h = _make_h({"num_utterances": 2, "seed": 0, "frames_per_utterance": None})

    # First construct -- writes the cache.
    VBDCalibrationReader(streaming, h, ds)
    cache_dir = tmp_path / "vbd-hashes"
    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1, f"expected 1 cache file in {cache_dir}, found {cache_files}"
    cache_path = cache_files[0]
    cache_payload = json.loads(cache_path.read_text())
    assert "train_hashes" in cache_payload and "test_hashes" in cache_payload
    assert "train_fingerprint" in cache_payload and "test_fingerprint" in cache_payload
    assert len(cache_payload["train_hashes"]) == len(audios_train)
    assert len(cache_payload["test_hashes"]) == len(audios_test)

    # Capture mtime, write a sentinel value to test we don't re-overwrite when valid.
    mtime_before = cache_path.stat().st_mtime_ns

    # Second construct -- must read existing cache (mtime unchanged).
    VBDCalibrationReader(streaming, h, ds)
    mtime_after = cache_path.stat().st_mtime_ns
    assert mtime_before == mtime_after, "hash cache was rewritten on second construct"


# ----------------------------------------------------------------------------
# CAL-03 test (h.calibration getattr defaults) -- D-07
# ----------------------------------------------------------------------------

def test_calibration_config_defaults(monkeypatch, tmp_path):
    """CAL-03 / D-07: a config without h.calibration uses defaults (200, h.seed, None).

    cp_baseline/config.json has NO h.calibration block - same shape exercised here.
    """
    _redirect_hf_cache(monkeypatch, tmp_path)
    streaming = _make_tiny_streaming()
    # 250 train utts so default num_utterances=200 is feasible
    audios_train = [_random_audio(2048, seed=i) for i in range(250)]
    audios_test  = [_random_audio(2048, seed=10000 + i) for i in range(5)]
    ds = _make_dataset_dict(audios_train, audios_test)
    h = _make_h(extra_calibration=None)   # NO h.calibration block

    reader = VBDCalibrationReader(streaming, h, ds)
    assert reader.num_utterances == 200, "default num_utterances must be 200"
    assert reader.seed == h.seed, "default seed must be h.seed"
    assert reader.frames_per_utterance is None, "default frames_per_utterance must be None"


# ----------------------------------------------------------------------------
# CAL-04 tests (calibration_hash determinism + log + integration smoke) -- D-12, D-13, D-14
# ----------------------------------------------------------------------------

def test_calibration_hash_stable(monkeypatch, tmp_path):
    """CAL-04 / D-14: identical inputs produce identical calibration_hash."""
    _redirect_hf_cache(monkeypatch, tmp_path)
    streaming = _make_tiny_streaming()
    audios_train = [_random_audio(2048, seed=i) for i in range(8)]
    audios_test  = [_random_audio(2048, seed=100 + i) for i in range(2)]
    ds = _make_dataset_dict(audios_train, audios_test)
    h = _make_h({"num_utterances": 4, "seed": 7, "frames_per_utterance": None})

    r1 = VBDCalibrationReader(streaming, h, ds)
    r2 = VBDCalibrationReader(streaming, h, ds)
    assert r1.calibration_hash == r2.calibration_hash, (
        f"calibration_hash drifted: {r1.calibration_hash} != {r2.calibration_hash}"
    )
    # 64-hex char (sha256) shape
    assert len(r1.calibration_hash) == 64
    int(r1.calibration_hash, 16)  # parse as hex; raises if not hex


def test_calibration_hash_logged(monkeypatch, tmp_path, capsys):
    """D-13: calibration_hash printed at construction in the documented format."""
    _redirect_hf_cache(monkeypatch, tmp_path)
    streaming = _make_tiny_streaming()
    audios_train = [_random_audio(2048, seed=i) for i in range(4)]
    audios_test  = [_random_audio(2048, seed=100 + i) for i in range(2)]
    ds = _make_dataset_dict(audios_train, audios_test)
    h = _make_h({"num_utterances": 2, "seed": 5, "frames_per_utterance": 11})

    reader = VBDCalibrationReader(streaming, h, ds)
    captured = capsys.readouterr().out
    expected_substring = (
        f"VBDCalibrationReader: calib_hash={reader.calibration_hash} "
        f"(utts=2, seed=5, frames_per_utterance=11)"
    )
    assert expected_substring in captured, (
        f"expected log substring {expected_substring!r} not found in: {captured!r}"
    )


def test_calibration_hash_responds_to_seed(monkeypatch, tmp_path):
    """CAL-04 / D-14: bumping seed produces a different hash."""
    _redirect_hf_cache(monkeypatch, tmp_path)
    streaming = _make_tiny_streaming()
    audios_train = [_random_audio(2048, seed=i) for i in range(8)]
    audios_test  = [_random_audio(2048, seed=100 + i) for i in range(2)]
    ds = _make_dataset_dict(audios_train, audios_test)
    h_a = _make_h({"num_utterances": 4, "seed": 0, "frames_per_utterance": None})
    h_b = _make_h({"num_utterances": 4, "seed": 1, "frames_per_utterance": None})

    r_a = VBDCalibrationReader(streaming, h_a, ds)
    r_b = VBDCalibrationReader(streaming, h_b, ds)
    assert r_a.calibration_hash != r_b.calibration_hash, (
        "calibration_hash MUST change when seed changes"
    )


def test_quantize_static_smoke(monkeypatch, tmp_path):
    """CAL-04 SC-4: end-to-end - synthetic FP32 ONNX + reader -> int8 ONNX written without crash.

    Exercises the same call seam Phase 4 quant.quantize_checkpoint will use. Reader pattern
    + quantize_static call were verified end-to-end in 03-RESEARCH.md (49KB int8 produced).
    """
    from onnxruntime.quantization import (
        quantize_static, QuantFormat, QuantType, CalibrationMethod,
    )
    _redirect_hf_cache(monkeypatch, tmp_path)
    streaming = _make_tiny_streaming()

    # Export tiny FP32 ONNX (mirrors Phase 2 test_export_onnx synthetic-checkpoint pattern)
    n_freq = _H_TINY_BASE["n_fft"] // 2 + 1
    fp32 = tmp_path / "mini_fp32.onnx"
    int8 = tmp_path / "mini_int8.onnx"
    example_frame = torch.zeros(1, n_freq, dtype=torch.float32)
    example_states = torch.zeros(streaming.num_layers, 1, streaming.hidden_size, dtype=torch.float32)
    torch.onnx.export(
        streaming,
        (example_frame, example_states),
        str(fp32),
        input_names=["frame_in", "states_in"],
        output_names=["mask", "states_out"],
        dynamic_axes={
            "frame_in":   {0: "B"},
            "states_in":  {1: "B"},
            "mask":       {0: "B"},
            "states_out": {1: "B"},
        },
        opset_version=17,
        dynamo=False,
    )
    assert fp32.exists() and fp32.stat().st_size > 0

    # Build reader against synthetic DatasetDict (5 train + 2 test, all distinct audio).
    audios_train = [_random_audio(2048, seed=i) for i in range(5)]
    audios_test  = [_random_audio(2048, seed=100 + i) for i in range(2)]
    ds = _make_dataset_dict(audios_train, audios_test)
    h = _make_h({"num_utterances": 3, "seed": 0, "frames_per_utterance": None})
    reader = VBDCalibrationReader(streaming, h, ds)

    # Run quantize_static. Phase 3 only asserts "doesn't crash + writes file".
    quantize_static(
        str(fp32),
        str(int8),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    assert int8.exists(), f"quantize_static did not write {int8}"
    assert int8.stat().st_size > 0, f"quantize_static wrote empty file at {int8}"
    # NOTE: int8-smaller-than-fp32 size check is intentionally OMITTED here. At this
    # tiny synthetic scale (n_fft=64, hidden=16, 2 GRU layers), the QDQ graph's fixed
    # overhead (per-channel scale/zero tensors + Q/DQ op pairs) exceeds the bytes saved
    # by int8 weights -- empirically we see int8 ~49KB vs fp32 ~25KB on the synthetic
    # mini model (matches 03-RESEARCH.md's end-to-end smoke at 49KB int8). The size
    # budget (<= 30% of fp32) is enforced at real-checkpoint scale by Phase 4
    # quant.quantize_checkpoint and its dedicated test, NOT here.


# ----------------------------------------------------------------------------
# Variant boundary (Phase 1 D-05 carryover -- guard since lifted)
# ----------------------------------------------------------------------------
# The original test_variant_hard_fail expected NotImplementedError on
# non-cuDNN gru_kind. Commit 8843372 ("unblock structured-variant int8 export
# pipeline") lifted that guard once the structured-FC export patches in
# export_onnx.py were proven to handle butterfly + monarch end-to-end.
# Test removed during the feat/quantization merge to main.
