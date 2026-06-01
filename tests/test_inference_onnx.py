"""tests/test_inference_onnx.py - Phase 5 ORT inference gate (INF-01..04 + D-11/D-12/D-14/D-15).

Eight pytest functions cover INF-01..04, the dual fail-fast fix-hints (D-11/D-12),
the dual warn-not-crash paths (D-14/D-15), the byte-identical-reruns gate
(INF-02 / Pitfall 8 closure), and the cp_baseline-skipif PESQ-sanity test.

Tests build a synthetic mini int8 ONNX end-to-end via the Phase 4 D-18 pattern
(tiny NSNet2 -> torch.save -> export_streaming -> quantize_checkpoint), then
drive inference_onnx.enhance_one_utterance and inference_onnx._check_ort_version
directly. The cp_baseline-skipif test (#8) runs on the real Phase 4 artifact
when present.

Test list (locked names per CONTEXT.md D-19):
  1. test_cli_help_lists_all_flags        - INF-01 / D-19 #1
  2. test_enhance_smoke                   - INF-01 / INF-03 / D-19 #2
  3. test_byte_identical_reruns           - INF-02 (Pitfall 8) / D-19 #3
  4. test_missing_int8_fail_fast          - D-11 / D-19 #4
  5. test_missing_fp32_sibling_fail_fast  - D-12 / D-19 #5
  6. test_ort_version_mismatch_warning    - INF-04 (Pitfall 9 / D-14) / D-19 #6
  7. test_missing_metadata_props_warning  - INF-04 (Pitfall 9 / D-15) / D-19 #7
  8. test_pesq_sanity_on_cp_baseline      - PESQ sanity (skipif) / D-19 #8

DET-02 sidestep preserved: zero deserialization-side load sites in this file.
The synthetic fixture uses torch.save (writing only); checkpoint reading happens
inside quant.quantize_checkpoint via NSNet2Streaming.from_checkpoint (DET-02 owner).

Test files are self-contained per the convention established by tests/test_quant.py
and tests/test_calibration.py - copy fixture helpers verbatim, do NOT cross-import
from tests/test_quant.py.

Determinism: tests/conftest.py (Plan 01-01) provides session-wide seeds + cudnn flags.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import soundfile as sf
import torch
from datasets import Dataset, DatasetDict

from common.env import AttrDict
from nsnet2.model import NSNet2
from nsnet2.export_onnx import export_streaming
from nsnet2.quant import quantize_checkpoint
import nsnet2.quant as quant_mod


# Repo root resolved once so subprocess-spawn tests don't depend on pytest invocation cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------
# Helper builders ported from tests/test_quant.py (do NOT cross-import).
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


def _random_audio(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n_samples).astype(np.float32)


def _redirect_hf_cache(monkeypatch, tmp_path: Path):
    """Point datasets_config.HF_DATASETS_CACHE at tmp_path so vbd-hashes/ writes are isolated per test.

    Mirrors tests/test_quant.py:94-100. Required for any test that constructs the
    VBDCalibrationReader (i.e., goes through quantize_checkpoint).
    """
    from datasets import config as datasets_config
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", str(tmp_path))
    # calibration.py imports `datasets_config` directly; patch that reference too.
    import nsnet2.calibration as calibration_mod
    monkeypatch.setattr(calibration_mod.datasets_config, "HF_DATASETS_CACHE", str(tmp_path))


# ----------------------------------------------------------------------------
# Phase 5 fixture: synthetic mini cp_dir with FP32 + int8 ONNX both present.
# ----------------------------------------------------------------------------

def _make_synthetic_cp_dir_with_int8(tmp_path: Path, monkeypatch) -> Path:
    """Build a synthetic cp_dir with: g_best (PyTorch ckpt), config.json, g_best_fp32.onnx, g_best.onnx (int8).

    Pipeline:
        1. Build tiny NSNet2(h) with normal_(0, 0.5) bias initialization.
        2. torch.save({"generator": ...}) and write config.json.
        3. export_streaming(...) -> writes g_best_fp32.onnx (Phase 2 D-13).
        4. quantize_checkpoint(cp_dir, n_calib_utts=3) -> writes g_best.onnx (int8) with metadata stamps.

    Returns the cp_dir Path.

    DET-02 invariant on this test file: only torch.save (writing). The torch.load happens
    INSIDE quant.quantize_checkpoint via NSNet2Streaming.from_checkpoint (Phase 2 D-09 owner).

    Mirrors tests/test_quant.py:_make_synthetic_cp_dir + adds the int8 quantize step.
    """
    _redirect_hf_cache(monkeypatch, tmp_path)
    torch.manual_seed(0)
    h_dict = dict(_H_TINY_BASE)
    h = AttrDict(h_dict)
    base = NSNet2(h)
    # Mirror tests/test_quant.py:122-127: normal_(0, 0.5) on every GRU bias tensor.
    for k in range(base.gru.num_layers):
        for name in (f"bias_ih_l{k}", f"bias_hh_l{k}"):
            torch.nn.init.normal_(getattr(base.gru, name), 0.0, 0.5)
    base.eval()

    cp_dir = tmp_path / "cp_synthetic"
    cp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cp_dir / "g_best"
    config_path = cp_dir / "config.json"
    torch.save({"generator": base.state_dict()}, str(ckpt_path))
    with open(config_path, "w") as f:
        json.dump(h_dict, f)

    # Phase 2 FP32 ONNX export (writes cp_dir/g_best_fp32.onnx)
    export_streaming(str(ckpt_path))

    # Phase 4 int8 quantize: patch HF dataset to synthetic offline payload.
    train_audios = [_random_audio(2048, seed=i) for i in range(5)]
    test_audios  = [_random_audio(2048, seed=100 + i) for i in range(2)]
    ds = _make_dataset_dict(train_audios, test_audios)
    monkeypatch.setattr(quant_mod, "load_voicebank_demand", lambda cache_dir=None: ds)
    quantize_checkpoint(cp_dir, n_calib_utts=3)                            # writes g_best.onnx (int8)
    return cp_dir


# ----------------------------------------------------------------------------
# Test 1: argparse --help lists all 7 expected flags (D-19 #1 / INF-01).
# ----------------------------------------------------------------------------

def test_cli_help_lists_all_flags():
    """D-19 #1: argparse --help lists all seven expected flags.

    INF-01 SC-1: CLI mirrors inference.py's flag set + Phase 5 additions
    (--fp32_checkpoint_file per D-09, --log_dir per D-01, --max_utterances per Discretion).
    """
    result = subprocess.run(
        [sys.executable, "-m", "nsnet2.inference_onnx", "--help"],
        capture_output=True, text=True, check=True, cwd=str(REPO_ROOT),
    )
    out = result.stdout
    for flag in (
        "--checkpoint_file",
        "--input_noisy_wavs_dir",
        "--hf_cache_dir",
        "--output_dir",
        "--log_dir",
        "--fp32_checkpoint_file",
        "--max_utterances",
    ):
        assert flag in out, f"--help output missing {flag!r}; got:\n{out}"


# ----------------------------------------------------------------------------
# Test 2: enhance_one_utterance smoke (D-19 #2 / INF-01 / INF-03).
# ----------------------------------------------------------------------------

def test_enhance_smoke(monkeypatch, tmp_path):
    """D-19 #2: synthetic int8 ONNX + synthetic noisy waveform produce non-NaN audio + finite RTF."""
    cp_dir = _make_synthetic_cp_dir_with_int8(tmp_path, monkeypatch)
    import nsnet2.inference_onnx as inf_mod
    fp32_session = inf_mod._make_session(cp_dir / "g_best_fp32.onnx")
    int8_session = inf_mod._make_session(cp_dir / "g_best.onnx")
    with open(cp_dir / "config.json") as f:
        h = AttrDict(json.load(f))

    noisy = _random_audio(8000, seed=42)                                   # 0.5s @ 16kHz
    fp32_audio, int8_audio, rtf_int8 = inf_mod.enhance_one_utterance(
        noisy, fp32_session, int8_session, h,
    )
    assert fp32_audio.shape == noisy.shape, (fp32_audio.shape, noisy.shape)
    assert int8_audio.shape == noisy.shape, (int8_audio.shape, noisy.shape)
    assert not np.isnan(fp32_audio).any(), "fp32_audio contains NaN"
    assert not np.isnan(int8_audio).any(), "int8_audio contains NaN"
    assert rtf_int8 > 0 and np.isfinite(rtf_int8), f"rtf_int8={rtf_int8} not positive-finite"


# ----------------------------------------------------------------------------
# Test 3: byte-identical reruns (D-19 #3 / INF-02 / Pitfall 8 closure).
# ----------------------------------------------------------------------------

def test_byte_identical_reruns(monkeypatch, tmp_path):
    """D-19 #3 / INF-02: two consecutive enhance_one_utterance() calls on same input -> byte-identical WAV.

    Pitfall 8 closure: states_in MUST reset to zeros INSIDE the per-utterance loop. If reset is
    above the loop (one-line oversight), the second call's first frame inherits tail state from
    the first call's last frame -> bytewise diverges from a fresh second call.
    """
    cp_dir = _make_synthetic_cp_dir_with_int8(tmp_path, monkeypatch)
    import nsnet2.inference_onnx as inf_mod
    fp32_session = inf_mod._make_session(cp_dir / "g_best_fp32.onnx")
    int8_session = inf_mod._make_session(cp_dir / "g_best.onnx")
    with open(cp_dir / "config.json") as f:
        h = AttrDict(json.load(f))

    noisy = _random_audio(8000, seed=42)
    _, int8_a, _ = inf_mod.enhance_one_utterance(noisy, fp32_session, int8_session, h)
    _, int8_b, _ = inf_mod.enhance_one_utterance(noisy, fp32_session, int8_session, h)

    # Bytewise comparison via WAV file write (the production path; D-21 PCM_16).
    out_a = tmp_path / "a.wav"
    out_b = tmp_path / "b.wav"
    sf.write(str(out_a), int8_a, h.sampling_rate, "PCM_16")
    sf.write(str(out_b), int8_b, h.sampling_rate, "PCM_16")
    assert out_a.read_bytes() == out_b.read_bytes(), (
        "INF-02 FAIL: byte-identical reruns broken. Likely Pitfall 8: states_in initialized "
        "outside per-utterance loop. Check enhance_one_utterance / _run_session_frame_by_frame."
    )


# ----------------------------------------------------------------------------
# Test 4: missing int8 ONNX -> FileNotFoundError with quant.py fix-hint (D-19 #4 / D-11).
# ----------------------------------------------------------------------------

def test_missing_int8_fail_fast(tmp_path):
    """D-19 #4 / D-11: missing int8 ONNX -> FileNotFoundError with verbatim quant.py fix-hint."""
    cp_dir = tmp_path / "empty_cp"
    cp_dir.mkdir()
    int8_path = cp_dir / "g_best.onnx"

    import nsnet2.inference_onnx as inf_mod
    with pytest.raises(FileNotFoundError) as excinfo:
        inf_mod._validate_paths(int8_path, None)
    msg = str(excinfo.value)
    assert "int8 ONNX missing at" in msg, msg
    assert "uv run python -m nsnet2.quant --checkpoint_dir" in msg, msg
    assert "--num_utterances" in msg, msg


# ----------------------------------------------------------------------------
# Test 5: missing FP32 sibling -> FileNotFoundError with export_onnx.py fix-hint (D-19 #5 / D-12).
# ----------------------------------------------------------------------------

def test_missing_fp32_sibling_fail_fast(monkeypatch, tmp_path):
    """D-19 #5 / D-12: int8 present, FP32 sibling missing -> FileNotFoundError with export_onnx.py fix-hint.

    Builds full synthetic fixture (which writes both int8 and FP32 sibling), then deletes the
    FP32 sibling to exercise the FP32-missing branch.
    """
    cp_dir = _make_synthetic_cp_dir_with_int8(tmp_path, monkeypatch)
    (cp_dir / "g_best_fp32.onnx").unlink()                                 # remove FP32 sibling

    import nsnet2.inference_onnx as inf_mod
    with pytest.raises(FileNotFoundError) as excinfo:
        inf_mod._validate_paths(cp_dir / "g_best.onnx", None)
    msg = str(excinfo.value)
    assert "FP32 ONNX missing at" in msg, msg
    assert "uv run python -m nsnet2.export_onnx --checkpoint_file" in msg, msg


# ----------------------------------------------------------------------------
# Test 6: ORT version mismatch -> WARNING, never raises (D-19 #6 / INF-04 / D-14).
# ----------------------------------------------------------------------------

def test_ort_version_mismatch_warning(monkeypatch, capsys, tmp_path):
    """D-19 #6 / D-14 / INF-04 (Pitfall 9): runtime ORT version mismatch -> WARNING, never raises."""
    cp_dir = _make_synthetic_cp_dir_with_int8(tmp_path, monkeypatch)
    int8_path = cp_dir / "g_best.onnx"

    # Spoof runtime ORT version BEFORE calling _check_ort_version.
    # Dual-patch defense (RESEARCH "Code Examples" lines 619): patch both the imported `ort` and the
    # already-bound module-level reference inside inference_onnx (via inf_mod.ort.__version__).
    import onnxruntime as live_ort
    monkeypatch.setattr(live_ort, "__version__", "9.99.99")
    import nsnet2.inference_onnx as inf_mod
    monkeypatch.setattr(inf_mod.ort, "__version__", "9.99.99")

    inf_mod._check_ort_version(int8_path)                                  # never raises (D-14)

    captured = capsys.readouterr()
    assert "WARNING: int8 ONNX was quantized with ORT" in captured.out, captured.out
    assert "9.99.99" in captured.out, captured.out


# ----------------------------------------------------------------------------
# Test 7: missing metadata_props -> WARNING, never raises (D-19 #7 / INF-04 / D-15).
# ----------------------------------------------------------------------------

def test_missing_metadata_props_warning(monkeypatch, capsys, tmp_path):
    """D-19 #7 / D-15 / INF-04: int8 ONNX missing ort_quantize_version stamp -> WARNING, never raises."""
    cp_dir = _make_synthetic_cp_dir_with_int8(tmp_path, monkeypatch)
    int8_path = cp_dir / "g_best.onnx"

    # Strip the ort_quantize_version stamp via protobuf pop+append dance.
    # protobuf RepeatedField doesn't support direct list assignment.
    model = onnx.load(str(int8_path))
    new_props = [p for p in model.metadata_props if p.key != "ort_quantize_version"]
    while len(model.metadata_props) > 0:
        model.metadata_props.pop()
    for p in new_props:
        model.metadata_props.append(p)
    onnx.save(model, str(int8_path))

    import nsnet2.inference_onnx as inf_mod
    inf_mod._check_ort_version(int8_path)                                  # never raises (D-15)

    captured = capsys.readouterr()
    assert "WARNING: int8 ONNX has no ort_quantize_version" in captured.out, captured.out


# ----------------------------------------------------------------------------
# Test 8: PESQ sanity on cp_baseline (D-19 #8 / [skipif]-gated).
# ----------------------------------------------------------------------------

@pytest.mark.skipif(
    not Path("cp_baseline/g_best.onnx").exists(),
    reason="cp_baseline/g_best.onnx missing; run quant.py on cp_baseline first",
)
def test_pesq_sanity_on_cp_baseline():
    """D-19 #8 / PESQ sanity: real cp_baseline + first 3 HF test utterances -> plausible PESQ + finite delta.

    Mirrors tests/test_quant.py:test_size_ratio_under_30pct shape:
      - skipif on cp_baseline artifact presence
      - NO _redirect_hf_cache (uses developer's real HF cache; VBD is ~6 GiB - redirecting forces
        re-copy and fails on small /tmp). Matches the locked Phase 4 convention.
      - Wideband PESQ plausible range: 1.0..4.5 (real cp_baseline expected ~2.5-3.0 per RESEARCH).
    """
    int8_path = Path("cp_baseline/g_best.onnx")
    fp32_path = Path("cp_baseline/g_best_fp32.onnx")
    if not fp32_path.exists():
        pytest.skip(f"{fp32_path} missing")

    import nsnet2.inference_onnx as inf_mod
    with open("cp_baseline/config.json") as f:
        h = AttrDict(json.load(f))
    fp32_session = inf_mod._make_session(fp32_path)
    int8_session = inf_mod._make_session(int8_path)

    from common.dataset import load_voicebank_demand
    from common.metrics import eval_pesq
    hf = load_voicebank_demand()
    test_split = hf["test"].select(range(3))
    fp32_scores, int8_scores = [], []
    for item in test_split:
        noisy = np.asarray(item["noisy"]["array"], dtype=np.float32)
        clean = np.asarray(item["clean"]["array"], dtype=np.float32)
        fp32_audio, int8_audio, _ = inf_mod.enhance_one_utterance(
            noisy, fp32_session, int8_session, h,
        )
        fp32_scores.append(eval_pesq(clean, fp32_audio, h.sampling_rate))
        int8_scores.append(eval_pesq(clean, int8_audio, h.sampling_rate))
    # Plausible-range gates (synthetic-noise PESQ would be nonsense; cp_baseline is real).
    for s in fp32_scores:
        assert s == -1 or 1.0 <= s <= 4.5, f"FP32 PESQ {s} outside plausible range [1.0, 4.5]"
    for s in int8_scores:
        assert s == -1 or 1.0 <= s <= 4.5, f"int8 PESQ {s} outside plausible range [1.0, 4.5]"
    deltas = [
        fp - i8 for fp, i8 in zip(fp32_scores, int8_scores)
        if fp != -1 and i8 != -1
    ]
    for d in deltas:
        assert np.isfinite(d), f"PESQ delta {d} not finite"


# ----------------------------------------------------------------------------
# Test 9 (gap closure): end-to-end CLI fail-fast with D-11 fix-hint
# (Plan 05-03 -- closes Phase 5 UAT gap; complements helper-only test #4).
# ----------------------------------------------------------------------------

def test_missing_int8_fail_fast_e2e_cli(tmp_path):
    """Plan 05-03 / D-11 + D-13 (UAT gap closure): missing int8 ONNX (no config.json sibling)
    -> CLI exits non-zero AND captured stderr contains the verbatim quant.py fix-hint substring.

    Why this test exists:
      The pre-existing test_missing_int8_fail_fast (test #4 above) calls _validate_paths(...)
      directly against a tmp_path-built empty cp_dir, bypassing main()'s D-13 sibling-config.json
      gate. UAT (test #5 in 05-UAT.md) reported the user-CLI scenario where BOTH the int8 ONNX
      AND the sibling config.json are missing -- main()'s D-13 check trips FIRST and (pre-fix)
      raised a corrupted-dir message WITHOUT the D-11 fix-hint substring. This test locks the
      end-to-end CLI contract: regardless of which gate trips first, the D-11 fix-hint substring
      must surface to the user.

      Defense-in-depth: this test does NOT replace test_missing_int8_fail_fast (which still
      exercises _validate_paths in isolation). Both tests fire on every run.

    Mirrors the subprocess.run shape from test_cli_help_lists_all_flags (test #1 above).
    """
    bogus_path = tmp_path / "does-not-exist.onnx"                          # parent dir has no config.json sibling
    assert not bogus_path.exists(), bogus_path
    assert not (tmp_path / "config.json").exists(), tmp_path

    result = subprocess.run(
        [sys.executable, "-m", "nsnet2.inference_onnx",
         "--checkpoint_file", str(bogus_path)],
        capture_output=True, text=True, check=False,                       # EXPECT non-zero exit
        cwd=str(REPO_ROOT),                                                # match the documented invocation
    )

    assert result.returncode != 0, (
        f"CLI returned 0 unexpectedly. stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "uv run python -m nsnet2.quant --checkpoint_dir" in result.stderr, (
        f"D-11 fix-hint substring missing from stderr. stderr:\n{result.stderr}"
    )
    assert "sibling config.json missing at" in result.stderr, (
        f"D-13 diagnostic substring missing from stderr (augmented-message contract broken). "
        f"stderr:\n{result.stderr}"
    )
