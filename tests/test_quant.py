"""tests/test_quant.py - Phase 4 static int8 quantization gate (QNT-01..04 + variant boundary).

Seven pytest functions cover QNT-01..04 plus Plan 04-01's in-function audits
(D-12 erratum semantics: opaque-RNN-only Pitfall 11 audit) + the D-03 fail-fast +
the cp_baseline-skipif size-budget gate. Tests exercise quant.quantize_checkpoint
end-to-end on a synthetic mini cp_dir (n_fft=64, hidden_dim=16, num_gru_layers=2;
mirrors tests/test_calibration.py and tests/test_export_onnx.py fixtures), with
test 7 cp_baseline-gated for the real-scale size budget.

Test list (locked names per VALIDATION.md):
  1. test_quantize_checkpoint_smoke              - QNT-01 SC-1 / D-16 #1
  2. test_no_unquantized_matmul_or_gemm          - QNT-02 SC-2 / D-12 erratum (opaque-RNN audit)
  3. test_metadata_props_stamped                 - QNT-03 / D-09 / D-10
  4. test_calibration_method_minmax_default      - QNT-04 default dispatch
  5. test_calibration_method_percentile_via_h    - QNT-04 Percentile dispatch
  6. test_missing_fp32_onnx_fail_fast            - D-03 fail-fast
  7. test_size_ratio_under_30pct                 - QNT-02 real-scale (cp_baseline-skipif)

DET-02 sidestep preserved: zero deserialization-side load sites in this file. The
synthetic fixture uses torch.save (writing only). Actual checkpoint reading happens
INSIDE quant.quantize_checkpoint via NSNet2Streaming.from_checkpoint (Phase 2 D-09)
which is the DET-02-sanctioned read path.

HF cache + dataset isolation: every test that constructs a reader calls
_redirect_hf_cache(monkeypatch, tmp_path); tests 1-5 also monkey-patch
quant.load_voicebank_demand to return a synthetic DatasetDict (offline tests).

See .planning/phases/04-static-int8-quantization/04-CONTEXT.md (D-15 fixture
strategy + D-16 seven-test list + D-12 erratum 2026-04-29) and 04-VALIDATION.md
(per-task verification map - test names locked).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from datasets import Dataset, DatasetDict

from common.env import AttrDict
from nsnet2.model import NSNet2
from nsnet2.streaming import NSNet2Streaming
from nsnet2.export_onnx import export_streaming
from nsnet2.calibration import VBDCalibrationReader
from nsnet2.quant import quantize_checkpoint
import nsnet2.quant as quant_mod


# ----------------------------------------------------------------------------
# Helper builders ported from tests/test_calibration.py (do NOT cross-import).
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
    """Point datasets_config.HF_DATASETS_CACHE at tmp_path so vbd-hashes/ writes are isolated per test."""
    from datasets import config as datasets_config
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", str(tmp_path))
    # calibration.py imports `datasets_config` directly; patch that reference too.
    import nsnet2.calibration as calibration_mod
    monkeypatch.setattr(calibration_mod.datasets_config, "HF_DATASETS_CACHE", str(tmp_path))


# ----------------------------------------------------------------------------
# Phase 4 specific helpers (synthetic cp_dir + load_voicebank_demand patch).
# ----------------------------------------------------------------------------

def _make_synthetic_cp_dir(tmp_path: Path, h_overrides: dict | None = None) -> Path:
    """Build a synthetic cp_dir in tmp_path with g_best + config.json + g_best_fp32.onnx.

    Mirrors tests/test_export_onnx.py:synthetic_ckpt + invokes export_onnx.export_streaming
    (Phase 2 D-13 default path) so the resulting cp_dir is the exact shape Plan 04-01's
    quantize_checkpoint expects to find.

    h_overrides allows tests to inject h.quantization.calibration_method etc.
    Returns the cp_dir Path.
    """
    torch.manual_seed(0)
    h_dict = dict(_H_TINY_BASE)
    if h_overrides:
        h_dict.update(h_overrides)
    h = AttrDict(h_dict)
    base = NSNet2(h)
    # Mirror tests/test_export_onnx.py: normal_(0, 0.5) on every GRU bias tensor.
    # Non-zero biases give activation-range realism for calibration.
    for k in range(base.gru.num_layers):
        for name in (f"bias_ih_l{k}", f"bias_hh_l{k}"):
            torch.nn.init.normal_(getattr(base.gru, name), 0.0, 0.5)
    base.eval()

    cp_dir = tmp_path / "cp_synthetic"
    cp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cp_dir / "g_best"
    config_path = cp_dir / "config.json"
    # Save side only - DET-02 invariant on this test file forbids deserialization sites here;
    # the only checkpoint reader is NSNet2Streaming.from_checkpoint (Plan 02-01 / DET-02 owner).
    torch.save({"generator": base.state_dict()}, str(ckpt_path))
    with open(config_path, "w") as f:
        json.dump(h_dict, f)

    # Produce sibling g_best_fp32.onnx via the public export path (Phase 2 D-13 default).
    export_streaming(str(ckpt_path))   # writes cp_dir/g_best_fp32.onnx
    return cp_dir


def _patch_load_voicebank_demand(monkeypatch, train_audios, test_audios):
    """Patch quant.load_voicebank_demand to return a synthetic DatasetDict (offline tests)."""
    ds = _make_dataset_dict(train_audios, test_audios)
    monkeypatch.setattr(quant_mod, "load_voicebank_demand", lambda cache_dir=None: ds)
    return ds


def _patch_from_checkpoint_attrdict_nested(monkeypatch):
    """Wrap NSNet2Streaming.from_checkpoint so nested h.quantization / h.calibration are AttrDict.

    The codebase's AttrDict (env.py) is a flat wrapper - JSON-loaded nested dicts stay as
    plain dicts. quant.py's QNT-04 dispatch reads h.quantization via getattr(...) which only
    works when h.quantization is itself an AttrDict.

    Phase 6's in-training hook will construct h in-memory and pass it through with
    AttrDict-wrapped nested blocks naturally; this test-time wrapper exercises the same
    dispatch path on a JSON-loaded synthetic cp_dir without editing quant.py (Plan 04-02
    boundary: tests/test_quant.py only).

    Tests 4 and 5 (QNT-04 dispatch) need this; tests 1, 2, 3 do not (no nested blocks
    exercised); test 6 fails before from_checkpoint runs; test 7 cp_baseline has no
    h.quantization block in config.json.
    """
    original = NSNet2Streaming.from_checkpoint

    def _wrapped(path):
        streaming = original(path)
        h = streaming.base.h
        for key in ("quantization", "calibration"):
            val = getattr(h, key, None)
            if val is not None and not isinstance(val, AttrDict):
                setattr(h, key, AttrDict(val))
        return streaming

    monkeypatch.setattr(quant_mod, "NSNet2Streaming", type(
        "NSNet2StreamingAttrDictPatched", (), {"from_checkpoint": staticmethod(_wrapped)},
    ))


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def test_quantize_checkpoint_smoke(monkeypatch, tmp_path):
    """QNT-01 SC-1 / D-16 #1: end-to-end pipeline on synthetic cp_dir produces an int8 ONNX file.

    Load-bearing assertions:
      - the int8 file exists at <cp_dir>/g_best.onnx
      - its size > 0
      - quantize_checkpoint returns the matching Path
      - ORT session loads the int8 ONNX with CPUExecutionProvider and produces non-NaN outputs
    No size-ratio check at synthetic scale (D-14 - QDQ overhead exceeds bytes saved on this tiny
    model; Phase 3 baseline shows 49KB int8 vs 25KB fp32). Real-scale size budget is enforced
    by test_size_ratio_under_30pct (cp_baseline-gated).
    """
    _redirect_hf_cache(monkeypatch, tmp_path)
    cp_dir = _make_synthetic_cp_dir(tmp_path)
    train_audios = [_random_audio(2048, seed=i) for i in range(5)]
    test_audios  = [_random_audio(2048, seed=100 + i) for i in range(2)]
    _patch_load_voicebank_demand(monkeypatch, train_audios, test_audios)

    out_path = quantize_checkpoint(cp_dir, n_calib_utts=3)
    assert out_path == cp_dir / "g_best.onnx"
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Sanity: ORT can load the int8 ONNX with CPUExecutionProvider (QNT-02 SC-2 partial).
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        str(out_path), sess_options=so, providers=["CPUExecutionProvider"],
    )
    n_freq = _H_TINY_BASE["n_fft"] // 2 + 1                                  # F = 33
    streaming_for_dims = NSNet2Streaming.from_checkpoint(str(cp_dir / "g_best"))
    L = streaming_for_dims.num_layers                                        # L = 2
    H = streaming_for_dims.hidden_size                                       # H = 16
    frame = np.zeros((1, n_freq), dtype=np.float32)                          # (B=1, F)
    states = np.zeros((L, 1, H), dtype=np.float32)                           # (L, B=1, H)
    out = sess.run(["mask", "states_out"], {"frame_in": frame, "states_in": states})
    assert not np.isnan(out[0]).any(), "int8 mask contains NaN"
    assert not np.isnan(out[1]).any(), "int8 states_out contains NaN"


def test_no_unquantized_matmul_or_gemm(monkeypatch, tmp_path):
    """QNT-02 SC-2 / D-12 erratum 2026-04-29: the in-function Pitfall 11 audit MUST pass.

    D-12 erratum semantics: the audit detects opaque GRU/LSTM/RNN op survival, NOT raw
    Gemm/MatMul count. QNT-01's locked QuantFormat.QDQ inherently keeps linear ops as
    Gemm/MatMul wrapped in DequantizeLinear -> Gemm -> QuantizeLinear triples - those are
    NOT a Pitfall 11 hit (the original "zero MatMul + zero Gemm" reading was a
    QOperator-format misread mutually exclusive with QNT-01's locked QDQ format).
    See 04-CONTEXT.md D-12 erratum block for the full reasoning.

    Defense in depth: also assert at least one DequantizeLinear op is present, proving the
    QDQ pattern actually fired (i.e., quantization happened, not just a pass-through).
    """
    _redirect_hf_cache(monkeypatch, tmp_path)
    cp_dir = _make_synthetic_cp_dir(tmp_path)
    _patch_load_voicebank_demand(
        monkeypatch,
        [_random_audio(2048, seed=i) for i in range(5)],
        [_random_audio(2048, seed=100 + i) for i in range(2)],
    )

    out_path = quantize_checkpoint(cp_dir, n_calib_utts=3)

    model = onnx.load(str(out_path))
    # Pitfall 11 audit (D-12 erratum): opaque RNN ops only.
    opaque_rnn = [n for n in model.graph.node if n.op_type in ("GRU", "LSTM", "RNN")]
    assert not opaque_rnn, (
        f"Pitfall 11: {len(opaque_rnn)} opaque RNN op(s) survived "
        f"(types: {sorted({n.op_type for n in opaque_rnn})}). "
        f"quant.py in-function audit should have caught this before write. "
        f"Expected unrolled MatMul/Gemm primitives wrapped with Q/DQ pairs - "
        f"the streaming wrapper must replace nn.GRU before export (Phase 1 D-02)."
    )

    # Defense in depth: QDQ pattern fired - DequantizeLinear / QuantizeLinear ops exist.
    # In QDQ format, these wrap the surviving Gemm/MatMul ops; QGemm/MatMulInteger/QLinearMatMul
    # are QOperator-format markers and would NOT appear here (mutually exclusive with QDQ).
    qdq_ops = [n for n in model.graph.node if n.op_type in ("DequantizeLinear", "QuantizeLinear")]
    assert len(qdq_ops) > 0, (
        f"Expected DequantizeLinear / QuantizeLinear ops in QDQ-format int8 graph "
        f"(QGemm / MatMulInteger / QLinearMatMul are the QOperator-format equivalents - "
        f"NOT expected here per QNT-01's locked QuantFormat.QDQ). "
        f"op_types present = {sorted({n.op_type for n in model.graph.node})}"
    )


def test_metadata_props_stamped(monkeypatch, tmp_path):
    """QNT-03 / D-09 / D-10: metadata_props stamps three keys; calibration_hash matches Phase 3 reader.

    Computes the expected calibration_hash by constructing an independent
    VBDCalibrationReader against the same synthetic DatasetDict + h.calibration override
    quantize_checkpoint applies internally (D-07), then asserts the metadata stamp
    matches the reader's hash exactly. Single source of truth: Phase 3 D-13.
    """
    _redirect_hf_cache(monkeypatch, tmp_path)
    cp_dir = _make_synthetic_cp_dir(tmp_path)
    ds = _patch_load_voicebank_demand(
        monkeypatch,
        [_random_audio(2048, seed=i) for i in range(5)],
        [_random_audio(2048, seed=100 + i) for i in range(2)],
    )

    # Compute the expected calibration_hash by constructing a sibling reader with the same
    # h.calibration override Plan 04-01 applies internally (D-07).
    streaming = NSNet2Streaming.from_checkpoint(str(cp_dir / "g_best"))
    h = streaming.base.h
    cal_cfg = getattr(h, "calibration", AttrDict({}))
    h.calibration = AttrDict({**dict(cal_cfg), "num_utterances": 3})
    expected_reader = VBDCalibrationReader(streaming, h, ds)
    expected_hash = expected_reader.calibration_hash

    # Reset h.calibration so quantize_checkpoint computes its own (and they match).
    del h.calibration
    out_path = quantize_checkpoint(cp_dir, n_calib_utts=3)

    model = onnx.load(str(out_path))
    props = {p.key: p.value for p in model.metadata_props}
    assert "ort_quantize_version" in props, props
    assert "torch_version"        in props, props
    assert "calibration_hash"     in props, props
    assert props["ort_quantize_version"] == ort.__version__
    assert props["torch_version"]        == torch.__version__
    assert props["calibration_hash"]     == expected_hash, (
        f"metadata_props.calibration_hash = {props['calibration_hash']} "
        f"!= expected reader.calibration_hash = {expected_hash}"
    )
    assert len(props["calibration_hash"]) == 64, "calibration_hash must be sha256 hex (64 chars)"


def test_calibration_method_minmax_default(monkeypatch, tmp_path):
    """QNT-04 default dispatch: a config WITHOUT h.quantization uses MinMax silently.

    Spies on quant.quantize_static to capture the calibrate_method kwarg (and confirm
    nodes_to_exclude defaults to the empty list per QNT-04).
    """
    _redirect_hf_cache(monkeypatch, tmp_path)
    cp_dir = _make_synthetic_cp_dir(tmp_path)   # _H_TINY_BASE has NO h.quantization block
    _patch_load_voicebank_demand(
        monkeypatch,
        [_random_audio(2048, seed=i) for i in range(5)],
        [_random_audio(2048, seed=100 + i) for i in range(2)],
    )
    # Wrap from_checkpoint so any nested h.quantization / h.calibration is AttrDict
    # (codebase AttrDict is flat; quant.py's getattr-based dispatch needs nested AttrDict).
    # MUST be applied BEFORE the spy (the spy is on quant.quantize_static, downstream).
    _patch_from_checkpoint_attrdict_nested(monkeypatch)

    # Spy on quantize_static to capture calibrate_method (set up AFTER the cp_dir +
    # load_voicebank_demand patch so export_streaming inside _make_synthetic_cp_dir is
    # not affected by the spy).
    seen = {}
    from nsnet2.quant import quantize_static as orig_quantize_static
    def _spy(*args, **kwargs):
        seen["calibrate_method"] = kwargs.get("calibrate_method")
        seen["nodes_to_exclude"] = kwargs.get("nodes_to_exclude")
        return orig_quantize_static(*args, **kwargs)
    monkeypatch.setattr(quant_mod, "quantize_static", _spy)

    from onnxruntime.quantization import CalibrationMethod
    quantize_checkpoint(cp_dir, n_calib_utts=3)
    assert seen["calibrate_method"] == CalibrationMethod.MinMax, seen
    assert seen["nodes_to_exclude"] == [], seen   # default empty list per QNT-04


def test_calibration_method_percentile_via_h(monkeypatch, tmp_path):
    """QNT-04 dispatch: h.quantization.calibration_method='Percentile' -> CalibrationMethod.Percentile.

    Proves the QNT-04 sweep ergonomics: a sweep can tune calibration_method without code edits
    by setting h.quantization.calibration_method in cp_dir/config.json.
    """
    _redirect_hf_cache(monkeypatch, tmp_path)
    cp_dir = _make_synthetic_cp_dir(tmp_path, h_overrides={
        "quantization": {
            "calibration_method": "Percentile",
            "exclude_op_patterns": [],
        },
    })
    _patch_load_voicebank_demand(
        monkeypatch,
        [_random_audio(2048, seed=i) for i in range(5)],
        [_random_audio(2048, seed=100 + i) for i in range(2)],
    )
    # Wrap from_checkpoint so the JSON-loaded h.quantization (plain dict) is AttrDict
    # before quant.py's QNT-04 dispatch reads it via getattr.
    _patch_from_checkpoint_attrdict_nested(monkeypatch)

    seen = {}
    from nsnet2.quant import quantize_static as orig_quantize_static
    def _spy(*args, **kwargs):
        seen["calibrate_method"] = kwargs.get("calibrate_method")
        return orig_quantize_static(*args, **kwargs)
    monkeypatch.setattr(quant_mod, "quantize_static", _spy)

    from onnxruntime.quantization import CalibrationMethod
    quantize_checkpoint(cp_dir, n_calib_utts=3)
    assert seen["calibrate_method"] == CalibrationMethod.Percentile, seen


def test_missing_fp32_onnx_fail_fast(monkeypatch, tmp_path):
    """D-03: empty cp_dir (no g_best_fp32.onnx) -> FileNotFoundError with verbatim fix-command hint.

    No load_voicebank_demand patch needed - the FileNotFoundError fires before reader
    construction (D-03 is the first guard inside quantize_checkpoint).
    """
    _redirect_hf_cache(monkeypatch, tmp_path)
    cp_dir = tmp_path / "empty_cp"
    cp_dir.mkdir()

    with pytest.raises(FileNotFoundError) as excinfo:
        quantize_checkpoint(cp_dir, n_calib_utts=3)
    msg = str(excinfo.value)
    assert "FP32 ONNX missing at" in msg, msg
    assert "uv run python -m nsnet2.export_onnx --checkpoint_file" in msg, msg


def test_size_ratio_under_30pct(monkeypatch, tmp_path):
    """QNT-02 SC-2 real-scale size budget. Skipped if cp_baseline/g_best_fp32.onnx is missing.

    End-to-end run on the real trained checkpoint; the in-function size-budget audit
    (D-13/D-14, ratio <= 0.35) is the load-bearing assertion in quant.py - this test
    asserts the produced int8 is below the 30% target as ROADMAP SC-2 specifies.

    No load_voicebank_demand monkey-patch: this test exercises the real VBD path. The
    HF cache is redirected to tmp_path so the test does NOT pollute the developer's
    HF cache root.
    """
    cp_baseline = Path("cp_baseline")
    fp32 = cp_baseline / "g_best_fp32.onnx"
    if not fp32.exists():
        pytest.skip(f"cp_baseline/g_best_fp32.onnx missing; produce it via export_onnx.py first")

    # NOTE: deliberately NOT calling _redirect_hf_cache here. VBD is ~6 GiB; redirecting
    # HF_DATASETS_CACHE to tmp_path forces datasets.load_dataset to re-copy the parquet
    # files into tmp_path (not just point at them), which fails on small /tmp filesystems
    # (OSError: No space left on device). The developer's actual HF cache already has VBD
    # cached - using it directly is the offline-friendly path. The vbd-hashes/ JSON cache
    # written by VBDCalibrationReader is small (~1 KB) and lands in the actual HF cache
    # (per Phase 3 D-10 documentation) - non-polluting.

    out = tmp_path / "g_best_int8.onnx"
    quantize_checkpoint(cp_baseline, n_calib_utts=3, out_path=out)

    fp32_size = fp32.stat().st_size
    int8_size = out.stat().st_size
    ratio = int8_size / fp32_size
    assert fp32_size >= 1 * 1024 * 1024, "scale-aware threshold inactive at this size"
    assert ratio <= 0.30, (
        f"int8/fp32 ratio = {ratio:.3f} exceeds 0.30 (ROADMAP SC-2). "
        f"int8 = {int8_size} bytes; fp32 = {fp32_size} bytes."
    )
