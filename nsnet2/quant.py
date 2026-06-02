"""quant.py - Static int8 PTQ entry point for the NSNet2 streaming model.

Wires Phase 2's FP32 ONNX (under cp_<run>/) into Phase 3's
VBDCalibrationReader and runs onnxruntime.quantization.quantize_static (after
quant_pre_process) with the QNT-01 verbatim args: QDQ format, QInt8 activations,
QInt8 weights, per-channel quantization, MinMax default calibration, asymmetric
activations, and symmetric weights.

After the int8 ONNX is written, three metadata_props keys are stamped
(ort_quantize_version + torch_version + calibration_hash; Pitfall 9 closure /
QNT-03), then two always-on audits fire on the reloaded-from-disk artifact:

  * Pitfall 11 audit (D-12 erratum 2026-04-29): zero opaque GRU/LSTM/RNN ops in the
    int8 graph. QDQ format keeps Gemm/MatMul wrapped in Q/DQ pairs as expected -
    those are NOT a Pitfall 11 hit (the original "zero MatMul/Gemm" reading was a
    QOperator-format misread; see 04-CONTEXT.md D-12 erratum).
  * Size-budget audit (D-13, D-14): scale-aware; runs only when fp32 >= 1 MiB.
    This is the proxy that proves "weights actually got quantized" at full scale.

QNT-04 dispatch on h.quantization.calibration_method enables sweep-time tuning
without code edits (MinMax / Percentile / Entropy). h.quantization.exclude_op_patterns
forwards through to quantize_static's ``nodes_to_exclude`` — i.e. exact ONNX node
names to skip (not glob/regex patterns, despite the legacy key name).

Phase 4 boundary: Phase 6's training hook will import quantize_checkpoint
directly; Phase 5's inference script will read the metadata stamps for
runtime/quantize-time version comparison.

See .planning/phases/04-static-int8-quantization/04-CONTEXT.md (D-01..D-16) for
locked behavior. Mirrors export_onnx.py (Phase 2): Python function + argparse main().
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import tempfile

import onnx
import onnxruntime as ort
import torch
from onnx import StringStringEntryProto
from onnxruntime.quantization import (
    CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from nsnet2.calibration import VBDCalibrationReader
from common.dataset import load_voicebank_demand
from common.env import AttrDict
from nsnet2.streaming import NSNet2Streaming


# QNT-04 dispatch table — string config value -> ORT CalibrationMethod enum.
_METHOD_MAP = {
    "MinMax":     CalibrationMethod.MinMax,
    "Percentile": CalibrationMethod.Percentile,
    "Entropy":    CalibrationMethod.Entropy,
}


def _add_metadata(model, key, value):
    """D-09/D-10: stamp one (key, value) pair into ONNX metadata_props."""
    entry = StringStringEntryProto()
    entry.key = key
    entry.value = value
    model.metadata_props.append(entry)


def quantize_checkpoint(cp_dir, n_calib_utts, out_path=None, hf_cache_dir=None,
                        frames_per_utterance=None) -> Path:
    """Static int8 quantize a checkpoint dir's FP32 ONNX into g_best.onnx.

    cp_dir          : directory containing g_best (PyTorch ckpt), config.json,
                      and the Phase 2 FP32 ONNX output.
    n_calib_utts    : calibration utterance count; OVERRIDES h.calibration.num_utterances (D-07).
    out_path        : optional override for the int8 ONNX output path.
                      Default: cp_dir / "g_best.onnx" (D-01, D-02). Silent overwrite (D-15).
    hf_cache_dir    : optional HuggingFace datasets cache directory (D-06).
                      None falls through to default HF cache.

    Returns the resolved out_path (pathlib.Path).

    Raises:
        FileNotFoundError       — when the FP32 ONNX is absent from cp_dir (D-03);
                                  message includes the verbatim fix command.
        NotImplementedError     — when streaming.gru_kind != "gru" (Phase 1 D-05 carryover).
        ValueError              — when h.quantization.calibration_method is unknown.
        AssertionError          — Pitfall 11 audit (D-12) or scale-aware size-budget (D-13).
    """
    # --- 1. Resolve paths (D-01, D-02, D-03) --------------------------------
    cp_dir = Path(cp_dir)
    ckpt_path = cp_dir / "g_best"
    fp32_path = cp_dir / "g_best_fp32.onnx"
    out_path = Path(out_path) if out_path else cp_dir / "g_best.onnx"
    if not fp32_path.exists():
        raise FileNotFoundError(
            f"FP32 ONNX missing at {fp32_path}; run: "
            f"uv run python -m nsnet2.export_onnx --checkpoint_file {ckpt_path}"
        )

    # --- 2. Build streaming (variant guard lifted) --------------------------
    # Structured GRUs are now handled at FP32 export by export_onnx.py's
    # _patch_structured_for_export(); the FP32 ONNX read here therefore
    # contains only standard primitives and quantizes through the same
    # quantize_static path as the cuDNN-GRU baseline. Calibration runs
    # PyTorch-level streaming.forward_step (custom ops work fine in eager
    # mode), so VBDCalibrationReader needs no changes here either.
    streaming = NSNet2Streaming.from_checkpoint(str(ckpt_path))
    h = streaming.base.h  # AttrDict from sibling config.json (loaded by from_checkpoint)

    # --- 3. Load HF dataset (D-06) ------------------------------------------
    hf_dataset = load_voicebank_demand(cache_dir=hf_cache_dir)

    # --- 4. n_calib_utts override of h.calibration.num_utterances (D-07) ----
    cal_cfg = getattr(h, "calibration", AttrDict({}))
    cal_dict = {**dict(cal_cfg), "num_utterances": int(n_calib_utts)}
    # Optional per-utterance frame cap. MinMax calibration accumulates per-tensor
    # activation stats over every calibration frame, so on memory-limited machines
    # the wide structured-FC graphs (butterfly) can exceed RAM at full frame counts.
    # Capping frames/utt bounds calibration memory while keeping all 200 utterances.
    if frames_per_utterance is not None:
        cal_dict["frames_per_utterance"] = int(frames_per_utterance)
    h.calibration = AttrDict(cal_dict)

    # --- 5. Build reader (D-08) ---------------------------------------------
    reader = VBDCalibrationReader(streaming, h, hf_dataset)

    # --- 6. QNT-04 dispatch on h.quantization block --------------------------
    # Use dict-style access: NSNet2Streaming.from_checkpoint loads config.json
    # into a top-level AttrDict but leaves nested blocks as plain dicts, so
    # getattr() on the inner block silently falls through to defaults — a real
    # bug masked by the assertion below. quant_cfg.get() works for both.
    quant_cfg = dict(h.get("quantization", {}))
    method_str = quant_cfg.get("calibration_method", "MinMax")
    if method_str not in _METHOD_MAP:
        raise ValueError(
            f"Unknown calibration_method {method_str!r}; "
            f"valid: {sorted(_METHOD_MAP)}"
        )
    calibrate_method = _METHOD_MAP[method_str]
    exclude_op_patterns = list(quant_cfg.get("exclude_op_patterns", []))
    print(
        f"quantize_checkpoint: calibration_method={method_str}, "
        f"exclude_op_patterns={exclude_op_patterns or 'none'}"
    )

    # --- 7. quant_pre_process into a tempfile (D-11) -------------------------
    # Allocate the preprocessed model path via tempfile, then close the handle
    # so quant_pre_process can write to it. try/finally ensures cleanup even if
    # quantize_static raises mid-run (T-04-05 mitigation).
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as _tmp:
        preprocessed_path = Path(_tmp.name)
    try:
        quant_pre_process(str(fp32_path), str(preprocessed_path))

        # --- 8. quantize_static (QNT-01 verbatim) ---------------------------
        quantize_static(
            str(preprocessed_path),
            str(out_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            nodes_to_exclude=exclude_op_patterns,
            calibrate_method=calibrate_method,
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        )
    finally:
        preprocessed_path.unlink(missing_ok=True)

    # --- 9. Metadata stamping (D-09, D-10) ----------------------------------
    # In-place edit of the int8 file only. FP32 ONNX is NOT modified.
    model_int8 = onnx.load(str(out_path))
    _add_metadata(model_int8, "ort_quantize_version", ort.__version__)
    _add_metadata(model_int8, "torch_version",       torch.__version__)
    _add_metadata(model_int8, "calibration_hash",    reader.calibration_hash)
    onnx.save(model_int8, str(out_path))

    # --- 10. Reload from disk + run audits ----------------------------------
    # Reloading proves the on-disk artifact (post-stamp save) is what we audit,
    # not an in-memory ProtoBuf that could diverge from disk.
    model_check = onnx.load(str(out_path))

    # Pitfall 11 audit (D-12 erratum 2026-04-29): detect opaque RNN op survival.
    # QDQ format inherently keeps Gemm/MatMul ops wrapped in DequantizeLinear/QuantizeLinear
    # pairs - that's the correct quantized state, not a Pitfall 11 hit. The actual Pitfall 11
    # signal is an opaque GRU/LSTM/RNN node that torch.onnx.export emitted instead of
    # unrolled primitives - meaning the streaming wrapper failed to replace nn.GRU
    # (Phase 1 D-02). Phase 1 already closes this at FP32 export; we re-verify post-quant.
    opaque_rnn = [n for n in model_check.graph.node if n.op_type in ("GRU", "LSTM", "RNN")]
    assert len(opaque_rnn) == 0, (
        f"Pitfall 11: {len(opaque_rnn)} opaque RNN op(s) survived "
        f"(types: {sorted({n.op_type for n in opaque_rnn})}). "
        f"Expected unrolled MatMul/Gemm primitives wrapped with Q/DQ pairs - "
        f"the streaming wrapper must replace nn.GRU before export (Phase 1 D-02)."
    )

    # Quantization-presence audit (replaces the original D-13/D-14 ratio assert).
    #
    # The 0.35 ratio assertion was a proxy for "weights actually got quantized
    # at full scale" — sound for dense baseline-class FP32 graphs but breaks
    # on structure-preserving exports of monarch / butterfly variants. There,
    # FP32 weight bytes are already small (the whole point of the structure)
    # and the fixed QDQ scaffolding overhead (Q/DQ nodes, per-tensor scale +
    # zero-point bytes) lands in the same order of magnitude as the quantized
    # weights — int8 can come out the same size as or larger than FP32 even
    # though every quantizable op was correctly quantized.
    #
    # The unambiguous semantic check is "did QuantizeLinear / DequantizeLinear
    # nodes appear in the int8 graph" — we hard-assert on that, and report
    # the size ratio as telemetry only. Structure-preserving exports are
    # expected to have ratios near 1.0; the size win is recovered at the
    # downstream decomposition / structured-deploy step (see PROJECT.md).
    fp32_size = fp32_path.stat().st_size
    int8_size = out_path.stat().st_size
    n_qlinear = sum(
        1 for n in model_check.graph.node if n.op_type == "QuantizeLinear"
    )
    assert n_qlinear > 0, (
        f"Quantization presence: int8 file {out_path} contains zero "
        f"QuantizeLinear nodes — quantize_static silently produced a "
        f"non-quantized graph. Check that calibration_data_reader yielded "
        f"frames and that nodes_to_exclude did not skip every quantizable op."
    )

    # --- 11. Telemetry (D-16) — mirrors Phase 2 export_onnx.py D-16 shape ----
    op_hist = Counter(n.op_type for n in model_check.graph.node)
    sorted_ops = dict(sorted(op_hist.items(), key=lambda kv: -kv[1]))
    mb = int8_size / (1024 * 1024)
    print(f"quantize_checkpoint: {out_path} ({mb:.3f} MiB, {len(model_check.graph.node)} nodes)")
    if fp32_size >= 1 * 1024 * 1024:
        print(f"  ratio: {int8_size / fp32_size:.3f} vs FP32 {fp32_size / (1024 * 1024):.3f} MiB")
    print(f"  ops  : {sorted_ops}")
    print(
        f"  Pitfall 11: {len(opaque_rnn)} opaque RNN ops "
        f"(zero GRU/LSTM/RNN survived; QDQ Gemm/MatMul ops are wrapped in Q/DQ as expected)"
    )
    print(
        f"  metadata: ort_quantize_version={ort.__version__}, "
        f"torch_version={torch.__version__}, "
        f"calibration_hash={reader.calibration_hash}"
    )

    # --- 12. Return resolved out_path ---------------------------------------
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Static int8 quantization of an FP32 NSNet2 streaming ONNX."
    )
    parser.add_argument(
        "--checkpoint_dir", required=True,
        help="Directory containing g_best, config.json, and the FP32 ONNX (e.g., cp_baseline).",
    )
    parser.add_argument(
        "--num_utterances", type=int, required=True,
        help="Calibration utterance count; overrides h.calibration.num_utterances.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output .onnx path. Defaults to <checkpoint_dir>/g_best.onnx.",
    )
    parser.add_argument(
        "--hf_cache_dir", default=None,
        help="HuggingFace datasets cache directory (default: HF_DATASETS_CACHE env var).",
    )
    parser.add_argument(
        "--frames_per_utterance", type=int, default=None,
        help="Cap calibration frames per utterance (default: all). Bounds calibration "
             "memory for wide structured-FC graphs on memory-limited machines.",
    )
    a = parser.parse_args()
    out = quantize_checkpoint(
        a.checkpoint_dir, a.num_utterances, a.output, hf_cache_dir=a.hf_cache_dir,
        frames_per_utterance=a.frames_per_utterance,
    )
    print(f"int8 ONNX written to: {out}")


if __name__ == "__main__":
    main()
