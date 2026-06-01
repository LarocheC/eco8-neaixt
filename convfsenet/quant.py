"""Static int8 PTQ for ConvFSENet streaming ONNX.

Mirrors quant.py (NSNet2) shape: load checkpoint → build streaming + ONNX
view → read sibling FP32 ONNX → quantize_static with QDQ format → stamp
metadata → audit → return path.

Hardened recipe (matches NSNet2's, proven loss-free on monarch variants):
  - QuantFormat.QDQ
  - QInt8 activations + weights
  - per_channel=True       (essential for the depthwise dconv)
  - ActivationSymmetric=False, WeightSymmetric=True
  - MinMax calibration (overridable via h.quantization.calibration_method)

ConvFSENet specifics:
  - 9 per-block FIFO state inputs (vs NSNet2's 1 GRU state) — handled
    transparently by ConvFSENetCalibrationReader.
  - Depthwise dconv (groups=C): per-channel quant applies one scale per
    output channel, which for a depthwise conv means one scale per group.
    ORT handles this correctly under QDQ + per_channel=True.
  - No GRU/LSTM/RNN, no BatchNorm (folded at FP32 export). Audited post-quant.

Two entry points:
  - quantize_fp32_onnx(...) : lower-level; takes a pre-existing FP32 ONNX
                              and pre-built fast/onnx_view/h/hf_dataset.
                              Easy to test without a real checkpoint.
  - quantize_checkpoint(...): high-level; loads checkpoint + sibling FP32
                              ONNX + HF dataset, then calls the lower one.
                              Mirrors quant.py:quantize_checkpoint CLI.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from onnx import StringStringEntryProto
from onnxruntime.quantization import (
    CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from convfsenet.calibration import ConvFSENetCalibrationReader
from convfsenet.model import ConvFSENet_QuantFriendly
from common.dataset import load_voicebank_demand
from common.env import AttrDict
from convfsenet.streaming import (
    ConvFSENetStreamingFast,
    ConvFSENetStreamingONNX,
)


_METHOD_MAP = {
    "MinMax":     CalibrationMethod.MinMax,
    "Percentile": CalibrationMethod.Percentile,
    "Entropy":    CalibrationMethod.Entropy,
}


def _add_metadata(model, key: str, value: str) -> None:
    entry = StringStringEntryProto()
    entry.key = key
    entry.value = value
    model.metadata_props.append(entry)


def _compression_prologue_nodes(model) -> list:
    """Node names of the magnitude-compression prologue — every op on the path
    from the real-magnitude graph input to (but not including) the first Conv.

    For a mag_compressed export that path is Add(+eps) -> Pow -> Unsqueeze; for
    a plain 'mag' export just Unsqueeze. These ops must run in FP32: the whole
    point of the (|stft|+eps)**0.3 compression is to squash speech's 60+ dB
    magnitude range into an int8-friendly one, so int8 quantization has to
    start AFTER it, on the compressed features at the frontend Conv. ORT's
    QDQ quantizer otherwise quantizes the eps-Add (a quantizable op), which
    drags the raw |stft| input onto a coarse int8 grid and throws away exactly
    the low-energy detail compression was added to preserve.

    Returns node names to feed quantize_static's nodes_to_exclude. Walk the
    graph AFTER quant_pre_process so the names match the quantized graph.
    """
    graph = model.graph
    data_inputs = [i.name for i in graph.input if not i.name.startswith("state_")]
    if not data_inputs:
        return []
    consumers = {}
    for n in graph.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)

    prologue, seen = [], set()
    cur = data_inputs[0]                       # the magnitude input (noisy_mag)
    while True:
        nxt = next(
            (n for n in consumers.get(cur, []) if n.op_type != "Conv"), None,
        )
        if nxt is None or id(nxt) in seen:
            break
        seen.add(id(nxt))
        if nxt.name:
            prologue.append(nxt.name)
        cur = nxt.output[0]
    return prologue


def quantize_fp32_onnx(
    fp32_path,
    int8_path,
    fast: ConvFSENetStreamingFast,
    onnx_view: ConvFSENetStreamingONNX,
    h: AttrDict,
    hf_dataset,
    calibration_method: str = "MinMax",
    exclude_op_patterns=None,
) -> Path:
    """Quantize an existing FP32 ONNX to int8 via ORT QDQ static PTQ.

    fp32_path / int8_path : input / output ONNX paths.
    fast / onnx_view      : PyTorch streaming wrappers used by the calibration
                            reader to propagate per-block state.
    h                     : AttrDict; reads `calibration` sub-block (num_utts,
                            seed, frames_per_utterance) and STFT params.
    hf_dataset            : VoiceBank-DEMAND DatasetDict (or any HF-shaped
                            object with ["train"] / ["test"] splits).
    calibration_method    : "MinMax" | "Percentile" | "Entropy". Default MinMax.
    exclude_op_patterns   : Optional iterable of exact ONNX node names to skip
                            (fed to quantize_static nodes_to_exclude; not
                            glob/regex patterns, despite the legacy key name).

    Returns the resolved int8_path.

    Raises:
        FileNotFoundError    — FP32 ONNX missing.
        ValueError           — unknown calibration_method.
        AssertionError       — post-quant audit failure (opaque RNN, BN
                               survival, or zero QuantizeLinear nodes).
    """
    fp32_path = Path(fp32_path)
    int8_path = Path(int8_path)
    if not fp32_path.exists():
        raise FileNotFoundError(
            f"FP32 ONNX missing at {fp32_path}; run: "
            f"python -m convfsenet.export_onnx --checkpoint_file <ckpt>"
        )
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    if calibration_method not in _METHOD_MAP:
        raise ValueError(
            f"Unknown calibration_method {calibration_method!r}; "
            f"valid: {sorted(_METHOD_MAP)}"
        )
    calibrate_method = _METHOD_MAP[calibration_method]
    exclude_list = list(exclude_op_patterns or [])

    reader = ConvFSENetCalibrationReader(fast, onnx_view, h, hf_dataset)
    print(
        f"quantize_fp32_onnx: calibration_method={calibration_method}, "
        f"exclude_op_patterns={exclude_list or 'none'}"
    )

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as _tmp:
        preprocessed_path = Path(_tmp.name)
    try:
        quant_pre_process(str(fp32_path), str(preprocessed_path))
        # Keep the magnitude-compression prologue in FP32 — int8 must start on
        # the compressed features, not the raw |stft| input (see
        # _compression_prologue_nodes).
        prologue = _compression_prologue_nodes(onnx.load(str(preprocessed_path)))
        if prologue:
            print(f"quantize_fp32_onnx: FP32 compression prologue (excluded): {prologue}")
            exclude_list = exclude_list + prologue
        quantize_static(
            str(preprocessed_path),
            str(int8_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            nodes_to_exclude=exclude_list,
            calibrate_method=calibrate_method,
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        )
    finally:
        preprocessed_path.unlink(missing_ok=True)

    # Stamp metadata (Pitfall 9 closure: detect runtime/quantize-time ORT
    # version skew at inference time; calibration_hash detects calibration
    # drift between train and deploy).
    model = onnx.load(str(int8_path))
    _add_metadata(model, "ort_quantize_version", ort.__version__)
    _add_metadata(model, "torch_version", torch.__version__)
    _add_metadata(model, "calibration_hash", reader.calibration_hash)
    onnx.save(model, str(int8_path))

    # Reload from disk + audit (proves the on-disk artifact is what we sign off).
    model_check = onnx.load(str(int8_path))
    op_hist = Counter(n.op_type for n in model_check.graph.node)

    opaque_rnn = [n for n in model_check.graph.node if n.op_type in ("GRU", "LSTM", "RNN")]
    assert len(opaque_rnn) == 0, (
        f"int8 ONNX contains {len(opaque_rnn)} opaque RNN op(s) "
        f"(types: {sorted({n.op_type for n in opaque_rnn})}); expected 0. "
        f"ConvFSENet has no recurrent layers — this is a tracing regression."
    )
    n_bn = op_hist.get("BatchNormalization", 0)
    assert n_bn == 0, (
        f"int8 ONNX contains {n_bn} BatchNormalization op(s); expected 0. "
        f"BN should have been folded into preceding Conv at FP32 export "
        f"(_fold_bn_into_conv in convfsenet/streaming.py)."
    )
    n_qlinear = op_hist.get("QuantizeLinear", 0)
    assert n_qlinear > 0, (
        f"int8 ONNX has zero QuantizeLinear nodes — quantize_static silently "
        f"produced an unquantized graph. Check that the CalibrationDataReader "
        f"yielded any frames and that exclude_op_patterns didn't skip every op."
    )

    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size
    sorted_ops = dict(sorted(op_hist.items(), key=lambda kv: -kv[1]))
    print(
        f"quantize_fp32_onnx: wrote {int8_path} "
        f"({int8_size / (1024 * 1024):.3f} MiB, {len(model_check.graph.node)} nodes)"
    )
    if fp32_size >= 1 * 1024 * 1024:
        print(
            f"  size ratio: {int8_size / fp32_size:.3f} "
            f"vs FP32 {fp32_size / (1024 * 1024):.3f} MiB"
        )
    print(f"  ops  : {sorted_ops}")
    print(f"  Q nodes: {n_qlinear}, BatchNorm: {n_bn}, opaque RNN: {len(opaque_rnn)}")
    print(
        f"  metadata: ort_quantize_version={ort.__version__}, "
        f"torch_version={torch.__version__}, "
        f"calibration_hash={reader.calibration_hash}"
    )

    return int8_path


def _load_offline_from_checkpoint(checkpoint_file: Path) -> tuple[ConvFSENet_QuantFriendly, AttrDict]:
    """Load a trained ConvFSENet_QuantFriendly + its config from a checkpoint dir."""
    checkpoint_file = Path(checkpoint_file)
    config_file = checkpoint_file.parent / "config.json"
    with open(config_file) as f:
        h = AttrDict(json.load(f))
    model = ConvFSENet_QuantFriendly(
        n_fft=h.n_fft, win_length=h.win_length, n_features=h.n_features,
        n_channels_res=h.n_channels_res, n_channels_conv=h.n_channels_conv,
        kernel_size=h.kernel_size, n_blocks=h.n_blocks, n_stacks=h.n_stacks,
        extractor_type=h.extractor_type, compress_factor=h.compress_factor,
        causal=h.causal,
        loss=None, preproc=None, postproc=None,
    )
    ckpt = torch.load(str(checkpoint_file), weights_only=True, map_location="cpu")
    model.load_state_dict(ckpt["generator"], strict=True)
    model.eval()
    return model, h


def quantize_checkpoint(
    cp_dir,
    n_calib_utts: int,
    out_path=None,
    hf_cache_dir=None,
    hf_dataset=None,
) -> Path:
    """Quantize a checkpoint dir's FP32 ONNX into g_best.onnx.

    cp_dir        : directory containing g_best, config.json, and the FP32 ONNX
                    output of convfsenet/export_onnx.py.
    n_calib_utts  : calibration utterance count; overrides h.calibration.num_utterances.
    out_path      : optional override for the int8 ONNX output path.
                    Default: <cp_dir>/g_best.onnx.
    hf_cache_dir  : optional HuggingFace datasets cache directory.
    hf_dataset    : optional pre-loaded HF dataset (test hook — skips
                    load_voicebank_demand when provided).

    Returns the resolved int8 ONNX path.
    """
    cp_dir = Path(cp_dir)
    ckpt_path = cp_dir / "g_best"
    fp32_path = cp_dir / "g_best_fp32.onnx"
    int8_path = Path(out_path) if out_path else cp_dir / "g_best.onnx"

    if not fp32_path.exists():
        raise FileNotFoundError(
            f"FP32 ONNX missing at {fp32_path}; run: "
            f"python -m convfsenet.export_onnx --checkpoint_file {ckpt_path}"
        )

    base, h = _load_offline_from_checkpoint(ckpt_path)

    # n_calib_utts overrides h.calibration.num_utterances.
    cal_cfg = getattr(h, "calibration", AttrDict({}))
    h.calibration = AttrDict({**dict(cal_cfg), "num_utterances": int(n_calib_utts)})

    fast = ConvFSENetStreamingFast(base).eval()
    onnx_view = ConvFSENetStreamingONNX(fast).eval()

    if hf_dataset is None:
        hf_dataset = load_voicebank_demand(cache_dir=hf_cache_dir)

    # Optional sweep-time tuning hooks.
    quant_cfg = dict(h.get("quantization", {})) if hasattr(h, "get") else {}
    method = quant_cfg.get("calibration_method", "MinMax")
    exclude = quant_cfg.get("exclude_op_patterns", [])

    return quantize_fp32_onnx(
        fp32_path, int8_path,
        fast, onnx_view, h, hf_dataset,
        calibration_method=method,
        exclude_op_patterns=exclude,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Static int8 quantization of an FP32 ConvFSENet streaming ONNX."
    )
    parser.add_argument(
        "--checkpoint_dir", required=True,
        help="Directory containing g_best, config.json, and g_best_fp32.onnx.",
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
        help="HuggingFace datasets cache directory (default: HF env var).",
    )
    a = parser.parse_args()
    out = quantize_checkpoint(
        a.checkpoint_dir, a.num_utterances, a.output, hf_cache_dir=a.hf_cache_dir,
    )
    print(f"int8 ONNX written to: {out}")


if __name__ == "__main__":
    main()
