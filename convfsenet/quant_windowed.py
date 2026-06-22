"""Static int8 PTQ for the stateless-windowed ConvFSENet ONNX (Track 1).

Same hardened QDQ recipe as convfsenet/quant.py (proven loss-free on the
streaming model), but for the windowed graph:

  - input is a single `noisy_mag_window` [B, n_freq, L+T] (no per-block state),
  - the calibration reader slides fixed windows instead of threading FIFO state,
  - the FP32 magnitude-compression prologue (Pow/Add for mag_compressed) is kept
    out of quantization exactly as in quant.py (reuses _compression_prologue_nodes),
  - a stricter post-quant audit asserts the rework's invariants: no opaque RNN,
    no BatchNorm, no Gather, no Pad, no state_* nodes.

Entry points mirror quant.py:
  - quantize_windowed_fp32_onnx(...) : low-level; pre-built onnx_view + FP32 ONNX.
  - quantize_windowed_checkpoint(...): high-level; loads checkpoint + sibling
                                       FP32 windowed ONNX + HF dataset.
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
from onnxruntime.quantization import (
    CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from convfsenet.calibration import ConvFSENetWindowedCalibrationReader
from convfsenet.model import ConvFSENet_QuantFriendly
from convfsenet.quant import _add_metadata, _compression_prologue_nodes
from convfsenet.streaming import ConvFSENetWindowedONNX
from common.dataset import load_voicebank_demand
from common.env import AttrDict


_METHOD_MAP = {
    "MinMax":     CalibrationMethod.MinMax,
    "Percentile": CalibrationMethod.Percentile,
    "Entropy":    CalibrationMethod.Entropy,
}


def quantize_windowed_fp32_onnx(
    fp32_path,
    int8_path,
    onnx_view: ConvFSENetWindowedONNX,
    h: AttrDict,
    hf_dataset,
    calibration_method: str = "MinMax",
    exclude_op_patterns=None,
) -> Path:
    """Quantize an existing windowed FP32 ONNX to int8 via ORT QDQ static PTQ.

    fp32_path / int8_path : input / output ONNX paths.
    onnx_view             : the ConvFSENetWindowedONNX used to build the FP32
                            graph (supplies L / T / drop_nyquist to the reader).
    h                     : AttrDict; reads `calibration` sub-block + STFT params.
    hf_dataset            : VoiceBank-DEMAND DatasetDict.
    calibration_method    : "MinMax" | "Percentile" | "Entropy". Default MinMax.
    exclude_op_patterns   : optional exact ONNX node names to skip.

    Returns the resolved int8_path.
    """
    fp32_path = Path(fp32_path)
    int8_path = Path(int8_path)
    if not fp32_path.exists():
        raise FileNotFoundError(
            f"windowed FP32 ONNX missing at {fp32_path}; run: "
            f"python -m convfsenet.export_onnx --windowed --checkpoint_file <ckpt>"
        )
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    if calibration_method not in _METHOD_MAP:
        raise ValueError(
            f"Unknown calibration_method {calibration_method!r}; valid: {sorted(_METHOD_MAP)}"
        )
    calibrate_method = _METHOD_MAP[calibration_method]
    exclude_list = list(exclude_op_patterns or [])

    reader = ConvFSENetWindowedCalibrationReader(onnx_view, h, hf_dataset)
    print(
        f"quantize_windowed_fp32_onnx: calibration_method={calibration_method}, "
        f"exclude_op_patterns={exclude_list or 'none'}"
    )

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as _tmp:
        preprocessed_path = Path(_tmp.name)
    try:
        quant_pre_process(str(fp32_path), str(preprocessed_path))
        # Keep the magnitude-compression prologue in FP32 (same rationale as
        # quant.py): int8 must start on the compressed features at the frontend
        # Conv, not on the raw |stft| window. _compression_prologue_nodes walks
        # from the data input (noisy_mag_window) to the first Conv.
        prologue = _compression_prologue_nodes(onnx.load(str(preprocessed_path)))
        if prologue:
            print(f"quantize_windowed_fp32_onnx: FP32 compression prologue (excluded): {prologue}")
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

    model = onnx.load(str(int8_path))
    # Replace-or-add: the FP32 export already stamps windowed_L/T/drop_nyquist,
    # and they propagate through quant; appending again would duplicate keys
    # (onnx.checker rejects duplicate metadata_props).
    def _set_meta(key, value):
        for i, p in enumerate(model.metadata_props):
            if p.key == key:
                p.value = value
                return
        _add_metadata(model, key, value)
    _set_meta("ort_quantize_version", ort.__version__)
    _set_meta("torch_version", torch.__version__)
    _set_meta("calibration_hash", reader.calibration_hash)
    _set_meta("windowed_L", str(onnx_view.L))
    _set_meta("windowed_T", str(onnx_view.T))
    _set_meta("drop_nyquist", str(int(onnx_view.drop_nyquist)))
    onnx.save(model, str(int8_path))

    # Reload + audit — the rework's invariants are a deploy gate, so enforce here.
    model_check = onnx.load(str(int8_path))
    op_hist = Counter(n.op_type for n in model_check.graph.node)

    opaque_rnn = [n for n in model_check.graph.node if n.op_type in ("GRU", "LSTM", "RNN")]
    assert len(opaque_rnn) == 0, (
        f"int8 windowed ONNX contains {len(opaque_rnn)} opaque RNN op(s); expected 0."
    )
    n_bn = op_hist.get("BatchNormalization", 0)
    assert n_bn == 0, (
        f"int8 windowed ONNX contains {n_bn} BatchNormalization op(s); expected 0 "
        f"(BN must fold into preceding Conv at FP32 export)."
    )
    n_gather = op_hist.get("Gather", 0)
    assert n_gather == 0, (
        f"int8 windowed ONNX contains {n_gather} Gather op(s); expected 0 — the "
        f"stateless-windowed rework removes the FIFO gather/state class entirely."
    )
    n_pad = op_hist.get("Pad", 0)
    assert n_pad == 0, (
        f"int8 windowed ONNX contains {n_pad} Pad op(s); expected 0 — valid "
        f"(padding-0) convs must introduce no Pad nodes."
    )
    n_state = sum(1 for n in model_check.graph.node if "state" in n.name.lower())
    assert n_state == 0, (
        f"int8 windowed ONNX contains {n_state} state-named node(s); expected 0 "
        f"(stateless graph)."
    )
    n_qlinear = op_hist.get("QuantizeLinear", 0)
    assert n_qlinear > 0, (
        f"int8 windowed ONNX has zero QuantizeLinear nodes — quantize_static "
        f"produced an unquantized graph. Check the calibration reader yielded windows."
    )

    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size
    sorted_ops = dict(sorted(op_hist.items(), key=lambda kv: -kv[1]))
    print(
        f"quantize_windowed_fp32_onnx: wrote {int8_path} "
        f"({int8_size / (1024 * 1024):.3f} MiB, {len(model_check.graph.node)} nodes)"
    )
    if fp32_size >= 1 * 1024 * 1024:
        print(f"  size ratio: {int8_size / fp32_size:.3f} vs FP32 {fp32_size / (1024 * 1024):.3f} MiB")
    print(f"  ops  : {sorted_ops}")
    print(f"  Q nodes: {n_qlinear}, Gather: {n_gather}, Pad: {n_pad}, "
          f"BatchNorm: {n_bn}, state nodes: {n_state}, opaque RNN: {len(opaque_rnn)}")
    print(
        f"  metadata: ort_quantize_version={ort.__version__}, "
        f"calibration_hash={reader.calibration_hash}, L={onnx_view.L}, T={onnx_view.T}, "
        f"drop_nyquist={onnx_view.drop_nyquist}"
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


def quantize_windowed_checkpoint(
    cp_dir,
    n_calib_utts: int,
    T: int = 1,
    drop_nyquist: bool = False,
    fp32_name: str = None,
    out_path=None,
    hf_cache_dir=None,
    hf_dataset=None,
    frames_per_utterance=None,
    calibration_method: str = "MinMax",
) -> Path:
    """Quantize a checkpoint dir's windowed FP32 ONNX into an int8 ONNX.

    cp_dir        : dir with g_best, config.json, and the windowed FP32 ONNX.
    n_calib_utts  : calibration utterance count.
    T             : emitted frames (must match the FP32 export).
    drop_nyquist  : 256-bin variant (must match the FP32 export).
    fp32_name     : windowed FP32 ONNX filename. Default derives from
                    T/drop_nyquist: g_best_win{256|257}_fp32.onnx.
    out_path      : int8 output. Default g_best_win{256|257}.onnx.
    """
    cp_dir = Path(cp_dir)
    ckpt_path = cp_dir / "g_best"
    bins = 256 if drop_nyquist else 257
    fp32_path = cp_dir / (fp32_name or f"g_best_win{bins}_fp32.onnx")
    int8_path = Path(out_path) if out_path else cp_dir / f"g_best_win{bins}.onnx"

    if not fp32_path.exists():
        raise FileNotFoundError(
            f"windowed FP32 ONNX missing at {fp32_path}; run: "
            f"python -m convfsenet.export_onnx --windowed --checkpoint_file {ckpt_path} "
            f"--emit_T {T}" + (" --drop_nyquist" if drop_nyquist else "")
        )

    base, h = _load_offline_from_checkpoint(ckpt_path)

    cal_cfg = getattr(h, "calibration", AttrDict({}))
    cal_dict = {**dict(cal_cfg), "num_utterances": int(n_calib_utts)}
    if frames_per_utterance is not None:
        cal_dict["frames_per_utterance"] = int(frames_per_utterance)
    h.calibration = AttrDict(cal_dict)

    onnx_view = ConvFSENetWindowedONNX(base, T=T, drop_nyquist=drop_nyquist).eval()

    if hf_dataset is None:
        hf_dataset = load_voicebank_demand(cache_dir=hf_cache_dir)

    return quantize_windowed_fp32_onnx(
        fp32_path, int8_path, onnx_view, h, hf_dataset,
        calibration_method=calibration_method,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Static int8 quantization of a stateless-windowed ConvFSENet ONNX."
    )
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Dir with g_best, config.json, and the windowed FP32 ONNX.")
    parser.add_argument("--num_utterances", type=int, required=True)
    parser.add_argument("--emit_T", type=int, default=1)
    parser.add_argument("--drop_nyquist", action="store_true")
    parser.add_argument("--fp32_name", default=None,
                        help="Windowed FP32 ONNX filename (default derives from T/bins).")
    parser.add_argument("--output", default=None)
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--frames_per_utterance", type=int, default=None,
                        help="Cap calibration windows per utterance (bounds memory).")
    parser.add_argument("--calibration_method", default="MinMax")
    a = parser.parse_args()
    out = quantize_windowed_checkpoint(
        a.checkpoint_dir, a.num_utterances, T=a.emit_T, drop_nyquist=a.drop_nyquist,
        fp32_name=a.fp32_name, out_path=a.output, hf_cache_dir=a.hf_cache_dir,
        frames_per_utterance=a.frames_per_utterance, calibration_method=a.calibration_method,
    )
    print(f"int8 windowed ONNX written to: {out}")


if __name__ == "__main__":
    main()
