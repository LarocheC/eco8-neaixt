"""FP32 ONNX export for ConvFSENetStreamingFast.

Mirrors export_onnx.py (NSNet2's) shape: Python function (`export_streaming_fp32`)
+ argparse main. Independent of NSNet2's export path — ConvFSENet has a
different state shape and IO contract (real-valued mask + per-block FIFO
buffers, no GRU hidden state).

The ONNX graph contains only Conv / Add / Relu / Sigmoid / Gather / Concat
primitives — no BatchNorm (folded into preceding Conv at trace time), no
opaque RNN nodes, no complex IO. The complex STFT / iSTFT and the
`stft_t * mask` multiply live in the Python wrapper around the session.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx
import torch

from convfsenet.model import ConvFSENet_QuantFriendly
from common.env import AttrDict
from convfsenet.streaming import (
    ConvFSENetStreamingFast,
    ConvFSENetStreamingONNX,
)


def export_streaming_fp32(
    base: ConvFSENet_QuantFriendly,
    output_path,
    batch_size: int = 1,
) -> Path:
    """Export the BN-folded streaming view of `base` to FP32 ONNX.

    base         : ConvFSENet_QuantFriendly with causal=True. Will be put in
                   eval() before export.
    output_path  : .onnx target file.
    batch_size   : example batch size for tracing. The exported graph has a
                   dynamic batch axis on every IO tensor, so any B works at
                   runtime; this only controls the example-shape used at trace.

    Returns the resolved output path (pathlib.Path).
    """
    base.eval()
    fast = ConvFSENetStreamingFast(base).eval()
    onnx_view = ConvFSENetStreamingONNX(fast).eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Example inputs — concrete shapes for the trace, dynamic_axes drops the
    # batch dim to a symbol at export.
    F = onnx_view.n_features
    mag_in = torch.zeros(batch_size, F, dtype=torch.float32)
    states_in = onnx_view.init_states(batch_size, "cpu", dtype=torch.float32)
    args = (mag_in, *states_in)

    input_names = ["noisy_mag"] + onnx_view.state_input_names
    output_names = ["mask"] + onnx_view.state_output_names
    # Only the batch axis is dynamic. Buffer widths (per-block (K-1)*D) are
    # static in the graph by design — the runtime always feeds exactly the
    # state shape the trace saw, just with a variable B.
    dynamic_axes = {name: {0: "B"} for name in (input_names + output_names)}

    torch.onnx.export(
        onnx_view,
        args,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,                         # matches NSNet2 export_onnx.py
        dynamo=False,                             # MANDATORY — Pitfall 10 closure
    )

    # Telemetry — mirrors export_onnx.py D-16 print shape.
    model = onnx.load(str(output_path))
    op_hist = Counter(n.op_type for n in model.graph.node)
    sorted_ops = dict(sorted(op_hist.items(), key=lambda kv: -kv[1]))
    size_mib = output_path.stat().st_size / (1024 * 1024)
    n_opaque_rnn = op_hist.get("GRU", 0) + op_hist.get("LSTM", 0) + op_hist.get("RNN", 0)
    n_bn = op_hist.get("BatchNormalization", 0)
    print(f"Exported FP32 ONNX to {output_path}")
    print(f"  size : {size_mib:.3f} MiB")
    print(f"  nodes: {len(model.graph.node)}")
    print(f"  ops  : {sorted_ops}")
    print(f"  opaque RNN: {n_opaque_rnn} (expected 0 — none in this model)")
    print(f"  BatchNorm : {n_bn} (expected 0 — folded at trace time)")
    return output_path


def _load_offline_from_checkpoint(checkpoint_file) -> ConvFSENet_QuantFriendly:
    """Load a trained ConvFSENet_QuantFriendly from a g_best PyTorch checkpoint.

    Reads sibling config.json next to the checkpoint, exactly like
    inference.py / export_onnx.py.
    """
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
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Export ConvFSENet streaming view (BN-folded) to FP32 ONNX."
    )
    parser.add_argument("--checkpoint_file", required=True,
                        help="Path to checkpoint file (e.g., cp_convfsenet/g_best). "
                             "Sibling config.json is auto-loaded.")
    parser.add_argument("--output", default=None,
                        help="Output .onnx path. Defaults to <ckpt_dir>/g_best_fp32.onnx.")
    a = parser.parse_args()
    base = _load_offline_from_checkpoint(a.checkpoint_file)
    out = Path(a.output) if a.output else Path(a.checkpoint_file).parent / "g_best_fp32.onnx"
    export_streaming_fp32(base, out)


if __name__ == "__main__":
    main()
