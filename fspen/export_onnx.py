"""FP32 ONNX export for FSPEN.

Exports the deployable core — noisy complex spectrum in, enhanced complex
spectrum out — to a portable ONNX graph. As with the other families the STFT /
iSTFT stay *outside* the graph, so the exported model is pure tensor ops that
ONNX Runtime and edge toolchains accept. The magnitude branch input is derived
in-graph from the spectrum (``sqrt(re^2 + im^2)``), giving a single input.

Two views:

  * **whole-utterance** (``export_fp32``): ``spec (B, T, 2, F)`` ->
    ``est_spec (B, T, 2, F)`` with dynamic batch and time axes. The DPE inter
    GRUs export without an initial-state input (ONNX GRU defaults to zeros),
    exactly matching ``enhance_spectrum(..., hidden=None)``.
  * **frame-by-frame streaming** (``export_streaming_fp32``): ``spec
    (B, 1, 2, F)`` + one ``state_i_in`` per DPE block -> ``est_spec`` + updated
    states (the ``fspen/streaming.py`` contract). This is the real-time graph:
    state shapes are static apart from the batch-scaled axis.

Like LiSenNet's RNN variant, the exported graphs carry GRU nodes — fine for
ONNX Runtime; an NPU-mappable (GRU-free) variant would be a separate hardening
effort, as it was for LiSenNet.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx
import torch
from torch import nn

from fspen.model import FSPEN, build_fspen
from fspen.streaming import FSPENStreamingONNX, _amplitude
from common.env import AttrDict


class FSPENONNX(nn.Module):
    """Graph IO wrapper: spec (B, T, 2, F) -> est_spec (B, T, 2, F).

    Runs the enhancer with zero initial state (whole-utterance semantics); the
    magnitude branch input is computed inside the graph.
    """

    def __init__(self, model: FSPEN):
        super().__init__()
        self.model = model.eval()

    def forward(self, spec):                                # (B, T, 2, F)
        est, _ = self.model.enhance_spectrum(spec, _amplitude(spec))
        return est


def export_fp32(model: FSPEN, output_path, batch_size: int = 1,
                n_frames: int = 32, opset: int = 17) -> Path:
    """Export the whole-utterance enhancer to FP32 ONNX with dynamic B and T axes.

    `n_frames` only sets the example length used to trace the graph; the dynamic
    time axis means any length runs at inference.
    """
    model.eval()
    wrapper = FSPENONNX(model).eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = torch.zeros(batch_size, n_frames, 2, model.n_freqs, dtype=torch.float32)
    dynamic_axes = {"spec": {0: "B", 1: "T"}, "est_spec": {0: "B", 1: "T"}}

    torch.onnx.export(
        wrapper, (spec,), str(output_path),
        input_names=["spec"], output_names=["est_spec"],
        dynamic_axes=dynamic_axes, opset_version=opset,
        dynamo=False,                                       # legacy tracer — stable for this graph
    )

    model_proto = onnx.load(str(output_path))
    ops = Counter(n.op_type for n in model_proto.graph.node)
    # The dual-path GRUs are the model's core sequence modules: they must
    # survive export (rather than being silently dropped/decomposed wrong).
    assert "GRU" in ops, "DPE GRUs missing from the exported graph"
    sorted_ops = dict(sorted(ops.items(), key=lambda kv: -kv[1]))
    size_mib = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported FP32 ONNX to {output_path}")
    print(f"  size : {size_mib:.3f} MiB")
    print(f"  nodes: {len(model_proto.graph.node)}")
    print(f"  ops  : {sorted_ops}")
    return output_path


def export_streaming_fp32(model: FSPEN, output_path, batch_size: int = 1,
                          opset: int = 17) -> Path:
    """Export the frame-by-frame streaming enhancer to FP32 ONNX.

    Graph IO: ``spec (B,1,2,F)`` + one ``state_i_in`` per DPE block ->
    ``est_spec (B,1,2,F)`` + ``state_i_out``. The state tensors follow the
    ``FSPEN.init_hidden`` layout ``(groups, B*bands, width)``; their first axis
    is static, the batch-scaled second axis is dynamic.
    """
    model.eval()
    view = FSPENStreamingONNX(model).eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = torch.zeros(batch_size, 1, 2, model.n_freqs, dtype=torch.float32)
    states = view.init_states(batch_size)
    args = (spec, *states)
    input_names = ["spec"] + view.state_input_names
    output_names = ["est_spec"] + view.state_output_names
    dynamic_axes = {"spec": {0: "B"}, "est_spec": {0: "B"}}
    for name in view.state_input_names + view.state_output_names:
        dynamic_axes[name] = {1: "B_bands"}                 # = B * dpe_bands

    torch.onnx.export(
        view, args, str(output_path),
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=opset,
        dynamo=False,                                       # legacy tracer — stable for this graph
    )

    model_proto = onnx.load(str(output_path))
    ops = Counter(n.op_type for n in model_proto.graph.node)
    assert "GRU" in ops, "DPE GRUs missing from the exported graph"
    sorted_ops = dict(sorted(ops.items(), key=lambda kv: -kv[1]))
    size_mib = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported streaming FP32 ONNX to {output_path}")
    print(f"  size   : {size_mib:.3f} MiB")
    print(f"  states : {view.n_states}  (spec + {view.n_states} state tensors -> est_spec + {view.n_states} states)")
    print(f"  nodes  : {len(model_proto.graph.node)}")
    print(f"  ops    : {sorted_ops}")
    return output_path


def _load_from_checkpoint(checkpoint_file) -> FSPEN:
    """Load a trained FSPEN from a g_best-style checkpoint + sibling config.json."""
    checkpoint_file = Path(checkpoint_file)
    with open(checkpoint_file.parent / "config.json") as f:
        h = AttrDict(json.load(f))
    model = build_fspen(h)
    ckpt = torch.load(str(checkpoint_file), weights_only=True, map_location="cpu")
    model.load_state_dict(ckpt["generator"], strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Export FSPEN to FP32 ONNX.")
    parser.add_argument("--checkpoint_file", required=True,
                        help="Path to a checkpoint (e.g. cp_fspen/g_best). "
                             "Sibling config.json is auto-loaded.")
    parser.add_argument("--output", default=None,
                        help="Output .onnx path (default: <ckpt_dir>/g_best_fp32.onnx, "
                             "or g_best_streaming_fp32.onnx with --streaming).")
    parser.add_argument("--streaming", action="store_true",
                        help="Export the frame-by-frame graph with explicit state I/O "
                             "instead of the whole-utterance graph.")
    a = parser.parse_args()
    model = _load_from_checkpoint(a.checkpoint_file)
    if a.streaming:
        out = Path(a.output) if a.output else Path(a.checkpoint_file).parent / "g_best_streaming_fp32.onnx"
        export_streaming_fp32(model, out)
    else:
        out = Path(a.output) if a.output else Path(a.checkpoint_file).parent / "g_best_fp32.onnx"
        export_fp32(model, out)


if __name__ == "__main__":
    main()
