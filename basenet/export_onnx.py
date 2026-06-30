"""FP32 ONNX export for the causal BASENet generator.

Exports the spectral network — (compressed magnitude, phase) -> (enhanced
magnitude, enhanced phase) — to a portable ONNX graph. As in
``convfsenet/export_onnx.py`` the STFT / iSTFT live *outside* the graph, so the
exported model is pure tensor ops that ONNX Runtime and edge toolchains accept.
Batch and time axes are dynamic; the frequency axis is fixed (n_features).

The model is ONNX-clean by construction: frequency pooling/upsampling uses
``FreqResample`` (a constant matmul) instead of adaptive pooling / Resize, which
ONNX rejects when the output size is not constant. See ``basenet/model.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx
import torch
from torch import nn

from basenet.model import BASENet, build_basenet
from common.env import AttrDict


class BASENetONNX(nn.Module):
    """Graph IO wrapper: (mag, pha) -> (mag_g, pha_g).

    Drops BASENet's derived complex output so the exported graph has a minimal,
    well-named interface; consumers rebuild the complex spectrogram from the
    enhanced magnitude and phase before the iSTFT.
    """

    def __init__(self, model: BASENet):
        super().__init__()
        self.model = model.eval()

    def forward(self, mag, pha):
        mag_g, pha_g, _ = self.model(mag, pha)
        return mag_g, pha_g


def export_fp32(model: BASENet, output_path, batch_size: int = 1,
                n_frames: int = 32, opset: int = 17) -> Path:
    """Export `model` (a BASENet, causal) to FP32 ONNX with dynamic B and T axes.

    `n_frames` only sets the example length used to trace the graph; the dynamic
    time axis means any length runs at inference.
    """
    model.eval()
    wrapper = BASENetONNX(model).eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    f = model.n_features
    mag = torch.zeros(batch_size, f, n_frames, dtype=torch.float32)
    pha = torch.zeros(batch_size, f, n_frames, dtype=torch.float32)
    io_names = ["mag", "pha", "mag_g", "pha_g"]
    dynamic_axes = {name: {0: "B", 2: "T"} for name in io_names}

    torch.onnx.export(
        wrapper, (mag, pha), str(output_path),
        input_names=["mag", "pha"], output_names=["mag_g", "pha_g"],
        dynamic_axes=dynamic_axes, opset_version=opset,
        dynamo=False,                                   # legacy tracer — stable for this graph
    )

    model_proto = onnx.load(str(output_path))
    ops = Counter(n.op_type for n in model_proto.graph.node)
    sorted_ops = dict(sorted(ops.items(), key=lambda kv: -kv[1]))
    size_mib = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported FP32 ONNX to {output_path}")
    print(f"  size : {size_mib:.3f} MiB")
    print(f"  nodes: {len(model_proto.graph.node)}")
    print(f"  ops  : {sorted_ops}")
    return output_path


def _load_from_checkpoint(checkpoint_file) -> BASENet:
    """Load a trained BASENet from a g_best-style checkpoint + sibling config.json."""
    checkpoint_file = Path(checkpoint_file)
    with open(checkpoint_file.parent / "config.json") as f:
        h = AttrDict(json.load(f))
    model = build_basenet(h)
    ckpt = torch.load(str(checkpoint_file), weights_only=True, map_location="cpu")
    model.load_state_dict(ckpt["generator"], strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Export causal BASENet to FP32 ONNX.")
    parser.add_argument("--checkpoint_file", required=True,
                        help="Path to a checkpoint (e.g. cp_basenet_causal/g_best). "
                             "Sibling config.json is auto-loaded.")
    parser.add_argument("--output", default=None,
                        help="Output .onnx path (default: <ckpt_dir>/g_best_fp32.onnx).")
    a = parser.parse_args()
    model = _load_from_checkpoint(a.checkpoint_file)
    out = Path(a.output) if a.output else Path(a.checkpoint_file).parent / "g_best_fp32.onnx"
    export_fp32(model, out)


if __name__ == "__main__":
    main()
