"""FP32 ONNX export for LiSenNet's mask sub-network.

Exports the deployable core — the 3-channel feature map ``feat`` (compressed
magnitude + group delay + instantaneous-frequency difference) -> enhanced
magnitude ``est_mag`` — to a portable ONNX graph. As in ``basenet/export_onnx.py``
the STFT, feature extraction and phase recovery (Griffin-Lim offline / noisy
phase for real-time) live *outside* the graph, so the exported model is pure
tensor ops that ONNX Runtime and edge toolchains accept. Batch and time axes are
dynamic; the frequency axis is fixed (n_freqs = n_fft // 2 + 1).

The graph folds in the magnitude combination (``apply_mask``): because the
network input's channel 0 *is* the compressed noisy magnitude, ``est_mag`` is
computed inside the graph from ``feat`` alone, giving a single clean output. The
recurrent DPR GRUs and the sub-band (transposed) convs survive export.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx
import torch
from torch import nn

from lisennet.model import LiSenNet, build_lisennet
from common.env import AttrDict


class LiSenNetONNX(nn.Module):
    """Graph IO wrapper: feat (B, 3, T, F) -> est_mag (B, T, F).

    Runs the mask sub-network and applies the magnitude combination, reusing the
    noisy magnitude carried as feature channel 0 — so consumers only feed the
    3-channel feature map and get the enhanced magnitude back.
    """

    def __init__(self, model: LiSenNet):
        super().__init__()
        self.model = model.eval()

    def forward(self, feat):                                # (B, 3, T, F)
        mask = self.model.predict_mask(feat)                # (B, 2, T, F)
        src_mag = feat[:, 0]                                # (B, T, F)  == compressed noisy mag
        return self.model.apply_mask(mask, src_mag)         # (B, T, F)


def export_fp32(model: LiSenNet, output_path, batch_size: int = 1,
                n_frames: int = 32, opset: int = 17) -> Path:
    """Export `model`'s mask sub-network to FP32 ONNX with dynamic B and T axes.

    `n_frames` only sets the example length used to trace the graph; the dynamic
    time axis means any length runs at inference.
    """
    model.eval()
    wrapper = LiSenNetONNX(model).eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    f = model.n_freqs
    feat = torch.zeros(batch_size, 3, n_frames, f, dtype=torch.float32)
    dynamic_axes = {"feat": {0: "B", 2: "T"}, "est_mag": {0: "B", 1: "T"}}

    torch.onnx.export(
        wrapper, (feat,), str(output_path),
        input_names=["feat"], output_names=["est_mag"],
        dynamic_axes=dynamic_axes, opset_version=opset,
        dynamo=False,                                       # legacy tracer — stable for this graph
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


def _load_from_checkpoint(checkpoint_file) -> LiSenNet:
    """Load a trained LiSenNet from a g_best-style checkpoint + sibling config.json."""
    checkpoint_file = Path(checkpoint_file)
    with open(checkpoint_file.parent / "config.json") as f:
        h = AttrDict(json.load(f))
    model = build_lisennet(h)
    ckpt = torch.load(str(checkpoint_file), weights_only=True, map_location="cpu")
    model.load_state_dict(ckpt["generator"], strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Export LiSenNet mask sub-network to FP32 ONNX.")
    parser.add_argument("--checkpoint_file", required=True,
                        help="Path to a checkpoint (e.g. cp_lisennet/g_best). "
                             "Sibling config.json is auto-loaded.")
    parser.add_argument("--output", default=None,
                        help="Output .onnx path (default: <ckpt_dir>/g_best_fp32.onnx).")
    a = parser.parse_args()
    model = _load_from_checkpoint(a.checkpoint_file)
    out = Path(a.output) if a.output else Path(a.checkpoint_file).parent / "g_best_fp32.onnx"
    export_fp32(model, out)


if __name__ == "__main__":
    main()
