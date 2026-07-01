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


def export_streaming_fp32(model: LiSenNet, output_path, batch_size: int = 1,
                          opset: int = 17) -> Path:
    """Export the frame-by-frame streaming mask sub-network to FP32 ONNX.

    Graph IO (à la ConvFSENet's ``export_streaming_fp32``):
    ``feat (B,3,1,F)`` + N ``state_i_in`` -> ``est_mag (B,1,F)`` + N ``state_i_out``.
    Only the batch axis is dynamic; the FIFO buffer widths are static (Neural-ART
    wants fixed state shapes). Requires the conv bottleneck (``bottleneck="conv"``)
    — the exported graph has no GRU and no 2-axis ``LayerNormalization``, which is
    asserted here so a regression can't slip a Neural-ART blocker back in.
    """
    from lisennet.streaming import LiSenNetStreamingONNX      # local: avoids a cycle

    model.eval()
    view = LiSenNetStreamingONNX(model).eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feat = torch.zeros(batch_size, 3, 1, model.n_freqs, dtype=torch.float32)
    states = view.init_states(batch_size)
    args = (feat, *states)
    input_names = ["feat"] + view.state_input_names
    output_names = ["est_mag"] + view.state_output_names
    dynamic_axes = {name: {0: "B"} for name in input_names + output_names}

    torch.onnx.export(
        view, args, str(output_path),
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=opset,
        dynamo=False,                                       # legacy tracer — stable for this graph
    )

    model_proto = onnx.load(str(output_path))
    ops = Counter(n.op_type for n in model_proto.graph.node)
    # Always-forbidden: recurrent ops + the 2-axis LayerNorm primitive (crash / not-mappable).
    forbidden = ["GRU", "LSTM", "RNN", "LayerNormalization"]
    # A deploy-hardened model (no PReLU/Mish modules) must also export none of their ops —
    # PRelu's per-channel float slope and Mish's Softplus block full int8 / the NPU. The
    # default (PReLU/Mish) conv model keeps the lenient check so its export is unaffected.
    hardened = not any(isinstance(mm, (nn.PReLU, nn.Mish)) for mm in model.modules())
    if hardened:
        forbidden += ["PRelu", "Softplus"]
    blockers = {op: ops[op] for op in forbidden if op in ops}
    assert not blockers, f"Neural-ART blocker op(s) survived export: {blockers}"
    sorted_ops = dict(sorted(ops.items(), key=lambda kv: -kv[1]))
    size_mib = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported streaming FP32 ONNX to {output_path}")
    print(f"  size   : {size_mib:.3f} MiB")
    print(f"  states : {view.n_states}  (feat + {view.n_states} state tensors -> est_mag + {view.n_states} states)")
    print(f"  nodes  : {len(model_proto.graph.node)}  (no GRU / LayerNormalization)")
    print(f"  ops    : {sorted_ops}")
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
                        help="Output .onnx path (default: <ckpt_dir>/g_best_fp32.onnx, "
                             "or g_best_streaming_fp32.onnx with --streaming).")
    parser.add_argument("--streaming", action="store_true",
                        help="Export the frame-by-frame graph with explicit state I/O "
                             "(requires bottleneck='conv') instead of the whole-utterance graph.")
    a = parser.parse_args()
    model = _load_from_checkpoint(a.checkpoint_file)
    default_name = "g_best_streaming_fp32.onnx" if a.streaming else "g_best_fp32.onnx"
    out = Path(a.output) if a.output else Path(a.checkpoint_file).parent / default_name
    if a.streaming:
        export_streaming_fp32(model, out)
    else:
        export_fp32(model, out)


if __name__ == "__main__":
    main()
