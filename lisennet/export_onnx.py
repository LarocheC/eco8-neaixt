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

from lisennet.model import DualPathConv, LiSenNet, build_lisennet
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


def _receptive_field(model) -> int:
    """Total causal time lookback (past frames the current output depends on) of the
    conv LiSenNet — the warmup length ``L`` a stateless windowed graph must prepend so
    its emitted frames are bit-exact to the offline model. Summed along the time-causal
    path: the encoder DSConvs, then each DPR block's dilated inter-time convs and the
    ConvolutionalGLU depthwise conv, then the decoder mask conv. (conv_1 is 1x1 and the
    USConv upsamplers have time-kernel 1 -> no time lookback.)"""
    L = 0
    enc = model.encoder
    for ds in (enc.conv_2, enc.conv_3, enc.conv_4):                  # DSConv: time kernel 2
        L += ds.low_conv[1].kernel_size[0] - 1
    if getattr(model, "bottleneck", "rnn") == "hybrid":
        raise ValueError(
            "windowed export is undefined for bottleneck='hybrid': its inter-time GRU has "
            "unbounded lookback, so no finite window reproduces the offline model (the conv "
            "bottleneck's finite receptive field is exactly what makes stateless recompute "
            "possible). Export the hybrid with --streaming — recurrence must carry state."
        )
    for blk in model.blocks:
        dpc = blk.dp_rnn_attn
        if not isinstance(dpc, DualPathConv):
            raise ValueError("windowed export requires bottleneck='conv' (DualPathConv)")
        for ct in dpc.inter:                                        # _CausalTimeConv: (kt-1)*dilation
            L += (ct.dw.kernel_size[0] - 1) * ct.dw.dilation[0]
        L += blk.conv_glu.dwconv[1].kernel_size[0] - 1              # ConvolutionalGLU depthwise (kt=3)
    L += model.decoder.mask_conv[1].kernel_size[0] - 1              # mask conv (kt=2)
    return L


class LiSenNetWindowedONNX(nn.Module):
    """Stateless windowed view: ``feat_window (B,3,L+T,F)`` -> ``est_mag (B,T,F)``.

    Carries NO FIFO state tensors — the Slice/Pad/Concat state I/O of the per-frame
    streaming export is what segfaults the Neural-ART compiler. Instead the app keeps a
    rolling ``L+T``-frame input window; the graph runs the offline causal mask over it
    and emits the last ``T`` frames, which have full in-window context and are bit-exact
    to the offline model (``L`` = receptive field). Same idea as ConvFSENet's
    ``ConvFSENetWindowedONNX``. Window and freq axes are static (Neural-ART wants fixed
    conv extents); only batch is dynamic.
    """

    def __init__(self, model, emit_T: int = 1):
        super().__init__()
        self.inner = LiSenNetONNX(model)
        self.L = _receptive_field(model)
        self.T = emit_T
        self.window_length = self.L + emit_T
        self.n_freqs = model.n_freqs

    def forward(self, feat_window):                         # (B, 3, L+T, F)
        est_mag = self.inner(feat_window)                   # (B, L+T, F)
        return est_mag[:, -self.T:, :]                      # (B, T, F)


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


def _strip_empty_pad_value_inputs(model_proto):
    """Drop the present-but-empty optional ``constant_value`` input from ``Pad`` nodes.

    ``F.pad`` (the streaming ring-buffer path) exports as ``Pad(data, pads, "")`` —
    a 3-input node whose third slot is the *empty* name. The Neural-ART compiler
    (atonn 4.0.1) segfaults (signo=11) on that form when the Pad sits in a
    multi-branch subgraph — this was the whole-graph blocker #4 of
    ``docs/targets/stm32n6-lisennet-npu.md``, isolated by bisection to exactly this. The
    offline/windowed graph is immune only because ``nn.ConstantPad2d`` exports an
    *explicit* constant_value tensor instead. Dropping the empty slot is bit-exact
    (the ONNX default constant_value is 0, which is what "" means).
    """
    n_fixed = 0
    for n in model_proto.graph.node:
        if n.op_type == "Pad" and len(n.input) == 3 and n.input[2] == "":
            del n.input[2]
            n_fixed += 1
    return n_fixed


def export_streaming_fp32(model: LiSenNet, output_path, batch_size: int = 1,
                          opset: int = 17) -> Path:
    """Export the frame-by-frame streaming mask sub-network to FP32 ONNX.

    Graph IO (à la ConvFSENet's ``export_streaming_fp32``):
    ``feat (B,3,1,F)`` + N ``state_i_in`` -> ``est_mag (B,1,F)`` + N ``state_i_out``.
    Only the batch axis is dynamic; the state widths are static (Neural-ART wants
    fixed state shapes).

    Works for both NPU bottlenecks — ``"conv"`` (all states are conv FIFOs) and
    ``"hybrid"`` (conv FIFOs plus one GRU hidden state per DPR block). In *both*
    cases the exported graph contains **no GRU op**: the hybrid's recurrence is
    traced through :class:`~lisennet.model.TimeGRU`'s single-step cell, which is 1x1
    convolutions over the hidden state, so recurrence reaches the NPU as
    Conv/Sigmoid/Tanh/Mul. That invariant, and the absence of 2-axis
    ``LayerNormalization``, are asserted below so a regression can't slip a
    Neural-ART blocker back in. The exported ``Pad`` nodes are normalized by
    :func:`_strip_empty_pad_value_inputs` (the atonn signo=11 fix); with that, this
    graph compiles to the NPU.
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
    n_pad_fixed = _strip_empty_pad_value_inputs(model_proto)
    if n_pad_fixed:
        onnx.save(model_proto, str(output_path))
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
    print(f"  pads   : {n_pad_fixed} empty Pad constant_value inputs stripped (atonn signo=11 fix)")
    return output_path


def export_windowed_fp32(model: LiSenNet, output_path, emit_T: int = 1,
                         batch_size: int = 1, opset: int = 17) -> Path:
    """Export the stateless-windowed mask sub-network to FP32 ONNX (the NPU deploy graph).

    Graph IO: ``feat_window (B,3,L+T,F)`` -> ``est_mag (B,T,F)``, where ``L`` is the
    receptive field (:func:`_receptive_field`) and ``T`` = ``emit_T``. No FIFO state I/O
    (that Slice/Pad/Concat class crashes atonn), so this is what maps to the Neural-ART
    NPU. Window/freq axes are static; only batch is dynamic. Requires the conv bottleneck.
    """
    model.eval()
    view = LiSenNetWindowedONNX(model, emit_T=emit_T).eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    W = view.window_length
    feat = torch.zeros(batch_size, 3, W, model.n_freqs, dtype=torch.float32)
    dynamic_axes = {"feat_window": {0: "B"}, "est_mag": {0: "B"}}

    torch.onnx.export(
        view, (feat,), str(output_path),
        input_names=["feat_window"], output_names=["est_mag"],
        dynamic_axes=dynamic_axes, opset_version=opset,
        dynamo=False,                                       # legacy tracer — stable for this graph
    )

    model_proto = onnx.load(str(output_path))
    # Stamp window geometry (L, T) so downstream tooling reads the contract exactly.
    for k, v in (("windowed_L", str(view.L)), ("windowed_T", str(view.T))):
        e = model_proto.metadata_props.add()
        e.key = k
        e.value = v
    onnx.save(model_proto, str(output_path))

    ops = Counter(n.op_type for n in model_proto.graph.node)
    # Same blocker guard as the streaming export: reject recurrent / 2-axis-norm ops,
    # and (for a hardened model) PReLU/Mish too. The windowed graph is also stateless.
    hardened = not any(isinstance(mm, (nn.PReLU, nn.Mish)) for mm in model.modules())
    forbidden = ["GRU", "LSTM", "RNN", "LayerNormalization"] + (["PRelu", "Softplus"] if hardened else [])
    blockers = {op: ops[op] for op in forbidden if op in ops}
    assert not blockers, f"Neural-ART blocker op(s) survived export: {blockers}"
    sorted_ops = dict(sorted(ops.items(), key=lambda kv: -kv[1]))
    size_mib = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported windowed FP32 ONNX to {output_path}")
    print(f"  window : L={view.L} + T={view.T} = {W} frames; n_freqs={model.n_freqs}")
    print(f"  size   : {size_mib:.3f} MiB")
    print(f"  nodes  : {len(model_proto.graph.node)}")
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
    parser.add_argument("--windowed", action="store_true",
                        help="Export the stateless windowed graph feat_window (B,3,L+T,F) -> "
                             "est_mag (B,T,F) — the NPU deploy target (no FIFO state, so it "
                             "maps to Neural-ART; requires bottleneck='conv').")
    parser.add_argument("--emit_T", type=int, default=1,
                        help="Windowed only: mask frames emitted per call (window = L + emit_T).")
    a = parser.parse_args()
    model = _load_from_checkpoint(a.checkpoint_file)
    if a.windowed:
        out = Path(a.output) if a.output else Path(a.checkpoint_file).parent / "g_best_windowed_fp32.onnx"
        export_windowed_fp32(model, out, emit_T=a.emit_T)
    elif a.streaming:
        out = Path(a.output) if a.output else Path(a.checkpoint_file).parent / "g_best_streaming_fp32.onnx"
        export_streaming_fp32(model, out)
    else:
        out = Path(a.output) if a.output else Path(a.checkpoint_file).parent / "g_best_fp32.onnx"
        export_fp32(model, out)


if __name__ == "__main__":
    main()
