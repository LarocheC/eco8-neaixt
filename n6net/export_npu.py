"""Export N6Net to the Neural-ART deploy form: one frame in, FIFO state threaded.

The deployed graph is the streaming step the architecture was designed around.
Frequency stays on the HEIGHT axis; the graph input is the power-law-compressed
magnitude of ONE frame, ``feat (1, 1, F, 1)`` (compression stays FP32 on the
host, as for ConvFSENet), and the output is its sigmoid ``mask (1, 1, F, 1)``.
Where the FIFO lives is the ``--layout`` (see ``N6NetStreamStep``):

    time_split (default)  inputs fifo_i_j (1, C, F, 1), outputs col_i (1, C, F, 1)
                          no Concat/Slice -> the whole net is one NPU EC blob
    time_w                inputs/outputs fifo_i (1, C, F, k_t-1); the design as
                          drawn, but Neural-ART runs its W-axis Concat and Slice
                          on the M55 (6 hybrid epochs at 3 blocks)
    time_c                inputs/outputs fifo_i (1, (k_t-1)*C, F, 1); channel
                          Slice goes HW, the Concat still does not

Pipeline: fp32 export -> ORT parity (streaming vs offline) -> static QDQ int8
(signed activations, per-channel weights, the Neural-ART recipe) calibrated on
VoiceBank-DEMAND frames with the *propagated* FIFO state -> NPU post-pass
(int8 PRelu slopes, FIFO qparams tied to the column they store).

    python -m n6net.export_npu --config configs/n6net_b3.json --out build/n6net_b3 \
        [--checkpoint cp_n6net/g_best] [--layout time_split]

then ``deploy/stm32n6/scripts/compile_n6net.sh build/n6net_b3/n6net_stream_int8.onnx``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from n6net.model import N6Net, build_causal_model, compress_magnitude, cost_summary


class N6NetStreamStep(nn.Module):
    """One streaming step of an N6Net, state as explicit tensors.

    ``layout`` picks where the FIFO lives. All three are the same arithmetic:

    * ``"time_w"`` — the design as written: state ``fifo_i (1, C, F, k_t-1)``,
      the window is a W-axis Concat, the temporal layer a 1 x k_t conv, and the
      graph returns the shifted state.
    * ``"time_c"`` — time folded into channels: state ``(1, (k_t-1)*C, F, 1)``,
      a channel-axis Concat, the temporal layer a 1 x 1 conv over ``k_t*C``
      inputs (576 at C=96), and a channel Slice for the shift.
    * ``"time_split"`` — no Concat and no Slice at all: each past column is its
      own input ``fifo_i_j (1, C, F, 1)`` (oldest first), the temporal layer is
      ``k_t`` 1 x 1 convs summed, and the graph returns only the block's new
      input column ``col_i``. The host owns the ring buffer: with compiler-
      allocated inputs that is a 370 kB int8 copy per frame at 3 blocks (the
      Concat it replaces copied as much on the M55); pointer rotation needs
      user-allocated input buffers.

    Neural-ART runs Concat on the M55 for both axes here, which is what the
    split layout exists to avoid.
    """

    LAYOUTS = ("time_w", "time_c", "time_split")

    def __init__(self, model: N6Net, layout: str = "time_w"):
        super().__init__()
        if layout not in self.LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}")
        self.m = model
        self.layout = layout
        C = model.channels
        if layout == "time_c":
            self.conv_t1 = nn.ModuleList()
            for b in model.blocks:
                w = b.conv_t.weight                             # (Co, Ci, 1, k_t)
                c = nn.Conv2d(w.shape[1] * b.k_t, w.shape[0], 1)
                # channel j*Ci + i of the folded window is input channel i at frame j
                c.weight.data = w.permute(0, 3, 1, 2).reshape(w.shape[0], -1, 1, 1).clone()
                c.bias.data = b.conv_t.bias.data.clone()
                self.conv_t1.append(c)
        elif layout == "time_split":
            self.conv_tj = nn.ModuleList()
            for b in model.blocks:
                w = b.conv_t.weight
                cols = nn.ModuleList()
                for j in range(b.k_t):
                    last = j == b.k_t - 1                       # bias rides on the current frame
                    c = nn.Conv2d(C, C, 1, bias=last)
                    c.weight.data = w[..., j:j + 1].clone()
                    if last:
                        c.bias.data = b.conv_t.bias.data.clone()
                    cols.append(c)
                self.conv_tj.append(cols)

    # -- I/O contract -------------------------------------------------------

    @property
    def input_names(self):
        if self.layout == "time_split":
            return ["feat"] + [f"fifo_{i}_{j}_in" for i, b in enumerate(self.m.blocks)
                               for j in range(b.fifo_frames)]
        return ["feat"] + [f"fifo_{i}_in" for i in range(len(self.m.blocks))]

    @property
    def output_names(self):
        if self.layout == "time_split":
            return ["mask"] + [f"col_{i}_out" for i in range(len(self.m.blocks))]
        return ["mask"] + [f"fifo_{i}_out" for i in range(len(self.m.blocks))]

    def init_states(self, n=1):
        F_, C = self.m.n_features, self.m.channels
        if self.layout == "time_split":
            return [torch.zeros(n, C, F_, 1) for b in self.m.blocks for _ in range(b.fifo_frames)]
        if self.layout == "time_c":
            return [torch.zeros(n, b.fifo_frames * C, F_, 1) for b in self.m.blocks]
        return [torch.zeros(n, C, F_, b.fifo_frames) for b in self.m.blocks]

    def advance(self, states, outs):
        """Next call's state from this call's state and non-mask outputs."""
        if self.layout != "time_split":
            return list(outs)
        nxt, k = [], 0
        for b, col in zip(self.m.blocks, outs):                 # ring buffer: drop oldest
            nxt += list(states[k + 1:k + b.fifo_frames]) + [col]
            k += b.fifo_frames
        return nxt

    def forward(self, feat, *states):
        x = self.m.stem(feat)                                   # (1, C, F, 1)
        C = self.m.channels
        extra, k = [], 0
        for i, blk in enumerate(self.m.blocks):
            if self.layout == "time_split":
                cols = list(states[k:k + blk.fifo_frames]) + [x]
                k += blk.fifo_frames
                extra.append(x)
                y = self.conv_tj[i][0](cols[0])
                for conv, col in zip(self.conv_tj[i][1:], cols[1:]):
                    y = y + conv(col)
                y = blk.act_t(y)
                x = blk.act_f(blk.conv_f(y)) + x
            elif self.layout == "time_c":
                win = torch.cat([states[i], x], dim=1)          # (1, k_t*C, F, 1)
                extra.append(win[:, C:])                        # drop the oldest frame
                y = blk.act_t(self.conv_t1[i](win))
                x = blk.act_f(blk.conv_f(y)) + x
            else:
                win = torch.cat([states[i], x], dim=3)          # (1, C, F, k_t)
                extra.append(win[..., 1:])                      # shift the FIFO
                x = blk.forward_window(win, x)
        mask = torch.sigmoid(self.m.head(x))
        return (mask, *extra)


def load_model(h, checkpoint=None, seed=0):
    torch.manual_seed(seed)
    model = build_causal_model(h).eval()
    if checkpoint:
        ck = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(ck.get("generator", ck), strict=True)
    return model


def export_fp32(step: N6NetStreamStep, path: Path, opset: int = 17) -> Path:
    m = step.m
    feat = torch.randn(1, 1, m.n_features, 1).abs()
    states = step.init_states()
    in_names, out_names = step.input_names, step.output_names
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(step, (feat, *states), str(path),
                      input_names=in_names, output_names=out_names,
                      opset_version=opset, dynamo=False, do_constant_folding=True)
    return path


def speech_frames(h, n_utts=4, split="train"):
    """Compressed-magnitude frames (1, 1, F, T) from VoiceBank-DEMAND noisy crops."""
    from common.dataset import Dataset, load_voicebank_demand

    model = build_causal_model(h)
    hf = load_voicebank_demand()
    ds = Dataset(hf[split], h["segment_size"], h["sampling_rate"],
                 split=True, shuffle=True, seed=0)
    out = []
    with torch.no_grad():
        for i in range(n_utts):
            _, noisy = ds[i]
            spec = model.preproc(noisy.view(1, 1, -1))           # (1, 1, F, T)
            out.append(compress_magnitude(spec.abs(), float(h.get("compress_factor", 0.3))))
    return out


def stream_feeds(step, feats, max_frames=400):
    """Per-frame ORT feeds with the real propagated FIFO state."""
    names = step.input_names[1:]
    items = []
    with torch.no_grad():
        for f in feats:
            states = step.init_states()
            for t in range(f.shape[-1]):
                ft = f[..., t:t + 1]
                d = {"feat": ft.numpy()}
                d.update({n: s.numpy() for n, s in zip(names, states)})
                items.append(d)
                states = step.advance(states, step(ft, *states)[1:])
                if len(items) >= max_frames:
                    return items
    return items


def parity(step, onnx_path, feat):
    """Streaming ONNX over T frames vs the offline left-padded model."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    m = step.m
    with torch.no_grad():
        x = m.stem(feat)
        for b in m.blocks:
            x = b(x)
        ref = torch.sigmoid(m.head(x)).numpy()
    states = [s.numpy() for s in step.init_states()]
    names = step.input_names[1:]
    got = []
    for t in range(feat.shape[-1]):
        outs = sess.run(None, {"feat": feat[..., t:t + 1].numpy(), **dict(zip(names, states))})
        got.append(outs[0])
        states = step.advance(states, outs[1:])
    got = np.concatenate(got, axis=-1)
    return float(np.abs(got - ref).max()), got, ref


class _Reader:
    def __init__(self, items):
        self.items = items
        self._it = iter(items)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self.items)


def quantize(fp32_path, out_path, items):
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                          QuantFormat, QuantType, quantize_static)

    class R(_Reader, CalibrationDataReader):
        pass

    quantize_static(str(fp32_path), str(out_path), R(items),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.Percentile,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    return out_path


def quantize_prelu_slopes(model):
    """Give every PRelu an int8 slope behind a DequantizeLinear.

    ORT's static quantizer leaves the slope as a float initializer, and Neural-ART
    only maps PRelu to HW when the slope is quantized — otherwise all six
    activations become pure-software epochs on the M55. Symmetric per-tensor.
    """
    from onnx import helper, numpy_helper

    g = model.graph
    inits = {i.name: i for i in g.initializer}
    # Equal-valued slopes (e.g. the 0.25 init) are deduplicated by the exporter
    # into one initializer plus Identity aliases; fold the aliases away first.
    alias = {n.output[0]: n.input[0] for n in g.node
             if n.op_type == "Identity" and n.input[0] in inits}
    keep = [n for n in g.node if n.output[0] not in alias]
    for n in keep:
        for k, i in enumerate(n.input):
            if i in alias:
                n.input[k] = alias[i]
    del g.node[:]
    g.node.extend(keep)
    new_nodes, done = [], {}
    for n in g.node:
        if n.op_type != "PRelu":
            continue
        if n.input[1] in done:                  # slope initializer shared by the exporter
            n.input[1] = done[n.input[1]]
            continue
        if n.input[1] in inits:
            a = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
            scale = np.float32(max(float(np.abs(a).max()), 1e-8) / 127.0)
            q = np.clip(np.round(a / scale), -127, 127).astype(np.int8)
            base = n.input[1].replace(":", "_")
            g.initializer.extend([
                numpy_helper.from_array(q, base + "_q"),
                numpy_helper.from_array(np.array(scale, np.float32), base + "_scale"),
                numpy_helper.from_array(np.array(0, np.int8), base + "_zp"),
            ])
            dq_out = base + "_dq"
            new_nodes.append(helper.make_node(
                "DequantizeLinear", [base + "_q", base + "_scale", base + "_zp"], [dq_out],
                name=base + "_DequantizeLinear"))
            g.initializer.remove(inits.pop(n.input[1]))
            done[n.input[1]] = dq_out
            n.input[1] = dq_out
    # DQ nodes have only initializer inputs, so prepending keeps topological order
    nodes = new_nodes + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    return len(new_nodes)


def tie_fifo_qparams(model):
    """Quantize each FIFO path with the qparams of the block input it stores.

    ``fifo_i`` holds past values of the block input ``x``, so both Concat inputs,
    the Concat output and the shifted state share one scale/zero-point. Separate
    calibration gives them four slightly different ones, which turns the Concat
    into a requantizing (software) op and the state I/O into a scale conversion.
    """
    from onnx import numpy_helper

    g = model.graph
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    def set_qp(node, scale, zp):
        inits[node.input[1]].CopyFrom(numpy_helper.from_array(scale, node.input[1]))
        inits[node.input[2]].CopyFrom(numpy_helper.from_array(zp, node.input[2]))

    def qdq_chain_after(tensor):
        """Q -> DQ pair that directly consumes ``tensor``."""
        q = next(n for n in cons[tensor] if n.op_type == "QuantizeLinear")
        dq = next(n for n in cons[q.output[0]] if n.op_type == "DequantizeLinear")
        return q, dq

    tied = 0
    # split layout: fifo_i_j_in carry past values of col_i_out
    outs = {o.name for o in g.output}
    for i in range(64):
        col = f"col_{i}_out"
        if col not in outs:
            break
        src = prod[col]
        while src.op_type == "Identity":
            src = prod[src.input[0]]
        scale = numpy_helper.to_array(inits[src.input[1]])
        zp = numpy_helper.to_array(inits[src.input[2]])
        for inp in (x.name for x in g.input if x.name.startswith(f"fifo_{i}_")):
            q, dq = qdq_chain_after(inp)
            set_qp(q, scale, zp)
            set_qp(dq, scale, zp)
        tied += 1

    for cat in (n for n in g.node if n.op_type == "Concat"):
        fifo_dq, x_dq = prod[cat.input[0]], prod[cat.input[1]]
        scale = numpy_helper.to_array(inits[x_dq.input[1]])
        zp = numpy_helper.to_array(inits[x_dq.input[2]])
        # fifo_i_in: graph input -> Q -> DQ
        set_qp(fifo_dq, scale, zp)
        set_qp(prod[fifo_dq.input[0]], scale, zp)
        # concat output Q/DQ, and the slice -> fifo_i_out Q/DQ behind it
        q, dq = qdq_chain_after(cat.output[0])
        set_qp(q, scale, zp)
        set_qp(dq, scale, zp)
        sl = next(n for n in cons[dq.output[0]] if n.op_type == "Slice")
        q, dq = qdq_chain_after(sl.output[0])
        set_qp(q, scale, zp)
        set_qp(dq, scale, zp)
        tied += 1
    return tied


def npu_postpass(in_path, out_path):
    import onnx

    m = onnx.load(str(in_path))
    n_prelu = quantize_prelu_slopes(m)
    n_fifo = tie_fifo_qparams(m)
    onnx.checker.check_model(m)
    onnx.save(m, str(out_path))
    print(f"postpass: quantized {n_prelu} PRelu slopes, tied {n_fifo} FIFO qparam groups")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/n6net_b3.json")
    ap.add_argument("--checkpoint", default=None,
                    help="trained g_best; omitted = seeded random init (compiler-mapping study)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--n_blocks", type=int, default=None)
    ap.add_argument("--layout", choices=list(N6NetStreamStep.LAYOUTS), default="time_split")
    ap.add_argument("--calib_utts", type=int, default=4)
    ap.add_argument("--calib_frames", type=int, default=400)
    a = ap.parse_args()

    h = json.load(open(a.config))
    if a.channels:
        h["channels"] = a.channels
    if a.n_blocks:
        h["n_blocks"] = a.n_blocks
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(h, a.checkpoint)
    step = N6NetStreamStep(model, a.layout).eval()
    print("cost:", cost_summary(model))

    fp32 = export_fp32(step, out / "n6net_stream_fp32.onnx")
    feats = speech_frames(h, a.calib_utts)
    err, _, _ = parity(step, fp32, feats[0][..., :40])
    print(f"fp32 streaming-vs-offline max|diff| = {err:.3e}")

    items = stream_feeds(step, feats, a.calib_frames)
    q = quantize(fp32, out / "n6net_stream_int8_ort.onnx", items)
    q = npu_postpass(q, out / "n6net_stream_int8.onnx")
    err_q, got, ref = parity(step, q, feats[0][..., :40])
    cos = float((got * ref).sum() / (np.linalg.norm(got) * np.linalg.norm(ref)))
    print(f"int8 streaming mask vs fp32 offline: max|diff|={err_q:.3e} cos={cos:.4f}")
    json.dump({"config": h, "checkpoint": a.checkpoint, "layout": a.layout, "cost": cost_summary(model),
               "fp32_parity_maxabs": err, "int8_mask_cos": cos, "int8_mask_maxabs": err_q},
              open(out / "export_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
