"""Member-correct channel-fold: K *distinct* ensemble members in one streaming graph.

`fold_channels.py` (v2, bd-dense) is latency-exact but interleaves members in two
places, so it can only carry K identical copies. This builder (v3) fixes both and
folds K genuinely different members — e.g. stochastic-rounding draws of the int8
weights on the shared deployed scales — into one graph whose per-member outputs
are bit-equivalent to running the K member graphs separately:

  * **GLU chunk** — `fc1` (1x1, emb -> 2h) folds grouped, but the traced
    `chunk(2, dim=1)` splits the folded channel axis at K*h, mixing member 0's
    value half with member 1's gate half. Fix: split each fc1 into TWO grouped
    1x1 convs (rows [0:h] / [h:2h] of every member's weight — the x/gate and v
    halves), delete the Shape->Gather->Add->Div->Slice chain, and rewire the two
    existing half-quantizers (they reuse fc1's output scale, so the member
    arithmetic is unchanged: Q∘DQ∘Q at one scale == Q). Grouped 1x1 is HW-proven.
  * **Decoder skips** — `cat([x, enc], 1)` gives [x·m0..x·mK, enc·m0..enc·mK];
    consumers are bd-dense (v2 rule), so the member blocks are simply *placed*
    at the segment-local offsets in the dense weight (a permuted block layout).
    Standard op shapes, zero extra nodes.
  * **Tail** — `apply_mask` gathers mask channels 0/1; per-member that becomes
    two stride-2 channel Slices (members' [0::2] / [1::2]), the noisy-mag Gather
    becomes a keep-dims Slice, and `est_mag` widens to (B, K, 1, F) — one
    enhanced magnitude per member (average + disagreement are the caller's).
  * **Head** (v2's `feat` K-replication + its K x FP32-prologue SW cost): gone.
    `conv_1` is 1x1 on the 3 shared input channels, so members stack along its
    *output* axis only — feat stays (B, 3, 1, F) and is quantized once.

Everything else follows the measured v2 rules: 1x1 pointwise and depthwise convs
fold grouped (group *= K); all other Convs and every ConvTranspose fold
block-diagonal dense (zero blocks quantize exactly under the symmetric grid);
BN vectors and per-channel weight scales/zps/biases concatenate member-major;
state I/O widens K x on channels. Activation Q/DQ scales are shared per-tensor
scalars — identical across members by the SR protocol, asserted below.

Usage:
  fold_members.py <out.onnx> --members m0.onnx [m1.onnx ...] [--check N]

With one member this emits the restructured K=1 twin (same arithmetic as the
member, new op structure) — the board-latency A/B for the fc1-split/tail cost.
`--check N` streams N frames of real-shaped random input through the fold and
every member separately and asserts per-member bit-closeness.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def _arr(t):
    return numpy_helper.to_array(t)


class Fold:
    def __init__(self, member_paths):
        self.K = len(member_paths)
        self.models = [onnx.load(p) for p in member_paths]
        self.m = self.models[0]           # folded in place; members supply arrays
        g = self.m.graph
        ops = [(n.op_type, n.name) for n in g.node]
        for mm in self.models[1:]:
            assert [(n.op_type, n.name) for n in mm.graph.node] == ops, \
                "member graphs differ structurally"
        self.g = g
        self.inits = {t.name: t for t in g.initializer}
        self.member_inits = [{t.name: t for t in mm.graph.initializer}
                             for mm in self.models]
        self.prod = {o: n for n in g.node for o in n.output}
        self.cons = {}
        for n in g.node:
            for x in n.input:
                self.cons.setdefault(x, []).append(n)
        # activation scales/zps and all non-weight params must agree across members
        for name, t in self.inits.items():
            arrs = [_arr(mi[name]) for mi in self.member_inits]
            if not all(np.array_equal(arrs[0], a) for a in arrs[1:]):
                assert t.data_type == onnx.TensorProto.INT8 and name.endswith("_quantized"), \
                    f"{name}: members differ but it is not an int8 weight tensor"
        # pre-fold channel width of every internal tensor (for segment sizing)
        inferred = onnx.shape_inference.infer_shapes(onnx.load(member_paths[0]))
        self.chan = {}
        for vi in list(inferred.graph.value_info) + list(inferred.graph.output) \
                + list(inferred.graph.input):
            dims = vi.type.tensor_type.shape.dim
            if len(dims) == 4 and dims[1].HasField("dim_value"):
                self.chan[vi.name] = dims[1].dim_value

    # ---------------------------------------------------------------- helpers
    def member_arrs(self, name):
        return [_arr(mi[name]) for mi in self.member_inits]

    def set_init(self, name, arr):
        self.inits[name].CopyFrom(numpy_helper.from_array(arr, name))

    def cat_init(self, name):
        """Member-concat a 1-D per-channel initializer (scale/zp/bias/BN vec)."""
        arrs = self.member_arrs(name)
        if arrs[0].ndim == 0:
            return                        # per-tensor scalar: shared, keep
        self.set_init(name, np.concatenate(arrs, axis=0))

    def cat_param(self, name):
        """Member-concat a param fed as initializer / Constant / DQ of either."""
        if name in self.inits:
            self.cat_init(name)
            return
        p = self.prod[name]
        if p.op_type == "DequantizeLinear":
            self.cat_param(p.input[0])
            sarr = _arr(self.inits[p.input[1]]) if p.input[1] in self.inits else None
            if sarr is not None and sarr.size > 1:
                self.cat_init(p.input[1])
                if len(p.input) > 2:
                    self.cat_init(p.input[2])
            return
        if p.op_type == "Constant":
            for a in p.attribute:
                if a.name == "value":
                    arrs = []
                    for mm in self.models:
                        pn = next(n for n in mm.graph.node if n.name == p.name)
                        arrs.append(_arr(next(aa.t for aa in pn.attribute
                                              if aa.name == "value")))
                    if arrs[0].ndim == 0:
                        return
                    a.t.CopyFrom(numpy_helper.from_array(
                        np.concatenate(arrs, axis=0), a.t.name))
                    return
        raise AssertionError(f"{name}: unsupported param feed {p.op_type}")

    def wdq(self, node):
        p = self.prod.get(node.input[1])
        assert p is not None and p.op_type == "DequantizeLinear", \
            f"{node.name}: weight not fed by DQ"
        return p

    def fold_wscale(self, dq):
        sarr = _arr(self.inits[dq.input[1]]) if dq.input[1] in self.inits else None
        if sarr is not None and sarr.size > 1:
            self.cat_init(dq.input[1])
            if len(dq.input) > 2:
                self.cat_init(dq.input[2])

    def fold_bias(self, node):
        if len(node.input) > 2 and node.input[2]:
            bdq = self.prod[node.input[2]]
            assert bdq.op_type == "DequantizeLinear"
            self.cat_init(bdq.input[0])
            self.fold_wscale(bdq)

    def input_segments(self, node):
        """Per-member channel segmentation of a conv's input.

        Walks input[0] back through channel-layout-preserving ops. An axis-1
        Concat splits the input into segments; each folded segment is
        member-contiguous with per-member width = folded_width / K. Returns a
        list of per-member segment widths (single-element if unsegmented, in
        which case the caller needs the member conv's own in-channel count).
        """
        transparent = ("DequantizeLinear", "QuantizeLinear", "Pad", "Clip")
        x = node.input[0]
        while True:
            p = self.prod.get(x)
            if p is None:
                return None
            if p.op_type in transparent:
                x = p.input[0]
                continue
            if p.op_type == "Slice":
                axes = self._const_of(p.input[3]) if len(p.input) > 3 else None
                assert axes is None or 1 not in np.atleast_1d(axes).tolist(), \
                    f"{p.name}: unexpected channel Slice feeding {node.name}"
                x = p.input[0]
                continue
            if p.op_type == "Concat":
                ax = next((a.i for a in p.attribute if a.name == "axis"), 0)
                if ax != 1:               # FIFO/time or freq concat: opaque
                    return None
                widths = []
                for s in p.input:
                    w = self.chan.get(s)
                    assert w is not None, f"{p.name}: cannot size segment {s}"
                    widths.append(w)
                return widths
            return None

    def _const_of(self, name):
        if name in self.inits:
            return _arr(self.inits[name])
        p = self.prod.get(name)
        if p is not None and p.op_type == "Constant":
            for a in p.attribute:
                if a.name == "value":
                    return _arr(a.t)
        return None

    # ------------------------------------------------------------- transforms
    def run(self):
        K = self.K
        g = self.g

        # --- locate the special nodes ------------------------------------
        feat_dq = None
        for n in g.node:
            if n.op_type == "QuantizeLinear" and n.input[0] == "feat":
                (feat_q_out,) = n.output
                feat_dq = next(c for c in self.cons[feat_q_out]
                               if c.op_type == "DequantizeLinear")
        assert feat_dq is not None, "feat quantizer not found"
        feat_dq_out = feat_dq.output[0]

        conv1 = None
        for n in g.node:
            if n.op_type == "Conv" and n.input[0] == feat_dq_out:
                conv1 = n
        assert conv1 is not None, "conv_1 (feat consumer) not found"

        # GLU fc1s: their output DQ feeds a Shape node
        glu = []                          # (fc1_node, dq_out, slice_x, slice_v)
        for shape_n in [n for n in g.node if n.op_type == "Shape"]:
            dq_out = shape_n.input[0]
            q = self.prod[dq_out].input[0]           # DQ <- Q output
            fc1 = self.prod[self.prod[q].input[0]]   # Q <- fc1 conv
            assert fc1.op_type == "Conv", f"Shape feeds non-conv {fc1.name}"
            slices = [c for c in self.cons[dq_out] if c.op_type == "Slice"]
            assert len(slices) == 2, f"{fc1.name}: expected 2 chunk Slices"
            sx = sv = None
            for s in slices:
                c = s
                for _ in range(6):        # walk fwd through the half's Q/DQ
                    c = next(cc for cc in self.cons[c.output[0]])
                    if c.op_type not in ("QuantizeLinear", "DequantizeLinear"):
                        break
                if c.op_type == "Concat":
                    sx = s                # x half: enters the FIFO concat
                elif c.op_type == "Mul":
                    sv = s                # v half: gates via Mul
            assert sx is not None and sv is not None, f"{fc1.name}: halves not identified"
            glu.append((fc1, sx, sv))
        glu_fc1_names = {fc1.name for fc1, _, _ in glu}

        # tail gathers: float-tensor Gathers (mask ch0/ch1 + noisy-mag ch0)
        tail = []
        for n in g.node:
            if n.op_type != "Gather":
                continue
            p = self.prod.get(n.input[0])
            if p is not None and p.op_type == "DequantizeLinear":
                idx = int(self._const_of(n.input[1]))
                tail.append((n, idx))
        assert len(tail) == 3, f"expected 3 tail Gathers, found {len(tail)}"

        # --- fold every conv ---------------------------------------------
        n_grouped = n_dense = 0
        for n in g.node:
            if n.op_type not in ("Conv", "ConvTranspose") or n.name in glu_fc1_names:
                continue
            wdq = self.wdq(n)
            wname = wdq.input[0]
            warrs = self.member_arrs(wname)
            wdims = warrs[0].shape
            ga = [a for a in n.attribute if a.name == "group"]
            old_group = ga[0].i if ga else 1
            depthwise = old_group > 1 and wdims[1] == 1
            pointwise = n.op_type == "Conv" and wdims[2:] == (1, 1)

            if n is conv1:
                # members stack on the OUT axis only; feat stays 3-channel
                self.set_init(wname, np.concatenate(warrs, axis=0))
                n_grouped += 1            # counts as a shared-input grouped fold
            elif n.op_type == "Conv" and (depthwise or pointwise):
                n_grouped += 1
                if ga:
                    ga[0].i = old_group * K
                else:
                    n.attribute.append(onnx.helper.make_attribute("group", K))
                self.set_init(wname, np.concatenate(warrs, axis=0))
            else:
                n_dense += 1
                assert old_group == 1, f"{n.name}: densify expects group=1"
                segs = self.input_segments(n)
                if n.op_type == "Conv":
                    A, B = wdims[0], wdims[1]
                    segs = segs or [B]
                    assert sum(segs) == B, f"{n.name}: segments {segs} != in {B}"
                    out = np.zeros((K * A, K * B) + wdims[2:], warrs[0].dtype)
                    for mi, w in enumerate(warrs):
                        lo = 0
                        for ws in segs:
                            fo = K * lo + mi * ws
                            out[mi * A:(mi + 1) * A, fo:fo + ws] = w[:, lo:lo + ws]
                            lo += ws
                else:                     # ConvTranspose: (in, out, kh, kw)
                    B, A = wdims[0], wdims[1]
                    segs = segs or [B]
                    assert sum(segs) == B, f"{n.name}: segments {segs} != in {B}"
                    out = np.zeros((K * B, K * A) + wdims[2:], warrs[0].dtype)
                    for mi, w in enumerate(warrs):
                        lo = 0
                        for ws in segs:
                            fo = K * lo + mi * ws
                            out[fo:fo + ws, mi * A:(mi + 1) * A] = w[lo:lo + ws, :]
                            lo += ws
                self.set_init(wname, out)
            self.fold_wscale(wdq)
            self.fold_bias(n)

        # --- BN params ----------------------------------------------------
        n_bn = 0
        for n in g.node:
            if n.op_type == "BatchNormalization":
                n_bn += 1
                for x in n.input[1:5]:
                    self.cat_param(x)

        # --- GLU: split each fc1 into grouped x/v convs -------------------
        new_nodes = []
        for fc1, sx, sv in glu:
            wdq = self.wdq(fc1)
            wname = wdq.input[0]
            warrs = self.member_arrs(wname)
            h = warrs[0].shape[0] // 2
            sarrs = self.member_arrs(wdq.input[1])
            zarrs = self.member_arrs(wdq.input[2]) if len(wdq.input) > 2 else None
            bdq = self.prod[fc1.input[2]] if len(fc1.input) > 2 and fc1.input[2] else None
            barrs = self.member_arrs(bdq.input[0]) if bdq is not None else None
            bsarrs = self.member_arrs(bdq.input[1]) if bdq is not None else None

            for half, rows, target_slice in (("x", slice(0, h), sx),
                                             ("v", slice(h, 2 * h), sv)):
                base = f"{fc1.name}_{half}"
                wn = f"{base}_w_quantized"
                sn = f"{base}_w_scale"
                zn = f"{base}_w_zp"
                g.initializer.extend([
                    numpy_helper.from_array(
                        np.concatenate([w[rows] for w in warrs], axis=0), wn),
                    numpy_helper.from_array(
                        np.concatenate([s[rows] for s in sarrs], axis=0), sn),
                ])
                dq_in = [wn, sn]
                if zarrs is not None:
                    g.initializer.append(numpy_helper.from_array(
                        np.concatenate([z[rows] for z in zarrs], axis=0), zn))
                    dq_in.append(zn)
                wdq_new = onnx.helper.make_node(
                    "DequantizeLinear", dq_in, [f"{base}_w"], name=f"{base}_wdq",
                    axis=0)
                conv_in = [fc1.input[0], f"{base}_w"]
                if barrs is not None:
                    bn_ = f"{base}_b_quantized"
                    bsn = f"{base}_b_scale"
                    g.initializer.extend([
                        numpy_helper.from_array(
                            np.concatenate([b[rows] for b in barrs], axis=0), bn_),
                        numpy_helper.from_array(
                            np.concatenate([s[rows] for s in bsarrs], axis=0), bsn),
                    ])
                    bzn = f"{base}_b_zp"
                    g.initializer.append(numpy_helper.from_array(
                        np.zeros(self.K * h, np.int32), bzn))
                    bdq_new = onnx.helper.make_node(
                        "DequantizeLinear", [bn_, bsn, bzn], [f"{base}_b"],
                        name=f"{base}_bdq", axis=0)
                    new_nodes.append(bdq_new)
                    conv_in.append(f"{base}_b")
                conv = onnx.helper.make_node(
                    "Conv", conv_in, [f"{base}_out"], name=base,
                    group=self.K, kernel_shape=[1, 1])
                new_nodes.extend([wdq_new, conv])
                # rewire the half's existing quantizer onto the new conv output
                q = next(c for c in self.cons[target_slice.output[0]]
                         if c.op_type == "QuantizeLinear")
                q.input[0] = f"{base}_out"

        g.node.extend(new_nodes)

        # --- tail: Gathers -> member-preserving Slices --------------------
        starts = {}
        for n, idx in tail:
            is_feat = self.prod[n.input[0]] is feat_dq or n.input[0] == feat_dq_out
            base = n.name.replace("/", "_")
            if is_feat:                   # noisy mag: keep-dims channel 0
                s, e, st = 0, 1, 1
            else:                         # mask ch idx of every member
                s, e, st = idx, 2 * K, 2
            for nm, val in ((f"{base}_s", [s]), (f"{base}_e", [e]),
                            (f"{base}_ax", [1]), (f"{base}_st", [st])):
                g.initializer.append(numpy_helper.from_array(
                    np.array(val, np.int64), nm))
            sl = onnx.helper.make_node(
                "Slice", [n.input[0], f"{base}_s", f"{base}_e", f"{base}_ax",
                          f"{base}_st"],
                list(n.output), name=f"{base}_slice")
            starts[n.name] = sl
        g.node.extend(starts.values())
        drop = {n.name for n, _ in tail}
        keep = [n for n in g.node if n.name not in drop]
        del g.node[:]
        g.node.extend(keep)

        # --- state I/O widens; est_mag becomes (B, K, 1, F) ---------------
        n_freqs = None
        for vi in list(g.input) + list(g.output):
            if vi.name.startswith("state_"):
                d = vi.type.tensor_type.shape.dim[1]
                assert d.HasField("dim_value"), f"{vi.name}: symbolic channel dim"
                d.dim_value *= K
            if vi.name == "feat":
                n_freqs = vi.type.tensor_type.shape.dim[3].dim_value
        for vi in g.output:
            if vi.name == "est_mag":
                new = onnx.helper.make_tensor_value_info(
                    "est_mag", onnx.TensorProto.FLOAT, ["B", K, 1, n_freqs])
                vi.CopyFrom(new)
        del g.value_info[:]

        # --- DCE: drop everything unreachable from the outputs ------------
        prod2 = {o: n for n in g.node for o in n.output}
        needed, stack = set(), [o.name for o in g.output]
        keep_nodes = []
        seen_nodes = set()
        while stack:
            t = stack.pop()
            if t in needed:
                continue
            needed.add(t)
            p = prod2.get(t)
            if p is not None and id(p) not in seen_nodes:
                seen_nodes.add(id(p))
                keep_nodes.append(p)
                stack.extend(p.input)
        order = {id(n): i for i, n in enumerate(g.node)}
        keep_nodes.sort(key=lambda n: order[id(n)])
        n_dropped = len(g.node) - len(keep_nodes)
        del g.node[:]
        g.node.extend(keep_nodes)
        used = {x for n in g.node for x in n.input}
        drop_inits = [t.name for t in g.initializer if t.name not in used]
        keep_inits = [t for t in g.initializer if t.name in used]
        del g.initializer[:]
        g.initializer.extend(keep_inits)

        # stable toposort (the new fc1/tail nodes were appended out of order)
        avail = {i.name for i in g.input} | {t.name for t in g.initializer} | {""}
        remaining, ordered = list(g.node), []
        while remaining:
            ready = [n for n in remaining if all(x in avail for x in n.input)]
            assert ready, "cycle while re-sorting graph"
            for n in ready:
                avail.update(n.output)
            ordered.extend(ready)
            remaining = [n for n in remaining if id(n) not in {id(r) for r in ready}]
        del g.node[:]
        g.node.extend(ordered)

        print(f"K={K}: {n_grouped} grouped + {n_dense} bd-dense convs, "
              f"{len(glu)} GLU fc1s split, {n_bn} BNs folded, "
              f"{n_dropped} nodes + {len(drop_inits)} initializers DCE'd")
        return self.m


def stream_check(fold_path, member_paths, n_frames, seed=0):
    """Stream random frames through the fold and each member; per-member compare."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    fold = ort.InferenceSession(str(fold_path), sess_options=so,
                                providers=["CPUExecutionProvider"])
    membs = [ort.InferenceSession(str(p), sess_options=so,
                                  providers=["CPUExecutionProvider"])
             for p in member_paths]
    K = len(membs)
    rng = np.random.default_rng(seed)
    state_in = [i.name for i in fold.get_inputs() if i.name != "feat"]
    fshape = [1 if isinstance(d, str) else d for d in fold.get_inputs()[0].shape]

    def zeros(sess):
        return {i.name: np.zeros([1 if isinstance(d, str) else d for d in i.shape],
                                 np.float32)
                for i in sess.get_inputs() if i.name != "feat"}

    fs = zeros(fold)
    ms = [zeros(s) for s in membs]
    worst = 0.0
    n_exact = n_tot = 0
    for t in range(n_frames):
        feat = (rng.standard_normal(fshape) * 0.5).astype(np.float32)
        feat[0, 0] = np.abs(feat[0, 0])          # mag channel is non-negative
        fres = fold.run(None, {"feat": feat, **fs})
        fs = dict(zip(state_in, fres[1:]))
        for k, (sess, st) in enumerate(zip(membs, ms)):
            mres = sess.run(None, {"feat": feat, **st})
            ms[k] = dict(zip(state_in, mres[1:]))
            em_f = fres[0][0, k, 0]              # (F,)
            em_m = mres[0][0, 0]                 # member est_mag (1,1,F)->(F,)
            d = np.abs(em_f - em_m)
            worst = max(worst, float(d.max()))
            n_exact += int((d == 0).sum())
            n_tot += d.size
            for i_s, nm in enumerate(state_in):
                sf = fs[nm]
                C = sf.shape[1] // K
                ds = np.abs(sf[:, k * C:(k + 1) * C] - ms[k][nm])
                worst = max(worst, float(ds.max()))
    print(f"check: {n_frames} frames x {K} members: est_mag exact "
          f"{n_exact}/{n_tot} ({100.0 * n_exact / n_tot:.2f}%), "
          f"worst |diff| (est_mag+states) = {worst:.3e}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--check", type=int, default=0,
                    help="stream N random frames and compare fold vs members")
    a = ap.parse_args()

    f = Fold(a.members)
    m = f.run()
    onnx.checker.check_model(m)
    onnx.save(m, a.out)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB)")

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(a.out, sess_options=so,
                                providers=["CPUExecutionProvider"])
    feeds = {}
    for i in sess.get_inputs():
        shape = [1 if isinstance(d, str) or d is None else d for d in i.shape]
        feeds[i.name] = np.zeros(shape, np.float32)
    outs = sess.run(None, feeds)
    print(f"ORT run OK: est_mag {outs[0].shape}, {len(outs)} outputs")

    if a.check:
        stream_check(a.out, a.members, a.check)


if __name__ == "__main__":
    main()
