"""Channel-fold K ensemble members into one streaming QDQ graph (v2: bd-dense).

Measured v1 lesson (see README): the Neural-ART mapper takes grouped 1x1 and
depthwise convs on HW, but grouped 2-D-kernel convs and any per-member
ConvTranspose workaround fall to SW on the M55 (~3.7 ms extra at K=2). So:
- 1x1 pointwise and depthwise convs fold as grouped (group *= K, HW-proven);
- every other Conv and all ConvTranspose fold as BLOCK-DIAGONAL DENSE weights
  (standard op shapes, no group attr change -> HW mapping preserved; the zero
  blocks quantize exactly and the extra MACs are negligible at these sizes).
BN vectors tile K x, state I/O widens K x on channels, `feat` replicates K x
via a head Concat. Channel-split slices are shape-derived (dynamic) and stay
shape-consistent; member semantics there interleave, irrelevant for latency.

Usage: fold_channels.py <in.onnx> <K> <out.onnx>
"""
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def main():
    src, K, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    m = onnx.load(src)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    tiled = {}  # name -> reps applied

    def tile(name, reps, axis=0):
        """Tile initializer `name` K x along axis (idempotent, conflict-checked)."""
        if name in tiled:
            assert tiled[name] == (reps, axis), f"{name}: conflicting tiling"
            return
        t = inits[name]
        arr = numpy_helper.to_array(t)
        shape = [1] * max(arr.ndim, 1)
        if arr.ndim == 0:
            return  # scalar scale/zp: per-tensor, nothing to tile
        shape[axis] = reps
        out = np.tile(arr, shape)
        t2 = numpy_helper.from_array(out, t.name)
        t.CopyFrom(t2)
        tiled[name] = (reps, axis)

    def dq_chain(tensor_name):
        """Return the DQ node feeding `tensor_name`, or None."""
        n = producer.get(tensor_name)
        return n if n is not None and n.op_type == "DequantizeLinear" else None

    def resolve_arr(name):
        if name in inits:
            return numpy_helper.to_array(inits[name])
        p = producer.get(name)
        if p is not None and p.op_type == "Constant":
            for a in p.attribute:
                if a.name == "value":
                    return numpy_helper.to_array(a.t)
        return None

    def tile_any(name, reps, axis=0):
        """Tile a param fed as initializer, Constant node, or DQ of either."""
        if name in inits:
            tile(name, reps, axis)
            return
        p = producer.get(name)
        assert p is not None, f"{name}: no producer and not an initializer"
        if p.op_type == "Constant":
            key = f"const::{name}"
            if key in tiled:
                assert tiled[key] == (reps, axis), f"{name}: conflicting tiling"
                return
            for a in p.attribute:
                if a.name == "value":
                    arr = numpy_helper.to_array(a.t)
                    if arr.ndim == 0:
                        return
                    shape = [1] * arr.ndim
                    shape[axis] = reps
                    a.t.CopyFrom(numpy_helper.from_array(
                        np.tile(arr, shape), a.t.name))
                    tiled[key] = (reps, axis)
                    return
            raise AssertionError(f"{name}: Constant without value")
        if p.op_type == "Identity":
            tile_any(p.input[0], reps, axis)
            return
        if p.op_type in ("DequantizeLinear", "QuantizeLinear"):
            tile_any(p.input[0], reps, axis)
            sarr = resolve_arr(p.input[1])
            if sarr is not None and sarr.size > 1:
                tile_any(p.input[1], reps, 0)
                if len(p.input) > 2:
                    tile_any(p.input[2], reps, 0)
            return
        raise AssertionError(f"{name}: fed by unsupported {p.op_type}")

    def block_diag(name, reps):
        """Densify [A,B,kh,kw] -> [reps*A, reps*B, kh, kw], members on the diagonal."""
        if name in tiled:
            assert tiled[name] == (reps, "bd"), f"{name}: conflicting tiling"
            return
        t = inits[name]
        arr = numpy_helper.to_array(t)
        assert arr.ndim == 4, f"{name}: expected 4-D conv weight"
        a, b = arr.shape[0], arr.shape[1]
        out = np.zeros((reps * a, reps * b) + arr.shape[2:], dtype=arr.dtype)
        for mi in range(reps):
            out[mi * a:(mi + 1) * a, mi * b:(mi + 1) * b] = arr
        t.CopyFrom(numpy_helper.from_array(out, t.name))
        tiled[name] = (reps, "bd")

    n_grouped = n_dense = n_bn = 0
    for n in g.node:
        if n.op_type in ("Conv", "ConvTranspose"):
            wdq = dq_chain(n.input[1])
            assert wdq is not None, f"{n.name}: weight not fed by DQ"
            wname = wdq.input[0]
            wdims = list(inits[wname].dims)
            ga = [a for a in n.attribute if a.name == "group"]
            old_group = ga[0].i if ga else 1
            kernel = next((list(a.ints) for a in n.attribute
                           if a.name == "kernel_shape"), None) or wdims[2:]
            depthwise = old_group > 1 and wdims[1] == 1
            pointwise = list(kernel) == [1, 1]
            if n.op_type == "Conv" and (depthwise or pointwise):
                # grouped fold — HW-proven for 1x1 and depthwise (probe + v1)
                n_grouped += 1
                if ga:
                    ga[0].i = old_group * K
                else:
                    n.attribute.append(onnx.helper.make_attribute("group", K))
                tile(wname, K, axis=0)
            else:
                # block-diagonal dense — standard shapes keep the HW mapping;
                # 2-D-kernel grouped convs and grouped/expanded ConvTranspose
                # measured as SW fallbacks in v1
                n_dense += 1
                assert old_group == 1, f"{n.name}: densify expects group=1"
                block_diag(wname, K)
            # per-channel weight scale/zp: every layout scales its axis dim by K
            sarr = resolve_arr(wdq.input[1])
            if sarr is not None and sarr.size > 1:
                tile(wdq.input[1], K)
                if len(wdq.input) > 2:
                    tile(wdq.input[2], K)
            # bias (optional input 2): per-out-channel behind DQ, tile K x
            if len(n.input) > 2 and n.input[2]:
                bdq = dq_chain(n.input[2])
                assert bdq is not None, f"{n.name}: bias not fed by DQ"
                tile(bdq.input[0], K, axis=0)
                sarr = resolve_arr(bdq.input[1])
                if sarr is not None and sarr.size > 1:
                    tile(bdq.input[1], K)
                    if len(bdq.input) > 2:
                        tile(bdq.input[2], K)
        elif n.op_type == "BatchNormalization":
            n_bn += 1
            for x in n.input[1:5]:
                tile_any(x, K)

    # ConstantOfShape zero-tensors (FIFO seeds/pads) bake [B,C,T,F] shape
    # vectors — scale the channel element of 4-long constant shape inputs
    patched = set()

    def patch_shape_vec(name):
        if name in patched:
            return
        arr = resolve_arr(name)
        if arr is None or arr.ndim != 1 or arr.size != 4 or arr[1] <= 1:
            return
        arr = arr.copy()
        arr[1] *= K
        if name in inits:
            inits[name].CopyFrom(numpy_helper.from_array(arr, name))
        else:
            p = producer[name]
            for a in p.attribute:
                if a.name == "value":
                    a.t.CopyFrom(numpy_helper.from_array(arr, a.t.name))
        patched.add(name)

    n_cos = 0
    for n in g.node:
        if n.op_type == "ConstantOfShape":
            patch_shape_vec(n.input[0])
            n_cos += 1

    # stale internal shape annotations would contradict the folded widths
    del g.value_info[:]

    # widen state I/O on the channel axis (dim 1); feat stays [B,3,1,257]
    for vi in list(g.input) + list(g.output):
        if vi.name.startswith("state_"):
            d = vi.type.tensor_type.shape.dim[1]
            assert d.HasField("dim_value"), f"{vi.name}: symbolic channel dim"
            d.dim_value *= K
        # est_mag / feat untouched

    # replicate feat K x on channels via a head Concat, rewire consumers
    feat_tiled = "feat_tiled_x%d" % K
    for n in g.node:
        for i, x in enumerate(n.input):
            if x == "feat":
                n.input[i] = feat_tiled
    cat = onnx.helper.make_node("Concat", ["feat"] * K, [feat_tiled],
                                name="feat_replicate", axis=1)
    g.node.insert(0, cat)

    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"wrote {dst}: K={K}, {n_grouped} grouped + {n_dense} bd-dense convs, "
          f"{n_bn} BNs tiled, {len(tiled)} initializers touched")

    import onnxruntime as ort
    sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    feeds = {}
    for i in sess.get_inputs():
        shape = [1 if isinstance(d, str) or d is None else d for d in i.shape]
        feeds[i.name] = np.random.default_rng(0).standard_normal(
            shape).astype(np.float32)
    outs = sess.run(None, feeds)
    print(f"ORT run OK: est_mag {outs[0].shape}, "
          f"{len(outs)} outputs, state_0_out {outs[1].shape}")


if __name__ == "__main__":
    main()
