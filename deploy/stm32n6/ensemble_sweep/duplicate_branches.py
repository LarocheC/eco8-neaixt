"""K-branch ensemble duplication of the streaming QDQ graph (worst-case route).

Copies the whole node graph K times inside one model: shared `feat` input,
per-member state I/O and initializers, K separate est_mag outputs. Structurally
exact for the compiler (epochs x K); member weights are byte-identical copies,
which is irrelevant for latency/epoch mapping.

Usage: duplicate_branches.py <in.onnx> <K> <out.onnx>
"""
import sys

import onnx


def member_name(name, k):
    if name == "" or name == "feat":
        return name
    return f"{name}__m{k}"


def main():
    src, k_members, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    m = onnx.load(src)
    g = m.graph

    new = onnx.GraphProto()
    new.name = f"{g.name}_x{k_members}"

    # shared feat input first, then per-member state inputs
    feat = [i for i in g.input if i.name == "feat"]
    states = [i for i in g.input if i.name != "feat"]
    assert feat, "expected a `feat` graph input"
    new.input.extend(feat)
    for k in range(k_members):
        for i in states:
            vi = onnx.ValueInfoProto()
            vi.CopyFrom(i)
            vi.name = member_name(i.name, k)
            new.input.append(vi)

    for k in range(k_members):
        for t in g.initializer:
            t2 = onnx.TensorProto()
            t2.CopyFrom(t)
            t2.name = member_name(t.name, k)
            new.initializer.append(t2)
        for n in g.node:
            n2 = onnx.NodeProto()
            n2.CopyFrom(n)
            if n2.name:
                n2.name = member_name(n2.name, k)
            for i, x in enumerate(n2.input):
                n2.input[i] = member_name(x, k)
            for i, x in enumerate(n2.output):
                n2.output[i] = member_name(x, k)
            new.node.append(n2)
        for o in g.output:
            vo = onnx.ValueInfoProto()
            vo.CopyFrom(o)
            vo.name = member_name(o.name, k)
            new.output.append(vo)

    m2 = onnx.ModelProto()
    m2.CopyFrom(m)
    m2.graph.CopyFrom(new)
    onnx.checker.check_model(m2)
    onnx.save(m2, dst)
    print(f"wrote {dst}: {len(new.node)} nodes, {len(new.input)} inputs, "
          f"{len(new.output)} outputs, {len(new.initializer)} initializers")

    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
        feeds = {}
        for i in sess.get_inputs():
            shape = [1 if isinstance(d, str) or d is None else d for d in i.shape]
            feeds[i.name] = np.zeros(shape, dtype=np.float32)
        outs = sess.run(None, feeds)
        print(f"ORT run OK: {len(outs)} outputs, est_mag shape {outs[0].shape}")
    except Exception as e:  # noqa: BLE001 - report and continue, compile is the real gate
        print(f"ORT check failed: {e}")


if __name__ == "__main__":
    main()
