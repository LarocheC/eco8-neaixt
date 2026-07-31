"""Minimal QDQ probe: dense vs group=2 vs depthwise conv, one graph.

Mirrors the real graph's QDQ pattern (per-tensor activation Q/DQ, per-channel
int8 weight DQ on axis 0). The generate report then shows how each conv maps
(HW / hybrid / SW) on Neural-ART.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

C, F = 40, 32
rng = np.random.default_rng(0)


def qdq_act(name, src):
    s = numpy_helper.from_array(np.float32(0.02), f"{name}_s")
    z = numpy_helper.from_array(np.int8(0), f"{name}_z")
    q = helper.make_node("QuantizeLinear", [src, s.name, z.name], [f"{name}_q"])
    dq = helper.make_node("DequantizeLinear", [f"{name}_q", s.name, z.name],
                          [f"{name}_dq"])
    return [q, dq], [s, z], f"{name}_dq"


def conv(name, src, cout, cin_per_group, group, kh=1, kw=1, pads=(0, 0, 0, 0)):
    w = rng.integers(-127, 127, size=(cout, cin_per_group, kh, kw), dtype=np.int8)
    wt = numpy_helper.from_array(w, f"{name}_w")
    ws = numpy_helper.from_array(
        rng.uniform(0.001, 0.02, size=cout).astype(np.float32), f"{name}_ws")
    wz = numpy_helper.from_array(np.zeros(cout, dtype=np.int8), f"{name}_wz")
    dqw = helper.make_node("DequantizeLinear", [wt.name, ws.name, wz.name],
                           [f"{name}_wdq"], axis=0)
    cv = helper.make_node("Conv", [src, f"{name}_wdq"], [f"{name}_y"], name=name,
                          group=group, kernel_shape=[kh, kw], pads=list(pads),
                          strides=[1, 1], dilations=[1, 1])
    return [dqw, cv], [wt, ws, wz], f"{name}_y"


nodes, inits = [], []
n, i, cur = qdq_act("in", "x")
nodes += n
inits += i

n, i, cur = conv("conv_dense", cur, C, C, 1)
nodes += n
inits += i
n, i, cur = qdq_act("a1", cur)
nodes += n
inits += i

n, i, cur = conv("conv_group2", cur, C, C // 2, 2)
nodes += n
inits += i
n, i, cur = qdq_act("a2", cur)
nodes += n
inits += i

n, i, cur = conv("conv_group4", cur, C, C // 4, 4)
nodes += n
inits += i
n, i, cur = qdq_act("a3", cur)
nodes += n
inits += i

n, i, cur = conv("conv_dw", cur, C, 1, C, kh=1, kw=3, pads=(0, 1, 0, 1))
nodes += n
inits += i
n, i, cur = qdq_act("out", cur)
nodes += n
inits += i

graph = helper.make_graph(
    nodes, "group_probe",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, 1, F])],
    [helper.make_tensor_value_info(cur, TensorProto.FLOAT, [1, C, 1, F])],
    initializer=inits)
model = helper.make_model(
    graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
onnx.checker.check_model(model)
onnx.save(model, "group_probe.onnx")
print("wrote group_probe.onnx")
