"""Strip trailing empty optional inputs (the Pad constant_value blocker-#4 fix).

The stored streaming export predates the exporter fix; atonn segfaults on
Pad nodes carrying an empty-string constant_value input. Trailing "" optional
inputs are semantically identical to omitted ones.

Usage: sanitize_pads.py <in.onnx> <out.onnx>
"""
import sys

import onnx

m = onnx.load(sys.argv[1])
n_fixed = 0
for n in m.graph.node:
    while n.input and n.input[-1] == "":
        del n.input[-1]
        n_fixed += 1
onnx.checker.check_model(m)
onnx.save(m, sys.argv[2])
print(f"wrote {sys.argv[2]}: stripped {n_fixed} trailing empty inputs")
