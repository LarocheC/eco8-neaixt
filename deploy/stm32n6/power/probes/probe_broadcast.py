"""The first two probes for the full-band hang (2026-09-17): a pooled
full-height branch added back by broadcast (bcast), and the same with a
same-shape Add (ctrl). bcast hangs the board, ctrl runs -- see N6NET_POWER.md.

`deploy/stm32n6/host/fullband_probes.py` supersedes these: same question at the
real shapes, and it splits the cause (the compiler's 8x1 sub-conv rewrite)
rather than just reproducing it. Kept because these two are the cells the
power campaign's measure/ logs name.

    python probe_broadcast.py            # writes {bcast,ctrl}_{fp32,int8}.onnx here
"""
import numpy as np, torch
from torch import nn
from onnxruntime.quantization import (quantize_static, CalibrationDataReader, QuantFormat,
                                      QuantType)

class P(nn.Module):
    def __init__(self, bcast):
        super().__init__()
        self.bcast = bcast
        self.a = nn.Conv2d(96, 96, (5, 1), padding=(2, 0))
        self.p1 = nn.Conv2d(96, 48, (4, 1), stride=(4, 1))
        self.p2 = nn.Conv2d(48, 48, (4, 1), stride=(4, 1))
        self.p3 = nn.Conv2d(48, 96, (8, 1)) if bcast else nn.Conv2d(48, 96, (1, 1))
        self.up = nn.Conv2d(96, 96, (1, 1))
    def forward(self, x):
        h = torch.relu(self.a(x))
        g = torch.relu(self.p3(torch.relu(self.p2(torch.relu(self.p1(h))))))
        if not self.bcast:  # same shape: undo the pooling with a 1x1 on h instead
            g = self.up(h)
        return torch.relu(h + g)

class R(CalibrationDataReader):
    def __init__(self): self.it = iter([{"x": np.random.rand(1, 96, 128, 1).astype(np.float32)} for _ in range(16)])
    def get_next(self): return next(self.it, None)

torch.manual_seed(0)
for name in ("bcast", "ctrl"):
    m = P(name == "bcast").eval()
    torch.onnx.export(m, torch.randn(1, 96, 128, 1), f"{name}_fp32.onnx", input_names=["x"],
                      output_names=["y"], opset_version=17, dynamo=False)
    quantize_static(f"{name}_fp32.onnx", f"{name}_int8.onnx", R(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True)
    print(name, "ok")
