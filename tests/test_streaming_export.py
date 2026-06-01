"""Phase 1 STR-04 ONNX export smoke test.

See:
- .planning/phases/01-streaming-forward-fp32-parity-gate/01-CONTEXT.md (D-04, D-05)
- .planning/phases/01-streaming-forward-fp32-parity-gate/01-RESEARCH.md (Pitfall 10, Pitfall 11)

Two failure modes guarded:
- Pitfall 10: nn.GRUCell.forward calls aten::_thnn_fused_gru_cell which has no
  ONNX symbolic in dynamo=False — export raises OnnxExporterError. Smoke test
  fails at export-time inside the exported_path fixture.
- Pitfall 11: nn.GRU(...) call inside forward_step emits an opaque ONNX `GRU`
  op that Phase 4 quantize_static cannot rewrite. Smoke test enumerates
  graph.node and asserts no node has op_type == 'GRU' (or aten::*gru*).

Determinism (manual_seed, cudnn flags, deterministic algos, CUBLAS env)
is provided by tests/conftest.py — autouse, session-scoped (Plan 01-01).
"""

from __future__ import annotations

import json

import onnx
import pytest
import torch

from common.env import AttrDict
from nsnet2.model import NSNet2
from nsnet2.streaming import NSNet2Streaming


@pytest.fixture
def baseline_h() -> AttrDict:
    """Load baseline.json — same path inference.py uses."""
    with open("configs/baseline.json") as f:
        return AttrDict(json.load(f))


@pytest.fixture
def fresh_nsnet2(baseline_h) -> NSNet2:
    """Fresh-init NSNet2 — D-06. No bias override needed for export smoke
    (the smoke test only verifies graph structure, not numerics)."""
    torch.manual_seed(0)
    return NSNet2(baseline_h)


@pytest.fixture
def streaming(fresh_nsnet2) -> NSNet2Streaming:
    """Streaming wrapper — D-01 (composes base by reference)."""
    return NSNet2Streaming(fresh_nsnet2).eval()


@pytest.fixture
def gru_only_skip(fresh_nsnet2):
    """D-05: ONNX export smoke test only runs for cuDNN gru_kind."""
    if fresh_nsnet2.gru_kind != "gru":
        pytest.skip(
            f"ONNX export smoke test only runs for gru_kind='gru' "
            f"(got {fresh_nsnet2.gru_kind!r}) — D-05."
        )


@pytest.fixture
def exported_path(streaming, fresh_nsnet2, baseline_h, gru_only_skip, tmp_path):
    """Export streaming.forward_step (via the `forward = forward_step` alias)
    under dynamo=False / opset=17 to a temp .onnx file. Returns the path.

    Failure modes:
    - OnnxExporterError mentioning aten::*gru* or aten::_thnn_fused_gru_cell
      -> Pitfall 10 (nn.GRUCell at export time). Fix Plan 02 output.
    - Any other OnnxExporterError -> generic export break (likely an
      unsupported op in the unrolled cell).
    """
    B = 1
    n_freq = baseline_h.n_fft // 2 + 1                  # (B, n_freq) example input
    H = streaming.hidden_size
    L = streaming.num_layers
    frame_in = torch.zeros(B, n_freq)                   # (B, n_freq)
    states_in = torch.zeros(L, B, H)                    # (L, B, H)

    out_path = tmp_path / "smoke.onnx"
    torch.onnx.export(
        streaming,
        (frame_in, states_in),
        str(out_path),
        input_names=["frame_in", "states_in"],
        output_names=["mask", "states_out"],
        dynamic_axes={
            "frame_in":  {0: "B"},                      # batch axis only — STR-04 / SC-3
            "states_in": {1: "B"},                      # (L, B=dyn, H)
            "mask":      {0: "B"},
            "states_out": {1: "B"},
        },
        opset_version=17,                               # DET-01 onnx 1.21.x compat
        dynamo=False,                                   # MANDATORY — STR-04 (Pitfall 10 prevention)
    )
    return out_path


def test_onnx_export_smoke(exported_path):
    """STR-04: torch.onnx.export(forward_step, dynamo=False, opset=17) succeeds.

    The exported_path fixture does the export; this test simply asserts the
    file exists and is non-empty. The export-time failure mode (OnnxExporterError)
    surfaces during fixture construction, which is the right place — pytest reports
    the OnnxExporterError as the test's error rather than a soft failure here.
    """
    assert exported_path.exists(), f"export produced no file at {exported_path}"
    assert exported_path.stat().st_size > 0, "exported .onnx is empty"


def test_onnx_no_opaque_gru(exported_path):
    """Pitfall 11 prevention: exported graph must contain zero opaque GRU ops.

    Also catches Pitfall 10 lookalikes (aten::*gru*) — though those usually fail
    at export time, not graph-inspection time.
    """
    model = onnx.load(str(exported_path))

    # Primary check: literal "GRU" op_type (Pitfall 11) — Phase 4 quantize_static
    # cannot rewrite an opaque GRU op, so the GRU compute would silently run FP32
    # internally even after quant. This assertion is the smoke-test gate that
    # Phase 4's QDQ quant depends on.
    gru_ops = [n for n in model.graph.node if n.op_type == "GRU"]
    op_types = [n.op_type for n in model.graph.node]
    assert len(gru_ops) == 0, (
        f"Pitfall 11: {len(gru_ops)} opaque GRU op(s) survived export. "
        f"Phase 4 quantize_static cannot rewrite this op — the GRU compute "
        f"would run FP32 internally even after quant. "
        f"Check NSNet2Streaming.forward_step in models/streaming.py: it must NOT "
        f"call self.base.gru(...) or any nn.GRU(...) — only inline MatMul/Add/Sigmoid/Tanh "
        f"using the per-gate nn.Parameter tensors set up in __init__."
    )

    # Secondary check: aten::*gru* (Pitfall 10 lookalike — should have failed
    # at export already, but covered for completeness).
    aten_gru_ops = [op for op in op_types if op.startswith("aten::") and "gru" in op.lower()]
    assert len(aten_gru_ops) == 0, (
        f"Pitfall 10 lookalike: {aten_gru_ops} — likely nn.GRUCell or nn.RNNCell "
        f"in NSNet2Streaming.forward_step. Replace with raw nn.Parameter MatMul + Add."
    )


def test_onnx_checker_passes(exported_path):
    """Graph-validity smoke: onnx.checker.check_model accepts the exported file."""
    model = onnx.load(str(exported_path))
    onnx.checker.check_model(model)  # raises onnx.checker.ValidationError on invalid graph
