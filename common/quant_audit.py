"""Post-quantization audit: catch compute nodes whose weights stayed FP32.

Model-agnostic, and it exists because of a specific, expensive failure. NSNet2's
structure-preserving export lowered every block-diagonal / Monarch matmul to an
``Einsum``; onnxruntime ships no QDQ handler for ``Einsum``, so
``quantize_static`` skipped those nodes entirely and quantized only the
activations and the residual dense ``MatMul``s. The graph still had plenty of
``QuantizeLinear`` nodes, so every "is it quantized?" smoke check passed — and
every structured int8 PESQ number published before 2026-07-11 was really a
hybrid-precision model. See the correction notice at the top of
``RESULTS_NSNET2.md`` and the workaround in ``nsnet2/qdq_einsum_quantizer.py``.

The lesson generalizes past ``Einsum``: any op the quantizer does not handle
(or that lands in ``nodes_to_exclude``) keeps a raw FLOAT initializer as its
weight, silently. This checks the weight operand of every compute node instead
of counting ``QuantizeLinear`` nodes.

Biases are deliberately not checked: onnxruntime's QDQ format leaves Conv/Gemm
bias as an FP32 initializer by design.
"""

from __future__ import annotations

from typing import Iterable, Optional

import onnx


# Which input index carries the weight, per op type. ``None`` means "any input
# that is a float initializer is a weight" (Einsum/MatMul have no fixed slot).
_WEIGHT_INPUT = {
    "Conv": 1,
    "ConvTranspose": 1,
    "Gemm": 1,
    "MatMul": None,
    "Einsum": None,
}


def float_weight_operands(model: onnx.ModelProto,
                          exclude_nodes: Optional[Iterable[str]] = None) -> list:
    """Return ``(op_type, node_name, initializer_name)`` for unquantized weights.

    An entry means: this compute node reads its weight straight from an FP32
    initializer, i.e. the quantizer never touched it.
    """
    exclude = set(exclude_nodes or ())
    float_inits = {
        init.name for init in model.graph.initializer
        if init.data_type == onnx.TensorProto.FLOAT
    }
    offenders = []
    for node in model.graph.node:
        if node.op_type not in _WEIGHT_INPUT or node.name in exclude:
            continue
        slot = _WEIGHT_INPUT[node.op_type]
        candidates = node.input if slot is None else node.input[slot:slot + 1]
        for name in candidates:
            if name in float_inits:
                offenders.append((node.op_type, node.name, name))
    return offenders


def assert_all_weights_quantized(model: onnx.ModelProto,
                                 exclude_nodes: Optional[Iterable[str]] = None,
                                 context: str = "") -> None:
    """Raise unless every compute node's weight went through the quantizer."""
    offenders = float_weight_operands(model, exclude_nodes)
    if offenders:
        listed = ", ".join(f"{op}[{name or '<unnamed>'}]<-{init}"
                           for op, name, init in offenders[:8])
        raise AssertionError(
            f"{context}int8 model has {len(offenders)} compute node(s) still "
            f"reading FP32 weight initializers: {listed}"
            f"{' ...' if len(offenders) > 8 else ''}. The graph is "
            f"hybrid-precision, not int8 — any quality number measured on it "
            f"would be an artifact (this is exactly the Einsum/QDQ failure that "
            f"forced the int8 retraction in RESULTS_NSNET2.md). Either the op "
            f"has no QDQ handler in this onnxruntime, or it landed in "
            f"nodes_to_exclude."
        )
