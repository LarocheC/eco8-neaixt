"""Register a QDQ quantizer for ``Einsum`` so the structured weight operands
actually get int8-quantized.

The structure-preserving ONNX export (``nsnet2/export_onnx.py``) lowers each
BlockdiagLinear / MonarchLinear matmul to an ``Einsum`` whose weight operand is
the raw structured factor tensor — ``[nblocks, out_blksz, in_blksz]`` for the
single block-diagonal factor, or the Monarch ``w1 [nblocks, q, p]`` /
``w2 [nblocks, s, r]``. onnxruntime ships no QDQ handler for ``Einsum``, so it
was never in ``quantize_static``'s default op set (that default is
``QLinearOpsRegistry.keys() | QDQRegistry.keys()``). The Einsum nodes were
skipped entirely: only activations and the residual dense ``MatMul`` weights
became int8, while the *dominant* structured weights stayed FP32. That made the
"int8" model a hybrid and its "int8 is loss-free" behaviour an artifact — the
weights that carry the model were never quantized.

Importing this module (for its side effect) registers ``QDQEinsum`` in
``QDQRegistry`` under ``"Einsum"``. Because ``quantize_static`` computes its
default ``op_types_to_quantize`` from the registry keys at call time, this both
opts ``Einsum`` into quantization and gives it a weight-aware handler — no
change to the ``quantize_static(...)`` call is needed. The registration MUST run
before ``quantize_static`` (import-time here satisfies that).

The weight operand is quantized per-channel along **axis=1**. That is the layout
axis for these tensors: for the block-diagonal factor ``[nblocks, out_blksz,
in_blksz]`` axis=1 is the per-output-row-within-block channel, and for the
Monarch factors the same axis applies. It matches the precedent in
``common/quant_fake.py`` (its ``_is_blockdiag_linear`` / ``MonarchLinear``
branches quantize these tensors on axis=1), and ``export_onnx.py`` feeds that
identical, untouched factor tensor straight into the ``Einsum`` — same tensor,
same layout, same axis.

Usage: ``import nsnet2.qdq_einsum_quantizer  # noqa: F401`` (side-effecting).
"""

import itertools

from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase
from onnxruntime.quantization.quant_utils import find_by_name
from onnxruntime.quantization.registry import QDQRegistry

# Per-channel axis for the structured (block-diagonal / Monarch) factor tensors.
# See module docstring and common/quant_fake.py for why axis=1.
_WEIGHT_CHANNEL_AXIS = 1


class QDQEinsum(QDQOperatorBase):
    """QDQ quantizer for ``Einsum`` nodes.

    Body mirrors ``onnxruntime.quantization.operators.matmul.QDQMatMul.quantize``
    (the op-type assert is swapped to ``"Einsum"``): walk the node's inputs (and
    outputs, unless output quantization is disabled for this node); quantize any
    graph-initializer operand as a per-channel weight (axis=1) and every other
    operand as a per-tensor activation.
    """

    def quantize(self):
        node = self.node
        assert node.op_type == "Einsum"

        if self.disable_qdq_for_node_output:
            nodes_to_iterate = node.input
        else:
            nodes_to_iterate = itertools.chain(node.input, node.output)

        for tensor_name in nodes_to_iterate:
            if find_by_name(tensor_name, self.quantizer.model.initializer()):
                self.quantizer.quantize_weight_tensor_per_channel(
                    tensor_name, _WEIGHT_CHANNEL_AXIS
                )
            else:
                self.quantizer.quantize_activation_tensor(tensor_name)


# Side effect on import: make quantize_static discover + weight-quantize Einsum.
QDQRegistry["Einsum"] = QDQEinsum
