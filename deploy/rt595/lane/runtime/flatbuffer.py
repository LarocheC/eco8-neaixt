"""Direct .tflite flatbuffer parsing.

Deliberately does NOT go through `ai_edge_litert.Interpreter`:

* `interpreter._get_ops_details()` segfaults on some of the RT595 artifacts, and
* one artifact segfaults the whole process merely on construction.

Everything here is a pure read of the buffer via the vendored
`schema_py_generated` module, so it is safe on a malformed / hostile model.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# --------------------------------------------------------------------------
# schema module discovery
# --------------------------------------------------------------------------

# Where the ai_edge_litert-vendored flatbuffer schema lives.  Import order:
#   1. plain `ai_edge_litert.schema_py_generated` (works in the rt595-export venv)
#   2. $LANE_SCHEMA_DIR/schema_py_generated.py
#   3. the directory of an importable `ai_edge_litert` package, on sys.path
_SCHEMA_ENV = "LANE_SCHEMA_DIR"

_schema = None
_schema_error: Optional[str] = None


def _load_schema():
    """Import the tflite flatbuffer schema module, memoised."""
    global _schema, _schema_error
    if _schema is not None:
        return _schema
    if _schema_error is not None:
        raise ImportError(_schema_error)

    tried = []

    override = os.environ.get(_SCHEMA_ENV)
    if override:
        tried.append(override)
        if override not in sys.path:
            sys.path.insert(0, override)

    try:
        from ai_edge_litert import schema_py_generated as mod  # type: ignore

        _schema = mod
        return _schema
    except Exception as exc:  # pragma: no cover - depends on the venv
        tried.append("ai_edge_litert.schema_py_generated (%s)" % exc)

    try:
        import schema_py_generated as mod  # type: ignore

        _schema = mod
        return _schema
    except Exception as exc:
        tried.append("bare schema_py_generated (%s)" % exc)

    try:
        import ai_edge_litert  # type: ignore

        pkg_dir = os.path.dirname(ai_edge_litert.__file__)
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)
        import schema_py_generated as mod  # type: ignore

        _schema = mod
        return _schema
    except Exception as exc:
        tried.append("ai_edge_litert package dir (%s)" % exc)

    _schema_error = (
        "could not import the tflite flatbuffer schema module. Tried: "
        + "; ".join(tried)
        + ". Set $%s to the directory holding schema_py_generated.py "
        "(in this repo: <venv>/lib/python3.10/site-packages/ai_edge_litert)."
        % _SCHEMA_ENV
    )
    raise ImportError(_schema_error)


def _enum_names(enum_cls) -> Dict[int, str]:
    """int -> NAME for a flatbuffers-generated enum class."""
    out: Dict[int, str] = {}
    for name in dir(enum_cls):
        if name.startswith("_"):
            continue
        val = getattr(enum_cls, name)
        if isinstance(val, int):
            # First name wins: the generated module has no aliases in practice,
            # but be deterministic if it ever grows one.
            out.setdefault(val, name)
    return out


# --------------------------------------------------------------------------
# public data model
# --------------------------------------------------------------------------


# Inclusive zero_point range implied by the tensor dtype.  A zero_point outside
# it cannot be represented by the tensor's own storage type, so the buffer is
# malformed by TFLite's own rules regardless of what any runtime does with it.
ZERO_POINT_RANGE: Dict[str, Tuple[int, int]] = {
    "INT4": (-8, 7),
    "UINT4": (0, 15),
    "INT8": (-128, 127),
    "UINT8": (0, 255),
    "INT16": (-32768, 32767),
    "UINT16": (0, 65535),
    "INT32": (-(2 ** 31), 2 ** 31 - 1),
    "UINT32": (0, 2 ** 32 - 1),
    "INT64": (-(2 ** 63), 2 ** 63 - 1),
}


@dataclass(frozen=True)
class TensorInfo:
    """One tensor of subgraph 0, with its quantization parameters."""

    index: int
    name: str
    dtype: str
    shape: tuple
    scales: tuple
    zero_points: tuple
    is_quantized: bool
    quantized_dimension: int = 0

    # -- convenience predicates used by the quant gates -------------------
    def has_nonfinite_scale(self) -> bool:
        return any(not math.isfinite(s) for s in self.scales)

    def has_nonpositive_scale(self) -> bool:
        """True if any scale is <= 0.

        Reads the raw tuple, deliberately NOT the `abs()`-ed view
        `min_abs_scale()` returns: the TFLite spec requires ``scale > 0``, and a
        negative scale of ordinary magnitude is invisible to both a finiteness
        test and a magnitude test.
        """
        return any(math.isfinite(s) and s <= 0.0 for s in self.scales)

    def nonpositive_scales(self) -> tuple:
        return tuple(s for s in self.scales if math.isfinite(s) and s <= 0.0)

    def min_abs_scale(self) -> Optional[float]:
        finite = [abs(s) for s in self.scales if math.isfinite(s)]
        return min(finite) if finite else None

    def min_positive_scale(self) -> Optional[float]:
        """Smallest strictly-positive scale, or None.

        This is what the collapsed-range (floor) gate wants: zero and negative
        scales are *invalid*, not *small*, and belong to G-SCALE-POSITIVE.
        """
        pos = [s for s in self.scales if math.isfinite(s) and s > 0.0]
        return min(pos) if pos else None

    def quantized_dim_size(self) -> Optional[int]:
        """Length of the axis the per-axis scale vector must match, or None."""
        d = self.quantized_dimension
        if not self.shape or d < 0 or d >= len(self.shape):
            return None
        return int(self.shape[d])


@dataclass(frozen=True)
class ModelInfo:
    """Everything the gates need, read straight out of the buffer."""

    path: str
    n_subgraphs: int
    op_histogram: dict          # {"CONV_2D": 20, ...}; custom ops as "CUSTOM:Name"
    tensors: tuple              # of TensorInfo, subgraph 0
    inputs: tuple               # positional tensor indices, subgraph 0
    outputs: tuple              # positional tensor indices, subgraph 0
    op_versions: dict           # {"CONV_2D": 3, ...} max version seen

    # -- convenience ------------------------------------------------------
    @property
    def op_set(self) -> set:
        return set(self.op_histogram)

    @property
    def n_ops(self) -> int:
        return sum(self.op_histogram.values())

    def tensor(self, index: int) -> TensorInfo:
        return self.tensors[index]


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def _decode(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return str(raw)


def _builtin_code(op_code, schema) -> int:
    """Resolve an OperatorCode to a BuiltinOperator value.

    Mirrors tflite's `GetBuiltinCode`: `deprecated_builtin_code` is an int8 that
    saturates at 127 (PLACEHOLDER_FOR_GREATER_OP_CODES), so the real code is the
    max of the two fields.
    """
    try:
        new = int(op_code.BuiltinCode())
    except Exception:
        new = 0
    try:
        old = int(op_code.DeprecatedBuiltinCode())
    except Exception:
        old = 0
    return max(new, old)


def load_model(path: str) -> ModelInfo:
    """Parse `path` and return a `ModelInfo` for subgraph 0.

    Raises FileNotFoundError / ValueError on a buffer that is not a tflite model.
    """
    schema = _load_schema()

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as fh:
        buf = bytearray(fh.read())

    if len(buf) < 8:
        raise ValueError("%s: too small to be a tflite flatbuffer" % path)
    # The identifier lives at bytes 4..8 for a size-prefixed-free root.
    ident = bytes(buf[4:8])
    if ident not in (b"TFL3", b"TFL2"):
        raise ValueError(
            "%s: file identifier %r is not TFL3/TFL2 -- not a tflite model" % (path, ident)
        )

    model = schema.Model.GetRootAsModel(buf, 0)

    builtin_names = _enum_names(schema.BuiltinOperator)
    type_names = _enum_names(schema.TensorType)

    n_subgraphs = model.SubgraphsLength()
    if n_subgraphs == 0:
        raise ValueError("%s: model has no subgraphs" % path)

    # ---- operator codes -------------------------------------------------
    op_code_names = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        code = _builtin_code(oc, schema)
        name = builtin_names.get(code, "UNKNOWN_%d" % code)
        if name == "CUSTOM":
            custom = _decode(oc.CustomCode())
            name = "CUSTOM:%s" % (custom or "<unnamed>")
        op_code_names.append(name)

    sub = model.Subgraphs(0)

    # ---- op histogram + versions ---------------------------------------
    op_histogram: Dict[str, int] = {}
    op_versions: Dict[str, int] = {}
    for i in range(sub.OperatorsLength()):
        op = sub.Operators(i)
        idx = op.OpcodeIndex()
        if 0 <= idx < len(op_code_names):
            name = op_code_names[idx]
            version = int(model.OperatorCodes(idx).Version())
        else:  # pragma: no cover - corrupt model
            name = "UNKNOWN_OPCODE_INDEX_%d" % idx
            version = 0
        op_histogram[name] = op_histogram.get(name, 0) + 1
        op_versions[name] = max(op_versions.get(name, 0), version)

    # ---- tensors --------------------------------------------------------
    tensors = []
    for i in range(sub.TensorsLength()):
        t = sub.Tensors(i)
        shape = tuple(int(t.Shape(j)) for j in range(t.ShapeLength()))
        dtype = type_names.get(int(t.Type()), "UNKNOWN_%d" % int(t.Type()))

        scales: Tuple[float, ...] = ()
        zeros: Tuple[int, ...] = ()
        qdim = 0
        q = t.Quantization()
        if q is not None:
            n_s = q.ScaleLength()
            if n_s:
                scales = tuple(float(q.Scale(j)) for j in range(n_s))
            n_z = q.ZeroPointLength()
            if n_z:
                zeros = tuple(int(q.ZeroPoint(j)) for j in range(n_z))
            try:
                qdim = int(q.QuantizedDimension())
            except Exception:  # pragma: no cover - very old schema
                qdim = 0

        tensors.append(
            TensorInfo(
                index=i,
                name=_decode(t.Name()),
                dtype=dtype,
                shape=shape,
                scales=scales,
                zero_points=zeros,
                is_quantized=bool(scales),
                quantized_dimension=qdim,
            )
        )

    inputs = tuple(int(sub.Inputs(i)) for i in range(sub.InputsLength()))
    outputs = tuple(int(sub.Outputs(i)) for i in range(sub.OutputsLength()))

    return ModelInfo(
        path=os.path.abspath(path),
        n_subgraphs=int(n_subgraphs),
        op_histogram=op_histogram,
        tensors=tuple(tensors),
        inputs=inputs,
        outputs=outputs,
        op_versions=op_versions,
    )


def builtin_operator_names() -> Dict[int, str]:
    """BuiltinOperator value -> name, straight from the vendored schema."""
    return _enum_names(_load_schema().BuiltinOperator)


def tensor_type_names() -> Dict[int, str]:
    """TensorType value -> name, straight from the vendored schema."""
    return _enum_names(_load_schema().TensorType)


__all__ = [
    "TensorInfo",
    "ModelInfo",
    "ZERO_POINT_RANGE",
    "load_model",
    "builtin_operator_names",
    "tensor_type_names",
]
