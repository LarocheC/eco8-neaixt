"""Fake-quant utilities for the int4/int8 weight + activation PTQ study.

A small, deliberately-non-invasive layer that lets us measure PESQ
degradation as we drop the weight bit depth (target: w4 weights, a8
activations) without touching training code or adding deployment
dependencies. Pure eager-mode PyTorch — no Triton kernel changes, no
ONNX, no quantize_static.

Two pieces:

1. ``fake_quant_tensor`` — symmetric quantize → dequantize on a tensor.
   Supports per-tensor (axis=None), per-channel (axis=int), and
   per-group (axis=int, group_size=int) granularity. Pure functional.

2. ``apply_weight_fake_quant`` / ``install_activation_fake_quant`` —
   apply ``fake_quant_tensor`` to a loaded model's weights (in place)
   and stamp pre-forward hooks on every weight-bearing leaf module to
   fake-quantize its input activation. After installation the model
   runs through the regular forward — every Linear / BlockdiagLinear /
   Butterfly etc. sees fake-quantized inputs and uses
   fake-quantized weights.

Scope: PTQ only (study question is "does w4 weight quant preserve
PESQ?"). For QAT we'd swap to gru_qat's FakeQuantize modules — same
math, different state lifecycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core fake-quant op
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QSpec:
    """Quantization specification for one tensor.

    Symmetric only. ``bits >= 32`` is identity (pass-through).
    """
    bits: int = 32
    axis: Optional[int] = None        # None = per-tensor
    group_size: Optional[int] = None  # None = per-channel along axis; int = per-group


class _STERound(torch.autograd.Function):
    """Round with straight-through gradient: forward=round, backward=identity."""
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class _STEClamp(torch.autograd.Function):
    """Clamp with the standard QAT gradient: zero outside [lo, hi], identity inside."""
    @staticmethod
    def forward(ctx, x, lo, hi):
        ctx.save_for_backward(((x >= lo) & (x <= hi)).to(x.dtype))
        return x.clamp(lo, hi)

    @staticmethod
    def backward(ctx, g):
        mask, = ctx.saved_tensors
        return g * mask, None, None


def fake_quant_tensor(x: torch.Tensor, spec: QSpec) -> torch.Tensor:
    """Symmetric quantize→dequantize round-trip. Returns a tensor of
    the same shape and dtype, holding the post-dequant values.

    Uses straight-through estimator on the round + clamped-passthrough on
    the clamp so the same op is valid for PTQ (no grad) and QAT (grad
    flows through to underlying weights/activations). Scales are computed
    from ``x.detach()`` so they don't enter the autograd graph — this is
    the standard "dynamic-scale" fake-quant; learnable scales (LSQ) would
    be a follow-up upgrade.
    """
    if spec.bits >= 32:
        return x
    qmin = -(2 ** (spec.bits - 1)) + 1
    qmax = 2 ** (spec.bits - 1) - 1

    if spec.axis is None:
        # Per-tensor: one absmax across the whole tensor.
        absmax = x.detach().abs().max().clamp_min(1e-12)
        scale = absmax / qmax
        q = _STEClamp.apply(_STERound.apply(x / scale), qmin, qmax)
        return q * scale

    # Move the quantization axis to position 0 for slicing.
    x_perm = x.movedim(spec.axis, 0)
    n = x_perm.shape[0]

    if spec.group_size is None:
        # Per-channel: one absmax per slice along axis.
        flat = x_perm.detach().reshape(n, -1)
        absmax = flat.abs().amax(dim=-1).clamp_min(1e-12)        # (n,)
        scale = (absmax / qmax).reshape((n,) + (1,) * (x_perm.ndim - 1))
        q = _STEClamp.apply(_STERound.apply(x_perm / scale), qmin, qmax) * scale
        return q.movedim(0, spec.axis)

    g = spec.group_size
    if n % g != 0:
        raise ValueError(
            f"per-group quant: axis {spec.axis} dim {n} not divisible by group_size {g}"
        )
    # Reshape (n, ...) -> (n//g, g, ...) and reduce over the g axis + all trailing.
    x_grp = x_perm.reshape(n // g, g, *x_perm.shape[1:])
    flat = x_grp.detach().reshape(n // g, -1)
    absmax = flat.abs().amax(dim=-1).clamp_min(1e-12)            # (n//g,)
    scale = (absmax / qmax).reshape((n // g,) + (1,) * (x_grp.ndim - 1))
    q = _STEClamp.apply(_STERound.apply(x_grp / scale), qmin, qmax) * scale
    return q.reshape(x_perm.shape).movedim(0, spec.axis)


# ---------------------------------------------------------------------------
# Module walker — identify what to quantize and on which axis
# ---------------------------------------------------------------------------


# Lazy imports so the file works even without the optional deps.
def _is_blockdiag_linear(m: nn.Module) -> bool:
    try:
        from torch_structured.monarch.blockdiag_linear import BlockdiagLinear
        return isinstance(m, BlockdiagLinear)
    except ImportError:
        return False


def _is_monarch_linear(m: nn.Module) -> bool:
    try:
        from torch_structured.monarch.monarch_linear import MonarchLinear
        return isinstance(m, MonarchLinear)
    except ImportError:
        return False


def _is_butterfly(m: nn.Module) -> bool:
    try:
        from torch_structured.butterfly.butterfly import Butterfly
        return isinstance(m, Butterfly)
    except ImportError:
        return False


def _is_gru_qat_cell(m: nn.Module) -> bool:
    try:
        from gru_qat.gru_cell import GRUCellQuant
        return isinstance(m, GRUCellQuant)
    except ImportError:
        return False


# Dense weight parameters inside gru_qat.GRUCellQuant — input projections
# (always present) and hidden projections (present only when the cell's
# hidden side is dense; ignored otherwise). They are raw nn.Parameter, not
# wrapped in nn.Linear, so the generic walker above misses them.
_GRU_QAT_CELL_WEIGHTS = (
    ("W_ir", 0),
    ("W_iz", 0),
    ("W_in", 0),
    ("W_hr", 0),
    ("W_hz", 0),
    ("W_hn", 0),
)


def _quantizable_weight_attrs(m: nn.Module):
    """Yield (attr_name, axis_for_per_channel) for the model-specific weight tensors.

    Returns an empty iterator if the module isn't a leaf we know how to handle.
    """
    if isinstance(m, nn.Linear):
        yield "weight", 0
    elif isinstance(m, nn.Conv1d):
        # weight shape (out_channels, in_channels/groups, kernel). axis=0 is
        # per-output-channel — matches onnxruntime per-channel weight quant,
        # and commutes with the BN-into-Conv fold ConvFSENet does at export
        # (a per-output-channel rescale; per-channel symmetric quant is
        # invariant to it). Covers depthwise convs (groups>1) too.
        yield "weight", 0
    elif isinstance(m, nn.Conv2d):
        # weight (out_channels, in_channels/groups, kh, kw): axis=0 is
        # per-output-channel, same rationale as Conv1d. Covers LiSenNet's
        # depthwise/pointwise convs and the BatchNorm-into-Conv fold.
        yield "weight", 0
    elif isinstance(m, nn.ConvTranspose2d):
        # transposed-conv weight is (in_channels, out_channels/groups, kh, kw):
        # the OUTPUT-channel axis is 1, not 0 — per-output-channel quant (matching
        # onnxruntime's per-channel ConvTranspose weight quant) uses axis=1.
        yield "weight", 1
    elif isinstance(m, nn.GRU):
        for k in range(m.num_layers):
            yield f"weight_ih_l{k}", 0
            yield f"weight_hh_l{k}", 0
    elif _is_blockdiag_linear(m):
        # weight shape (nblocks, out_blksz, in_blksz). axis=1 is the
        # per-output-channel within each block — closest analogue to
        # nn.Linear's axis=0.
        yield "weight", 1
    elif _is_monarch_linear(m):
        # Genuine two-factor Monarch: w1 (nblocks, in_blksz, in_blksz) and
        # w2 (nblocks, out_blksz, in_blksz). Quantize each factor per
        # output-row within its block (axis=1), the same per-output-channel
        # analogue used for the single block-diagonal factor above.
        yield "w1", 1
        yield "w2", 1
    elif _is_gru_qat_cell(m):
        for name, axis in _GRU_QAT_CELL_WEIGHTS:
            w = getattr(m, name, None)
            if isinstance(w, nn.Parameter):
                yield name, axis
    elif type(m).__name__ == "NSNet2Streaming":
        # The cuDNN streaming wrapper unpacks nn.GRU into 12
        # nn.Parameters per layer (W_ir_<k>, ..., W_hn_<k>, b_ir_<k>...).
        # We only quantize weights (not biases) per the convention used
        # for the other module types in this walker. Skip if the source
        # gru_kind was non-"gru" (then these attributes don't exist and
        # the structured FC + GRUCellQuant branches above handle it).
        for k in range(int(getattr(m, "num_layers", 0))):
            for gate in ("ir", "iz", "in", "hr", "hz", "hn"):
                attr = f"W_{gate}_{k}"
                w = getattr(m, attr, None)
                if isinstance(w, nn.Parameter):
                    yield attr, 0
    elif _is_butterfly(m):
        # twiddle shape (nstacks, butterfly_nblocks, log_n, n//2, 2, 2).
        # Per-tensor (axis=None) for v1 — the 2x2 twiddles are tiny so
        # per-channel doesn't buy much. The QSpec passed in by the
        # caller may override this default; we just need to flag this
        # as a weight tensor.
        yield "twiddle", None


# ---------------------------------------------------------------------------
# In-place weight quant
# ---------------------------------------------------------------------------


@torch.no_grad()
def apply_weight_fake_quant(model: nn.Module, weight_spec: QSpec) -> int:
    """Walk model leaves, fake-quantize each weight tensor in place.

    Returns the count of tensors quantized. Idempotent (re-applying with
    the same spec yields the same numbers, because fake-quant is its own
    fixed point at fp32 dequant).

    For BlockdiagLinear / Butterfly, the per-channel axis above is used
    in place of weight_spec.axis when weight_spec.axis is None and the
    user wants per-channel; otherwise the user's axis is respected.
    """
    n = 0
    for m in model.modules():
        for attr, default_axis in _quantizable_weight_attrs(m):
            spec = weight_spec
            if weight_spec.bits >= 32:
                continue
            # If caller didn't pick an axis, use the per-module default. This
            # applies for BOTH per-channel (group_size=None) and per-group
            # (group_size=N) since "group" only makes sense along an axis —
            # axis=None + group_size=N otherwise falls through to per-tensor
            # in fake_quant_tensor, which is almost certainly not what the
            # user meant when they passed --w_group_size.
            if spec.axis is None and default_axis is not None:
                spec = QSpec(bits=spec.bits, axis=default_axis,
                             group_size=spec.group_size)
            w = getattr(m, attr)
            # If group_size doesn't divide the chosen axis (e.g. monarch
            # axis=1 dim 50 with group_size=64), silently fall back to
            # per-channel along the same axis. Better than crashing mid-
            # sweep; the per-channel result is what you'd want anyway when
            # the axis is smaller than the requested group.
            if (
                spec.group_size is not None
                and spec.axis is not None
                and w.shape[spec.axis] % spec.group_size != 0
            ):
                spec = QSpec(bits=spec.bits, axis=spec.axis, group_size=None)
            w.copy_(fake_quant_tensor(w, spec))
            n += 1
    return n


# ---------------------------------------------------------------------------
# Activation fake-quant via pre-forward hooks
# ---------------------------------------------------------------------------


class _FakeQuantSwap(nn.Module):
    """nn.Module that fake-quantizes its input via fake_quant_tensor.

    Used to replace the Identity placeholder modules that gru_qat.GRUCellQuant
    keeps around for activation quant (quant_x, quant_h_in, quant_h_out, and
    the per-gate quant_W_*). The cell's per-step body calls these modules
    explicitly, so swapping them in is the cleanest way to fake-quant
    activations that aren't on a Linear/BlockdiagLinear/Butterfly forward
    boundary.
    """
    def __init__(self, spec: QSpec):
        super().__init__()
        self.spec = spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fake_quant_tensor(x, self.spec)


# Activation quantizer slots inside gru_qat.GRUCellQuant. These three are
# called every step regardless of dense vs structured topology, so swapping
# them covers both the cell's input activation (quant_x) and the hidden
# carry on both read and write sides.
_GRU_QAT_CELL_ACT_SLOTS = ("quant_x", "quant_h_in", "quant_h_out")


def install_activation_fake_quant(model: nn.Module, act_spec: QSpec) -> int:
    """Stamp a pre-forward hook on every weight-bearing leaf so its first
    positional input gets fake-quantized. Also swaps the activation quant
    slots inside gru_qat.GRUCellQuant so the cell's per-step body sees
    quantized x / h on its dense-input matmul path.

    Per-tensor activation quant is recomputed every forward (dynamic
    absmax) — matches typical int8 inference semantics where activation
    scales aren't frozen.

    Returns the count of hooks installed. The hooks are NOT removable
    through this API; reload the checkpoint to reset.
    """
    if act_spec.bits >= 32:
        return 0
    n = 0
    targets = []
    for m in model.modules():
        if (
            isinstance(m, (nn.Linear, nn.GRU, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d))
            or _is_blockdiag_linear(m)
            or _is_monarch_linear(m)
            or _is_butterfly(m)
        ):
            targets.append(m)

    def _make_hook(spec):
        def _hook(_mod, args, _kwargs):
            if not args:
                return None
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return None
            return (fake_quant_tensor(x, spec),) + args[1:], _kwargs
        return _hook

    for m in targets:
        m.register_forward_pre_hook(_make_hook(act_spec), with_kwargs=True)
        n += 1

    # Swap the GRUCellQuant activation slots. These are nn.Modules
    # (Identity by default for our fp32 checkpoints), so a clean setattr
    # replaces them without invalidating any module hierarchy.
    for m in model.modules():
        if _is_gru_qat_cell(m):
            for slot in _GRU_QAT_CELL_ACT_SLOTS:
                if hasattr(m, slot):
                    setattr(m, slot, _FakeQuantSwap(act_spec))
                    n += 1
    return n


# ---------------------------------------------------------------------------
# Convenience preset
# ---------------------------------------------------------------------------


def apply_ptq(model: nn.Module, *, weight: QSpec, activation: QSpec) -> tuple[int, int]:
    """Combined: in-place weight quant + activation pre-hooks. Returns counts."""
    nw = apply_weight_fake_quant(model, weight)
    na = install_activation_fake_quant(model, activation)
    return nw, na


# ---------------------------------------------------------------------------
# QAT preparation (weights stay trainable; quant happens on every forward)
# ---------------------------------------------------------------------------


class _FakeQuantWeightParam(nn.Module):
    """Parametrization that applies fake-quant on a weight every forward.

    Used with ``torch.nn.utils.parametrize.register_parametrization`` so
    the underlying nn.Parameter stays trainable while every read of
    ``module.weight`` (or whichever attr we parametrize) yields a
    fake-quantized tensor. Gradients flow through STE in
    ``fake_quant_tensor`` back to the underlying weight.
    """
    def __init__(self, spec: QSpec):
        super().__init__()
        self.spec = spec

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return fake_quant_tensor(w, self.spec)


# ---------------------------------------------------------------------------
# Static (observer-based) activation fake-quant
# ---------------------------------------------------------------------------


class StaticActFakeQuant(nn.Module):
    """Asymmetric int8 activation fake-quant with a STATIC (observer) scale.

    Unlike install_activation_fake_quant's dynamic per-forward absmax, this
    EMA-tracks the activation [min, max] during train() and fake-quantizes
    with the *accumulated* range — the same frozen-scale, asymmetric-int8
    semantics that onnxruntime.quantize_static deploys. In eval() the range
    is frozen. This is the fake-quant QAT must train against when the
    deployment target is static int8 PTQ.

    The first forward (any mode) seeds the observer from the current tensor
    so an un-calibrated module degrades to dynamic rather than to garbage.
    """

    def __init__(self, bits: int = 8, momentum: float = 0.01):
        super().__init__()
        self.bits = int(bits)
        self.momentum = float(momentum)
        self.register_buffer("running_min", torch.zeros(()))
        self.register_buffer("running_max", torch.zeros(()))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits >= 32:
            return x
        if self.training or int(self.initialized) == 0:
            with torch.no_grad():
                cur_min = x.min()
                cur_max = x.max()
            if int(self.initialized) == 0:
                self.running_min.copy_(cur_min)
                self.running_max.copy_(cur_max)
                self.initialized.fill_(1)
            elif self.training:
                m = self.momentum
                self.running_min.mul_(1 - m).add_(m * cur_min)
                self.running_max.mul_(1 - m).add_(m * cur_max)
        qmin = -(2 ** (self.bits - 1))
        qmax = 2 ** (self.bits - 1) - 1
        # Asymmetric range must straddle 0 so the zero-point is representable.
        lo = torch.minimum(self.running_min, torch.zeros_like(self.running_min))
        hi = torch.maximum(self.running_max, torch.zeros_like(self.running_max))
        scale = ((hi - lo) / (qmax - qmin)).clamp_min(1e-12)
        zero_point = torch.round(qmin - lo / scale)
        q = _STEClamp.apply(_STERound.apply(x / scale + zero_point), qmin, qmax)
        return (q - zero_point) * scale


def install_static_activation_fake_quant(model: nn.Module, bits: int) -> int:
    """Stamp a StaticActFakeQuant on every weight-bearing leaf's input.

    The per-module quantizers live in an nn.ModuleList registered as
    ``model._act_fake_quant`` so their observer buffers move with .to() and
    flip train/eval with the model. Reuses an existing same-length list so a
    re-install after a checkpoint save keeps the accumulated observers.

    Returns the count of activation quant points installed.
    """
    if bits >= 32:
        return 0
    targets = [
        m for m in model.modules()
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d))
        or _is_blockdiag_linear(m) or _is_monarch_linear(m) or _is_butterfly(m)
    ]
    existing = getattr(model, "_act_fake_quant", None)
    if isinstance(existing, nn.ModuleList) and len(existing) == len(targets):
        quants = existing
    else:
        quants = nn.ModuleList([StaticActFakeQuant(bits) for _ in targets])
        model.add_module("_act_fake_quant", quants)

    def _make_hook(q):
        def _hook(_mod, args, _kwargs):
            if not args or not isinstance(args[0], torch.Tensor):
                return None
            return (q(args[0]),) + args[1:], _kwargs
        return _hook

    for m, q in zip(targets, quants):
        m.register_forward_pre_hook(_make_hook(q), with_kwargs=True)
    return len(targets)


def prepare_for_qat(
    model: nn.Module,
    *,
    weight: QSpec,
    activation: QSpec,
    static_activation: bool = False,
) -> tuple[int, int]:
    """Install quantization-aware-training scaffolding in-place.

    Two differences from ``apply_ptq``:

    1. Weight quant goes through ``nn.utils.parametrize`` instead of
       in-place mutation, so the underlying nn.Parameter remains
       trainable and every forward reads a freshly-fake-quantized view.
       Gradients flow through STE back to the original weight.
    2. Sets ``model._qat_prepared = True`` so downstream eval scripts
       know to skip ``apply_ptq`` (which would double-quantize).

    Returns ``(n_weight_params, n_act_quant_points)``.
    """
    from torch.nn.utils import parametrize

    nw = 0
    if weight.bits < 32:
        for m in model.modules():
            for attr, default_axis in _quantizable_weight_attrs(m):
                spec = weight
                if spec.axis is None and default_axis is not None:
                    spec = QSpec(bits=spec.bits, axis=default_axis,
                                 group_size=spec.group_size)
                # Apply the group_size-divisibility fallback (same as PTQ).
                w_tensor = getattr(m, attr)
                if (
                    spec.group_size is not None
                    and spec.axis is not None
                    and w_tensor.shape[spec.axis] % spec.group_size != 0
                ):
                    spec = QSpec(bits=spec.bits, axis=spec.axis, group_size=None)
                # Only register once per attr to keep idempotency on repeated calls.
                if not parametrize.is_parametrized(m, attr):
                    parametrize.register_parametrization(
                        m, attr, _FakeQuantWeightParam(spec),
                    )
                    nw += 1

    # static_activation=True: deployment-matching static/asymmetric int8
    # activation fake-quant (observer-based). Default False keeps the legacy
    # dynamic per-tensor symmetric path.
    if static_activation:
        na = install_static_activation_fake_quant(model, activation.bits)
    else:
        na = install_activation_fake_quant(model, activation)
    model._qat_prepared = True  # type: ignore[assignment]
    model._qat_weight_spec = weight
    model._qat_activation_spec = activation
    return nw, na


def is_qat_prepared(model: nn.Module) -> bool:
    """True iff ``prepare_for_qat`` has been called on this model."""
    return bool(getattr(model, "_qat_prepared", False))
