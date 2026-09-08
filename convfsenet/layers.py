"""Structured pointwise (1x1) convolutions for ConvFSENet.

ConvFSENet spends 92.4% of its per-frame MACs in eighteen 1x1 convolutions
(``conv1x1`` 192->384 and ``conv1x1_out`` 384->192 in each of the nine TCM
blocks). A 1x1 conv over ``(B, C, T)`` is a linear map applied independently
per time step, so every structured-matrix factorization that applies to a
``nn.Linear`` applies here — which is what makes the NSNet2 block-count sweep
(``EXPERIMENT_MONARCH.md``, ``RESULTS_NSNET2.md``) repeatable on this model.

Two structured kinds, both selected from a config block:

* ``blockdiag`` — a SINGLE block-diagonal factor, ``nblocks`` blocks, zero
  cross-block mixing. Exactly ``nn.Conv1d(..., groups=nblocks)``.
* ``monarch``   — the GENUINE two-factor Monarch (block-diagonal x permutation
  x block-diagonal, Dao et al. 2022, arXiv:2204.00595). Its permutation gives
  cross-block mixing that ``blockdiag`` has none of, but "full" mixing only
  while ``nblocks <= sqrt(in_channels)``: each output channel sees
  ``min(in_channels/nblocks, nblocks)`` of the ``nblocks`` input blocks. On
  this model's 192->384 layer that is 100% at nblocks 4 and 8, 75% at 16 and
  18.8% at 32; on the 384->192 layer, 100% through 16 and 37.5% at 32. Still
  far above ``blockdiag``'s 1/nblocks, and NSNet2's published ``monarch_40``
  (25% on its GRU projection) reached dense-parity PESQ from there — but it
  is a covariate to report, not a constant.

Relationship to ``torch_structured.monarch.MonarchLinear`` (what NSNet2 uses)
----------------------------------------------------------------------------
``MonarchPointwise`` is numerically the SAME layer — same factor shapes
``w1 (nblocks, q, p)`` / ``w2 (nblocks, s, q)`` with ``q = p = in/nblocks``,
same variance-matched init drawn from the same RNG sequence, same composed map
— but it is written channels-first as *two grouped 1x1 convolutions separated
by a channel shuffle* instead of ``reshape -> einsum -> reshape`` on a
channels-last tensor. ``tests/test_convfsenet_monarch.py`` asserts both the
bit-identical init and the forward equality against ``MonarchLinear``.

The lowering is not cosmetic. The einsum form exports to ONNX as ``Einsum``
nodes, which onnxruntime ships no QDQ handler for — the bug that silently left
every structured NSNet2 weight in FP32 and forced the int8 retraction recorded
at the top of ``RESULTS_NSNET2.md`` (worked around there by
``nsnet2/qdq_einsum_quantizer.py``). Grouped convolutions are ordinary ``Conv``
nodes: onnxruntime quantizes them per-channel with no registry patching, and
the deploy targets already lower ``Conv`` natively. Measured op histogram per
Monarch layer: 2 ``Conv`` + 4 ``Reshape`` + 2 ``Transpose`` (two of each per
shuffle), i.e. 47 Conv / 72 Reshape / 36 Transpose for the whole 18-layer
model, and zero ``Shape`` / ``Gather`` / ``Pad`` / ``If`` / ``Einsum``.

Export shapes (``t_size``)
--------------------------
The channel shuffle between the two Monarch factors needs to view the channel
axis as a 2-D grid. In eager mode ``unflatten`` does that for any ``T``. Under
``torch.jit`` tracing (``torch.onnx.export``) it lowers to
``Shape``/``Slice``/``Concat`` reshape targets, which are both the ops
``convfsenet/quant_windowed.py`` hard-asserts against AND fatal to
quantization: they abort onnxruntime's symbolic shape inference, so
``quant_pre_process`` — which both quant paths call — dies on an
``AssertionError`` and **no int8 model can be built from a dynamically-traced
graph at all** (measured at T = 1, 4 and 126). The dynamic graph is numerically
correct at any batch size; it simply cannot be quantized.
So for export the reshape targets
must be static: set ``t_size`` to the traced time width (1 for the per-frame
streaming graph, ``L + T`` shrinking per block for the windowed graph) via
:func:`static_t_sizes`, which the export wrappers do for you. ``t_size = None``
(the default) is the dynamic eager path used for training.
"""

from __future__ import annotations

import contextlib
import math
from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


POINTWISE_KINDS = ("conv", "blockdiag", "monarch")
POINTWISE_SCOPES = ("tcm",)


# ---------------------------------------------------------------------------
# Structured pointwise layers
# ---------------------------------------------------------------------------


class _StructuredPointwiseBase(nn.Module):
    """Shared plumbing for the structured 1x1 convs.

    Deliberately does NOT expose ``.weight`` / ``.kernel_size`` / ``.groups``:
    the streaming BN-fold and clone helpers dispatch on module type, and a
    half-plausible ``.weight`` would let them fabricate a wrong dense conv
    instead of failing loudly.
    """

    in_channels: int
    out_channels: int
    nblocks: int

    def __init__(self):
        super().__init__()
        # None = eager (any T). An int pins the reshape targets for tracing.
        self.t_size: Optional[int] = None

    def _check_channels(self, in_channels: int, out_channels: int, nblocks: int) -> None:
        if nblocks < 1:
            raise ValueError(f"nblocks must be >= 1; got {nblocks}")
        if in_channels % nblocks or out_channels % nblocks:
            raise ValueError(
                f"nblocks={nblocks} must divide both in_channels={in_channels} "
                f"and out_channels={out_channels}. (Unlike MonarchLinear this "
                "layer refuses to zero-pad: padding would put Pad/Slice nodes "
                "in the deploy graph and silently change the MAC count the "
                "sweep matches on.)"
            )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, nblocks={self.nblocks}, "
            f"bias={self.bias is not None}"
        )


class BlockdiagPointwise(_StructuredPointwiseBase):
    """Single block-diagonal factor as a grouped 1x1 conv.

    ``weight`` is ``(nblocks, out_blksz, in_blksz)`` — the same layout as
    ``torch_structured.monarch.BlockdiagLinear.weight`` — so a checkpoint is
    interchangeable with that layer's. Zero cross-block mixing by construction:
    output channels ``[g*s, (g+1)*s)`` see only input channels ``[g*p, (g+1)*p)``.

    Provided for a matched block-diagonal control arm; the sweep itself uses
    ``monarch``. Params/MACs per frame: ``in*out/nblocks``.
    """

    def __init__(self, in_channels: int, out_channels: int, nblocks: int, bias: bool = True):
        super().__init__()
        self._check_channels(in_channels, out_channels, nblocks)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.nblocks = int(nblocks)
        self.in_blksz = self.in_channels // self.nblocks
        self.out_blksz = self.out_channels // self.nblocks
        self.weight = nn.Parameter(torch.empty(self.nblocks, self.out_blksz, self.in_blksz))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Matches BlockdiagLinear.set_weights_from_dense_init: draw a dense
        # (out_ext, in_ext) Kaiming tensor, rescale so the surviving block
        # carries the same total energy, and take one block-row of it.
        dense = torch.empty(self.out_channels, self.in_channels,
                            device=self.weight.device, dtype=self.weight.dtype)
        init.kaiming_uniform_(dense, a=math.sqrt(5))
        dense *= math.sqrt(dense.numel() / self.weight.numel())
        with torch.no_grad():
            # BlockdiagLinear's einops '(b o) (b1 i) -> b b1 o i' then [0]:
            # the first block-ROW of the dense draw, split column-wise.
            blocks = dense.reshape(self.nblocks, self.out_blksz,
                                   self.nblocks, self.in_blksz)
            self.weight.copy_(blocks[0].permute(1, 0, 2))
        _reset_bias_(self.bias, self.out_channels)

    @property
    def saving(self) -> float:
        return self.weight.numel() / (self.in_channels * self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, in, T) -> (B, out, T)
        y = F.conv1d(
            x,
            self.weight.reshape(self.nblocks * self.out_blksz, self.in_blksz, 1),
            groups=self.nblocks,
        )
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1)
        return y


class MonarchPointwise(_StructuredPointwiseBase):
    """Genuine two-factor Monarch as a 1x1 conv over ``(B, C, T)``.

    ``block-diagonal (w1) -> channel permutation -> block-diagonal (w2)``, i.e.
    two grouped 1x1 convolutions with a channel shuffle between them. Factor
    shapes follow ``MonarchLinear``: ``w1 (b, q, p)`` and ``w2 (b, s, q)`` with
    ``p = q = in/b`` and ``s = out/b``, so the intermediate width equals the
    input width and the composed map can reach full rank ``min(in, out)``.
    Note full RANK is not full SUPPORT: each output channel depends on
    ``min(p, b)`` of the ``b`` input blocks (see the module docstring).

    Params/MACs per frame: ``b*p*(p + s) = in*(in + out)/b``. Note the
    asymmetry this creates — ``w1`` scales with ``in**2``, so a CONTRACTING
    layer (384->192) compresses only ``b/3`` while an expanding one (192->384)
    compresses ``2b/3``. On ConvFSENet's TCM pair that puts the break-even at
    ``nblocks > 3``: at ``nblocks=2`` a Monarch model is 1.12x LARGER than the
    dense one it replaces.
    """

    def __init__(self, in_channels: int, out_channels: int, nblocks: int, bias: bool = True):
        super().__init__()
        self._check_channels(in_channels, out_channels, nblocks)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.nblocks = int(nblocks)
        b = self.nblocks
        self.in_blksz = self.in_channels // b                 # p, and q == p
        self.out_blksz = self.out_channels // b               # s
        self.w1 = nn.Parameter(torch.empty(b, self.in_blksz, self.in_blksz))
        self.w2 = nn.Parameter(torch.empty(b, self.out_blksz, self.in_blksz))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Variance-matched two-factor init, identical to MonarchLinear's.

        Kaiming-initializing both factors independently compounds the
        variance-preservation twice — w1 and w2 are two halves of ONE linear
        map with no nonlinearity between them, not a 2-layer MLP — and
        undershoots the dense-equivalent output variance by orders of
        magnitude. So: measure the per-element variance a dense Kaiming init
        would give, then rescale both factors so the COMPOSED variance matches.

        The composed map contracts twice, each over ``in_blksz``, against a
        dense layer's single contraction over ``in_channels``; hence the
        ``in_channels / in_blksz**2`` correction.

        The draw order (dense reference, then w1, then w2, then bias) is the
        same as ``MonarchLinear.reset_parameters``, so from a common seed this
        layer initializes bit-identically to it — asserted in
        ``tests/test_convfsenet_monarch.py``.
        """
        dense_ref = torch.empty(self.out_channels, self.in_channels,
                                device=self.w1.device, dtype=self.w1.dtype)
        init.kaiming_uniform_(dense_ref, a=math.sqrt(5))
        v_target = dense_ref.pow(2).mean()
        fan_in_ratio = self.in_channels / (self.in_blksz * self.in_blksz)
        with torch.no_grad():
            init.kaiming_uniform_(self.w1, a=math.sqrt(5))
            init.kaiming_uniform_(self.w2, a=math.sqrt(5))
            v1 = self.w1.pow(2).mean().clamp_min(1e-12)
            v2 = self.w2.pow(2).mean().clamp_min(1e-12)
            target = (fan_in_ratio * v_target).sqrt()
            self.w1.mul_((target / v1).sqrt())
            self.w2.mul_((target / v2).sqrt())
        _reset_bias_(self.bias, self.out_channels)

    @property
    def saving(self) -> float:
        return (self.w1.numel() + self.w2.numel()) / (self.in_channels * self.out_channels)

    # -- the permutation ----------------------------------------------------

    def _shuffle(self, y: torch.Tensor, g1: int, g2: int) -> torch.Tensor:
        """Read the channel axis as ``(g1, g2)`` and emit it as ``(g2, g1)``.

        Eager uses ``unflatten`` (correct for any B and T). Traced, the reshape
        targets must be constants — see the module docstring — so the static
        path pins the time width and leaves only the batch axis free.
        """
        if self.t_size is None:
            return y.unflatten(1, (g1, g2)).transpose(1, 2).flatten(1, 2)
        t = int(self.t_size)
        return (y.reshape(-1, g1, g2, t)
                 .transpose(1, 2)
                 .reshape(-1, g1 * g2, t))

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, in, T) -> (B, out, T)
        b, p, s = self.nblocks, self.in_blksz, self.out_blksz
        # Factor 1: block-diagonal, block g maps input channels [g*p,(g+1)*p).
        y = F.conv1d(x, self.w1.reshape(b * p, p, 1), groups=b)          # (B, b*p, T) as (k, q)
        # Permutation: (k, q) -> read as (r, l) -> transpose -> (l, r), with
        # r == q == p and l == k == b. This is the step that makes the layer a
        # Monarch rather than a block-diagonal: every output block now draws on
        # min(p, b) input blocks instead of exactly one -- all b of them only
        # when p >= b, i.e. nblocks <= sqrt(in_channels).
        y = self._shuffle(y, p, b)
        # Factor 2: block-diagonal over the permuted channels.
        y = F.conv1d(y, self.w2.reshape(b * s, p, 1), groups=b)          # (B, b*s, T) as (l, s)
        # MonarchLinear emits its output in (s, l) order; match it exactly so
        # the composed map — and the block boundaries the NEXT structured layer
        # sees — are identical to the library layer's.
        y = self._shuffle(y, b, s)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1)
        return y


def _reset_bias_(bias: Optional[nn.Parameter], fan_in: int) -> None:
    """MonarchLinear/StructuredLinear bias init: U(-1/sqrt(out), 1/sqrt(out)).

    Note this differs from ``nn.Conv1d``'s default (which uses fan_in =
    in_channels). Kept aligned with the structured layers so a structured arm
    initializes identically to the NSNet2 reference implementation.
    """
    if bias is None:
        return
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    init.uniform_(bias, -bound, bound)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_pointwise(in_channels: int, out_channels: int, bias: bool = True, *,
                   cfg: Optional[dict] = None) -> nn.Module:
    """Build a 1x1 conv: dense ``nn.Conv1d`` or a structured replacement.

    ``cfg`` is the config's ``pointwise`` block, e.g.
    ``{"kind": "monarch", "nblocks": 8, "scope": "tcm"}``. A missing/empty cfg
    or ``kind="conv"`` gives a plain ``nn.Conv1d``, bit-identical to the model
    before this module existed.
    """
    cfg = dict(cfg or {})
    kind = cfg.get("kind", "conv")
    if kind not in POINTWISE_KINDS:
        raise ValueError(f"unknown pointwise kind {kind!r}; expected one of {POINTWISE_KINDS}")
    if kind == "conv":
        return nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    scope = cfg.get("scope", "tcm")
    if scope not in POINTWISE_SCOPES:
        raise ValueError(
            f"unknown pointwise scope {scope!r}; expected one of {POINTWISE_SCOPES}. "
            "Only the TCM blocks' pointwise convs are structured: the frontend "
            "(257->C) and backend (C->257) carry 7.6% of the MACs, 257 is prime "
            "so every nblocks would zero-pad, and convfsenet/quant.py's "
            "compression-prologue walk and streaming's _slice_nyquist both "
            "assume a dense Conv there."
        )
    nblocks = int(cfg.get("nblocks", 4))
    if kind == "monarch":
        return MonarchPointwise(in_channels, out_channels, nblocks, bias=bias)
    return BlockdiagPointwise(in_channels, out_channels, nblocks, bias=bias)


def is_structured_pointwise(m: nn.Module) -> bool:
    return isinstance(m, _StructuredPointwiseBase)


# ---------------------------------------------------------------------------
# Streaming/export helpers: BatchNorm folding and cloning
# ---------------------------------------------------------------------------


def fold_bn_into_pointwise(mod: nn.Module, bn: nn.BatchNorm1d) -> nn.Module:
    """Return a copy of ``mod`` with an eval-mode BatchNorm1d absorbed.

    For ``y = BN(W x + b)`` with ``BN(z) = (z - mu)/sigma * gamma + beta``:
    scale each output channel's weight row by ``gamma/sigma`` and set
    ``b' = (b - mu) * gamma/sigma + beta``.

    Which row drives output channel ``c`` is the whole difficulty for a Monarch
    layer. After the final shuffle the output is ordered ``(s, l)``, so channel
    ``c`` comes from ``w2[c % nblocks, c // nblocks, :]`` and from nothing else
    (verified by per-channel perturbation in the tests). ``w1`` is untouched —
    it feeds every output channel and cannot carry a per-channel scale.

    The ``- mu`` term is mandatory and its omission is not loud: the resulting
    model still trains, exports and quantizes, and is simply wrong (max abs
    error ~1.5 on the fold gate). The tests carry that negative control.
    """
    if bn.training:
        raise ValueError(
            "BN must be in eval mode before folding; running_mean / running_var "
            "only stabilize after .eval()."
        )
    if bn.num_features != mod.out_channels:
        raise ValueError(
            f"BN.num_features={bn.num_features} != out_channels={mod.out_channels}"
        )
    folded = clone_pointwise(mod)
    with torch.no_grad():
        sigma = torch.sqrt(bn.running_var + bn.eps)
        scale = bn.weight / sigma                                  # (C_out,)
        beta = bn.bias - bn.running_mean * scale                   # (C_out,)
        if isinstance(mod, MonarchPointwise):
            b = folded.nblocks
            c = torch.arange(folded.out_channels)
            folded.w2[c % b, c // b, :] *= scale.unsqueeze(-1)
        elif isinstance(mod, BlockdiagPointwise):
            # weight is (nblocks, out_blksz, in_blksz); channel c is block
            # c // out_blksz, row c % out_blksz — contiguous, unlike Monarch.
            folded.weight *= scale.reshape(folded.nblocks, folded.out_blksz, 1)
        else:
            raise TypeError(f"not a structured pointwise layer: {type(mod).__name__}")
        b_w = folded.bias if folded.bias is not None else torch.zeros_like(beta)
        new_bias = b_w * scale + beta
        if folded.bias is None:
            folded.bias = nn.Parameter(new_bias)
        else:
            folded.bias.copy_(new_bias)
    return folded


def clone_pointwise(src: nn.Module) -> nn.Module:
    """Deep-copy a structured pointwise layer (parameters, device and dtype)."""
    if isinstance(src, MonarchPointwise):
        dst = MonarchPointwise(src.in_channels, src.out_channels, src.nblocks,
                               bias=src.bias is not None)
    elif isinstance(src, BlockdiagPointwise):
        dst = BlockdiagPointwise(src.in_channels, src.out_channels, src.nblocks,
                                 bias=src.bias is not None)
    else:
        raise TypeError(f"not a structured pointwise layer: {type(src).__name__}")
    # Construct on the default device/dtype, then match the source. The layers
    # take no device=/dtype= kwargs precisely so this stays a single code path.
    ref = next(src.parameters())
    dst = dst.to(device=ref.device, dtype=ref.dtype)
    with torch.no_grad():
        for name, p in src.named_parameters():
            getattr(dst, name).copy_(p)
    dst.t_size = src.t_size
    return dst


@contextlib.contextmanager
def static_t_sizes(pairs: Iterable[tuple]):
    """Temporarily pin ``t_size`` on structured layers for tracing.

    ``pairs`` yields ``(module, t)``. Every structured pointwise layer inside
    each module gets ``t_size = t`` for the duration; restored afterwards, so a
    failed export cannot leave the model in a traced-only state.
    """
    saved = []
    try:
        for mod, t in pairs:
            for sub in mod.modules():
                if isinstance(sub, _StructuredPointwiseBase):
                    saved.append((sub, sub.t_size))
                    sub.t_size = None if t is None else int(t)
        yield
    finally:
        for sub, old in saved:
            sub.t_size = old
