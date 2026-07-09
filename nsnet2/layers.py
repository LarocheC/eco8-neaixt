"""Pluggable structured-linear and structured-GRU factories.

Two registries:

- ``make_linear(in_size, out_size, *, cfg)`` — drop-in replacement for
  ``nn.Linear``. ``cfg["kind"]`` selects the backend:
    * ``"linear"``    — dense ``nn.Linear``                 (O(N^2)).
    * ``"butterfly"`` — ``torch_structured.Butterfly``       (~O(N log N)).
    * ``"monarch"``   — ``torch_structured.monarch.BlockdiagLinear``
                        single block-diagonal factor, no permutation
                        (~O(N^2 / nblocks)); zero cross-block mixing by
                        construction.
    * ``"monarch2"``  — ``torch_structured.monarch.MonarchLinear``
                        genuine two-factor Monarch (block-diagonal x
                        permutation x block-diagonal); full cross-channel
                        mixing by construction, unlike ``"monarch"``.

- ``make_gru(input_size, hidden_size, num_layers, *, cfg)`` — same idea for
  the GRU stack. ``cfg["kind"]`` selects:
    * ``"gru"``       — cuDNN-fused ``nn.GRU``.
    * ``"butterfly"`` / ``"monarch"`` / ``"monarch2"`` — ``StructuredGRU``: a
                        Python time-loop wrapping ``StructuredGRUCell``, whose
                        ``W_ih`` and ``W_hh`` projections (packed across
                        r/z/n gates) come from ``make_linear`` with the same
                        backend. The h->3h projection defaults to
                        ``init='ortho'`` for butterfly — recurrent stability
                        needs near-orthogonal weights. ``"monarch"``/
                        ``"monarch2"`` have no ``init`` knob (BlockdiagLinear/
                        MonarchLinear don't accept one); with ``"monarch"``
                        the GRU's cross-block mixing is an accidental
                        byproduct of nblocks/hidden_size not dividing the
                        r/z/n gate width evenly, whereas ``"monarch2"``
                        mixes fully regardless of that alignment.

All ``cfg`` blocks accept the per-backend knobs (e.g. ``nblocks`` for
butterfly/monarch, ``init`` for butterfly).

The ``triton``/``triton_monarch``/``triton_butterfly`` GRU kinds route the
recurrence through ``gru_qat.GRULayer`` (persistent Triton hidden kernel on
CUDA, per-step fallback elsewhere). ``triton_*`` structures the GRU *hidden*
projection; add ``"struct_input": true`` to also structure the *input*
projection so the triton path mirrors a config that structured BOTH GRU
projections (the HF ``monarch_full`` / ``butterfly_full`` family). Requires
gru-qat >= 0.2.0.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from torch_structured import Butterfly
    HAVE_BUTTERFLY = True
except ImportError:
    Butterfly = None
    HAVE_BUTTERFLY = False

try:
    from torch_structured.monarch.blockdiag_linear import BlockdiagLinear
    HAVE_MONARCH = True
except ImportError:
    BlockdiagLinear = None
    HAVE_MONARCH = False

try:
    from torch_structured.monarch.monarch_linear import MonarchLinear
    HAVE_MONARCH2 = True
except ImportError:
    MonarchLinear = None
    HAVE_MONARCH2 = False

try:
    from gru_qat import GRULayer as _GRULayer
    from gru_qat import QuantRecipe as _QuantRecipe
    from gru_qat import QuantizerConfig as _QuantizerConfig
    HAVE_TRITON_GRU = True
except ImportError:
    _GRULayer = None
    HAVE_TRITON_GRU = False


# ---------------------------------------------------------------------------
# Linear factory
# ---------------------------------------------------------------------------

LINEAR_KINDS = ("linear", "butterfly", "monarch", "monarch2")


def make_linear(in_size: int, out_size: int, bias: bool = True, *,
                cfg: Optional[dict] = None) -> nn.Module:
    cfg = dict(cfg or {})
    kind = cfg.get("kind", "linear")

    if kind == "linear":
        return nn.Linear(in_size, out_size, bias=bias)

    if kind == "butterfly":
        if not HAVE_BUTTERFLY:
            raise ImportError("torch_structured.Butterfly unavailable; rebuild torch-butterfly.")
        return Butterfly(
            in_size=in_size, out_size=out_size, bias=bias,
            nblocks=cfg.get("nblocks", 1),
            init=cfg.get("init", "randn"),
        )

    if kind == "monarch":
        if not HAVE_MONARCH:
            raise ImportError("torch_structured.monarch.BlockdiagLinear unavailable; rebuild torch-butterfly.")
        return BlockdiagLinear(
            in_features=in_size, out_features=out_size, bias=bias,
            nblocks=cfg.get("nblocks", 4),
        )

    if kind == "monarch2":
        if not HAVE_MONARCH2:
            raise ImportError("torch_structured.monarch.MonarchLinear unavailable; rebuild torch-butterfly.")
        return MonarchLinear(
            in_features=in_size, out_features=out_size, bias=bias,
            nblocks=cfg.get("nblocks", 4),
        )

    raise ValueError(f"Unknown linear kind: {kind!r} (expected one of {LINEAR_KINDS})")


# ---------------------------------------------------------------------------
# Structured GRU
# ---------------------------------------------------------------------------

class StructuredGRUCell(nn.Module):
    """Single GRU cell with pluggable W_ih and W_hh projections.

    Both projections produce ``3 * hidden_size`` outputs, packed as
    ``(r-gate, z-gate, n-gate)``. Standard GRU equations from the PyTorch
    docs (note: the n-gate mixes the reset gate with the *hidden* projection,
    not the input projection):

        gx = W_ih @ x + b_ih       # split into x_r, x_z, x_n
        gh = W_hh @ h + b_hh       # split into h_r, h_z, h_n
        r  = sigmoid(x_r + h_r)
        z  = sigmoid(x_z + h_z)
        n  = tanh   (x_n + r * h_n)
        h' = (1 - z) * n + z * h
    """

    def __init__(self, input_size: int, hidden_size: int, *,
                 x_cfg: Optional[dict] = None, h_cfg: Optional[dict] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.x_proj = make_linear(input_size, 3 * hidden_size, bias=True, cfg=x_cfg)
        self.h_proj = make_linear(hidden_size, 3 * hidden_size, bias=True, cfg=h_cfg)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gx = self.x_proj(x)
        gh = self.h_proj(h)
        x_r, x_z, x_n = gx.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(x_r + h_r)
        z = torch.sigmoid(x_z + h_z)
        n = torch.tanh(x_n + r * h_n)
        return (1 - z) * n + z * h


class StructuredGRU(nn.Module):
    """Multi-layer GRU using ``StructuredGRUCell`` at each layer.

    API matches ``nn.GRU(input_size, hidden_size, num_layers, batch_first=True)``:
    ``forward(x, h_0=None)`` accepts ``x`` of shape ``(B, T, input_size)`` and
    returns ``(output, h_n)`` with shapes ``(B, T, hidden_size)`` and
    ``(num_layers, B, hidden_size)``.

    The time loop is in Python so this is much slower than cuDNN; use only
    when you actually need a structured recurrence.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, *,
                 x_cfg: Optional[dict] = None, h_cfg: Optional[dict] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            StructuredGRUCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                x_cfg=x_cfg, h_cfg=h_cfg,
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        if h_0 is None:
            h = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h = list(h_0.unbind(0))

        outputs = []
        for t in range(T):
            xt = x[:, t]
            for li, cell in enumerate(self.cells):
                h[li] = cell(xt, h[li])
                xt = h[li]
            outputs.append(xt)
        out = torch.stack(outputs, dim=1)
        h_n = torch.stack(h, dim=0)
        return out, h_n


# ---------------------------------------------------------------------------
# Triton GRU wrapper (multi-layer stack of gru_qat.GRULayer modules)
# ---------------------------------------------------------------------------


class TritonGRU(nn.Module):
    """Multi-layer GRU built from ``gru_qat.GRULayer`` modules.

    Drop-in replacement for ``nn.GRU(input_size, hidden_size, num_layers,
    batch_first=True)``: ``forward(x, h_0=None)`` accepts ``x`` of shape
    ``(B, T, input_size)`` and returns ``(out, h_n)`` with shapes
    ``(B, T, hidden_size)`` and ``(num_layers, B, hidden_size)``.

    Each layer uses ``gate_layout="fused"``. ``pre_batch_input=True`` (which
    hoists the dense input GEMM out of the time loop) is only used for the
    dense ``kind="triton"`` path; it is forced off whenever ``structure_input``
    or ``structure_hidden`` is set, since gru_qat then owns the (batched)
    structured input projection internally. The hidden recurrence runs through
    the gru_qat per-step body (or the persistent Triton kernel when the hidden
    side is structured).
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 *, structure_hidden=None, structure_input=None, use_triton="auto",
                 compile_step: bool = False, pre_batch_input: bool = True):
        super().__init__()
        if not HAVE_TRITON_GRU:
            raise ImportError("gru_qat unavailable; pip install 'gru-qat[triton]'.")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # fp32 recipe — weights/activations identity. Triton kernel still runs.
        recipe = _QuantRecipe(
            weight=_QuantizerConfig(bits=32, axis=0, name="W_id"),
            input_act=_QuantizerConfig(bits=32, name="x_id"),
            hidden=_QuantizerConfig(bits=32, name="h_id"),
        )
        # pre_batch_input hoists the dense input GEMM out of the time loop;
        # it requires gate_layout='fused' AND a dense input. With a structured
        # input or hidden, gru_qat routes through its per-step structured path
        # (gru-qat >=0.2.0 still hoists the structured input projection as a
        # single batched GEMM internally), so pre_batch_input must be off.
        pbi = pre_batch_input and structure_hidden is None and structure_input is None
        self.layers = nn.ModuleList([
            _GRULayer(
                input_size if i == 0 else hidden_size,
                hidden_size,
                recipe=recipe,
                batch_first=True,
                gate_layout="fused",
                pre_batch_input=pbi,
                structure_hidden=structure_hidden,
                structure_input=structure_input,
                use_triton=use_triton,
                compile_step=compile_step,
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None):
        # h_0: (num_layers, B, H) like nn.GRU; default zeros handled per-layer.
        h_in = [None] * self.num_layers if h_0 is None else list(h_0.unbind(0))
        h_outs = []
        out = x
        for i, layer in enumerate(self.layers):
            out, h_T = layer(out, h_in[i])
            h_outs.append(h_T)
        h_n = torch.stack(h_outs, dim=0)
        return out, h_n


# ---------------------------------------------------------------------------
# GRU factory
# ---------------------------------------------------------------------------

GRU_KINDS = ("gru", "butterfly", "monarch", "monarch2", "triton", "triton_monarch", "triton_butterfly")


def make_gru(input_size: int, hidden_size: int, num_layers: int = 1, *,
             cfg: Optional[dict] = None) -> nn.Module:
    """Build either a cuDNN ``nn.GRU`` or a ``StructuredGRU`` with structured
    projections. ``cfg["kind"]`` selects the backend; remaining ``cfg`` keys
    are forwarded to ``make_linear`` for the per-cell W_ih and W_hh.

    For butterfly recurrence, the W_hh projection defaults to ``init='ortho'``
    to keep the recurrent dynamics stable; override by setting
    ``cfg["h_init"]``.
    """
    cfg = dict(cfg or {})
    kind = cfg.get("kind", "gru")

    if kind == "gru":
        return nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    if kind == "triton":
        return TritonGRU(
            input_size, hidden_size, num_layers,
            structure_hidden=None,
            use_triton=cfg.get("use_triton", "auto"),
            compile_step=bool(cfg.get("compile_step", False)),
            pre_batch_input=bool(cfg.get("pre_batch_input", True)),
        )

    if kind in ("triton_monarch", "triton_butterfly"):
        from gru_qat import StructureConfig
        sub = "monarch" if kind == "triton_monarch" else "butterfly"

        def _struct(role: str) -> "StructureConfig":
            sk = {"kind": sub}
            if sub == "monarch":
                sk["nblocks"] = cfg.get("nblocks", 4)
            else:
                # butterfly_nblocks controls the stacked-butterfly depth
                # (torch_structured's `nblocks` argument).
                sk["butterfly_nblocks"] = cfg.get("nblocks", 1)
                # Map the original config's per-side init onto gru_qat's
                # StructureConfig.init, mirroring the python StructuredGRU
                # butterfly path (make_gru below): hidden defaults to ortho
                # for recurrent stability, input to randn.
                if role == "hidden":
                    sk["init"] = cfg.get("h_init", cfg.get("init", "ortho"))
                else:
                    sk["init"] = cfg.get("x_init", cfg.get("init", "randn"))
            return StructureConfig(**sk)

        sh = _struct("hidden")
        # struct_input=True structures the GRU *input* projection too, so the
        # triton path faithfully mirrors a config that structured BOTH GRU
        # projections (e.g. the HF monarch_full / butterfly_full runs). The
        # input GEMM is hoisted out of the recurrence and streamed into the
        # hidden kernel (gru-qat >=0.2.0). Default False keeps the historical
        # dense-input triton behaviour for the standalone triton_* configs.
        si = _struct("input") if cfg.get("struct_input", False) else None
        return TritonGRU(
            input_size, hidden_size, num_layers,
            structure_hidden=sh,
            structure_input=si,
            use_triton=cfg.get("use_triton", "auto"),
            compile_step=bool(cfg.get("compile_step", False)),
            pre_batch_input=False,
        )

    if kind not in ("butterfly", "monarch", "monarch2"):
        raise ValueError(f"Unknown gru kind: {kind!r} (expected one of {GRU_KINDS})")

    base = {k: v for k, v in cfg.items() if k not in ("kind", "h_init", "x_init")}
    x_cfg = dict(base, kind=kind)
    h_cfg = dict(base, kind=kind)
    if kind == "butterfly":
        x_cfg["init"] = cfg.get("x_init", cfg.get("init", "randn"))
        h_cfg["init"] = cfg.get("h_init", "ortho")
    return StructuredGRU(input_size, hidden_size, num_layers, x_cfg=x_cfg, h_cfg=h_cfg)


# ---------------------------------------------------------------------------
# Butterfly orthogonality penalty (regularizer for int8-friendly training)
# ---------------------------------------------------------------------------


def butterfly_ortho_penalty(twiddle: torch.Tensor) -> torch.Tensor:
    """Mean ||T^T T - I||_F^2 across all 2x2 factors of a Butterfly twiddle.

    Pulls every 2x2 twiddle factor toward orthogonality. The cumulative
    log_n-stage butterfly transform stays spectrally bounded as a result —
    activation magnitudes do not compound across stages, which is what makes
    int8 quantization tractable for butterfly variants (each stage's QDQ
    rounding error stays small in absolute terms).

    Empirical motivation: with randn-init twiddles, fc_in's stage-8 max_abs
    grows ~3x across log_n=9 stages (e.g. 3.5 -> 11). With ortho-init it
    stays bounded (~3.5 -> 5). The 2x activation magnitude difference at
    int8 scale = step_size = max_abs/127 directly translates to ~1 bit of
    effective resolution lost per stage — compounding to ~9 bits across
    fc_in alone. See investigation in 2026-05-01 session notes.

    twiddle : (..., 2, 2)  — Butterfly.twiddle parameter (real-valued).
    Returns a scalar penalty (mean across factors so the lambda hyperparam
    does not scale with model size).
    """
    TtT = twiddle.transpose(-1, -2) @ twiddle           # (..., 2, 2)
    eye = torch.eye(2, device=twiddle.device, dtype=twiddle.dtype)
    return (TtT - eye).pow(2).sum(dim=(-1, -2)).mean()
