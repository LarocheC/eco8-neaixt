"""CPU coverage for the structured / triton GRU + structured-linear paths.

The streaming / export / quant suites all load configs/baseline.json
(gru.kind='gru') and skip structured variants, so the butterfly / blockdiag /
monarch / triton GRU families — including the structure_input support and the
configs/*_triton.json variants — otherwise have no automated coverage. These
are construction + forward smoke tests on CPU: the gru_qat Triton kernel needs
CUDA, so on CPU ``use_triton='auto'`` falls back to an equivalent per-step
loop, which is exactly what we exercise here.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_structured")
pytest.importorskip("gru_qat")

from nsnet2.layers import make_gru, make_linear  # noqa: E402
from nsnet2.model import NSNet2  # noqa: E402


# ---------------------------------------------------------------------------
# Structured linear factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [
    {"kind": "linear"},
    {"kind": "blockdiag", "nblocks": 4},
    {"kind": "monarch", "nblocks": 4},
    {"kind": "butterfly", "nblocks": 1, "init": "randn"},
    {"kind": "butterfly", "nblocks": 1, "init": "ortho"},
])
def test_make_linear_forward_shape(cfg):
    torch.manual_seed(0)
    layer = make_linear(64, 96, cfg=cfg)
    y = layer(torch.randn(3, 7, 64))
    assert y.shape == (3, 7, 96)
    assert torch.isfinite(y).all()


def test_blockdiag_and_monarch_are_distinct_layers():
    """'blockdiag' is a single block-diagonal factor (no cross-block mixing);
    'monarch' is the genuine two-factor construction (full cross-channel
    mixing). They must map to different torch_structured classes, and monarch
    must actually mix across blocks."""
    bd = make_linear(64, 64, bias=False, cfg={"kind": "blockdiag", "nblocks": 4})
    mo = make_linear(64, 64, bias=False, cfg={"kind": "monarch", "nblocks": 4})
    assert type(bd).__name__ == "BlockdiagLinear"
    assert type(mo).__name__ == "MonarchLinear"
    # A single input channel: block-diagonal lights up at most one block
    # (<=16 outputs); a true monarch mixes across all blocks (all 64).
    x = torch.zeros(1, 64)
    x[0, 0] = 1.0
    with torch.no_grad():
        bd_nz = (bd(x).abs() > 1e-9).sum().item()
        mo_nz = (mo(x).abs() > 1e-9).sum().item()
    assert bd_nz <= 16, f"blockdiag should not mix across blocks, got {bd_nz} nonzero"
    assert mo_nz == 64, f"monarch should mix across all channels, got {mo_nz} nonzero"


# ---------------------------------------------------------------------------
# GRU factory — every kind builds and forwards with nn.GRU-compatible shapes
# ---------------------------------------------------------------------------

GRU_CFGS = [
    {"kind": "gru"},
    {"kind": "blockdiag", "nblocks": 4},
    {"kind": "monarch", "nblocks": 4},
    {"kind": "butterfly", "nblocks": 1, "h_init": "ortho"},
    {"kind": "triton"},
    {"kind": "triton_blockdiag", "nblocks": 4, "struct_input": True},
    {"kind": "triton_monarch", "nblocks": 4, "struct_input": True},
    {"kind": "triton_butterfly", "nblocks": 1, "struct_input": True, "h_init": "ortho"},
]


@pytest.mark.parametrize("cfg", GRU_CFGS, ids=lambda c: c["kind"] + ("_si" if c.get("struct_input") else ""))
def test_make_gru_forward_shape(cfg):
    torch.manual_seed(0)
    B, T, H, L = 2, 5, 64, 2
    gru = make_gru(H, H, num_layers=L, cfg=cfg).eval()
    x = torch.randn(B, T, H)
    with torch.no_grad():
        out, h_n = gru(x)
    assert out.shape == (B, T, H)
    assert h_n.shape == (L, B, H)
    assert torch.isfinite(out).all() and torch.isfinite(h_n).all()


def test_struct_input_engages_structured_input_projection():
    """triton_* + struct_input must structure BOTH projections (not just hidden)."""
    gru = make_gru(64, 64, num_layers=1,
                   cfg={"kind": "triton_monarch", "nblocks": 4, "struct_input": True})
    cell = gru.layers[0].cell
    assert cell._input_dense is False
    assert cell._hidden_dense is False


def test_triton_without_struct_input_keeps_dense_input():
    """Default (no struct_input) preserves the historical dense-input behaviour."""
    gru = make_gru(64, 64, num_layers=1, cfg={"kind": "triton_monarch", "nblocks": 4})
    cell = gru.layers[0].cell
    assert cell._input_dense is True
    assert cell._hidden_dense is False


# ---------------------------------------------------------------------------
# Full NSNet2 model with structured / triton GRU blocks
# ---------------------------------------------------------------------------


def _cfg(linear, gru):
    return SimpleNamespace(
        n_fft=64, hidden_dim=64, fc_hidden_dim=64, num_gru_layers=1,
        linear=linear, gru=gru,
    )


@pytest.mark.parametrize("linear,gru", [
    ({"kind": "blockdiag", "nblocks": 4}, {"kind": "blockdiag", "nblocks": 4}),
    ({"kind": "blockdiag", "nblocks": 4},
     {"kind": "triton_blockdiag", "nblocks": 4, "struct_input": True}),
    ({"kind": "monarch", "nblocks": 4}, {"kind": "monarch", "nblocks": 4}),
    ({"kind": "monarch", "nblocks": 4},
     {"kind": "triton_monarch", "nblocks": 4, "struct_input": True}),
    ({"kind": "butterfly", "nblocks": 1, "init": "randn"},
     {"kind": "triton_butterfly", "nblocks": 1, "struct_input": True, "h_init": "ortho"}),
])
def test_nsnet2_structured_forward(linear, gru):
    torch.manual_seed(0)
    model = NSNet2(_cfg(linear, gru)).eval()
    n_freq = 64 // 2 + 1
    mag = torch.rand(2, n_freq, 5)
    pha = torch.rand(2, n_freq, 5)
    with torch.no_grad():
        denoised_mag, _, _ = model(mag, pha)
    assert denoised_mag.shape == (2, n_freq, 5)
    assert torch.isfinite(denoised_mag).all()
