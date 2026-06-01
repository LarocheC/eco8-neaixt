"""Coverage for quant_fake's static-activation observer and structured-weight
/ per-group-fallback branches, which test_quant_fake.py (nn.Linear only) does
not exercise. The static observer is intricate (EMA range, asymmetric int8
zero-point, first-forward seeding, eval-mode freeze) and is exactly the
fake-quant QAT trains against when the deployment target is static int8 PTQ.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from common.quant_fake import (
    QSpec,
    StaticActFakeQuant,
    apply_weight_fake_quant,
    install_static_activation_fake_quant,
    is_qat_prepared,
    prepare_for_qat,
)


# ---------------------------------------------------------------------------
# StaticActFakeQuant observer math
# ---------------------------------------------------------------------------


def test_static_act_seeds_on_first_forward():
    q = StaticActFakeQuant(bits=8)
    assert int(q.initialized) == 0
    x = torch.randn(16, 32) * 3.0
    q.eval()
    q(x)
    assert int(q.initialized) == 1
    # seeded from the tensor's actual range
    assert torch.isclose(q.running_min, x.min())
    assert torch.isclose(q.running_max, x.max())


def test_static_act_eval_freezes_range():
    q = StaticActFakeQuant(bits=8).eval()
    q(torch.randn(8, 8))                       # seed
    lo0, hi0 = q.running_min.clone(), q.running_max.clone()
    q(torch.randn(8, 8) * 100.0)               # huge input must NOT move range in eval
    assert torch.equal(q.running_min, lo0)
    assert torch.equal(q.running_max, hi0)


def test_static_act_ema_updates_in_train():
    q = StaticActFakeQuant(bits=8, momentum=0.5).train()
    q(torch.zeros(4, 4) + 1.0)                 # seed at constant 1.0 -> min=max=1
    before = q.running_max.clone()
    q(torch.zeros(4, 4) + 5.0)                 # larger -> EMA pulls running_max up
    assert q.running_max > before


def test_static_act_int8_levels_and_roundtrip():
    q = StaticActFakeQuant(bits=8).eval()
    x = torch.randn(64, 64)
    y = q(x)                                    # seeds + quantizes against x's range
    assert y.unique().numel() <= 256
    lo = torch.minimum(q.running_min, torch.zeros(()))
    hi = torch.maximum(q.running_max, torch.zeros(()))
    scale = ((hi - lo) / 255).clamp_min(1e-12)
    assert (y - x).abs().max() <= scale + 1e-6   # in-range error bounded by one step


def test_static_act_passthrough_when_32bit():
    q = StaticActFakeQuant(bits=32)
    x = torch.randn(4, 4)
    assert torch.equal(q(x), x)


# ---------------------------------------------------------------------------
# install / prepare plumbing
# ---------------------------------------------------------------------------


class _MixedModel(nn.Module):
    """Explicit-forward module (not nn.Sequential) so the registered
    ``_act_fake_quant`` ModuleList is not swept into the forward chain."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def _mixed_model():
    return _MixedModel()


def test_install_static_activation_counts_and_registers():
    model = _mixed_model()
    n = install_static_activation_fake_quant(model, bits=8)
    assert n == 2                              # two Linear leaves
    assert isinstance(model._act_fake_quant, nn.ModuleList)
    assert len(model._act_fake_quant) == 2
    out = model(torch.randn(3, 16))
    assert out.shape == (3, 4) and torch.isfinite(out).all()


def test_prepare_for_qat_static_activation():
    model = _mixed_model()
    nw, na = prepare_for_qat(
        model, weight=QSpec(bits=4, axis=0), activation=QSpec(bits=8),
        static_activation=True,
    )
    assert nw == 2 and na == 2
    assert is_qat_prepared(model)
    assert isinstance(model._act_fake_quant, nn.ModuleList)
    out = model(torch.randn(2, 16))
    assert out.shape == (2, 4) and torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Structured-weight quant + per-group -> per-channel divisibility fallback
# ---------------------------------------------------------------------------


def test_blockdiag_weight_quant_pergroup_fallback():
    """group_size that doesn't divide the per-block axis must silently fall
    back to per-channel rather than crash (the monarch axis=1 dim 8 case)."""
    bd = pytest.importorskip("torch_structured.monarch.blockdiag_linear")
    layer = bd.BlockdiagLinear(64, 64, bias=True, nblocks=8)   # weight (8, 8, 8), axis=1 dim 8
    x = torch.randn(2, 64)
    ref_unique = layer.weight.detach().unique().numel()
    n = apply_weight_fake_quant(layer, QSpec(bits=4, axis=1, group_size=64))
    assert n == 1
    y = layer(x)
    assert y.shape == (2, 64) and torch.isfinite(y).all()
    # quantization actually happened (int4 -> far fewer distinct values)
    assert layer.weight.detach().unique().numel() < ref_unique


def test_prepare_for_qat_blockdiag_pergroup_fallback():
    bd = pytest.importorskip("torch_structured.monarch.blockdiag_linear")
    layer = bd.BlockdiagLinear(64, 64, bias=True, nblocks=8)
    nw, _ = prepare_for_qat(
        layer, weight=QSpec(bits=4, axis=1, group_size=64), activation=QSpec(bits=8),
    )
    assert nw == 1
    y = layer(torch.randn(2, 64))
    assert y.shape == (2, 64) and torch.isfinite(y).all()
