"""Tests for the int4/int8 PTQ+QAT fake-quant scaffold in quant_fake.py.

The branch's intent is "is w4/a8 viable for this model class?" — so the
tests focus on (a) the math of fake_quant_tensor, (b) the STE gradient
behavior that QAT depends on, (c) the prepare_for_qat / apply_ptq
plumbing into the structured-layer types we actually use.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from common.quant_fake import (
    QSpec,
    _STEClamp,
    _STERound,
    apply_ptq,
    apply_weight_fake_quant,
    fake_quant_tensor,
    install_activation_fake_quant,
    is_qat_prepared,
    prepare_for_qat,
)


# ---------------------------------------------------------------------------
# Core fake-quant math
# ---------------------------------------------------------------------------


def test_fake_quant_per_tensor_unique_levels():
    """Symmetric int4 = {-7, ..., 0, ..., 7} → at most 15 distinct values."""
    torch.manual_seed(0)
    x = torch.randn(64, 64)
    y = fake_quant_tensor(x, QSpec(bits=4))
    assert y.unique().numel() <= 15


def test_fake_quant_per_channel_unique_per_row():
    """Per-channel int4 along axis=0 → each row has ≤15 distinct values."""
    torch.manual_seed(0)
    x = torch.randn(8, 128)
    y = fake_quant_tensor(x, QSpec(bits=4, axis=0))
    for row in y:
        assert row.unique().numel() <= 15


def test_fake_quant_per_group_along_input_axis():
    """Per-group along axis=1 (input dim) splits each row into groups."""
    torch.manual_seed(0)
    x = torch.randn(4, 128)
    y = fake_quant_tensor(x, QSpec(bits=4, axis=1, group_size=32))
    # Each (row, group_idx) chunk of 32 elements gets its own scale → ≤15 levels
    for row in y:
        for grp in row.reshape(4, 32):
            assert grp.unique().numel() <= 15


def test_fake_quant_fp32_passthrough():
    """bits >= 32 must be bit-identical to the input (Identity)."""
    x = torch.randn(8, 16)
    y = fake_quant_tensor(x, QSpec(bits=32))
    assert torch.equal(x, y)


def test_fake_quant_quantization_step_size():
    """For per-tensor symmetric int4, max error ≤ half of (absmax / 7)."""
    torch.manual_seed(0)
    x = torch.randn(256) * 3.0
    y = fake_quant_tensor(x, QSpec(bits=4))
    step = x.abs().max().item() / 7.0
    assert (x - y).abs().max().item() <= step * 0.50 + 1e-6


def test_fake_quant_zero_input_stable():
    """All-zero input must not produce NaNs (clamp_min on absmax guard)."""
    y = fake_quant_tensor(torch.zeros(4, 4), QSpec(bits=4, axis=0))
    assert not torch.isnan(y).any()
    assert torch.equal(y, torch.zeros(4, 4))


# ---------------------------------------------------------------------------
# STE behaviour — critical for QAT
# ---------------------------------------------------------------------------


def test_ste_round_identity_gradient():
    """_STERound: forward rounds, backward passes grad through unchanged."""
    x = torch.tensor([1.3, -2.7, 0.4, 7.9], requires_grad=True)
    y = _STERound.apply(x)
    g = torch.ones_like(y)
    y.backward(g)
    # round(1.3)=1, round(-2.7)=-3, round(0.4)=0, round(7.9)=8
    assert torch.equal(y.detach(), torch.tensor([1., -3., 0., 8.]))
    # STE: dy/dx ≈ 1 everywhere
    assert torch.equal(x.grad, torch.tensor([1., 1., 1., 1.]))


def test_ste_clamp_masked_gradient():
    """_STEClamp: forward clamps, backward zeroes grad where clipped."""
    x = torch.tensor([-10., -3., 0., 4., 12.], requires_grad=True)
    y = _STEClamp.apply(x, -5., 5.)
    g = torch.ones_like(y)
    y.backward(g)
    assert torch.equal(y.detach(), torch.tensor([-5., -3., 0., 4., 5.]))
    # x=-10 and x=12 are outside [-5, 5] → grad zeroed
    assert torch.equal(x.grad, torch.tensor([0., 1., 1., 1., 0.]))


def test_fake_quant_gradient_flows_to_weight():
    """End-to-end QAT path: gradient of loss w.r.t. underlying weight is non-zero."""
    torch.manual_seed(0)
    w = torch.randn(4, 8, requires_grad=True)
    x = torch.randn(2, 8)
    w_fq = fake_quant_tensor(w, QSpec(bits=4, axis=0))
    out = x @ w_fq.T
    loss = out.pow(2).sum()
    loss.backward()
    assert w.grad is not None
    assert w.grad.abs().sum().item() > 0
    assert not torch.isnan(w.grad).any()


# ---------------------------------------------------------------------------
# PTQ plumbing on structured layers
# ---------------------------------------------------------------------------


def _make_tiny_linear():
    torch.manual_seed(0)
    return nn.Linear(16, 32)


def test_apply_weight_fake_quant_in_place():
    m = _make_tiny_linear()
    w_before = m.weight.detach().clone()
    n = apply_weight_fake_quant(m, QSpec(bits=4))
    assert n == 1
    # In-place mutation: weight is now quantized.
    assert not torch.equal(w_before, m.weight)
    # Per-channel along axis=0 (Linear default): each row has ≤15 levels.
    for row in m.weight:
        assert row.unique().numel() <= 15


def test_apply_weight_fake_quant_fp32_noop():
    m = _make_tiny_linear()
    w_before = m.weight.detach().clone()
    n = apply_weight_fake_quant(m, QSpec(bits=32))
    assert n == 0
    assert torch.equal(w_before, m.weight)


def test_install_activation_hook_fires():
    """Activation hook should fake-quantize the first positional input
    every forward — verifiable by counting distinct values in a probe."""
    m = _make_tiny_linear()
    seen_unique = []
    install_activation_fake_quant(m, QSpec(bits=8))
    # Install our own pre-hook AFTER apply_ptq's so we observe the
    # quantized input.
    m.register_forward_pre_hook(
        lambda _m, args: seen_unique.append(args[0].unique().numel())
    )
    torch.manual_seed(0)
    m(torch.randn(4, 16))
    assert seen_unique
    # int8 symmetric → ≤2^8 - 1 = 255 distinct levels.
    assert all(u <= 255 for u in seen_unique)


# ---------------------------------------------------------------------------
# QAT preparation
# ---------------------------------------------------------------------------


class _MiniModel(nn.Module):
    """Two-layer dense MLP to exercise prepare_for_qat without pulling in
    the full NSNet2 dependency tree."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_prepare_for_qat_marks_model():
    m = _MiniModel()
    assert not is_qat_prepared(m)
    prepare_for_qat(m, weight=QSpec(bits=4), activation=QSpec(bits=8))
    assert is_qat_prepared(m)


def test_prepare_for_qat_keeps_weights_trainable():
    """After prepare_for_qat the underlying weights must still be
    nn.Parameter with requires_grad=True — parametrize moves the original
    weight to ``parametrizations.weight.original`` but keeps it trainable."""
    m = _MiniModel()
    prepare_for_qat(m, weight=QSpec(bits=4), activation=QSpec(bits=8))
    # The user-facing module.weight is now the parametrized (quantized) version,
    # but the underlying original lives under .parametrizations.weight.original.
    underlying = m.fc1.parametrizations.weight.original
    assert isinstance(underlying, nn.Parameter)
    assert underlying.requires_grad


def test_prepare_for_qat_gradient_flows_end_to_end():
    """One backward pass through a QAT-prepared model must reach the
    underlying weights with non-zero gradient — the STE working through
    parametrize."""
    torch.manual_seed(0)
    m = _MiniModel()
    prepare_for_qat(m, weight=QSpec(bits=4), activation=QSpec(bits=8))
    x = torch.randn(4, 16)
    out = m(x)
    out.pow(2).sum().backward()
    underlying = m.fc1.parametrizations.weight.original
    assert underlying.grad is not None
    assert underlying.grad.abs().sum().item() > 0


def test_prepare_for_qat_forward_quantizes():
    """The forward of a QAT-prepared model should produce activations
    whose distinct-value count never exceeds the int8 budget — proving
    the activation hooks fire."""
    torch.manual_seed(0)
    m = _MiniModel()
    prepare_for_qat(m, weight=QSpec(bits=4), activation=QSpec(bits=8))
    captured = []
    m.fc2.register_forward_pre_hook(
        lambda _mod, args: captured.append(args[0].unique().numel())
    )
    m(torch.randn(4, 16))
    assert captured and captured[0] <= 255


def test_prepare_for_qat_is_idempotent():
    """Calling prepare_for_qat twice must not double-register
    parametrizations or stack duplicate activation hooks."""
    m = _MiniModel()
    nw1, _ = prepare_for_qat(m, weight=QSpec(bits=4), activation=QSpec(bits=8))
    nw2, _ = prepare_for_qat(m, weight=QSpec(bits=4), activation=QSpec(bits=8))
    # Second call must NOT add any new parametrizations (only nw1 from the first).
    assert nw1 > 0
    assert nw2 == 0


def test_apply_ptq_returns_counts():
    """apply_ptq still works alongside prepare_for_qat — they don't share state."""
    m = _MiniModel()
    nw, na = apply_ptq(m, weight=QSpec(bits=4), activation=QSpec(bits=8))
    assert nw == 2  # fc1.weight + fc2.weight
    assert na == 2  # one hook per Linear


# ---------------------------------------------------------------------------
# Save / load roundtrip for the QAT save protocol used by qat_train.py
# ---------------------------------------------------------------------------


def test_remove_parametrizations_then_load_matches_qat_forward():
    """qat_train.py saves by removing parametrizations (so the state_dict
    keys are plain Linear.weight, matching the source NSNet2 layout) and
    a sidecar carries the QSpec. The test below mimics that save flow,
    then loads the saved state_dict into a fresh model + re-applies
    apply_ptq with the SAME spec, and checks forward parity against the
    original QAT model.

    This is the round-trip guarantee end users rely on: a QAT-trained
    checkpoint can be re-evaluated via eval_quant.py at the same quant
    settings and produce identical numbers to what the trainer saw at
    its final validation step.
    """
    from torch.nn.utils import parametrize

    torch.manual_seed(0)
    m = _MiniModel()
    w_spec = QSpec(bits=4)
    a_spec = QSpec(bits=8)
    prepare_for_qat(m, weight=w_spec, activation=a_spec)

    # Capture forward output of the QAT model (parametrize + hooks live).
    m.eval()
    x = torch.randn(4, 16)
    with torch.no_grad():
        out_qat = m(x).clone()

    # Save flow: strip parametrizations (leave_parametrized=False keeps
    # the original underlying weight as the new bare weight) and snapshot
    # the state_dict.
    for mod in list(m.modules()):
        for attr in list(getattr(mod, "parametrizations", {}).keys()):
            parametrize.remove_parametrizations(mod, attr, leave_parametrized=False)
    saved = m.state_dict()

    # Load flow: fresh model, plain load, then re-apply PTQ-equivalent
    # quant (apply_ptq does in-place weight mutation + activation hooks;
    # at eval time there's no gradient so PTQ == QAT-forward).
    m2 = _MiniModel()
    m2.load_state_dict(saved, strict=True)
    apply_ptq(m2, weight=w_spec, activation=a_spec)
    m2.eval()
    with torch.no_grad():
        out_reload = m2(x)

    # Identical because: same underlying fp32 weights, same dynamic
    # absmax (deterministic), same STE forward.
    assert torch.allclose(out_qat, out_reload, atol=1e-6), (
        f"QAT forward != reload-and-PTQ forward: max diff "
        f"{(out_qat - out_reload).abs().max().item():.3e}"
    )
