"""Twiddle parameter grouping (nsnet2/optim_groups.py)."""

import pytest
import torch

pytest.importorskip("torch_structured")

from nsnet2.bfly_arch import ButterflyGRU            # noqa: E402
from nsnet2.optim_groups import build_param_groups, is_twiddle  # noqa: E402


@pytest.fixture(scope="module")
def gru():
    return ButterflyGRU(64, 64, 2, nblocks=1)


def test_is_twiddle_matches_both_implementations():
    assert is_twiddle("x_proj.0.twiddle")
    assert is_twiddle("twiddle")
    assert not is_twiddle("x_proj.0.bias")
    assert not is_twiddle("some.twiddle.weight")


def test_every_parameter_lands_in_exactly_one_group(gru):
    groups = build_param_groups(gru, 1e-3)
    seen = [p for g in groups for p in g["params"]]
    assert len(seen) == len(list(gru.parameters()))
    assert len({id(p) for p in seen}) == len(seen)


def test_defaults_are_behaviourally_a_no_op(gru):
    groups = build_param_groups(gru, 3e-3)
    assert {g["lr"] for g in groups} == {3e-3}
    assert all("weight_decay" not in g for g in groups)


def test_multiplier_applies_only_to_twiddles(gru):
    groups = {g["group_name"]: g for g in build_param_groups(gru, 1e-3, twiddle_lr_mult=10.0)}
    assert groups["twiddle"]["lr"] == pytest.approx(1e-2)
    assert groups["default"]["lr"] == pytest.approx(1e-3)


def test_twiddle_weight_decay_can_be_zeroed_alone(gru):
    groups = {g["group_name"]: g
              for g in build_param_groups(gru, 1e-3, twiddle_weight_decay=0.0,
                                          weight_decay=0.01)}
    assert groups["twiddle"]["weight_decay"] == 0.0
    assert groups["default"]["weight_decay"] == 0.01


def test_groups_drive_a_real_adamw_step(gru):
    """The grouping must survive contact with the optimiser, and the twiddle
    group must actually move at its own rate."""
    groups = build_param_groups(gru, 1e-3, twiddle_lr_mult=100.0, twiddle_weight_decay=0.0)
    opt = torch.optim.AdamW(groups, lr=1e-3)
    tw0 = [p.detach().clone() for n, p in gru.named_parameters() if is_twiddle(n)]
    other0 = [p.detach().clone() for n, p in gru.named_parameters() if not is_twiddle(n)]
    out, _ = gru(torch.randn(2, 5, 64))
    out.square().mean().backward()
    opt.step()
    tw1 = [p.detach() for n, p in gru.named_parameters() if is_twiddle(n)]
    other1 = [p.detach() for n, p in gru.named_parameters() if not is_twiddle(n)]
    d_tw = max((a - b).abs().max().item() for a, b in zip(tw0, tw1))
    d_other = max((a - b).abs().max().item() for a, b in zip(other0, other1))
    # Adam's update is ~lr per step regardless of gradient scale, so a 100x
    # multiplier should show up as roughly two orders of magnitude.
    assert d_tw > 10 * d_other, (d_tw, d_other)
