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


# ---------------------------------------------------------------------------
# Per-role multiplier. The stage-1 sweep showed a global 10x multiplier helps
# the randn feed-forward butterflies but degrades the ortho-initialised
# recurrent matrix (rank 438-447 -> 230-282), so the recurrent twiddles need
# their own rate.
# ---------------------------------------------------------------------------

from nsnet2.optim_groups import is_recurrent  # noqa: E402


def test_is_recurrent_separates_the_two_gru_projections():
    assert is_recurrent("rnn.h_proj.0.twiddle")
    assert not is_recurrent("rnn.x_proj.0.twiddle")
    assert not is_recurrent("fc_in.twiddle")


def test_absent_recurrent_mult_keeps_the_two_group_layout(gru):
    groups = build_param_groups(gru, 1e-3, twiddle_lr_mult=10.0)
    assert [g["group_name"] for g in groups] == ["default", "twiddle"]


def test_recurrent_twiddles_can_hold_the_base_rate_while_others_scale(gru):
    groups = {g["group_name"]: g for g in build_param_groups(
        gru, 1e-3, twiddle_lr_mult=10.0, recurrent_twiddle_lr_mult=1.0)}
    assert set(groups) == {"default", "twiddle", "twiddle_recurrent"}
    assert groups["twiddle"]["lr"] == pytest.approx(1e-2)       # x_proj, scaled
    assert groups["twiddle_recurrent"]["lr"] == pytest.approx(1e-3)  # h_proj, held
    assert groups["default"]["lr"] == pytest.approx(1e-3)


def test_split_still_covers_every_parameter_exactly_once(gru):
    groups = build_param_groups(gru, 1e-3, twiddle_lr_mult=10.0,
                                recurrent_twiddle_lr_mult=1.0)
    seen = [p for g in groups for p in g["params"]]
    assert len({id(p) for p in seen}) == len(seen) == len(list(gru.parameters()))


def test_split_puts_the_right_tensors_in_the_recurrent_group(gru):
    groups = {g["group_name"]: g for g in build_param_groups(
        gru, 1e-3, twiddle_lr_mult=10.0, recurrent_twiddle_lr_mult=1.0)}
    expected = {id(p) for n, p in gru.named_parameters()
                if is_twiddle(n) and is_recurrent(n)}
    assert {id(p) for p in groups["twiddle_recurrent"]["params"]} == expected
    assert len(expected) > 0
