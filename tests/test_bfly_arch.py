"""Butterfly-native NSNet2 arms (nsnet2/bfly_arch.py): the engine contract and causality."""

import glob
import json

import pytest
import torch

pytest.importorskip("torch_structured")
from torch_structured import Butterfly  # noqa: E402

from common.env import AttrDict  # noqa: E402
from nsnet2.bfly_arch import ButterflyGRU, StageButterfly, build_generator, engine_pairs  # noqa: E402
from nsnet2.layers import make_gru  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("n_in,n_out,inc", [(256, 256, True), (256, 1024, True), (200, 512, True),
                                            (512, 256, True), (256, 256, False)])
def test_stage_butterfly_matches_torch_structured(n_in, n_out, inc):
    """Full stride list == torch_structured.Butterfly: same pair indexing as the engine packer."""
    torch.manual_seed(0)
    b = Butterfly(n_in, n_out, bias=True, init="randn", increasing_stride=inc).double()
    strides = [2 ** k for k in range(b.log_n)]
    s = StageButterfly(b.n, strides if inc else strides[::-1], nstacks=b.nstacks,
                       in_size=n_in, out_size=n_out).double()
    with torch.no_grad():
        s.twiddle.copy_(b.twiddle[:, 0].permute(1, 0, 2, 3, 4))
        s.bias.copy_(b.bias)
    x = torch.randn(5, 3, n_in, dtype=torch.float64)
    assert torch.allclose(b(x), s(x), atol=1e-12)


def test_butterfly_gru_matches_structured_gru():
    torch.manual_seed(0)
    H = 64
    ref = make_gru(H, H, 2, cfg={"kind": "butterfly", "nblocks": 1, "h_init": "ortho"}).to(DEV)
    me = ButterflyGRU(H, H, 2).to(DEV)
    with torch.no_grad():
        for i, cell in enumerate(ref.cells):
            for a, b in ((cell.x_proj, me.x_proj[i]), (cell.h_proj, me.h_proj[i])):
                b.twiddle.copy_(a.twiddle)
                b.bias.copy_(a.bias)
    x = torch.randn(4, 30, H, device=DEV)
    assert (ref(x)[0] - me(x)[0]).abs().max().item() < 1e-5


@pytest.mark.parametrize("cfg", sorted(glob.glob("configs/ba_*.json")))
def test_arm_is_causal_and_costed(cfg):
    torch.manual_seed(0)
    m = build_generator(AttrDict(json.load(open(cfg)))).to(DEV).eval()
    assert engine_pairs(m) > 0
    with torch.no_grad():
        mag = torch.rand(2, 257, 30, device=DEV) * 3
        pha = torch.rand_like(mag)
        a, _, com = m(mag, pha)
        mag2 = mag.clone()
        mag2[..., 20:] = torch.rand_like(mag2[..., 20:]) * 3
        b = m(mag2, pha)[0]
    assert a.shape == mag.shape and com.shape == (*mag.shape, 2)
    assert torch.equal(a[..., :20], b[..., :20])          # nothing leaks from the future
    assert torch.all(a <= mag + 1e-6)                      # a gain in [0, 1]


# ---------------------------------------------------------------------------
# arch.lin: the dense control. With stage-2 showing every butterfly variant
# inside the reference's own seed spread, "does the butterfly cost anything
# against dense at matched budget" is the question the study now rests on --
# and it is the arm the compression literature most often omits.
# ---------------------------------------------------------------------------

def _cfg(**arch):
    import copy
    h = AttrDict(copy.deepcopy(json.load(open("configs/ba_R.json"))))
    h["arch"] = dict(h["arch"], **arch)
    return h


def test_default_arch_is_unchanged_by_the_lin_switch():
    """Regression guard: omitting arch.lin must build exactly what it always did."""
    m = build_generator(_cfg())
    assert sum(p.numel() for p in m.parameters()) == 154368
    assert type(m.rnn.x_proj[0]).__name__ == "Butterfly"
    assert type(m.rnn.h_proj[0]).__name__ == "Butterfly"


def test_lin_linear_builds_a_fully_dense_control():
    m = build_generator(_cfg(hidden=88, lin={"kind": "linear"}))
    assert isinstance(m.rnn.x_proj[0], torch.nn.Linear)
    assert isinstance(m.rnn.h_proj[0], torch.nn.Linear)
    assert isinstance(m.fc_in, torch.nn.Linear)
    assert isinstance(m.fc_out, torch.nn.Linear)
    assert not any("twiddle" in n for n, _ in m.named_parameters())


def test_dense_control_is_budget_matched_to_the_butterfly_reference():
    """H=88 is the iso-parameter control, H=85 the iso-MAC one; the study has to
    report both axes, and the two must not differ much from each other."""
    n88 = sum(p.numel() for p in build_generator(_cfg(hidden=88, lin={"kind": "linear"})).parameters())
    n85 = sum(p.numel() for p in build_generator(_cfg(hidden=85, lin={"kind": "linear"})).parameters())
    assert abs(n88 - 154368) / 154368 < 0.02      # within 2% on parameters
    assert n85 < n88


def test_dense_control_runs_a_forward_pass():
    m = build_generator(_cfg(hidden=88, lin={"kind": "linear"})).eval()
    mag = torch.rand(2, 257, 7).clamp_min(1e-4)
    pha = torch.zeros(2, 257, 7)
    with torch.no_grad():
        out_mag, out_pha, out_com = m(mag, pha)
    assert out_mag.shape == mag.shape and torch.isfinite(out_mag).all()
