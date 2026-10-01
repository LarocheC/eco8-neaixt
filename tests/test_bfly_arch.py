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
