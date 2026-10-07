"""MixedRadixButterfly (nsnet2/mixed_radix.py) and its wiring in make_gru.

CPU only. The layer is checked against an explicit product of its stage
matrices, against the library's radix-2 Butterfly and base-4 ButterflyBase4,
and its three initialisations against their laws; the factory against the
published configs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_structured")

from torch_structured import Butterfly  # noqa: E402
from torch_structured.butterfly.butterfly_base4 import ButterflyBase4  # noqa: E402

from common.env import AttrDict  # noqa: E402
from nsnet2.layers import make_gru, make_linear  # noqa: E402
from nsnet2.mixed_radix import MixedRadixButterfly, init_blocks, radix_strides  # noqa: E402
from nsnet2.model import NSNet2  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LAYOUTS = ([2] * 9, [2, 4, 4, 4, 4], [4, 4, 4, 4, 2])
GRID_ARMS = ("ig_r2_randn", "ig_r2_ortho", "ig_r24_randn", "ig_r24_ortho", "ig_r42_randn",
             "ig_r42_ortho", "ig_r2_rownorm")
PUBLISHED = ("butterfly_full", "butterfly_ortho", "butterfly_2blocks")


def load_config(name: str) -> AttrDict:
    return AttrDict(json.loads((REPO_ROOT / "configs" / f"{name}.json").read_text()))


# ---------------------------------------------------------------------------
# The layer against slow references
# ---------------------------------------------------------------------------


def stage_matrix(T: torch.Tensor, b: int, s: int, n: int) -> torch.Tensor:
    """Dense n x n matrix of one stage of one stack, entry by entry:
    out[g, p, l] = sum_q T[g s + l, p, q] in[g, q, l], position (g, q, l) = g b s + q s + l."""
    M = torch.zeros(n, n, dtype=torch.float64)
    for g in range(n // (b * s)):
        for lo in range(s):
            for p in range(b):
                for q in range(b):
                    M[g * b * s + p * s + lo, g * b * s + q * s + lo] = T[g * s + lo, p, q]
    return M


def reference_matrix(layer: MixedRadixButterfly) -> torch.Tensor:
    """(out_size, in_size): per stack the product of the stage matrices on the zero-padded input,
    stacks concatenated and truncated."""
    rows = []
    for k in range(layer.nstacks):
        M = torch.eye(layer.n, dtype=torch.float64)
        for T, b, s in zip(layer.stages, layer.radices, radix_strides(layer.radices)):
            M = stage_matrix(T[k].detach().double(), b, s, layer.n) @ M
        rows.append(M[:, :layer.in_size])
    return torch.cat(rows, 0)[:layer.out_size]


def layer_matrix(layer: MixedRadixButterfly) -> torch.Tensor:
    """The layer applied to the identity, bias removed: (out_size, in_size)."""
    with torch.no_grad():
        eye = torch.eye(layer.in_size, dtype=torch.float64)
        return (layer(eye) - layer.bias).T


@pytest.mark.parametrize("radices,in_size,out_size", [
    ((2, 3), 5, 14),        # padding 6 -> 5, three stacks, truncation 18 -> 14
    ((3, 2, 2), 10, 30),    # padding, three stacks, truncation 36 -> 30
    ((4, 2), 7, 20),        # padding, three stacks, truncation 24 -> 20
    ((3, 2), 6, 6),         # one stack, no padding, no truncation
    ((2, 4, 2), 13, 9),     # one stack, padding, truncation
])
@pytest.mark.parametrize("init", ["randn", "ortho", "rownorm"])
def test_matrix_is_the_product_of_the_stage_matrices(radices, in_size, out_size, init):
    torch.manual_seed(0)
    layer = MixedRadixButterfly(in_size, out_size, radices, init=init).double()
    torch.testing.assert_close(layer_matrix(layer), reference_matrix(layer), rtol=1e-12, atol=1e-12)


def test_leading_dimensions_and_bias():
    torch.manual_seed(0)
    layer = MixedRadixButterfly(5, 14, [2, 3])
    x = torch.randn(2, 3, 5)
    y = layer(x)
    assert y.shape == (2, 3, 14)
    torch.testing.assert_close(y[1, 2], layer(x[1, 2][None])[0])
    no_bias = MixedRadixButterfly(5, 14, [2, 3], bias=False)
    assert no_bias.bias is None and no_bias(x).shape == (2, 3, 14)


def test_radix2_equals_library_butterfly():
    torch.manual_seed(1)
    lib = Butterfly(400, 1200, bias=True, init="randn")
    layer = MixedRadixButterfly(400, 1200, [2] * 9)
    with torch.no_grad():
        for j, t in enumerate(layer.stages):
            t.copy_(lib.twiddle[:, 0, j])
        layer.bias.copy_(lib.bias)
    x = torch.randn(16, 400)
    torch.testing.assert_close(layer(x), lib(x), rtol=1e-5, atol=1e-5)


def test_radices_4444_2_equal_library_butterfly_base4():
    torch.manual_seed(2)
    twiddle4 = torch.randn(3, 1, 4, 128, 4, 4) / 2
    twiddle2 = torch.randn(3, 1, 1, 256, 2, 2) / 2 ** 0.5
    lib = ButterflyBase4(400, 1200, bias=True, init=(twiddle4, twiddle2))
    layer = MixedRadixButterfly(400, 1200, [4, 4, 4, 4, 2])
    with torch.no_grad():
        for j in range(4):
            layer.stages[j].copy_(lib.twiddle4[:, 0, j])
        layer.stages[4].copy_(lib.twiddle2[:, 0, 0])
        layer.bias.copy_(lib.bias)
    x = torch.randn(16, 400)
    torch.testing.assert_close(layer(x), lib(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("radices", LAYOUTS)
def test_9216_coefficients_per_stack(radices):
    layer = MixedRadixButterfly(400, 1200, radices)
    assert layer.n == 512 and layer.nstacks == 3
    assert sum(t.numel() for t in layer.stages) == 3 * 9216
    assert [tuple(t.shape) for t in layer.stages] == [(3, 512 // b, b, b) for b in radices]


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        MixedRadixButterfly(400, 1200, [2] * 8)          # 256 < 400
    with pytest.raises(ValueError):
        MixedRadixButterfly(16, 48, [2, 1, 8])
    with pytest.raises(ValueError):
        MixedRadixButterfly(16, 48, [4, 4], init="identity")


# ---------------------------------------------------------------------------
# Initialisations
# ---------------------------------------------------------------------------


def test_randn_entries_have_variance_one_over_b():
    torch.manual_seed(3)
    layer = MixedRadixButterfly(400, 1200, [2, 4, 4, 4, 4], init="randn")
    for t, b in zip(layer.stages, layer.radices):
        w = t.detach().double().reshape(-1)
        assert abs(w.mean().item()) < 0.03
        assert abs(w.var().item() * b - 1) < 0.08, (b, w.var().item())


@pytest.mark.parametrize("radices", LAYOUTS)
def test_ortho_blocks_are_haar_orthogonal(radices):
    torch.manual_seed(4)
    layer = MixedRadixButterfly(400, 1200, radices, init="ortho")
    for t, b in zip(layer.stages, layer.radices):
        B = t.detach().double().reshape(-1, b, b)
        eye = torch.eye(b, dtype=torch.float64).expand_as(B)
        torch.testing.assert_close(B.transpose(-1, -2) @ B, eye, rtol=0, atol=1e-6)
        # A missing sign fix still passes B^T B = I; it shows in the determinants and the entries.
        pos = (torch.linalg.det(B) > 0).double().mean().item()
        assert 0.4 < pos < 0.6, (b, pos)
        assert abs(B.mean().item()) < 0.05
        assert abs(torch.diagonal(B, dim1=-2, dim2=-1).mean().item()) < 0.08


@pytest.mark.parametrize("radices", LAYOUTS)
def test_rownorm_rows_have_unit_norm(radices):
    torch.manual_seed(5)
    layer = MixedRadixButterfly(400, 1200, radices, init="rownorm")
    for t in layer.stages:
        norms = torch.linalg.vector_norm(t.detach().double(), dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=0, atol=1e-6)


@pytest.mark.parametrize("init", ["randn", "ortho", "rownorm"])
def test_blocks_of_different_stacks_and_stages_differ(init):
    torch.manual_seed(6)
    layer = MixedRadixButterfly(400, 1200, [2] * 9, init=init)
    s0, s1 = layer.stages[0].detach(), layer.stages[1].detach()
    assert not torch.allclose(s0, s1)
    assert not torch.allclose(s0[0], s0[1]) and not torch.allclose(s0[1], s0[2])


def test_initialisations_of_one_layout_share_their_gaussian_blocks():
    built = {}
    for init in ("randn", "ortho", "rownorm"):
        torch.manual_seed(7)
        built[init] = MixedRadixButterfly(400, 1200, [2, 4, 4, 4, 4], init=init)
    for j, b in enumerate(built["randn"].radices):
        gauss = built["randn"].stages[j].detach() * b ** 0.5
        for init in ("ortho", "rownorm"):
            torch.testing.assert_close(built[init].stages[j].detach(), init_blocks(gauss, init),
                                       rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(built["randn"].bias, built["ortho"].bias, rtol=0, atol=0)


def test_different_layouts_share_no_coefficient():
    torch.manual_seed(8)
    a = MixedRadixButterfly(400, 1200, [2] * 9)
    torch.manual_seed(8)
    b = MixedRadixButterfly(400, 1200, [2, 4, 4, 4, 4])
    assert a.stages[0].shape == b.stages[0].shape
    assert not torch.isclose(a.stages[0], b.stages[0]).any()


def test_same_seed_same_values():
    torch.manual_seed(9)
    a = MixedRadixButterfly(400, 1200, [4, 4, 4, 4, 2], init="ortho")
    torch.manual_seed(9)
    b = MixedRadixButterfly(400, 1200, [4, 4, 4, 4, 2], init="ortho")
    for ta, tb in zip(a.parameters(), b.parameters()):
        assert torch.equal(ta, tb)


def test_global_generator_advances_by_the_same_amount_for_every_layout_and_init():
    states = []
    for radices in LAYOUTS:
        for init in ("randn", "ortho", "rownorm"):
            torch.manual_seed(10)
            MixedRadixButterfly(400, 1200, radices, init=init)
            states.append(torch.get_rng_state())
    torch.manual_seed(10)
    MixedRadixButterfly(16, 48, [2, 2, 4], init="randn")       # another shape too
    states.append(torch.get_rng_state())
    for s in states[1:]:
        assert torch.equal(s, states[0])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", GRID_ARMS)
def test_grid_configs_build_mixed_radix_input_projections(arm):
    h = load_config(arm)
    torch.manual_seed(11)
    model = NSNet2(h)
    for cell in model.gru.cells:
        assert isinstance(cell.x_proj, MixedRadixButterfly)
        assert cell.x_proj.radices == list(h.gru["x_radices"])
        assert cell.x_proj.init == h.gru["x_init"]
        assert type(cell.h_proj) is Butterfly and cell.h_proj.init == "ortho"
    for fc in (model.fc_in, model.fc1, model.fc2, model.fc_out):
        assert type(fc) is Butterfly and fc.init == "ortho"
    a, b = model.gru.cells[0].x_proj, model.gru.cells[1].x_proj
    assert not torch.allclose(a.stages[1], b.stages[1])        # the two input projections differ


def test_grid_arms_at_one_seed_share_every_other_initial_weight():
    sds = {}
    for arm in GRID_ARMS:
        torch.manual_seed(12)
        sds[arm] = NSNet2(load_config(arm)).state_dict()
    ref = sds[GRID_ARMS[0]]
    others = [k for k in ref if ".x_proj." not in k]
    assert len(others) == 12      # fc_in, fc1, fc2, fc_out and the two recurrent projections
    for arm in GRID_ARMS[1:]:
        for k in others:
            assert torch.equal(sds[arm][k], ref[k]), (arm, k)


def _published_reference(h) -> dict:
    """The published architecture built directly from library Butterfly modules, in the order of
    NSNet2.__init__ (fc_in, GRU cells with x_proj then h_proj, fc1, fc2, fc_out)."""
    lin, gru = h.linear, h.gru
    nb_l, nb_g = lin.get("nblocks", 1), gru.get("nblocks", 1)
    x_init = gru.get("x_init", gru.get("init", "randn"))
    mods = {"fc_in": Butterfly(257, 400, init=lin["init"], nblocks=nb_l)}
    for i in range(2):
        mods[f"gru.cells.{i}.x_proj"] = Butterfly(400, 1200, init=x_init, nblocks=nb_g)
        mods[f"gru.cells.{i}.h_proj"] = Butterfly(400, 1200, init=gru.get("h_init", "ortho"), nblocks=nb_g)
    mods["fc1"] = Butterfly(400, 600, init=lin["init"], nblocks=nb_l)
    mods["fc2"] = Butterfly(600, 600, init=lin["init"], nblocks=nb_l)
    mods["fc_out"] = Butterfly(600, 257, init=lin["init"], nblocks=nb_l)
    return {f"{m}.{k}": v for m, mod in mods.items() for k, v in mod.state_dict().items()}


@pytest.mark.parametrize("name", PUBLISHED)
def test_published_configs_build_as_before(name):
    h = load_config(name)
    assert "x_radices" not in h.gru
    torch.manual_seed(1234)
    sd = NSNet2(h).state_dict()
    torch.manual_seed(1234)
    ref = _published_reference(h)
    assert list(sd) == list(ref)
    for k in sd:
        assert torch.equal(sd[k], ref[k]), k


def test_x_radices_needs_the_butterfly_kind():
    with pytest.raises(ValueError, match="x_radices"):
        make_gru(16, 16, 1, cfg={"kind": "blockdiag", "nblocks": 4, "x_radices": [4, 4]})
    with pytest.raises(ValueError, match="x_radices"):
        make_gru(16, 16, 1, cfg={"kind": "gru", "x_radices": [4, 4]})


def test_make_linear_mixed_radix():
    torch.manual_seed(13)
    layer = make_linear(16, 48, cfg={"kind": "mixed_radix", "radices": [2, 4, 2], "init": "ortho"})
    assert isinstance(layer, MixedRadixButterfly) and layer.init == "ortho"
    with pytest.raises(ValueError, match="radices"):
        make_linear(16, 48, cfg={"kind": "mixed_radix"})


@pytest.mark.parametrize("arm", GRID_ARMS + ("ig_full",))
def test_forward_and_backward_are_finite(arm):
    h = load_config(arm)
    torch.manual_seed(14)
    model = NSNet2(h)
    mag = torch.rand(2, 257, 6)
    pha = torch.rand(2, 257, 6) * 6.28 - 3.14
    out_mag, _, out_com = model(mag, pha)
    loss = out_mag.pow(2).mean() + out_com.abs().mean()
    loss.backward()
    assert torch.isfinite(out_mag).all() and torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
