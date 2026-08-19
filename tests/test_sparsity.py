"""Fixed structured-sparsity masks: pattern conformance and training behaviour."""

import json

import pytest
import torch
import torch.nn as nn

from common.env import AttrDict
from nsnet2.export_sparse import collect_matrices, dims_table
from nsnet2.model import NSNet2
from nsnet2.sparsity import (MaskedLinear, SparsityController, build_mask,
                             parse_pattern, tail_elements, verify_pattern)


# ---------------------------------------------------------------------------
# Pattern parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pattern,family,sparsity", [
    ("2:4", "nm", 0.5),
    ("4:8", "nm", 0.5),
    ("1:4", "nm", 0.75),
    ("1x4:80", "block", 0.8),
    ("unstructured:80", "unstructured", 0.8),
    ("dense", "dense", 0.0),
])
def test_parse_pattern(pattern, family, sparsity):
    d = parse_pattern(pattern)
    assert d["family"] == family
    assert d["sparsity"] == pytest.approx(sparsity)


@pytest.mark.parametrize("bad", ["4:4", "5:4", "2/4", "1x4:100", "garbage"])
def test_parse_pattern_rejects(bad):
    with pytest.raises(ValueError):
        parse_pattern(bad)


# ---------------------------------------------------------------------------
# N:M masks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pattern,n,group", [("2:4", 2, 4), ("4:8", 4, 8)])
def test_nm_mask_exact_per_group(pattern, n, group):
    """Every complete group of `group` columns keeps exactly `n` weights."""
    torch.manual_seed(0)
    w = torch.randn(12, 4 * group)                 # K divides evenly
    m = build_mask(w, pattern)
    counts = m.reshape(12, -1, group).sum(dim=-1)
    assert torch.all(counts == n)


def test_nm_mask_keeps_largest_magnitude():
    w = torch.tensor([[0.1, -0.9, 0.5, 0.2]])
    m = build_mask(w, "2:4")
    assert m.tolist() == [[0.0, 1.0, 1.0, 0.0]]


def test_nm_mask_ragged_tail_kept_dense():
    """K=257 (NSNet2's fc_in) is 64 groups of 4 plus one leftover column."""
    torch.manual_seed(0)
    w = torch.randn(8, 257)
    m = build_mask(w, "2:4", tail="keep")
    assert torch.all(m[:, 256] == 1.0)                       # tail untouched
    assert torch.all(m[:, :256].reshape(8, 64, 4).sum(-1) == 2)
    assert tail_elements(w, "2:4") == 8                      # one column x 8 rows
    # achieved sparsity is a hair under the nominal 50% because of that column
    assert 0.49 < (1 - m.mean().item()) < 0.5


def test_nm_mask_ragged_tail_dropped():
    torch.manual_seed(0)
    w = torch.randn(8, 257)
    m = build_mask(w, "2:4", tail="drop")
    assert torch.all(m[:, 256] == 0.0)


def test_nm_mask_axis_out():
    """axis='out' groups down a column instead of along a row."""
    torch.manual_seed(0)
    w = torch.randn(8, 12)
    m = build_mask(w, "2:4", axis="out")
    assert m.shape == w.shape
    counts = m.t().reshape(12, -1, 4).sum(dim=-1)
    assert torch.all(counts == 2)


# ---------------------------------------------------------------------------
# Block and unstructured masks
# ---------------------------------------------------------------------------

def test_block_mask_is_all_or_nothing_per_block():
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    m = build_mask(w, "1x4:80")
    blocks = m.reshape(16, 16, 4).sum(dim=-1)
    assert torch.all((blocks == 0) | (blocks == 4))
    assert (1 - m.mean().item()) == pytest.approx(0.8, abs=0.02)


def test_block_mask_row_scope_balances_rows():
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    m = build_mask(w, "1x4:80", scope="row")
    per_row = m.sum(dim=-1)
    assert torch.all(per_row == per_row[0])          # identical work per row


def test_unstructured_mask_hits_target_sparsity():
    torch.manual_seed(0)
    w = torch.randn(32, 64)
    m = build_mask(w, "unstructured:80")
    assert (1 - m.mean().item()) == pytest.approx(0.8, abs=1e-3)


# ---------------------------------------------------------------------------
# MaskedLinear
# ---------------------------------------------------------------------------

def test_masked_linear_forward_matches_manual_masking():
    torch.manual_seed(0)
    layer = MaskedLinear(16, 8, pattern="2:4")
    x = torch.randn(3, 16)
    expected = torch.nn.functional.linear(x, layer.weight * layer.mask, layer.bias)
    assert torch.allclose(layer(x), expected)


def test_masked_linear_pruned_weights_get_no_gradient():
    torch.manual_seed(0)
    layer = MaskedLinear(16, 8, pattern="2:4")
    layer(torch.randn(4, 16)).sum().backward()
    assert torch.all(layer.weight.grad[layer.mask == 0] == 0)


def test_masked_linear_mask_survives_state_dict_roundtrip():
    torch.manual_seed(0)
    a = MaskedLinear(16, 8, pattern="2:4")
    b = MaskedLinear(16, 8, pattern="2:4")
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.mask, b.mask)
    assert "mask" in a.state_dict()


def test_masked_linear_via_layer_factory():
    from nsnet2.layers import make_linear
    layer = make_linear(16, 8, cfg={"kind": "masked", "pattern": "4:8"})
    assert isinstance(layer, MaskedLinear)
    assert torch.all(layer.mask.reshape(8, 2, 8).sum(-1) == 4)


# ---------------------------------------------------------------------------
# SparsityController on the real model
# ---------------------------------------------------------------------------

@pytest.fixture
def baseline_h():
    with open("configs/baseline.json") as f:
        return AttrDict(json.load(f))


def test_controller_covers_every_nsnet2_matmul(baseline_h):
    torch.manual_seed(0)
    model = NSNet2(baseline_h)
    ctrl = SparsityController(model, "2:4")
    names = set(ctrl.masks)
    expected = {
        "fc_in.weight", "fc1.weight", "fc2.weight", "fc_out.weight",
        "gru.weight_ih_l0", "gru.weight_hh_l0",
        "gru.weight_ih_l1", "gru.weight_hh_l1",
    }
    assert names == expected


def test_controller_apply_makes_weights_conform(baseline_h):
    torch.manual_seed(0)
    model = NSNet2(baseline_h)
    ctrl = SparsityController(model, "2:4")
    ctrl.apply()
    for name, p in model.named_parameters():
        if name not in ctrl.masks:
            continue
        K = p.shape[1]
        n_groups = K // 4
        head = p.data[:, :n_groups * 4].reshape(p.shape[0], n_groups, 4)
        assert torch.all((head != 0).sum(-1) <= 2), name


def _one_train_step(model, ctrl, *, mask_grads):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    mag = torch.rand(2, 257, 5)
    pha = torch.zeros(2, 257, 5)
    out, _, _ = model(mag, pha)
    out.sum().backward()
    if mask_grads:
        ctrl.mask_grads()
    opt.step()


def test_controller_reprojects_after_unmasked_optimizer_step(baseline_h):
    """Without mask_grads, AdamW drifts pruned weights; apply() zeroes them again."""
    torch.manual_seed(0)
    model = NSNet2(baseline_h)
    ctrl = SparsityController(model, "2:4")
    ctrl.apply()
    _one_train_step(model, ctrl, mask_grads=False)

    w = model.fc1.weight
    mask = ctrl.masks["fc1.weight"]
    assert torch.any(w.data[mask == 0] != 0), "expected AdamW to drift pruned weights"
    ctrl.apply()
    assert torch.all(w.data[mask == 0] == 0)


def test_controller_mask_grads_keeps_pruned_weights_pinned(baseline_h):
    """With zero grad and a weight already at zero, AdamW's update and its
    decoupled weight decay are both zero — nothing drifts in the first place."""
    torch.manual_seed(0)
    model = NSNet2(baseline_h)
    ctrl = SparsityController(model, "2:4")
    ctrl.apply()
    _one_train_step(model, ctrl, mask_grads=True)

    for name, p in model.named_parameters():
        if name in ctrl.masks:
            assert torch.all(p.data[ctrl.masks[name] == 0] == 0), name


def test_controller_disabled_by_config(baseline_h):
    model = NSNet2(baseline_h)
    assert SparsityController.from_config(model, None) is None
    assert SparsityController.from_config(model, {"enabled": False}) is None


def test_controller_exclude_pattern(baseline_h):
    model = NSNet2(baseline_h)
    ctrl = SparsityController(model, "2:4", exclude=[r"fc_out"])
    assert "fc_out.weight" not in ctrl.masks
    assert "fc_in.weight" in ctrl.masks


def test_controller_skips_masked_linear_modules(baseline_h):
    """MaskedLinear masks itself in forward; the controller must not double up."""
    h = AttrDict(dict(baseline_h))
    h["linear"] = {"kind": "masked", "pattern": "2:4"}
    model = NSNet2(h)
    ctrl = SparsityController(model, "2:4")
    assert not any(n.startswith("fc_in.") for n in ctrl.masks)
    assert "gru.weight_ih_l0" in ctrl.masks       # nn.GRU still handled here


def test_config_files_load_and_build(baseline_h):
    for name in ("sparse_2to4", "sparse_4to8", "sparse_block1x4_80",
                 "sparse_unstructured_80"):
        with open(f"configs/{name}.json") as f:
            h = AttrDict(json.load(f))
        model = NSNet2(h)
        ctrl = SparsityController.from_config(model, h.sparsity)
        assert ctrl is not None and len(ctrl) == 8, name
        ctrl.apply()
        target = parse_pattern(h.sparsity["pattern"])["sparsity"]
        report = ctrl.report()
        overall = 1 - sum(r["nonzero"] for r in report) / sum(r["numel"] for r in report)
        assert overall == pytest.approx(target, abs=0.01), name


# ---------------------------------------------------------------------------
# Pattern verification (the contract with the generated kernel)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pattern", ["2:4", "4:8", "1x4:80"])
def test_verify_accepts_a_conforming_weight(pattern):
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    w = w * build_mask(w, pattern)
    check = verify_pattern(w, pattern)
    assert check["ok"] and check["violations"] == 0


@pytest.mark.parametrize("pattern", ["2:4", "4:8"])
def test_verify_rejects_a_dense_weight(pattern):
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    check = verify_pattern(w, pattern)
    assert not check["ok"]
    assert check["violations"] == check["groups"]


@pytest.mark.parametrize("pattern", ["1x4:80", "unstructured:80"])
def test_verify_rejects_a_dense_weight_on_density(pattern):
    """A dense weight trivially satisfies "blocks are whole" — the density
    claim is the part that catches it."""
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    check = verify_pattern(w, pattern)
    assert not check["ok"]
    assert check["violations"] == 0
    assert check["sparsity_shortfall"] == pytest.approx(0.8)


def test_verify_catches_a_single_revived_weight():
    """One pruned weight drifting off zero breaks the kernel's assumption."""
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    mask = build_mask(w, "2:4")
    w = w * mask
    zero_idx = (mask == 0).nonzero()[0]
    w[zero_idx[0], zero_idx[1]] = 1e-9
    check = verify_pattern(w, "2:4")
    assert not check["ok"] and check["violations"] == 1


def test_verify_tolerates_an_incidental_zero():
    """Fewer nonzeros than allowed is fine; more is not."""
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    w = w * build_mask(w, "2:4")
    kept = (w != 0).nonzero()[0]
    w[kept[0], kept[1]] = 0.0
    assert verify_pattern(w, "2:4")["ok"]


def test_verify_block_rejects_a_partially_kept_block():
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    w = w * build_mask(w, "1x4:80")
    block = (w.reshape(16, 16, 4).abs().sum(-1) != 0).nonzero()[0]
    w.reshape(16, 16, 4)[block[0], block[1], 2] = 0.0
    assert not verify_pattern(w, "1x4:80")["ok"]


def test_trained_checkpoint_survives_a_state_dict_roundtrip(baseline_h, tmp_path):
    """The saved file is what ships, so verify the file and not the live model."""
    torch.manual_seed(0)
    model = NSNet2(baseline_h)
    ctrl = SparsityController(model, "2:4")
    ctrl.apply()
    torch.save({"generator": model.state_dict()}, tmp_path / "g_test")

    reloaded = NSNet2(baseline_h)
    reloaded.load_state_dict(torch.load(tmp_path / "g_test")["generator"])
    for name, p in reloaded.named_parameters():
        if name in ctrl.masks:
            assert verify_pattern(p.data, "2:4")["ok"], name


# ---------------------------------------------------------------------------
# int8 orientation logic
# ---------------------------------------------------------------------------

def test_int8_violations_detects_the_right_orientation():
    """ONNX stores Gemm weights (M,K) and MatMul weights (K,M), so the checker
    tests both ways and passes if either conforms."""
    from nsnet2.verify_int8_sparsity import _violations

    torch.manual_seed(0)
    w = torch.randn(16, 64)
    w = (w * build_mask(w, "2:4")).numpy()
    desc = parse_pattern("2:4")

    assert _violations(w, desc, 1) == 0        # groups along the last axis
    assert _violations(w, desc, 0) > 0         # wrong axis: does not conform
    assert _violations(w.T, desc, 0) == 0      # transposed: now axis 0 is right


def test_int8_extra_zeros_never_violate_nm():
    """Quantization rounds small weights to zero; N:M allows *at most* N, so
    extra zeros must stay legal."""
    from nsnet2.verify_int8_sparsity import _violations

    torch.manual_seed(0)
    w = torch.randn(16, 64)
    w = w * build_mask(w, "2:4")
    kept = (w != 0).nonzero()
    for idx in kept[:20]:
        w[idx[0], idx[1]] = 0.0
    assert _violations(w.numpy(), parse_pattern("2:4"), 1) == 0


@pytest.mark.parametrize("pattern", ["1x4:80", "unstructured:80"])
def test_int8_density_shortfall_catches_a_dense_graph(pattern):
    """A dense matrix trivially satisfies "every block is whole", so structure
    alone cannot catch it — density is what does."""
    from nsnet2.verify_int8_sparsity import density_shortfall

    desc = parse_pattern(pattern)
    assert density_shortfall(0.80, desc) == pytest.approx(0.0)
    assert density_shortfall(0.02, desc) == pytest.approx(0.78)
    # N:M pins density through its group constraint, so it is exempt
    assert density_shortfall(0.02, parse_pattern("2:4")) == 0.0


def test_int8_block_support_tolerates_a_split_block():
    """Quantization zeroing a value inside a kept block must not fail the check:
    a block-packed kernel still stores that block whole."""
    from nsnet2.verify_int8_sparsity import block_support

    torch.manual_seed(0)
    w = torch.randn(16, 64)
    w = w * build_mask(w, "1x4:80")
    desc = parse_pattern("1x4:80")
    before = block_support(w.numpy(), desc)

    kept = (w.reshape(16, 16, 4).abs().sum(-1) != 0).nonzero()[0]
    w.reshape(16, 16, 4)[kept[0], kept[1], 1] = 0.0
    assert block_support(w.numpy(), desc) == before      # support is unchanged


def test_int8_block_support_counts_a_dense_matrix_as_fully_live():
    from nsnet2.verify_int8_sparsity import block_support

    torch.manual_seed(0)
    w = torch.randn(16, 64)
    desc = parse_pattern("1x4:80")
    live, total = block_support(w.numpy(), desc)
    assert live == total                                  # nothing is dropped
    sparse = (w * build_mask(w, "1x4:80")).numpy()
    live_s, total_s = block_support(sparse, desc)
    assert live_s / total_s == pytest.approx(0.2, abs=0.02)


# ---------------------------------------------------------------------------
# Export metadata
# ---------------------------------------------------------------------------

def test_dims_table_matches_the_instantiated_model(baseline_h):
    """The shape table handed to the compiler side must match real parameters."""
    model = NSNet2(baseline_h)
    actual = {name: (int(w.shape[0]), int(w.shape[1]))
              for name, w, _ in collect_matrices(model)}
    table = {r["name"]: (r["M"], r["K"]) for r in dims_table(baseline_h)}
    table = {("fc_in.weight" if k == "fc_in" else
              "fc1.weight" if k == "fc1" else
              "fc2.weight" if k == "fc2" else
              "fc_out.weight" if k == "fc_out" else k): v
             for k, v in table.items()}
    assert actual == table


def test_export_roundtrip(tmp_path, baseline_h):
    import subprocess
    import sys

    import numpy as np

    cfg = tmp_path / "cfg.json"
    h = dict(baseline_h)
    h["sparsity"] = {"enabled": True, "pattern": "2:4", "axis": "in",
                     "tail": "keep", "scope": "matrix", "min_numel": 4096}
    cfg.write_text(json.dumps(h))
    out = tmp_path / "export"

    subprocess.run([sys.executable, "-m", "nsnet2.export_sparse",
                    "--config", str(cfg), "--out", str(out)],
                   check=True, capture_output=True)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["pattern"] == "2:4"
    assert manifest["N_inference"] == 1
    assert len(manifest["matrices"]) == 8

    npz = np.load(out / "weights.npz")
    for entry in manifest["matrices"]:
        w = npz[f"{entry['name']}.weight"]
        m = npz[f"{entry['name']}.mask"]
        assert w.shape == (entry["M"], entry["K"])
        assert int((m != 0).sum()) == entry["nonzero"]
        assert np.all(w[m == 0] == 0)          # explicit zeros where masked
