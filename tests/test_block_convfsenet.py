"""Block-design options for ConvFSENet: flags-off identity, fold exactness, lr warmup.

* Flags off (every existing config) must build the model of the pre-block-design
  code exactly: same modules, state_dict keys / shapes / values under a fixed
  seed (so the same init RNG draws), same forward -- the pilot control arm and
  every older checkpoint rely on it. The reference is convfsenet/model.py as of
  BASE_COMMIT, loaded from git.
* convfsenet/fold.py turns a frontend_norm='batch' + input_norm model into a
  plain model with the same function (max relative difference <= 1e-5 fp32),
  whose state_dict loads strict into the flags-off model class, and which the
  existing streaming wrapper accepts.
* LinearWarmup scales the generator lr per step without touching the per-epoch
  cosine schedule, and is a no-op at warmup_steps=0.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
from torch import nn

from common.env import AttrDict
from convfsenet.fold import fold_gate, fold_model, fold_state_dict, plain_config
from convfsenet.model import build_causal_model
from convfsenet.streaming import ConvFSENetStreamingFast
from convfsenet.train import LinearWarmup

REPO = Path(__file__).resolve().parents[1]
BASE_COMMIT = "e27adc5583161e7c6d41c0db74d14ab1cfff6f41"     # last commit before the block-design options
DENSE_CFG = Path("/home/clement/eco8-neaixt/configs/cfs_c96_dense.json")
if not DENSE_CFG.is_file():
    DENSE_CFG = REPO / "configs" / "cfs_c96_dense.json"
STATS = REPO / "configs" / "stats" / "convfsenet_vbd_train_magc0.3_nfft512.json"
TRAINED = Path("/home/clement/eco8-neaixt/cp_cfs_c96_dense/g_best")
TOL = 1e-5


def _h(path=DENSE_CFG) -> AttrDict:
    with open(path) as f:
        return AttrDict(json.load(f))


def _block_h() -> AttrDict:
    h = _h()
    h["frontend_norm"] = "batch"
    h["input_norm"] = {"kind": "mean", "path": str(STATS)}
    h["warmup_steps"] = 2000
    return h


@pytest.fixture(scope="module")
def base_module(tmp_path_factory):
    """convfsenet/model.py as of BASE_COMMIT, imported under another name."""
    try:
        src = subprocess.check_output(["git", "-C", str(REPO), "show", f"{BASE_COMMIT}:convfsenet/model.py"],
                                      stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"base commit {BASE_COMMIT[:8]} not reachable from {REPO}")
    p = tmp_path_factory.mktemp("base") / "convfsenet_model_base.py"
    p.write_bytes(src)
    spec = importlib.util.spec_from_file_location("convfsenet_model_base", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stft(model, B=2, S=16000, seed=1):
    g = torch.Generator().manual_seed(seed)
    return model.preproc(torch.randn(B, 1, S, generator=g))


def _randomize_block_stats(model, seed=0):
    """Non-trivial frontend BN stats + a random mu (so the fold has something to absorb)."""
    g = torch.Generator().manual_seed(seed)
    bn = model.frontend[1]
    C = bn.num_features
    with torch.no_grad():
        bn.weight.copy_(0.5 + torch.rand(C, generator=g))
        bn.bias.copy_(0.3 * torch.randn(C, generator=g))
        bn.running_mean.copy_(0.5 * torch.randn(C, generator=g))
        bn.running_var.copy_(0.2 + torch.rand(C, generator=g))
        model.input_mean.copy_(0.2 + 2.0 * torch.rand(model.input_mean.numel(), generator=g))
        for blk in model.tcm:                                    # TCM BNs too: the fold must leave them alone
            for norm in (blk.norm1, blk.norm2):
                norm.running_mean.copy_(0.1 * torch.randn(norm.num_features, generator=g))
                norm.running_var.copy_(0.5 + torch.rand(norm.num_features, generator=g))


# ----------------------------------------------------------------------------- flags off


@pytest.mark.parametrize("off", [None, {}, {"frontend_norm": None, "input_norm": None, "warmup_steps": 0},
                                 {"frontend_norm": "none", "input_norm": None}])
def test_flags_off_identical_to_base_commit(base_module, off):
    h = _h()
    if off:
        h.update(off)
    torch.manual_seed(0)
    ref = base_module.build_causal_model(_h())
    torch.manual_seed(0)
    new = build_causal_model(h)
    assert repr(new) == repr(ref)
    sr, sn = ref.state_dict(), new.state_dict()
    assert list(sn) == list(sr)
    for k in sr:
        assert sn[k].shape == sr[k].shape and sn[k].dtype == sr[k].dtype, k
        assert torch.equal(sn[k], sr[k]), f"{k}: different init values (RNG draws changed)"
    assert [n for n, _ in new.named_buffers()] == [n for n, _ in ref.named_buffers()]
    assert not new.input_centering and new.frontend_norm is None

    stft = _stft(new)
    for train in (False, True):                               # eval (running stats) and train (batch stats)
        ref.train(train); new.train(train)
        with torch.no_grad():
            assert torch.equal(new(stft), ref(stft))
    # one full training step: same loss, same grads
    torch.manual_seed(3); x_n = torch.randn(2, 1, 8000); x_c = torch.randn(2, 1, 8000)
    ref.train(); new.train()
    lr, ln = ref.train_step(x_n, x_c)["loss"], new.train_step(x_n, x_c)["loss"]
    assert torch.equal(lr, ln)
    lr.backward(); ln.backward()
    for (k, pr), (_, pn) in zip(ref.named_parameters(), new.named_parameters()):
        assert torch.equal(pr.grad, pn.grad), k


@pytest.mark.skipif(not TRAINED.is_file(), reason="trained cp_cfs_c96_dense/g_best not present")
def test_existing_checkpoint_loads_strict_into_flags_off_model():
    sd = torch.load(TRAINED, map_location="cpu", weights_only=True)["generator"]
    build_causal_model(_h()).load_state_dict(sd, strict=True)


# ----------------------------------------------------------------------------- flags on


def test_block_model_structure_and_masker_uses_raw_stft():
    torch.manual_seed(0)
    m = build_causal_model(_block_h()).eval()
    assert isinstance(m.frontend[0], nn.Conv1d) and m.frontend[0].kernel_size == (1,)
    assert isinstance(m.frontend[1], nn.BatchNorm1d) and isinstance(m.frontend[2], nn.ReLU)
    sd = m.state_dict()
    assert sd["input_mean"].shape == (257,)
    stats = json.load(open(STATS))
    assert torch.equal(sd["input_mean"], torch.tensor(stats["mean"], dtype=torch.float32))
    extra = set(sd) - set(build_causal_model(_h()).state_dict())
    assert extra == {"input_mean", "frontend.1.weight", "frontend.1.bias", "frontend.1.running_mean",
                     "frontend.1.running_var", "frontend.1.num_batches_tracked"}
    _randomize_block_stats(m)
    stft = _stft(m).squeeze(1)
    with torch.no_grad():
        feats = m.features_extractor(stft)
        assert torch.equal(m.frontend_features(stft), feats - m.input_mean.view(-1, 1))
        mask = m.backend(m.tcm(m.frontend(feats - m.input_mean.view(-1, 1))))
        assert torch.equal(m(stft.unsqueeze(1)).squeeze(1), stft * mask)   # mask applied to the ORIGINAL stft


def test_input_norm_refuses_stats_of_other_features(tmp_path):
    h = _block_h()
    h["compress_factor"] = 0.5
    with pytest.raises(ValueError, match="compress_factor"):
        build_causal_model(h)
    h = _block_h(); h["input_norm"] = {"kind": "mean", "path": str(tmp_path / "missing.json")}
    with pytest.raises(FileNotFoundError):
        build_causal_model(h)
    h = _block_h(); h["input_norm"] = {"kind": "meanstd", "path": str(STATS)}
    with pytest.raises(ValueError):
        build_causal_model(h)
    h = _block_h(); h["frontend_norm"] = "layer"
    with pytest.raises(ValueError):
        build_causal_model(h)


def test_stats_file_provenance_matches_pilot_config():
    s = json.load(open(STATS))
    assert s["kind"] == "mean" and len(s["mean"]) == 257
    assert s["convergence"]["pass"]
    assert s["provenance"]["split"] == "train"
    b1 = json.load(open(REPO / "configs" / "pilot_cf_B1.json"))
    assert b1["input_norm"]["path"] == "configs/stats/convfsenet_vbd_train_magc0.3_nfft512.json"


# ----------------------------------------------------------------------------- fold


def _block_model(seed=0):
    torch.manual_seed(seed)
    m = build_causal_model(_block_h())
    _randomize_block_stats(m, seed)
    return m.eval()


def test_fold_state_dict_loads_strict_into_plain_model():
    m = _block_model()
    sd = fold_state_dict(m)
    plain = build_causal_model(_h())                          # flags-off class, from the dense config
    assert list(sd) == list(plain.state_dict())
    assert {k: v.shape for k, v in sd.items()} == {k: v.shape for k, v in plain.state_dict().items()}
    plain.load_state_dict(sd, strict=True)
    assert "frontend_norm" not in plain_config(_block_h()) and "input_norm" not in plain_config(_block_h())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fold_exact_fp32(seed):
    m = _block_model(seed)
    folded = fold_model(m, _block_h())
    assert folded.frontend_norm is None and not folded.input_centering
    r = fold_gate(m, folded, _stft(m, B=3, S=24000, seed=seed + 10))
    assert r["mask_max_rel"] <= TOL and r["stft_max_rel"] <= TOL, r


def test_fold_exact_fp64():
    m = _block_model().double()
    folded = fold_model(m, _block_h())
    assert next(folded.parameters()).dtype == torch.float64
    stft = _stft(m).to(torch.complex128)
    r = fold_gate(m, folded, stft)
    assert r["mask_max_rel"] <= 1e-12 and r["stft_max_rel"] <= 1e-12, r


def test_fold_without_bn_or_without_mu():
    for drop in ("frontend_norm", "input_norm"):
        h = _block_h(); del h[drop]
        torch.manual_seed(0)
        m = build_causal_model(h).eval()
        with torch.no_grad():
            if m.input_centering:
                m.input_mean.uniform_(0.2, 2.0)
            else:
                m.frontend[1].running_var.uniform_(0.2, 1.2); m.frontend[1].running_mean.normal_()
        folded = fold_model(m, h)
        r = fold_gate(m, folded, _stft(m))
        assert r["mask_max_rel"] <= TOL, (drop, r)


def test_fold_leaves_model_mode_and_weights_untouched():
    m = _block_model().train()
    before = copy.deepcopy(m.state_dict())
    fold_model(m, _block_h())
    assert m.training
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_folded_model_goes_through_existing_streaming_path():
    m = _block_model()
    folded = fold_model(m, _block_h())
    stft = _stft(m, B=1, S=16000).squeeze(1)
    with torch.no_grad():
        ref = m(stft.unsqueeze(1)).squeeze(1)
        got = ConvFSENetStreamingFast(folded).eval().forward_full(stft)
    rel = float((got - ref).abs().max() / ref.abs().max())
    assert rel <= TOL, rel


# ----------------------------------------------------------------------------- warmup


def _simulate(warmup_steps, epochs=4, steps_per_epoch=5, lr=8e-4, lr_min=1e-5):
    p = nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([p], lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr_min)
    w = LinearWarmup(opt, warmup_steps)
    per_step, per_epoch = [], []
    steps = 0
    for _ in range(epochs):
        w.epoch_start()
        per_epoch.append(opt.param_groups[0]["lr"])
        for _ in range(steps_per_epoch):
            w.step(steps)
            per_step.append(opt.param_groups[0]["lr"])
            opt.step()
            steps += 1
        w.epoch_end()
        sched.step()
    return per_step, per_epoch


@pytest.mark.parametrize("W", [3, 5, 12, 20])                   # within, at, across, to the end of epochs
def test_warmup_lr_trajectory(W):
    E, N = 4, 5
    base_step, base_epoch = _simulate(0, E, N)
    step, epoch = _simulate(W, E, N)
    assert epoch == base_epoch                                   # the cosine schedule is untouched
    for s, (lr, lr0) in enumerate(zip(step, base_step)):
        assert lr == pytest.approx(lr0 * min(1.0, (s + 1) / W), rel=1e-12), s
    assert step[W - 1] == pytest.approx(base_step[W - 1])        # peak reached at step W-1
    assert step[0] == pytest.approx(base_step[0] / W)


def test_warmup_off_never_touches_lr():
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], 1e-3)
    w = LinearWarmup(opt, 0)
    opt.param_groups[0]["lr"] = 0.123                            # sentinel: an off warmup must not write it
    w.epoch_start(); w.step(0); w.step(10**6); w.epoch_end()
    assert opt.param_groups[0]["lr"] == 0.123 and not w.enabled
    assert _simulate(0) == _simulate(None)
    with pytest.raises(ValueError):
        LinearWarmup(opt, -1)


# ----------------------------------------------------------------------------- pilot configs


def test_pilot_configs():
    base = json.load(open(REPO / "configs" / "cfs_c96_dense.json"))
    b0 = json.load(open(REPO / "configs" / "pilot_cf_B0.json"))
    b1 = json.load(open(REPO / "configs" / "pilot_cf_B1.json"))
    assert b0 == base
    assert {k: v for k, v in b1.items() if k not in ("frontend_norm", "input_norm", "warmup_steps")} == base
    assert b1["frontend_norm"] == "batch" and b1["warmup_steps"] == 2000 and b1["input_norm"]["kind"] == "mean"
    cwd = os.getcwd()
    try:
        os.chdir(REPO)                                            # the stats path is repo-relative
        m = build_causal_model(AttrDict(b1))
    finally:
        os.chdir(cwd)
    assert m.frontend_norm == "batch" and m.input_centering
