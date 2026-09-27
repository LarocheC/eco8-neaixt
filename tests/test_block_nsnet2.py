"""Block-design NSNet2: flags-off identity, flags-on shape, fold exactness, warmup.

Covers the four config keys added for the block-design pilot (all default OFF):
``input_norm`` (frozen per-bin input mean), ``block_norm: "batch"``
(fc -> BN -> ReLU on fc_in/fc1/fc2), ``warmup_steps`` (generator LR warmup in
nsnet2/train.py) and the export fold (nsnet2/fold.py).

Fold gate: max relative output difference <= 1e-5 in fp32, where the relative
difference is ``max|y_block - y_folded| / max|y_block|`` over the denoised
magnitude; the element-wise worst case is asserted at the same bound on the
mask (sigmoid output, bounded away from 0 in practice).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from common.env import AttrDict
from nsnet2.fold import fold_for_export, strip_block_config
from nsnet2.model import NSNet2

REPO = Path(__file__).resolve().parent.parent
SQ192 = REPO / "configs" / "sq192_dense.json"
STATS = "configs/stats/nsnet2_nfft512_hop256_cf0.3.json"
N_FREQ = 257
GATE = 1e-5


def _cfg(path=SQ192, **extra) -> AttrDict:
    with open(path) as f:
        h = AttrDict(json.load(f))
    h.update(extra)
    return h


def _block_cfg(input_norm=True, block_norm=True):
    extra = {}
    if input_norm:
        extra["input_norm"] = {"kind": "mean", "path": STATS}
    if block_norm:
        extra["block_norm"] = "batch"
    return _cfg(**extra)


def _main_model_class():
    """NSNet2 as defined on ``main`` (before the block-design keys)."""
    try:
        src = subprocess.check_output(
            ["git", "-C", str(REPO), "show", "main:nsnet2/model.py"],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:
        pytest.skip("git / main ref unavailable")
    spec = importlib.util.spec_from_loader("_nsnet2_model_main", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, "main:nsnet2/model.py", "exec"), mod.__dict__)
    return mod.NSNet2


def _inputs(B=2, T=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    mag = torch.rand(B, N_FREQ, T, generator=g) * 3.0 + 1e-3
    pha = (torch.rand(B, N_FREQ, T, generator=g) * 2 - 1) * np.pi
    return mag, pha


# --------------------------------------------------------------------------
# 1. flags off == main
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_name", ["sq192_dense.json", "pilot_ns_A0.json"])
def test_flags_off_identical_to_main(cfg_name):
    MainNSNet2 = _main_model_class()
    h = _cfg(REPO / "configs" / cfg_name)
    torch.manual_seed(1234)
    ref = MainNSNet2(h).eval()
    torch.manual_seed(1234)
    new = NSNet2(h).eval()

    sd_ref, sd_new = ref.state_dict(), new.state_dict()
    assert list(sd_ref.keys()) == list(sd_new.keys())
    for k in sd_ref:
        assert sd_ref[k].shape == sd_new[k].shape, k
        assert torch.equal(sd_ref[k], sd_new[k]), k
    assert [n for n, _ in ref.named_modules()] == [n for n, _ in new.named_modules()]
    assert not hasattr(new, "in_mean") and not hasattr(new, "bn_in")

    mag, pha = _inputs()
    with torch.no_grad():
        out_ref, out_new = ref(mag, pha), new(mag, pha)
    for a, b in zip(out_ref, out_new):
        assert torch.equal(a, b)


def test_pilot_configs_differ_only_in_block_keys():
    base = _cfg()
    for arm, keys in {"A0": set(), "A1": {"input_norm", "warmup_steps"},
                      "A2": {"block_norm"},
                      "A3": {"input_norm", "block_norm", "warmup_steps"}}.items():
        h = _cfg(REPO / "configs" / f"pilot_ns_{arm}.json")
        assert set(h) - set(base) == keys, arm
        assert all(h[k] == base[k] for k in base), arm
    assert (REPO / "configs" / "pilot_ns_A0.json").read_bytes() == SQ192.read_bytes()


# --------------------------------------------------------------------------
# 2. flags on
# --------------------------------------------------------------------------

@pytest.mark.parametrize("input_norm,block_norm", [(True, False), (False, True), (True, True)])
def test_flags_on_forward_and_keys(input_norm, block_norm):
    torch.manual_seed(0)
    model = NSNet2(_block_cfg(input_norm, block_norm))
    sd = model.state_dict()
    base_keys = set(NSNet2(_cfg()).state_dict())
    extra = set(sd) - base_keys
    assert base_keys <= set(sd)
    want = set()
    if input_norm:
        want.add("in_mean")
        assert sd["in_mean"].shape == (N_FREQ,)
        stats = json.load(open(REPO / STATS))["mean"]
        assert torch.equal(sd["in_mean"], torch.tensor(stats, dtype=torch.float32))
    if block_norm:
        for bn, c in (("bn_in", 192), ("bn1", 192), ("bn2", 192)):
            for p in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                want.add(f"{bn}.{p}")
            assert sd[f"{bn}.weight"].shape == (c,)
    assert extra == want

    mag, pha = _inputs(B=3, T=25)
    for mode in (model.train, model.eval):
        mode()
        dm, dp, dc = model(mag, pha)
        assert dm.shape == mag.shape and dp.shape == pha.shape
        assert dc.shape == (*mag.shape, 2)
        assert torch.isfinite(dm).all()
        # the mask multiplies the ORIGINAL (un-centred) magnitude
        mask = dm / mag
        assert ((mask > 0) & (mask < 1)).all()


def test_sparsity_controller_still_finds_linears():
    from nsnet2.sparsity import SparsityController
    model = NSNet2(_block_cfg())
    ctl = SparsityController.from_config(model, {"enabled": True, "pattern": "2:4"})
    ref = SparsityController.from_config(NSNet2(_cfg()), {"enabled": True, "pattern": "2:4"})
    assert list(ctl.masks) == list(ref.masks)
    assert {"fc_in.weight", "fc1.weight", "fc2.weight", "fc_out.weight"} <= set(ctl.masks)


# --------------------------------------------------------------------------
# 3. fold
# --------------------------------------------------------------------------

def _randomise(model, seed=0):
    """Non-trivial BN running stats + affine params and a random mu."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if getattr(model, "input_norm", False):
            model.in_mean.copy_(torch.rand(N_FREQ, generator=g) * 2.6 + 0.15)
        if getattr(model, "block_norm", False):
            for bn in (model.bn_in, model.bn1, model.bn2):
                c = bn.num_features
                bn.running_mean.copy_(torch.randn(c, generator=g) * 2.0)
                bn.running_var.copy_(torch.rand(c, generator=g) * 4.0 + 0.05)
                bn.weight.copy_(torch.randn(c, generator=g) * 1.5)
                bn.bias.copy_(torch.randn(c, generator=g) * 0.5)
                bn.num_batches_tracked.fill_(1000)
    return model.eval()


def _rel(a, b):
    return ((a - b).abs().max() / a.abs().max()).item()


def _max_elem_rel(a, b):
    return ((a - b).abs() / a.abs().clamp_min(1e-12)).max().item()


def _masks(model, mag, pha):
    dm, _, _ = model(mag, pha)
    return dm, dm / mag


@pytest.mark.parametrize("input_norm,block_norm", [(True, False), (False, True), (True, True)])
def test_fold_exact_random_inputs(input_norm, block_norm):
    torch.manual_seed(1)
    model = _randomise(NSNet2(_block_cfg(input_norm, block_norm)), seed=3)
    plain, sd = fold_for_export(model)

    # strict load into NSNet2 built from the config with the new keys removed
    stripped = strip_block_config(model.h)
    assert not set(stripped) & {"input_norm", "block_norm"}
    target = NSNet2(stripped)
    target.load_state_dict(sd, strict=True)
    assert set(sd) == set(NSNet2(_cfg()).state_dict())
    assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in plain.modules())

    worst = 0.0
    for seed in range(4):
        mag, pha = _inputs(B=4, T=60, seed=seed)
        with torch.no_grad():
            dm_b, _ = _masks(model, mag, pha)
            dm_f, _ = _masks(target.eval(), mag, pha)
        worst = max(worst, _rel(dm_b, dm_f))
    print(f"fold random-input max rel diff ({input_norm=}, {block_norm=}): {worst:.3e}")
    assert worst <= GATE


def _real_utterances(n=20):
    try:
        from common.dataset import load_voicebank_demand, mag_pha_stft
        hf = load_voicebank_demand()
    except Exception as e:  # pragma: no cover - offline box
        pytest.skip(f"VoiceBank-DEMAND unavailable: {e}")
    h = _cfg()
    out = []
    for item in hf["test"].select(range(n)):
        noisy = torch.tensor(np.asarray(item["noisy"]["array"], dtype=np.float32))
        norm = torch.sqrt(len(noisy) / (torch.sum(noisy ** 2.0) + 1e-8))
        mag, pha, _ = mag_pha_stft((noisy * norm).unsqueeze(0), h.n_fft, h.hop_size,
                                   h.win_size, h.compress_factor)
        out.append((mag, pha))
    return out


def test_fold_exact_real_utterances():
    torch.manual_seed(2)
    model = _randomise(NSNet2(_block_cfg()), seed=5)
    with torch.no_grad():
        model.in_mean.copy_(torch.tensor(json.load(open(REPO / STATS))["mean"]))
    _, sd = fold_for_export(model)
    target = NSNet2(strip_block_config(model.h))
    target.load_state_dict(sd, strict=True)
    target.eval()
    worst = worst_mask = 0.0
    utts = _real_utterances(20)
    assert len(utts) == 20
    for mag, pha in utts:
        with torch.no_grad():
            dm_b, m_b = _masks(model, mag, pha)
            dm_f, m_f = _masks(target, mag, pha)
        worst = max(worst, _rel(dm_b, dm_f))
        worst_mask = max(worst_mask, _max_elem_rel(m_b, m_f))
    print(f"fold 20 real utts: max rel diff {worst:.3e}, element-wise mask {worst_mask:.3e}")
    assert worst <= GATE
    assert worst_mask <= GATE


def test_fold_is_noop_without_flags():
    torch.manual_seed(4)
    model = NSNet2(_cfg()).eval()
    _, sd = fold_for_export(model)
    for k, v in model.state_dict().items():
        assert torch.equal(v, sd[k]), k


def _export(tmp_dir, h, state_dict):
    from nsnet2.export_onnx import export_streaming
    import onnx
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with open(tmp_dir / "config.json", "w") as f:
        json.dump(dict(h), f)
    torch.save({"generator": state_dict}, tmp_dir / "g_best")
    path = export_streaming(tmp_dir / "g_best", tmp_dir / "model.onnx")
    return path, onnx.load(str(path))


def test_fold_through_existing_onnx_export(tmp_path):
    import onnxruntime as ort
    torch.manual_seed(6)
    model = _randomise(NSNet2(_block_cfg()), seed=7)
    _, sd = fold_for_export(model)
    path_f, onnx_f = _export(tmp_path / "folded", strip_block_config(model.h), sd)

    torch.manual_seed(6)
    old = NSNet2(_cfg())
    _, onnx_o = _export(tmp_path / "old", _cfg(), old.state_dict())

    ops_f = Counter(n.op_type for n in onnx_f.graph.node)
    ops_o = Counter(n.op_type for n in onnx_o.graph.node)
    print(f"folded ONNX ops: {dict(ops_f)}")
    assert ops_f.get("BatchNormalization", 0) == 0
    assert ops_f == ops_o
    assert [i.name for i in onnx_f.graph.input] == [i.name for i in onnx_o.graph.input]

    # the exported folded graph reproduces the block model (streamed per frame)
    sess = ort.InferenceSession(str(path_f), providers=["CPUExecutionProvider"])
    mag, pha = _inputs(B=1, T=50, seed=9)
    with torch.no_grad():
        dm_b, _, _ = model(mag, pha)
    states = np.zeros((2, 1, 192), dtype=np.float32)
    masks = []
    for t in range(mag.shape[-1]):
        m, states = sess.run(None, {"frame_in": mag[:, :, t].numpy(), "states_in": states})
        masks.append(m)
    dm_onnx = mag * torch.from_numpy(np.stack(masks, axis=-1))
    rel = _rel(dm_b, dm_onnx)
    print(f"block torch vs folded ONNX (streamed): max rel diff {rel:.3e}")
    assert rel <= 1e-5


# --------------------------------------------------------------------------
# 4. warmup lr trajectory
# --------------------------------------------------------------------------

def _lr_trajectory(warmup_steps, epochs=10, steps_per_epoch=45, lr=3e-3, gamma=0.99,
                   start_step=0, start_epoch=0):
    from nsnet2.train import LinearWarmup
    p = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([p], lr, betas=[0.8, 0.99])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    warm = LinearWarmup(opt, warmup_steps) if warmup_steps is not None else None
    step_lrs, epoch_lrs = [], []
    steps = start_step
    for _ in range(start_epoch, epochs):
        if warm:
            warm.epoch_start()
        epoch_lrs.append(opt.param_groups[0]["lr"])
        for _ in range(steps_per_epoch):
            if warm:
                warm.before_step(steps)
            step_lrs.append(opt.param_groups[0]["lr"])
            p.grad = torch.ones_like(p)
            opt.step()
            steps += 1
        if warm:
            warm.epoch_end()
        sched.step()
    return step_lrs, epoch_lrs, sched


def test_warmup_lr_trajectory():
    # 45 steps/epoch = VoiceBank train (11572) // batch 256: the 200-step ramp
    # crosses four epoch boundaries, where ExponentialLR decays the base.
    lr0, gamma, W, spe, E = 3e-3, 0.99, 200, 45, 10
    ref_steps, ref_epochs, _ = _lr_trajectory(None, E, spe, lr0, gamma)
    got_steps, got_epochs, sched = _lr_trajectory(W, E, spe, lr0, gamma)

    # the per-epoch ExponentialLR values are bit-identical to the no-warmup run
    assert got_epochs == ref_epochs
    for e, v in enumerate(ref_epochs):
        assert v == pytest.approx(lr0 * gamma ** e, rel=1e-12)
    # warmup ramp: lr_epoch * (s+1)/W for s < W, then exactly lr_epoch
    for s, v in enumerate(got_steps):
        e = s // spe
        if s < W:
            assert v == pytest.approx(ref_epochs[e] * (s + 1) / W, rel=1e-12), s
        else:
            assert v == ref_steps[s] == ref_epochs[e], s
    assert got_steps[0] == pytest.approx(lr0 / W)
    assert got_steps[W - 1] == pytest.approx(ref_epochs[(W - 1) // spe])
    # the ramp is taken on the CURRENT epoch's lr, so an epoch boundary late in
    # the ramp can dip it (0.99 * 181 < 180 at step 180): by design, the
    # scheduler owns the base value and the warmup only scales it.
    assert got_steps[180] < got_steps[179]


def test_warmup_off_is_noop():
    from nsnet2.train import LinearWarmup
    assert not LinearWarmup(torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], 1.0), 0).enabled
    ref = _lr_trajectory(None)
    off = _lr_trajectory(0)
    assert off[0] == ref[0] and off[1] == ref[1]


def test_warmup_resume_mid_ramp():
    """Resuming at a later epoch/step continues the same ramp (base lr = the
    scheduler's restored value, never a saved scaled one)."""
    lr0, gamma, W, spe = 3e-3, 0.99, 200, 45
    full, _, _ = _lr_trajectory(W, 6, spe, lr0, gamma)
    from nsnet2.train import LinearWarmup
    p = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([p], lr0)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    for _ in range(2):        # scheduler state as restored from a checkpoint at epoch 2
        sched.step()
    warm = LinearWarmup(opt, W)
    warm.epoch_start()
    got = []
    for s in range(2 * spe, 3 * spe):
        warm.before_step(s)
        got.append(opt.param_groups[0]["lr"])
    assert got == pytest.approx(full[2 * spe:3 * spe], rel=1e-12)


# --------------------------------------------------------------------------
# 5. train.py wiring smoke (tiny synthetic model + data, CPU, 2 epochs x 4 steps)
# --------------------------------------------------------------------------

def test_train_loop_smoke_all_flags(tmp_path):
    """train.py runs with every flag on: BN in train mode, centring, warmup
    across an epoch boundary; the checkpoint carries the new keys and folds.
    Synthetic 8-utterance dataset, hidden 16, n_fft 64 -- a wiring check,
    not a training run."""
    from datasets import Dataset as HFDataset, DatasetDict
    from nsnet2.fold import fold_checkpoint

    n_freq = 33
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"mean": [0.5] * n_freq, "n_frames": 1}))
    h_dict = {
        "num_gpus": 0, "batch_size": 2, "learning_rate": 1e-4, "adam_b1": 0.8,
        "adam_b2": 0.99, "lr_decay": 0.99, "seed": 1234, "hidden_dim": 16,
        "fc_hidden_dim": 16, "num_gru_layers": 2, "compress_factor": 0.3,
        "linear": {"kind": "linear"}, "gru": {"kind": "gru"},
        "sampling_rate": 16000, "segment_size": 4096, "n_fft": 64, "hop_size": 16,
        "win_size": 64, "num_workers": 0,
        "dist_config": {"dist_backend": "nccl", "dist_url": "tcp://localhost:54321",
                        "world_size": 1},
        "input_norm": {"kind": "mean", "path": str(stats)},
        "block_norm": "batch",
        "warmup_steps": 6,
    }
    config_path = tmp_path / "config_synth.json"
    config_path.write_text(json.dumps(h_dict))

    rng = np.random.default_rng(0)

    def _split(n, prefix):
        audios = [rng.standard_normal(8192).astype(np.float32) for _ in range(n)]
        return {"id": [f"{prefix}_{i}" for i in range(n)],
                "clean": [{"path": f"{prefix}{i}c", "array": a, "sampling_rate": 16000}
                          for i, a in enumerate(audios)],
                "noisy": [{"path": f"{prefix}{i}n", "array": a, "sampling_rate": 16000}
                          for i, a in enumerate(audios)]}

    ds_path = tmp_path / "synthetic_ds"
    DatasetDict({"train": HFDataset.from_dict(_split(8, "tr")),
                 "test": HFDataset.from_dict(_split(2, "te"))}).save_to_disk(str(ds_path))
    (tmp_path / "sitecustomize.py").write_text(
        "from datasets import load_from_disk\n"
        "import common.dataset as _ds_mod\n"
        f"_ds = load_from_disk(r'{ds_path}')\n"
        "_ds_mod.load_voicebank_demand = lambda cache_dir=None: _ds\n")

    env = dict(os.environ)
    env["HF_DATASETS_CACHE"] = str(tmp_path / "hf_cache")
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(REPO), env.get("PYTHONPATH", "")])
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cp_dir = tmp_path / "cp"
    result = subprocess.run(
        [sys.executable, "-m", "nsnet2.train", "--config", str(config_path),
         "--checkpoint_path", str(cp_dir), "--training_epochs", "2",
         "--validation_interval", "4", "--checkpoint_interval", "7",
         "--best_checkpoint_start_epoch", "999"],
        capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=300)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "Generator LR warmup: 6 steps" in result.stdout
    assert "Epoch: 2" in result.stdout

    g = torch.load(cp_dir / "g_00000007", map_location="cpu", weights_only=True)["generator"]
    assert {"in_mean", "bn_in.running_mean", "bn1.weight", "bn2.running_var"} <= set(g)
    assert int(g["bn_in.num_batches_tracked"]) == 8     # BN saw every train step
    do = torch.load(cp_dir / "do_00000007", map_location="cpu", weights_only=False)
    # checkpoint saved after the ramp: the scheduler state holds the unscaled lr
    assert do["scheduler_g"]["_last_lr"][0] == pytest.approx(1e-4 * 0.99)

    folded = fold_checkpoint(cp_dir / "g_00000007", tmp_path / "folded")
    h2 = AttrDict(json.loads((tmp_path / "folded" / "config.json").read_text()))
    assert "input_norm" not in h2 and "block_norm" not in h2
    NSNet2(h2).load_state_dict(
        torch.load(folded, map_location="cpu", weights_only=True)["generator"], strict=True)
