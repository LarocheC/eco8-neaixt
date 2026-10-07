"""The opt-in options of nsnet2.train: --seed, --snapshot_epochs, --state_epochs,
--init_only, --init_from / --init_keys, the run records and the refusal to
resume.

Subprocess runs on the CPU against a synthetic in-memory dataset (the
sitecustomize route of tests/test_train_quant_smoke.py), with a small NSNet2
whose GRU input projections are MixedRadixButterfly layers.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("torch_structured")
from datasets import Dataset, DatasetDict  # noqa: E402

from common.env import AttrDict  # noqa: E402
from nsnet2.model import NSNet2  # noqa: E402
from nsnet2.run_record import load_generator_state, parse_epoch_list, state_fingerprint  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

H = {
    "num_gpus": 0, "batch_size": 2, "learning_rate": 1e-3, "adam_b1": 0.8, "adam_b2": 0.99,
    "lr_decay": 0.99, "seed": 1234, "hidden_dim": 16, "fc_hidden_dim": 16, "num_gru_layers": 2,
    "compress_factor": 0.3,
    "linear": {"kind": "linear"},
    "gru": {"kind": "butterfly", "nblocks": 1, "x_radices": [2, 4, 2], "x_init": "randn",
            "h_init": "ortho"},
    "sampling_rate": 16000, "segment_size": 4096, "n_fft": 64, "hop_size": 16, "win_size": 64,
    "num_workers": 0,
    "dist_config": {"dist_backend": "nccl", "dist_url": "tcp://localhost:54321", "world_size": 1},
}
STEPS_PER_EPOCH = 4          # 8 training utterances, batch 2


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Config file, synthetic dataset and sitecustomize shim shared by the runs of this module."""
    root = tmp_path_factory.mktemp("train_snapshots")
    cfg = root / "mini_arm.json"
    cfg.write_text(json.dumps(H))
    rng = np.random.default_rng(0)

    def split(n, prefix):
        audios = [(0.1 * rng.standard_normal(8192)).astype(np.float32) for _ in range(n)]
        return {"id": [f"{prefix}_{i}" for i in range(n)],
                "clean": [{"path": f"{prefix}_{i}_c", "array": a, "sampling_rate": 16000}
                          for i, a in enumerate(audios)],
                "noisy": [{"path": f"{prefix}_{i}_n", "array": a + 0.05 * rng.standard_normal(8192).astype(np.float32),
                           "sampling_rate": 16000} for i, a in enumerate(audios)]}

    ds = DatasetDict({"train": Dataset.from_dict(split(8, "tr")), "test": Dataset.from_dict(split(2, "te"))})
    ds.save_to_disk(str(root / "ds"))
    (root / "sitecustomize.py").write_text(
        "from datasets import load_from_disk\n"
        "import common.dataset as _ds_mod\n"
        f"_ds = load_from_disk(r'{root / 'ds'}')\n"
        "_ds_mod.load_voicebank_demand = lambda cache_dir=None: _ds\n")
    return root, cfg


def train(env, run_dir, *args, check=True):
    root, cfg = env
    e = dict(os.environ)
    e["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + e.get("PYTHONPATH", "")
    e["HF_DATASETS_CACHE"] = str(root / "hf_cache")
    e["CUDA_VISIBLE_DEVICES"] = ""                     # CPU
    res = subprocess.run([sys.executable, "-m", "nsnet2.train", "--config", str(cfg),
                          "--checkpoint_path", str(run_dir), *args],
                         capture_output=True, text=True, cwd=str(root), env=e, timeout=300)
    if check:
        assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


def fresh_model(seed: int) -> dict:
    """The model as train.py builds it at this seed, before any optimiser step."""
    torch.manual_seed(seed)
    return NSNet2(AttrDict(json.loads(json.dumps(dict(H, seed=seed))))).state_dict()


def test_parse_epoch_list():
    assert parse_epoch_list("0-20,25-200:5") == list(range(21)) + list(range(25, 201, 5))
    assert len(parse_epoch_list("0-20,25-200:5")) == 57
    assert parse_epoch_list("50,100,150,200") == [50, 100, 150, 200]
    assert parse_epoch_list("") == []
    for bad in ("3-1", "x", "1-5:0", "-2"):
        with pytest.raises(ValueError):
            parse_epoch_list(bad)


def test_init_only_writes_the_untrained_model_with_the_overridden_seed(env, tmp_path):
    run = tmp_path / "cp_init"
    train(env, run, "--init_only", "--seed", "77")
    assert json.loads((run / "config.json").read_text())["seed"] == 77
    ck = torch.load(run / "snapshots" / "g_e000", map_location="cpu", weights_only=True)
    assert ck["epoch"] == 0 and ck["steps"] == 0
    rec = json.loads((run / "run.json").read_text())
    assert rec["seed"] == 77 and rec["arm"] == "mini_arm" and rec["run"] == "cp_init"
    assert rec["end_time"] is not None and rec["start_time"] is not None
    assert rec["g_e000_fingerprint"] == state_fingerprint(ck["generator"])
    assert rec["g_e000_fingerprint"] == state_fingerprint(fresh_model(77))
    assert rec["g_e000_fingerprint"] != state_fingerprint(fresh_model(78))
    assert not (run / "logs").exists() and sorted(os.listdir(run / "snapshots")) == ["g_e000"]


def test_snapshots_states_and_records_of_a_short_run(env, tmp_path):
    run = tmp_path / "cp_short"
    res = train(env, run, "--seed", "5", "--training_epochs", "3", "--snapshot_epochs", "0-1,3",
                "--state_epochs", "2", "--validation_interval", "2", "--best_checkpoint_start_epoch", "0",
                "--stdout_interval", "1", "--checkpoint_interval", "1000")
    assert "Initial weights:" in res.stdout
    snaps = sorted(os.listdir(run / "snapshots"))
    assert snaps == ["g_e000", "g_e001", "g_e003"]
    rec = json.loads((run / "run.json").read_text())
    assert rec["end_time"] is not None and rec["steps_per_epoch"] == STEPS_PER_EPOCH
    assert rec["snapshot_epochs"] == [0, 1, 3] and rec["state_epochs"] == [2]
    sds = {}
    for name in snaps:
        ck = torch.load(run / "snapshots" / name, map_location="cpu", weights_only=True)
        epoch = int(name[3:])
        assert ck["epoch"] == epoch and ck["steps"] == epoch * STEPS_PER_EPOCH
        sds[epoch] = ck["generator"]
    # g_e000 is the model before the first optimiser step; training changes it afterwards
    assert state_fingerprint(sds[0]) == rec["g_e000_fingerprint"] == state_fingerprint(fresh_model(5))
    assert state_fingerprint(sds[1]) != state_fingerprint(sds[0])
    state = torch.load(run / "states" / "s_e002", map_location="cpu", weights_only=False)
    assert {"generator", "discriminator", "optim_g", "optim_d", "scheduler_g", "scheduler_d", "steps",
            "epoch", "best_pesq", "rng"} <= set(state)
    assert state["epoch"] == 2 and state["steps"] == 2 * STEPS_PER_EPOCH
    lines = [json.loads(x) for x in (run / "val.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines] == list(range(2, 3 * STEPS_PER_EPOCH, 2))
    assert {"step", "epochs_finished", "pesq", "loss_mag", "loss_com", "loss_stft"} <= set(lines[0])
    best = json.loads((run / "best.json").read_text())
    assert best["pesq"] == max(x["pesq"] for x in lines)
    assert (run / "g_best").exists()


def test_init_from_changes_the_named_tensors_and_nothing_else(env, tmp_path):
    donor, base, mixed = tmp_path / "cp_donor", tmp_path / "cp_base", tmp_path / "cp_mixed"
    train(env, donor, "--init_only", "--seed", "11")
    train(env, base, "--init_only", "--seed", "12")
    train(env, mixed, "--init_only", "--seed", "12", "--init_from", str(donor / "snapshots" / "g_e000"),
          "--init_keys", "gru.cells.0.x_proj")
    d, b, m = (load_generator_state(p / "snapshots" / "g_e000") for p in (donor, base, mixed))
    moved = [k for k in m if k.startswith("gru.cells.0.x_proj")]
    assert sorted(moved) == ["gru.cells.0.x_proj.bias"] + [f"gru.cells.0.x_proj.stages.{j}" for j in range(3)]
    for k in m:
        assert torch.equal(m[k], d[k] if k in moved else b[k]), k
        if k in moved:
            assert not torch.equal(d[k], b[k]), k
    rec = json.loads((mixed / "run.json").read_text())
    assert rec["init_from"]["keys"] == moved and rec["init_from"]["prefix"] == "gru.cells.0.x_proj"
    assert rec["g_e000_fingerprint"] == state_fingerprint(m)


def test_expect_fingerprint_stops_a_run_that_starts_elsewhere(env, tmp_path):
    res = train(env, tmp_path / "cp_x", "--init_only", "--seed", "3", "--expect_fingerprint", "0" * 64,
                check=False)
    assert res.returncode != 0 and "differs from --expect_fingerprint" in res.stdout + res.stderr


@pytest.mark.parametrize("leftover", ["rolling", "snapshot", "run_json"])
def test_refuses_to_start_in_a_used_directory(env, tmp_path, leftover):
    run = tmp_path / "cp_used"
    if leftover == "rolling":
        run.mkdir()
        torch.save({"generator": fresh_model(1)}, run / "g_00000004")
    elif leftover == "snapshot":
        train(env, run, "--init_only", "--seed", "1")
        (run / "run.json").unlink()
    else:
        run.mkdir()
        (run / "run.json").write_text("{}")
    res = train(env, run, "--seed", "1", "--training_epochs", "1", "--snapshot_epochs", "0-1", check=False)
    assert res.returncode != 0
    assert "fresh directory" in res.stdout + res.stderr
