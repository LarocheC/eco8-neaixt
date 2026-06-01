"""tests/test_train_quant_smoke.py - Phase 6 end-to-end subprocess integration smoke.

Spawns train.py via a subprocess invocation ([sys.executable, str(REPO_ROOT / 'train.py'),
...]) with h.quant.enabled=true, --validation_interval=1, --training_epochs=1 against a
synthetic mini cp_dir. Verifies:
  - Subprocess exits cleanly (returncode == 0)
  - Stdout contains the verbatim TRN-03 cycle-1 breakdown 'Quant eval: export='
  - The resulting logs/ TB event file contains both new scalar names

This is the load-bearing TRN-01..05 end-to-end gate. The unit tests in
tests/test_quant_hook.py drive run_quant_eval directly; this test exercises
the lazy-import boundary at train.py:242 + the entire validation block in a
real subprocess context.

DET-02 sidestep preserved: zero deserialization-side load sites in this file.
The synthetic HF DatasetDict is built in-memory and persisted via
Dataset.save_to_disk() (writing only); subprocess reads it via load_from_disk
inside a sitecustomize.py shim that monkeypatches dataset.load_voicebank_demand
at interpreter startup (Route A per 06-02-PLAN decision (g)).

See .planning/phases/06-in-training-quant-eval-hook/06-CONTEXT.md (D-01..D-04) and
06-VALIDATION.md (06-02-01).
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
from datasets import Dataset, DatasetDict

from common.env import AttrDict
from nsnet2.model import NSNet2


# Repo root resolved once so subprocess-spawn tests don't depend on pytest invocation cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_train_quant_enabled_subprocess_smoke(tmp_path):
    """VALIDATION 06-02-01 / TRN-01..05: end-to-end subprocess. train.py with h.quant.enabled=true,
    --validation_interval=1, --training_epochs=1 produces TB event file with both new scalars
    and stdout containing the TRN-03 cycle-1 breakdown.

    Cross-process patch via sitecustomize.py (Route A per 06-02-PLAN decision (g)):
    monkeypatch.setattr does not cross subprocess boundaries, so the synthetic
    DatasetDict is persisted via Dataset.save_to_disk(...) and a sitecustomize.py
    in tmp_path replaces dataset.load_voicebank_demand at interpreter startup.

    Subprocess uses cwd=str(tmp_path) so sitecustomize.py is auto-imported, plus
    PYTHONPATH=str(REPO_ROOT):... so the train.py import resolves the repo modules.
    """
    # 1. Synthesize cp_dir + config.json (mini NSNet2 dimensions for fast test).
    cp_dir = tmp_path / "cp_synth"
    cp_dir.mkdir()
    h_dict = {
        "num_gpus": 0,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": 0.99,
        "seed": 1234,
        "hidden_dim": 16,
        "fc_hidden_dim": 16,
        "num_gru_layers": 2,
        "compress_factor": 0.3,
        "linear": {"kind": "linear"},
        "gru": {"kind": "gru"},
        "sampling_rate": 16000,
        "segment_size": 4096,
        "n_fft": 64,
        "hop_size": 16,
        "win_size": 64,
        "num_workers": 0,
        "dist_config": {
            "dist_backend": "nccl",
            "dist_url": "tcp://localhost:54321",
            "world_size": 1,
        },
        "quant": {"enabled": True, "n_calib_utts": 3},
    }
    config_path = tmp_path / "config_synth.json"
    with open(config_path, "w") as f:
        json.dump(h_dict, f)

    # 2. Synthesize HF DatasetDict and save_to_disk (offline payload for the subprocess).
    rng = np.random.default_rng(0)
    train_audios = [rng.standard_normal(8192).astype(np.float32) for _ in range(8)]
    test_audios  = [rng.standard_normal(8192).astype(np.float32) for _ in range(2)]

    def _make_split(audios, prefix):
        return {
            "id":    [f"{prefix}_{i}" for i in range(len(audios))],
            "clean": [{"path": f"{prefix}_{i}_c", "array": a, "sampling_rate": 16000}
                      for i, a in enumerate(audios)],
            "noisy": [{"path": f"{prefix}_{i}_n", "array": a, "sampling_rate": 16000}
                      for i, a in enumerate(audios)],
        }

    ds = DatasetDict({
        "train": Dataset.from_dict(_make_split(train_audios, "tr")),
        "test":  Dataset.from_dict(_make_split(test_audios,  "te")),
    })
    ds_path = tmp_path / "synthetic_ds"
    ds.save_to_disk(str(ds_path))

    # 3. sitecustomize.py auto-imports at subprocess startup (cwd=tmp_path) and
    #    monkeypatches dataset.load_voicebank_demand AND quant.load_voicebank_demand
    #    (defensive double-patch -- the hook reaches the loader transitively via
    #    quant.quantize_checkpoint -> calibration.VBDCalibrationReader).
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "import os\n"
        "from datasets import load_from_disk\n"
        "import common.dataset as _ds_mod\n"
        f"_ds = load_from_disk(r'{ds_path}')\n"
        "def _patched_loader(cache_dir=None):\n"
        "    return _ds\n"
        "_ds_mod.load_voicebank_demand = _patched_loader\n"
        "try:\n"
        "    import nsnet2.quant as _q_mod\n"
        "    _q_mod.load_voicebank_demand = _patched_loader\n"
        "except ImportError:\n"
        "    pass\n"
    )

    # 4. Spawn train.py. cwd=tmp_path so sitecustomize.py is auto-imported.
    #    HF_DATASETS_CACHE redirected to tmp_path/hf_cache for vbd-hashes/ isolation.
    env = dict(os.environ)
    env["HF_DATASETS_CACHE"] = str(tmp_path / "hf_cache")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # CUBLAS_WORKSPACE_CONFIG must be set BEFORE torch import in the subprocess
    # (mirrors tests/conftest.py session-scoped determinism).
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    result = subprocess.run(
        [sys.executable, "-m", "nsnet2.train",
         "--config", str(config_path),
         "--checkpoint_path", str(cp_dir),
         "--training_epochs", "1",
         "--validation_interval", "1",
         "--best_checkpoint_start_epoch", "999",
         ],
        capture_output=True, text=True, check=False,
        cwd=str(tmp_path),
        env=env,
        timeout=240,
    )

    # 5. Subprocess exit gate.
    assert result.returncode == 0, (
        f"train.py exited {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    # 6. Stdout assertion: TRN-03 cycle-1 breakdown fired.
    assert "Quant eval: export=" in result.stdout, (
        f"Missing TRN-03 stdout breakdown. stdout:\n{result.stdout}"
    )

    # 7. TB event file contains both new scalars (TRN-02 verbatim names).
    log_dir = cp_dir / "logs"
    assert log_dir.exists(), f"TB log dir missing: {log_dir}"
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    assert "Validation/PESQ Score (int8)"      in tags, tags
    assert "Validation/PESQ Delta (FP32-int8)" in tags, tags
