"""End-to-end subprocess smoke for basenet/train.py.

Spawns ``python -m basenet.train`` against a synthetic in-memory VoiceBank-DEMAND
stand-in (no download) and asserts the whole training orchestration runs: data
loading, a MetricGAN training step, a rolling checkpoint, and a validation pass
that reconstructs audio and scores PESQ. This covers the loop wiring that the
unit-level step tests in test_basenet_train_smoke.py deliberately don't.

Dataset injection follows the repo's established Route A (see
tests/test_train_quant_smoke.py): the synthetic DatasetDict is saved to disk and
a sitecustomize.py in the subprocess cwd monkeypatches
common.dataset.load_voicebank_demand at interpreter startup, since
monkeypatch.setattr does not cross the subprocess boundary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _voiced(samples, sr, f0, seed):
    """A voiced-ish utterance (two sinusoids + light noise) so PESQ is scorable."""
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / sr
    sig = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * 2.7 * f0 * t)
    return (sig + 0.05 * rng.standard_normal(samples)).astype(np.float32)


def test_basenet_train_subprocess_smoke(tmp_path):
    from datasets import Dataset, DatasetDict

    cp_dir = tmp_path / "cp_basenet_synth"
    cp_dir.mkdir()

    # 1. Tiny causal-BASENet config (real band layout, miniature widths/STFT).
    h_dict = {
        "num_gpus": 0,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "lr_min": 1e-5,
        "scheduler": "exponential",
        "adam_b1": 0.9, "adam_b2": 0.999, "lr_decay": 0.99,
        "seed": 1234,
        "sampling_rate": 16000,
        "segment_size": 8192,
        "n_fft": 128, "hop_size": 32, "win_size": 128, "compress_factor": 0.3,
        "base_channels": 8, "expand_ratio": 2, "attn_reduction": 4,
        "band_edges_hz": [0, 1000, 4000, 8000], "band_depths": [4, 3, 2],
        "crn_freq": 4, "crn_hidden": 16, "dec_depth": 1,
        "causal": True,
        "loss": {"magnitude": 0.9, "phase": 0.3, "complex": 0.1, "time": 0.2},
        "gan": {"enabled": True, "metric_loss_lambda": 0.05},
        "num_workers": 0,
        "dist_config": {"dist_backend": "nccl",
                        "dist_url": "tcp://localhost:54321", "world_size": 1},
    }
    config_path = tmp_path / "config_synth.json"
    config_path.write_text(json.dumps(h_dict))

    # 2. Synthetic VBD-shaped DatasetDict saved to disk for the subprocess.
    def _split(prefix, n, seed0):
        audios = [_voiced(8192, 16000, 180 + 40 * i, seed0 + i) for i in range(n)]
        return {
            "id": [f"{prefix}_{i}" for i in range(n)],
            "clean": [{"path": f"{prefix}_{i}_c", "array": a, "sampling_rate": 16000}
                      for i, a in enumerate(audios)],
            "noisy": [{"path": f"{prefix}_{i}_n", "array": a, "sampling_rate": 16000}
                      for i, a in enumerate(audios)],
        }

    ds = DatasetDict({
        "train": Dataset.from_dict(_split("tr", 8, 0)),
        "test": Dataset.from_dict(_split("te", 2, 100)),
    })
    ds_path = tmp_path / "synthetic_ds"
    ds.save_to_disk(str(ds_path))

    # 3. sitecustomize.py patches the loader at subprocess interpreter startup.
    (tmp_path / "sitecustomize.py").write_text(
        "from datasets import load_from_disk\n"
        "import common.dataset as _ds_mod\n"
        f"_ds = load_from_disk(r'{ds_path}')\n"
        "def _patched(cache_dir=None):\n"
        "    return _ds\n"
        "_ds_mod.load_voicebank_demand = _patched\n"
    )

    # 4. Spawn. cwd=tmp_path so sitecustomize.py auto-imports; PYTHONPATH finds the repo.
    env = dict(os.environ)
    env["HF_DATASETS_CACHE"] = str(tmp_path / "hf_cache")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    result = subprocess.run(
        [sys.executable, "-m", "basenet.train",
         "--config", str(config_path),
         "--checkpoint_path", str(cp_dir),
         "--training_epochs", "1",
         "--stdout_interval", "1",
         "--checkpoint_interval", "2",
         "--validation_interval", "2",
         "--best_checkpoint_start_epoch", "0"],
        capture_output=True, text=True, check=False,
        cwd=str(tmp_path), env=env, timeout=300,
    )

    assert result.returncode == 0, (
        f"basenet.train exited {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # Orchestration evidence: params printed, a training step ran, validation ran.
    assert "Total Parameters:" in result.stdout, result.stdout
    assert "PESQ Score:" in result.stdout, result.stdout
    # A rolling checkpoint was written (checkpoint_interval=2 -> g_00000002).
    ckpts = list(cp_dir.glob("g_0*"))
    assert ckpts, f"no rolling checkpoint written; stdout:\n{result.stdout}"
    # Validation scalar landed in the TensorBoard event file.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(cp_dir / "logs"))
    ea.Reload()
    assert "Validation/PESQ Score" in set(ea.Tags()["scalars"])
