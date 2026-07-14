"""End-to-end subprocess smoke for the LiSenNet and ConvFSENet trainers.

Both were switched to the MP-SENet loss (common/losses.py), which also renamed
their config `loss` keys — LiSenNet's `mag`/`adv` became `magnitude`/`metric`,
and ConvFSENet gained a `loss` block it never had. `loss_weights()` rejects
unknown keys, so a config left un-migrated fails at startup rather than training
silently at the default weights. These tests spawn the real trainers against the
real shipped configs (with only the model size and STFT shrunk for speed) so that
failure mode is caught here rather than an hour into a retraining run.

Dataset injection follows the repo's Route A (see tests/test_basenet_train_e2e.py):
a synthetic DatasetDict is saved to disk and a sitecustomize.py in the subprocess
cwd monkeypatches common.dataset.load_voicebank_demand at interpreter startup,
since monkeypatch.setattr does not cross the subprocess boundary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _voiced(samples, sr, f0, seed):
    """A voiced-ish utterance (two sinusoids + light noise) so PESQ is scorable."""
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / sr
    sig = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * 2.7 * f0 * t)
    return (sig + 0.05 * rng.standard_normal(samples)).astype(np.float32)


def _write_synthetic_dataset(tmp_path, segment):
    from datasets import Dataset, DatasetDict

    def _split(prefix, n, seed0):
        audios = [_voiced(segment, 16000, 180 + 40 * i, seed0 + i) for i in range(n)]
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

    (tmp_path / "sitecustomize.py").write_text(
        "from datasets import load_from_disk\n"
        "import common.dataset as _ds_mod\n"
        f"_ds = load_from_disk(r'{ds_path}')\n"
        "def _patched(cache_dir=None):\n"
        "    return _ds\n"
        "_ds_mod.load_voicebank_demand = _patched\n"
    )


def _run_trainer(module, config_path, cp_dir, tmp_path):
    env = dict(os.environ)
    env["HF_DATASETS_CACHE"] = str(tmp_path / "hf_cache")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # The box's GPU is usually busy with a real run; these are CPU-sized anyway.
    env["CUDA_VISIBLE_DEVICES"] = ""

    return subprocess.run(
        [sys.executable, "-m", module,
         "--config", str(config_path),
         "--checkpoint_path", str(cp_dir),
         "--training_epochs", "1",
         "--stdout_interval", "1",
         "--checkpoint_interval", "2",
         "--validation_interval", "2",
         "--best_checkpoint_start_epoch", "0"],
        capture_output=True, text=True, check=False,
        cwd=str(tmp_path), env=env, timeout=600,
    )


def _shrink(shipped: Path, tmp_path, **overrides):
    """The real shipped config — crucially including its `loss` block — with only
    the model size / STFT / batch shrunk so it runs on CPU in seconds."""
    h = json.loads(shipped.read_text())
    h.update(overrides)
    out = tmp_path / f"config_{shipped.stem}.json"
    out.write_text(json.dumps(h))
    return out


@pytest.mark.parametrize("module,shipped,overrides", [
    (
        # LiSenNet keeps its real STFT: its encoder downsamples frequency, so a
        # miniature n_fft leaves fewer bins than the kernels need. At ~37k params
        # the full-size STFT is cheap enough on CPU anyway.
        "lisennet.train", "configs/lisennet_conv_hardened_nc24.json",
        dict(batch_size=2, segment_size=16384, n_blocks=1, num_workers=0),
    ),
    (
        "convfsenet.train", "configs/convfsenet_win.json",
        dict(batch_size=2, segment_size=8192, n_fft=128, hop_size=64, win_size=128,
             win_length=128, n_features=65, n_channels_res=16, n_channels_conv=32,
             n_blocks=2, n_stacks=1, num_workers=0),
    ),
])
def test_trainer_runs_with_shipped_loss_block(module, shipped, overrides, tmp_path):
    cp_dir = tmp_path / f"cp_{module.split('.')[0]}"
    cp_dir.mkdir()
    _write_synthetic_dataset(tmp_path, overrides["segment_size"])

    config_path = _shrink(REPO_ROOT / shipped, tmp_path, **overrides)

    result = _run_trainer(module, config_path, cp_dir, tmp_path)

    assert result.returncode == 0, (
        f"{module} exited {result.returncode} with the shipped loss block.\n"
        f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
    )
    assert "PESQ Score:" in result.stdout, result.stdout[-4000:]
    assert list(cp_dir.glob("g_0*")), f"no checkpoint written:\n{result.stdout[-4000:]}"

    # Every MP-SENet term this model trains on must reach TensorBoard — if one is
    # missing from the summary, it is missing from the objective.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(cp_dir / "logs"))
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    for term in ("magnitude", "complex", "consistency", "time", "metric"):
        assert f"Training/{term}" in tags, f"{term} term never logged; tags={sorted(tags)}"
    # Mask-only models: no phase decoder, so no phase term.
    assert "Training/phase" not in tags


def test_stale_loss_keys_are_rejected(tmp_path):
    """A pre-port LiSenNet config (`mag`/`adv`) must fail loudly, not silently
    fall back to the default weights."""
    from common.env import AttrDict
    from common.losses import loss_weights

    stale = AttrDict({"loss": {"complex": 0.1, "mag": 0.9, "adv": 0.05}})
    with pytest.raises(ValueError, match="unknown loss weight"):
        loss_weights(stale)
