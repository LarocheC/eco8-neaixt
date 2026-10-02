"""The optional differentiable-PESQ loss and the fine-tune runner (finetune_pesq.py).

Runs on synthetic audio with the CPU (torch) backend of triton-pesq; skipped when
triton-pesq isn't installed (``uv sync --group pesq-loss``). The Triton backend
itself needs a GPU and is covered by triton-pesq's own tests.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytest.importorskip("torch_pesq")

from common.discriminator import MetricDiscriminator
from common.env import AttrDict
from common.pesq_loss import build_pesq_loss, pesq_loss, pesq_loss_config
from finetune_pesq import finetune_config, load_config, paired_delta, render_markdown
from lisennet.model import build_lisennet
from lisennet.train import _loss_weights, gan_step


def _pair(batch=2, samples=32000, silent_tail_from=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(samples) / 16000
    clean = (0.3 * torch.sin(2 * torch.pi * 300 * t) + 0.1 * torch.sin(2 * torch.pi * 1200 * t)).repeat(batch, 1)
    noisy = clean + 0.05 * torch.randn(clean.shape, generator=g)
    if silent_tail_from is not None:  # a zero-padded segment, as the Dataset makes for short utterances
        clean[-1, silent_tail_from:] = 0
        noisy[-1, silent_tail_from:] = 0
    return clean, noisy


@pytest.fixture(scope="module")
def pesq_fn():
    return build_pesq_loss(16000, "cpu", "torch")


def test_config_parsing():
    assert pesq_loss_config(AttrDict({})) == (0.0, "auto")
    assert pesq_loss_config(AttrDict({"pesq_loss": {"weight": 0.3, "backend": "torch"}})) == (0.3, "torch")
    with pytest.raises(ValueError):
        pesq_loss_config(AttrDict({"pesq_loss": {"backend": "cuda"}}))
    with pytest.raises(ValueError):
        pesq_loss_config(AttrDict({"pesq_loss": {"weight": -1}}))


def test_triton_backend_refuses_cpu():
    with pytest.raises(ValueError):
        build_pesq_loss(16000, "cpu", "triton")


def test_loss_ranks_cleaner_estimates_lower(pesq_fn):
    clean, noisy = _pair()
    assert pesq_loss(pesq_fn, clean, clean) < pesq_loss(pesq_fn, clean, noisy)


def test_gradient_is_finite_on_digital_silence(pesq_fn):
    clean, noisy = _pair(silent_tail_from=20000)
    est = noisy.clone().requires_grad_(True)
    pesq_loss(pesq_fn, clean, est).backward()
    assert torch.isfinite(est.grad).all()
    est.grad = None
    pesq_loss(pesq_fn, clean, est, dither=0.0).backward()   # the failure the dither prevents
    assert not torch.isfinite(est.grad).all()


def test_accepts_channel_dim_and_trims_lengths(pesq_fn):
    clean, noisy = _pair()
    a = pesq_loss(pesq_fn, clean, noisy, dither=0.0)
    b = pesq_loss(pesq_fn, clean[:, None, :], noisy[:, None, :-100], dither=0.0)
    assert torch.isfinite(b) and abs(float(a) - float(b)) < 0.5


def test_lisennet_gan_step_with_pesq_term(pesq_fn):
    torch.manual_seed(0)
    h = AttrDict({"loss": {"complex": 0.1, "mag": 0.9, "adv": 0.05}})
    model, disc = build_lisennet(), MetricDiscriminator(dim=16)
    opt_g = torch.optim.AdamW(model.parameters(), 1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), 1e-4)
    clean, noisy = _pair(silent_tail_from=24000)
    before = [p.detach().clone() for p in model.parameters()]
    out = gan_step(model, disc, opt_g, opt_d, noisy, clean, h, _loss_weights(h), 5.0,
                   pesq_fn=pesq_fn, pesq_weight=0.2)
    assert "pesq" in out and np.isfinite(out["pesq"]) and np.isfinite(out["loss"])
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))


def test_lisennet_gan_step_unchanged_without_pesq():
    torch.manual_seed(0)
    h = AttrDict({"loss": {"complex": 0.1, "mag": 0.9, "adv": 0.05}})
    model, disc = build_lisennet(), MetricDiscriminator(dim=16)
    opt_g = torch.optim.AdamW(model.parameters(), 1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), 1e-4)
    clean, noisy = _pair()
    out = gan_step(model, disc, opt_g, opt_d, noisy, clean, h, _loss_weights(h), 5.0)
    assert "pesq" not in out


def test_finetune_config_keeps_architecture_and_sets_recipe(tmp_path):
    base = {"num_channels": 24, "bottleneck": "conv", "learning_rate": 5e-4, "loss": {"mag": 0.9}}
    (tmp_path / "config.json").write_text(json.dumps(base))
    (tmp_path / "g_best").write_bytes(b"")
    cfg = finetune_config(load_config(tmp_path / "g_best"), base="x/g_best", pesq_weight=0.2,
                          backend="auto", lr=1e-4, epochs=5)
    assert cfg["num_channels"] == 24 and cfg["bottleneck"] == "conv" and cfg["loss"] == {"mag": 0.9}
    assert cfg["learning_rate"] == 1e-4
    assert cfg["pesq_loss"] == {"weight": 0.2, "backend": "auto"}
    assert cfg["_finetune"]["arm"] == "pesq"
    control = finetune_config(base, base="x", pesq_weight=0.0, backend="auto", lr=1e-4, epochs=5)
    assert control["_finetune"]["arm"] == "control"


def test_paired_delta_and_report():
    a = np.array([3.0, 3.2, np.nan, 3.1])
    b = np.array([2.9, 3.0, 3.0, 3.0])
    mean, ci, n = paired_delta(a, b)
    assert n == 3 and abs(mean - (0.1 + 0.2 + 0.1) / 3) < 1e-9 and ci > 0
    result = {
        "family": "lisennet", "n_utterances": 4, "columns": ["pesq"],
        "checkpoints": {"control": {}, "pesq": {}},
        "means": {"control": {"pesq": 3.0}, "pesq": {"pesq": 3.1}},
        "deltas": {"pesq − control": {"pesq": {"mean": mean, "ci95": ci, "n": n}}},
    }
    md = render_markdown(result)
    assert "| pesq | 3.100 |" in md and "pesq − control" in md


def _convfsenet_cfg():
    return AttrDict({
        "n_fft": 512, "win_length": 512, "hop_size": 256, "win_size": 512, "n_features": 257,
        "n_channels_res": 32, "n_channels_conv": 64, "kernel_size": 3, "n_blocks": 2, "n_stacks": 1,
        "extractor_type": "mag", "compress_factor": None, "causal": True, "seed": 0,
    })


def test_convfsenet_gan_step_with_pesq_term(pesq_fn):
    from convfsenet.model import build_causal_model
    from convfsenet.train import _gan_step

    torch.manual_seed(0)
    h = _convfsenet_cfg()
    model, disc = build_causal_model(h), MetricDiscriminator()
    opt, opt_d = torch.optim.AdamW(model.parameters(), 1e-4), torch.optim.AdamW(disc.parameters(), 1e-4)
    clean, noisy = _pair(samples=16000, silent_tail_from=12000)
    out = _gan_step(model, disc, opt, opt_d, noisy[:, None], clean[:, None], h, 0.05, 0.3,
                    torch.device("cpu"), pesq_fn=pesq_fn, pesq_weight=0.2)
    assert "pesq" in out and np.isfinite(out["pesq"]) and np.isfinite(out["loss"])
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_train_command_wires_the_trainer(tmp_path, monkeypatch):
    """`finetune_pesq.py train` writes the merged config and calls the family's trainer
    with --init_from, the epoch count and best-checkpoint tracking from epoch 0."""
    import finetune_pesq
    import lisennet.train as real

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "config.json").write_text(json.dumps({"num_channels": 16, "learning_rate": 5e-4}))
    (base_dir / "g_best").write_bytes(b"")
    calls = {}

    class Trainer:
        build_parser = staticmethod(real.build_parser)

        @staticmethod
        def train(a, h):
            calls["a"], calls["h"] = a, h

    monkeypatch.setattr(finetune_pesq.importlib, "import_module", lambda name: Trainer)
    out = tmp_path / "arm"
    finetune_pesq.main(["train", "--model", "lisennet", "--base", str(base_dir / "g_best"),
                        "--out", str(out), "--pesq_weight", "0.3", "--epochs", "4", "--lr", "2e-4"])
    a, h = calls["a"], calls["h"]
    assert a.init_from == str(base_dir / "g_best") and a.training_epochs == 4
    assert a.best_checkpoint_start_epoch == 0 and a.checkpoint_path == str(out)
    assert h.learning_rate == 2e-4 and h.pesq_loss == {"weight": 0.3, "backend": "auto"}
    assert json.loads((out / "config.json").read_text())["_finetune"]["arm"] == "pesq"
    with pytest.raises(SystemExit):   # a different arm in the same directory is refused
        finetune_pesq.main(["train", "--model", "lisennet", "--base", str(base_dir / "g_best"),
                            "--out", str(out), "--pesq_weight", "0"])
