"""Training loop for LiSenNet (arXiv:2409.13285, CMGAN-based) in this framework.

Ports the upstream PyTorch-Lightning recipe to the plain-PyTorch convention used
by the other models here (CLI, resume, g_<steps>/g_best checkpoints), wired to
the shared VoiceBank-DEMAND dataset. LiSenNet is end-to-end (waveform in/out;
STFT + Griffin-Lim phase live inside the model), so a batch is just
(clean, noisy) waveforms.

Losses (upstream model/model.py):
  generator   = w_com * (MSE(re) + MSE(im)) + w_mag * MSE(mag) + w_adv * adv
  discriminator (MetricGAN) = MSE(D(tgt,tgt), 1) + MSE(D(tgt,est), PESQ(tgt,est))
The CMGAN MetricDiscriminator is reused from common/discriminator.py (identical
to LiSenNet's). Phase is not learned (Griffin-Lim), so there is no phase loss.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lisennet.model import build_lisennet
from common.dataset import Dataset, data_generator, load_voicebank_demand, seed_worker
from common.env import AttrDict, build_env
from common.discriminator import MetricDiscriminator, batch_pesq
from common.metrics import pesq_score
from common.utils import load_checkpoint, save_checkpoint, scan_checkpoint

warnings.simplefilter(action="ignore", category=FutureWarning)

_DEFAULT_LOSS_WEIGHTS = {"complex": 0.1, "mag": 0.9, "adv": 0.05}


def _to_dev(t, device):
    return t.to(device, non_blocking=True)


def _loss_weights(h):
    cfg = h.get("loss", {}) or {}
    return {k: float(cfg.get(k, v)) for k, v in _DEFAULT_LOSS_WEIGHTS.items()}


def generator_loss(results, discriminator, weights):
    """CMGAN generator loss: complex + magnitude + adversarial (metric)."""
    est_spec, tgt_spec = results["est_spec"], results["tgt_spec"]
    loss_complex = F.mse_loss(est_spec.real, tgt_spec.real) + F.mse_loss(est_spec.imag, tgt_spec.imag)
    loss_mag = F.mse_loss(results["est_mag"], results["tgt_mag"])
    metric = discriminator(results["tgt_mag"], results["est_mag"]).flatten()
    loss_adv = F.mse_loss(metric, torch.ones_like(metric))
    loss = (weights["complex"] * loss_complex
            + weights["mag"] * loss_mag
            + weights["adv"] * loss_adv)
    parts = {"complex": float(loss_complex.item()), "mag": float(loss_mag.item()),
             "adv": float(loss_adv.item())}
    return loss, parts


def discriminator_loss(results, discriminator, device):
    """MetricGAN discriminator loss: real -> 1, fake -> normalised PESQ(tgt, est)."""
    tgt_mag = results["tgt_mag"]
    est_mag = results["est_mag"].detach()
    tgt_wav = results["tgt"]
    est_wav = results["est"].detach()

    pesq_scaled = batch_pesq(
        list(tgt_wav.cpu().numpy()),
        list(est_wav.cpu().numpy()),
    )
    metric_real = discriminator(tgt_mag, tgt_mag).flatten()
    loss_real = F.mse_loss(metric_real, torch.ones_like(metric_real))
    if pesq_scaled is not None:
        metric_fake = discriminator(tgt_mag, est_mag).flatten()
        loss_fake = F.mse_loss(metric_fake, pesq_scaled.to(device))
    else:
        # PESQ failed on some utterance (silence/too-short) — skip the fake branch.
        loss_fake = torch.zeros((), device=device)
    return loss_real + loss_fake


def gan_step(model, discriminator, optim_g, optim_d, noisy, clean, h, weights, clip,
             teacher=None, distill_weight=0.0):
    """One CMGAN step: discriminator then generator (one shared forward).

    With ``teacher`` set, adds ``distill_weight * MSE(est_mag, teacher_est_mag)``
    to the generator loss — mask-level knowledge distillation from a stronger
    (e.g. longer-receptive-field) model into this deployment architecture.
    """
    results = model(noisy, clean)
    device = clean.device

    # ----- discriminator -----
    optim_d.zero_grad()
    d_loss = discriminator_loss(results, discriminator, device)
    d_loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip)
    optim_d.step()

    # ----- generator -----
    optim_g.zero_grad()
    g_loss, parts = generator_loss(results, discriminator, weights)
    if teacher is not None:
        with torch.no_grad():
            teacher_mag = teacher(noisy, clean)["est_mag"]
        loss_distill = F.mse_loss(results["est_mag"], teacher_mag)
        g_loss = g_loss + distill_weight * loss_distill
        parts["distill"] = float(loss_distill.item())
    g_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optim_g.step()

    return {"loss": float(g_loss.item()), "d_loss": float(d_loss.item()), **parts}


def train(a, h):
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = build_lisennet(h).to(device)
    discriminator = MetricDiscriminator(dim=int(h.get("num_channels", 16))).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Total Parameters (generator): {n_params / 1e6:.4f}M")
    os.makedirs(a.checkpoint_path, exist_ok=True)
    os.makedirs(os.path.join(a.checkpoint_path, "logs"), exist_ok=True)
    print(f"checkpoints directory : {a.checkpoint_path}")

    weights = _loss_weights(h)
    clip = float(h.get("gradient_clip", 5.0))

    # ----- distillation teacher (optional, frozen) ---------------------------
    teacher = None
    if a.distill_from:
        with open(os.path.join(os.path.dirname(a.distill_from), "config.json")) as f:
            teacher_h = AttrDict(json.load(f))
        teacher = build_lisennet(teacher_h).to(device)
        teacher.load_state_dict(load_checkpoint(a.distill_from, device)["generator"])
        teacher.eval()
        teacher.requires_grad_(False)
        n_teacher = sum(p.numel() for p in teacher.parameters())
        print(f"Distilling from {a.distill_from} ({n_teacher / 1e6:.4f}M params, "
              f"weight={a.distill_weight}).")

    # ----- resume ------------------------------------------------------------
    cp_g = scan_checkpoint(a.checkpoint_path, "g_")
    steps, last_epoch, state = 0, -1, None
    if cp_g is not None:
        state = load_checkpoint(cp_g, device)
        model.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        steps = int(state.get("steps", 0)) + 1
        last_epoch = int(state.get("epoch", -1))
    elif a.init_from:
        init_state = load_checkpoint(a.init_from, device)
        model.load_state_dict(init_state["generator"])
        print(f"Warm-started weights from {a.init_from}.")

    optim_g = torch.optim.AdamW(model.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(discriminator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    if state is not None:
        if "optim_g" in state:
            optim_g.load_state_dict(state["optim_g"])
        if "optim_d" in state:
            optim_d.load_state_dict(state["optim_d"])

    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    # ----- datasets ----------------------------------------------------------
    hf = load_voicebank_demand(cache_dir=a.hf_cache_dir)
    trainset = Dataset(hf["train"], h.segment_size, h.sampling_rate, split=True, shuffle=True, seed=h.seed)
    train_loader = DataLoader(
        trainset, num_workers=h.num_workers, shuffle=False, batch_size=h.batch_size,
        pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=data_generator(h.seed))
    validset = Dataset(hf["test"], h.segment_size, h.sampling_rate, split=False, shuffle=False, seed=h.seed)
    validation_loader = DataLoader(
        validset, num_workers=1, shuffle=False, batch_size=1, pin_memory=True,
        drop_last=True, worker_init_fn=seed_worker, generator=data_generator(h.seed))

    sw = SummaryWriter(os.path.join(a.checkpoint_path, "logs"))
    best_pesq = float(state["best_pesq"]) if state is not None and "best_pesq" in state else 0.0

    for epoch in range(max(0, last_epoch), a.training_epochs):
        model.train()
        discriminator.train()
        start = time.time()
        print(f"Epoch: {epoch + 1}")
        for clean_audio, noisy_audio in train_loader:
            start_b = time.time()
            clean_audio = _to_dev(clean_audio, device)        # (B, samples)
            noisy_audio = _to_dev(noisy_audio, device)
            metrics = gan_step(model, discriminator, optim_g, optim_d,
                               noisy_audio, clean_audio, h, weights, clip,
                               teacher=teacher, distill_weight=a.distill_weight)

            if steps % a.stdout_interval == 0:
                print(f"Steps : {steps:d}, G: {metrics['loss']:.4f}, D: {metrics['d_loss']:.4f}, "
                      f"Mag: {metrics['mag']:.4f}, Com: {metrics['complex']:.4f}, "
                      f"s/b : {time.time() - start_b:.3f}")
            if steps % a.summary_interval == 0:
                sw.add_scalar("Training/Generator Loss", metrics["loss"], steps)
                sw.add_scalar("Training/Discriminator Loss", metrics["d_loss"], steps)
                for k in ("mag", "complex", "adv"):
                    sw.add_scalar(f"Training/{k}", metrics[k], steps)

            if steps % a.checkpoint_interval == 0 and steps != 0:
                save_checkpoint(f"{a.checkpoint_path}/g_{steps:08d}", {
                    "generator": model.state_dict(), "discriminator": discriminator.state_dict(),
                    "optim_g": optim_g.state_dict(), "optim_d": optim_d.state_dict(),
                    "steps": steps, "epoch": epoch, "best_pesq": best_pesq,
                })

            if steps % a.validation_interval == 0 and steps != 0:
                val_pesq = _validate(model, validation_loader, device, h)
                print(f"Steps : {steps:d}, PESQ Score: {val_pesq:.3f}")
                sw.add_scalar("Validation/PESQ Score", val_pesq, steps)
                if epoch >= a.best_checkpoint_start_epoch and val_pesq > best_pesq:
                    best_pesq = val_pesq
                    save_checkpoint(f"{a.checkpoint_path}/g_best", {"generator": model.state_dict()})
                model.train()
                discriminator.train()

            steps += 1

        sched_g.step()
        sched_d.step()
        print(f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n")


@torch.no_grad()
def _validate(model, validation_loader, device, h) -> float:
    """Enhance the VBD test split and return mean wideband PESQ vs clean."""
    model.eval()
    audios_r, audios_g = [], []
    for clean_audio, noisy_audio in validation_loader:
        clean_audio = _to_dev(clean_audio, device)            # (1, samples)
        noisy_audio = _to_dev(noisy_audio, device)
        est = model(noisy_audio)["est"]                       # (1, samples)
        n = min(est.shape[-1], clean_audio.shape[-1])
        audios_r.append(clean_audio[..., :n])
        audios_g.append(est[..., :n])
    return float(pesq_score(audios_r, audios_g, h))


def main():
    print("Initializing LiSenNet Training Process..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lisennet.json")
    parser.add_argument("--checkpoint_path", default="cp_lisennet")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--training_epochs", default=100, type=int)
    parser.add_argument("--stdout_interval", default=20, type=int)
    parser.add_argument("--checkpoint_interval", default=2000, type=int)
    parser.add_argument("--summary_interval", default=50, type=int)
    parser.add_argument("--validation_interval", default=1000, type=int)
    parser.add_argument("--best_checkpoint_start_epoch", default=1, type=int)
    parser.add_argument("--init_from", default=None)
    parser.add_argument("--distill_from", default=None,
                        help="Teacher g_best for mask-level knowledge distillation "
                             "(its config.json is read from the sibling directory).")
    parser.add_argument("--distill_weight", default=0.45, type=float,
                        help="Weight of the MSE(est_mag, teacher_est_mag) distillation term.")
    a = parser.parse_args()

    with open(a.config) as f:
        h = AttrDict(json.load(f))
    build_env(a.config, "config.json", a.checkpoint_path)
    train(a, h)


if __name__ == "__main__":
    main()
