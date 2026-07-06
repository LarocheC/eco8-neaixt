"""Training loop for the causal BASENet (arXiv:2606.12662).

Follows MP-SENet's training recipe (the paper says BASENet reuses it): the
generator predicts a magnitude mask + enhanced phase from the noisy STFT, and is
optimised with magnitude / anti-wrapping-phase / complex / time-domain losses
plus an optional MetricGAN term. The MetricDiscriminator (common/discriminator)
is trained to predict normalised PESQ and the generator is pulled toward
"perfect PESQ" — training-only, never on the inference path.

CLI surface, resume order (rolling checkpoint > --init_from), checkpoint dict
layout (``g_<steps>`` rolling + ``g_best`` PESQ-tracked) and scheduler handling
mirror ``convfsenet/train.py`` so the existing sweep / scan_checkpoint plumbing
works unchanged.

Validation runs the offline causal forward on the VBD test split and reports
mean PESQ on the reconstructed waveform; the PESQ-best model is saved to g_best.
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

from basenet.model import build_basenet
from basenet.losses import spectral_losses
from common.dataset import (
    Dataset, data_generator, load_voicebank_demand,
    mag_pha_stft, mag_pha_istft, seed_worker,
)
from common.env import AttrDict, build_env
from common.discriminator import MetricDiscriminator, batch_pesq
from common.metrics import pesq_score
from common.utils import load_checkpoint, save_checkpoint, scan_checkpoint

warnings.simplefilter(action="ignore", category=FutureWarning)

_DEFAULT_LOSS_WEIGHTS = {"magnitude": 0.9, "phase": 0.3, "complex": 0.1, "time": 0.2,
                         "consistency": 0.1}


def _to_dev(t, device):
    return t.to(device, non_blocking=True)


def _loss_weights(h):
    cfg = h.get("loss", {}) or {}
    return {k: float(cfg.get(k, v)) for k, v in _DEFAULT_LOSS_WEIGHTS.items()}


def enhance(model, noisy_audio, h):
    """Run BASENet end-to-end: noisy waveform -> enhanced waveform.

    Accepts ``(B, samples)`` or ``(B, 1, samples)`` and returns ``(B, samples)``.
    Shared by validation and any offline inference so the STFT round-trip stays
    in one place.
    """
    cf = float(h.get("compress_factor", 0.3))
    y = noisy_audio.squeeze(1) if noisy_audio.dim() == 3 else noisy_audio
    mag, pha, _ = mag_pha_stft(y, h.n_fft, h.hop_size, h.win_size, cf)
    mag_g, pha_g, _ = model(mag, pha)
    return mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, cf)


def generator_loss(model, noisy_audio, clean_audio, h, weights=None):
    """Forward BASENet and compute the GAN-free generator loss.

    Returns ``(loss, parts)`` where ``loss`` is the weighted magnitude + phase +
    complex + time objective (a tensor with grad) and ``parts`` carries the
    scalar sub-losses plus the compressed clean/predicted magnitudes and the
    reconstructed audio that the GAN step and logging need.
    """
    weights = weights or _loss_weights(h)
    cf = float(h.get("compress_factor", 0.3))
    n_fft, hop, win = h.n_fft, h.hop_size, h.win_size

    noisy = noisy_audio.squeeze(1)
    clean = clean_audio.squeeze(1)
    noisy_mag, noisy_pha, _ = mag_pha_stft(noisy, n_fft, hop, win, cf)
    clean_mag, clean_pha, clean_com = mag_pha_stft(clean, n_fft, hop, win, cf)

    mag_g, pha_g, com_g = model(noisy_mag, noisy_pha)
    sl = spectral_losses(mag_g, pha_g, com_g, clean_mag, clean_pha, clean_com)

    audio_g = mag_pha_istft(mag_g, pha_g, n_fft, hop, win, cf)
    n = min(audio_g.shape[-1], clean.shape[-1])
    loss_time = F.l1_loss(audio_g[..., :n], clean[..., :n])

    # MP-SENet's STFT-consistency loss: the (mag, phase) prediction must survive
    # the iSTFT -> STFT round trip. mag_g_hat (the consistent magnitude) is also
    # what MP-SENet shows the metric discriminator, not the raw network output.
    mag_g_hat, _, com_g_hat = mag_pha_stft(audio_g, n_fft, hop, win, cf)
    loss_consistency = F.mse_loss(com_g, com_g_hat) * 2.0

    loss = (weights["magnitude"] * sl["magnitude"]
            + weights["phase"] * sl["phase"]
            + weights["complex"] * sl["complex"]
            + weights["time"] * loss_time
            + weights["consistency"] * loss_consistency)

    parts = {
        "magnitude": float(sl["magnitude"].item()),
        "phase": float(sl["phase"].item()),
        "complex": float(sl["complex"].item()),
        "time": float(loss_time.item()),
        "consistency": float(loss_consistency.item()),
        "clean_mag": clean_mag,
        "pred_mag": mag_g_hat,
        "audio_g": audio_g,
        "clean_audio": clean,
    }
    return loss, parts


def gan_step(model, discriminator, optim, optim_d, noisy_audio, clean_audio,
             h, metric_lambda, device, weights=None,
             accum=1, is_first=True, do_step=True):
    """One MetricGAN micro-step (generator + discriminator), accumulation-aware.

    The discriminator learns to map (clean_mag, x_mag) -> normalised PESQ: 1 for
    clean-vs-clean, ``batch_pesq`` for pred-vs-clean. The generator adds a metric
    term pulling ``disc(clean_mag, pred_mag)`` toward 1. The forward graph is
    built once in ``generator_loss``; the discriminator branch uses a detached
    magnitude so its backward doesn't free the generator graph (same pattern as
    convfsenet/train.py).

    Gradient accumulation: losses are scaled by ``1/accum`` and grads accumulate
    over ``accum`` micro-batches; ``zero_grad`` fires on ``is_first`` and the
    optimizers step on ``do_step``. Because every module + the discriminator is
    per-sample (no batch statistics), accum micro-batches of size B/accum are
    numerically equal to one batch of size B. ``accum=1`` reproduces the original
    per-batch step exactly.
    """
    weights = weights or _loss_weights(h)
    base_loss, parts = generator_loss(model, noisy_audio, clean_audio, h, weights)
    clean_mag, pred_mag = parts["clean_mag"], parts["pred_mag"]
    audio_g, clean = parts["audio_g"], parts["clean_audio"]
    one_labels = torch.ones(clean_mag.shape[0], device=device)

    # ----- discriminator step ----------------------------------------------
    n = min(audio_g.shape[-1], clean.shape[-1])
    pesq_target = batch_pesq(
        list(clean[..., :n].detach().cpu().numpy()),
        list(audio_g[..., :n].detach().cpu().numpy()),
    )
    if is_first:
        optim_d.zero_grad()
    metric_r = discriminator(clean_mag, clean_mag)
    metric_g = discriminator(clean_mag, pred_mag.detach())
    loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
    if pesq_target is not None:
        loss_disc_g = F.mse_loss(pesq_target.to(device), metric_g.flatten())
    else:
        # batch_pesq returns None when any utterance's PESQ failed (silence,
        # too-short, NaN) — skip the fake branch for this batch.
        loss_disc_g = torch.zeros((), device=device)
    loss_disc = loss_disc_r + loss_disc_g
    (loss_disc / accum).backward()
    if do_step:
        optim_d.step()

    # ----- generator step ---------------------------------------------------
    if is_first:
        optim.zero_grad()
    metric_g = discriminator(clean_mag, pred_mag)              # keeps generator grad
    loss_metric = F.mse_loss(metric_g.flatten(), one_labels)
    loss_gen = base_loss + metric_lambda * loss_metric
    (loss_gen / accum).backward()
    if do_step:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optim.step()

    return {
        "loss": float(loss_gen.item()),
        "base_loss": float(base_loss.item()),
        "loss_metric": float(loss_metric.item()),
        "loss_disc": float(loss_disc.item()),
        "magnitude": parts["magnitude"],
        "phase": parts["phase"],
        "complex": parts["complex"],
        "time": parts["time"],
        "consistency": parts["consistency"],
    }


def train(a, h):
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = build_basenet(h).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Total Parameters: {n_params / 1e6:.3f}M")
    os.makedirs(a.checkpoint_path, exist_ok=True)
    os.makedirs(os.path.join(a.checkpoint_path, "logs"), exist_ok=True)
    print(f"checkpoints directory : {a.checkpoint_path}")

    weights = _loss_weights(h)

    # ----- resume from latest rolling checkpoint if present ------------------
    cp_g = scan_checkpoint(a.checkpoint_path, "g_")
    steps = 0
    last_epoch = -1
    state = None
    if cp_g is not None:
        state = load_checkpoint(cp_g, device)
        model.load_state_dict(state["generator"])
        steps = int(state.get("steps", 0)) + 1
        last_epoch = int(state.get("epoch", -1))
    elif a.init_from:
        init_state = load_checkpoint(a.init_from, device)
        model.load_state_dict(init_state["generator"])
        print(f"Warm-started weights from {a.init_from} (fresh optim + step=0).")

    optim = torch.optim.AdamW(
        model.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    if state is not None and "optim" in state:
        optim.load_state_dict(state["optim"])

    sched_kind = h.get("scheduler", "exponential")
    if sched_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=a.training_epochs,
            eta_min=float(h.get("lr_min", 1e-5)), last_epoch=last_epoch,
        )
    elif sched_kind == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optim, gamma=h.lr_decay, last_epoch=last_epoch
        )
    else:
        raise ValueError(f"unknown scheduler {sched_kind!r}; use 'exponential' or 'cosine'")

    # ----- optional metric-GAN discriminator --------------------------------
    gan_cfg = h.get("gan", {}) or {}
    use_gan = bool(gan_cfg.get("enabled", False))
    metric_lambda = float(gan_cfg.get("metric_loss_lambda", 0.05))
    discriminator = optim_d = scheduler_d = None
    if use_gan:
        discriminator = MetricDiscriminator().to(device)
        if state is not None and "discriminator" in state:
            discriminator.load_state_dict(state["discriminator"])
        optim_d = torch.optim.AdamW(
            discriminator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
        )
        if state is not None and "optim_d" in state:
            optim_d.load_state_dict(state["optim_d"])
        if sched_kind == "cosine":
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_d, T_max=a.training_epochs,
                eta_min=float(h.get("lr_min", 1e-5)), last_epoch=last_epoch,
            )
        else:
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                optim_d, gamma=h.lr_decay, last_epoch=last_epoch
            )
        print(f"metric-GAN enabled: lambda={metric_lambda}")

    # Restore the LR schedule exactly (see convfsenet/train.py for the rationale).
    if state is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
        for grp, lr in zip(optim.param_groups, scheduler.get_last_lr()):
            grp["lr"] = lr
        if use_gan and scheduler_d is not None and "scheduler_d" in state:
            scheduler_d.load_state_dict(state["scheduler_d"])
            for grp, lr in zip(optim_d.param_groups, scheduler_d.get_last_lr()):
                grp["lr"] = lr

    # ----- datasets ----------------------------------------------------------
    hf = load_voicebank_demand(cache_dir=a.hf_cache_dir)
    trainset = Dataset(
        hf["train"], h.segment_size, h.sampling_rate,
        split=True, shuffle=True, seed=h.seed,
    )
    train_loader = DataLoader(
        trainset, num_workers=h.num_workers, shuffle=False,
        batch_size=h.batch_size, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=data_generator(h.seed),
    )
    validset = Dataset(
        hf["test"], h.segment_size, h.sampling_rate,
        split=False, shuffle=False, seed=h.seed,
    )
    validation_loader = DataLoader(
        validset, num_workers=1, shuffle=False,
        batch_size=1, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=data_generator(h.seed),
    )

    sw = SummaryWriter(os.path.join(a.checkpoint_path, "logs"))
    best_pesq = float(state["best_pesq"]) if state is not None and "best_pesq" in state else 0.0

    accum = max(1, int(getattr(a, "accum_steps", 1)))
    if accum > 1:
        print(f"gradient accumulation: {accum} micro-batches "
              f"(effective batch = {h.batch_size * accum})")

    # ----- training loop -----------------------------------------------------
    for epoch in range(max(0, last_epoch), a.training_epochs):
        model.train()
        micro = 0
        start = time.time()
        print(f"Epoch: {epoch + 1}")

        for i, batch in enumerate(train_loader):
            start_b = time.time()
            clean_audio, noisy_audio = batch
            clean_audio = _to_dev(clean_audio, device).unsqueeze(1)   # (B, 1, samples)
            noisy_audio = _to_dev(noisy_audio, device).unsqueeze(1)

            is_first = micro == 0
            micro += 1
            do_step = micro >= accum

            if use_gan:
                metrics = gan_step(
                    model, discriminator, optim, optim_d,
                    noisy_audio, clean_audio, h, metric_lambda, device, weights,
                    accum=accum, is_first=is_first, do_step=do_step,
                )
            else:
                if is_first:
                    optim.zero_grad()
                loss, parts = generator_loss(model, noisy_audio, clean_audio, h, weights)
                (loss / accum).backward()
                if do_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optim.step()
                metrics = {"loss": float(loss.item()), **{
                    k: parts[k] for k in ("magnitude", "phase", "complex", "time", "consistency")
                }}

            # Accumulate until the effective batch is complete; only then
            # log / checkpoint / validate / advance the optimizer-step counter.
            if not do_step:
                continue
            micro = 0

            if steps % a.stdout_interval == 0:
                extra = ""
                if use_gan:
                    extra = (f", Metric: {metrics['loss_metric']:.4f}"
                             f", Disc: {metrics['loss_disc']:.4f}")
                print(
                    f"Steps : {steps:d}, Loss: {metrics['loss']:.4f}{extra}, "
                    f"Mag: {metrics['magnitude']:.4f}, Pha: {metrics['phase']:.4f}, "
                    f"s/b : {time.time() - start_b:.3f}"
                )
            if steps % a.summary_interval == 0:
                sw.add_scalar("Training/Loss", metrics["loss"], steps)
                for k in ("magnitude", "phase", "complex", "time", "consistency"):
                    sw.add_scalar(f"Training/{k}", metrics[k], steps)
                if use_gan:
                    sw.add_scalar("Training/Base Loss", metrics["base_loss"], steps)
                    sw.add_scalar("Training/Metric Loss", metrics["loss_metric"], steps)
                    sw.add_scalar("Training/Discriminator Loss", metrics["loss_disc"], steps)

            # rolling checkpoint
            if steps % a.checkpoint_interval == 0 and steps != 0:
                cp_path = f"{a.checkpoint_path}/g_{steps:08d}"
                ckpt = {
                    "generator": model.state_dict(),
                    "optim": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "steps": steps,
                    "epoch": epoch,
                    "best_pesq": best_pesq,
                }
                if use_gan:
                    ckpt["discriminator"] = discriminator.state_dict()
                    ckpt["optim_d"] = optim_d.state_dict()
                    ckpt["scheduler_d"] = scheduler_d.state_dict()
                save_checkpoint(cp_path, ckpt)

            # validation + best
            if steps % a.validation_interval == 0 and steps != 0:
                val_pesq = _validate(model, validation_loader, device, h)
                print(f"Steps : {steps:d}, PESQ Score: {val_pesq:.3f}")
                sw.add_scalar("Validation/PESQ Score", val_pesq, steps)
                if epoch >= a.best_checkpoint_start_epoch and val_pesq > best_pesq:
                    best_pesq = val_pesq
                    save_checkpoint(
                        f"{a.checkpoint_path}/g_best",
                        {"generator": model.state_dict()},
                    )
                model.train()

            steps += 1

        scheduler.step()
        if use_gan:
            scheduler_d.step()
        print(f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n")


@torch.no_grad()
def _validate(model, validation_loader, device, h) -> float:
    """Reconstruct the VBD test split and return mean PESQ vs clean."""
    model.eval()
    audios_r, audios_g = [], []
    for batch in validation_loader:
        clean_audio, noisy_audio = batch
        clean_audio = _to_dev(clean_audio, device)                 # (1, samples)
        noisy_audio = _to_dev(noisy_audio, device)
        audio_g = enhance(model, noisy_audio, h)
        n = min(audio_g.shape[-1], clean_audio.shape[-1])
        audios_r.append(clean_audio[..., :n])
        audios_g.append(audio_g[..., :n])
    return float(pesq_score(audios_r, audios_g, h))


def main():
    print("Initializing BASENet Training Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/basenet_causal.json")
    parser.add_argument("--checkpoint_path", default="cp_basenet")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--training_epochs", default=100, type=int)
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--checkpoint_interval", default=2000, type=int)
    parser.add_argument("--summary_interval", default=50, type=int)
    parser.add_argument("--validation_interval", default=2000, type=int)
    parser.add_argument("--best_checkpoint_start_epoch", default=5, type=int)
    parser.add_argument("--accum_steps", default=1, type=int,
                        help="Gradient-accumulation micro-batches. Effective batch "
                             "= batch_size * accum_steps. Numerically exact here "
                             "(all norms are per-sample), so it recovers a larger "
                             "effective batch on a memory-limited GPU. Validation / "
                             "checkpoint intervals count optimizer steps.")
    parser.add_argument("--init_from", default=None,
                        help="Optional path to a g_best-style checkpoint "
                             "({'generator': state_dict}). Loads weights, starts "
                             "with a fresh optimizer + step counter. Ignored if "
                             "--checkpoint_path already has a rolling checkpoint.")
    a = parser.parse_args()

    with open(a.config) as f:
        h = AttrDict(json.load(f))
    build_env(a.config, "config.json", a.checkpoint_path)

    train(a, h)


if __name__ == "__main__":
    main()
