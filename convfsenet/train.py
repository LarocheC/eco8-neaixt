"""Training loop for ConvFSENet_QuantFriendly_TD (causal=True).

Single-process, single-GPU/CPU. Mirrors train.py's CLI surface and
checkpoint conventions (g_<steps> rolling + g_best PESQ-tracked) so the
existing run_sweep.sh / utils.scan_checkpoint plumbing works untouched —
this script can be slotted into the same sweep pattern.

Training is end-to-end time-domain: audio -> Spectrogram -> ConvFSENet
mask + complex multiply -> InverseSpectrogram -> audio_pred.

Loss: the MP-SENet objective (common/losses.py), shared with every other model
in this repo — magnitude + complex + STFT-consistency + time + MetricGAN. It
replaces the model's own PreProcLoss(DynCompMSE), which used a different
formulation (active-speech-level normalisation, an alpha=0.3 mag/complex split,
and no time or consistency term).

The magnitude and complex terms are taken from the network's *pre-iSTFT* complex
spectrum (`mask * noisy_stft`), not from a re-STFT of the output waveform — that
is what makes the consistency term meaningful rather than identically zero.

The anti-wrapping phase loss is omitted: ConvFSENet is mask-only and reuses the
noisy phase, so it has no phase to train. See common/losses.py.

The MetricDiscriminator (common/discriminator.py) is trained to predict
normalized PESQ, and the generator's metric term pulls disc(clean, pred) toward
1. It is TRAINING-ONLY — it never touches the streaming / ONNX / int8 inference
path. Set `gan.enabled` false to drop the metric term.

Validation:
  - Offline causal forward (ConvFSENet.forward, with chomp) on the VBD test
    split. For this strictly-causal model that is numerically equivalent to
    the streaming wrapper, but note the deployed streaming path itself is not
    exercised here.
  - PESQ computed against clean. PESQ-best checkpoint -> g_best.
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

from convfsenet.model import build_causal_model
from common.dataset import (
    Dataset, data_generator, load_voicebank_demand, mag_pha_stft, seed_worker,
)
from common.env import AttrDict, build_env
from common.discriminator import MetricDiscriminator, pesq_target_from_config
from common.losses import (
    discriminator_loss, generator_loss, generator_terms, loss_weights,
    mag_pha_from_complex,
)
from common.metrics import pesq_score
from common.utils import load_checkpoint, save_checkpoint, scan_checkpoint

warnings.simplefilter(action="ignore", category=FutureWarning)

# No `phase`: ConvFSENet is mask-only and reuses the noisy phase.
_LOGGED_TERMS = ("magnitude", "complex", "consistency", "time", "metric")


def _to_dev(t, device):
    return t.to(device, non_blocking=True)


def forward_spectra(model, noisy_audio, clean_audio, h):
    """Forward ConvFSENet and produce every tensor the MP-SENet loss needs.

    The model is time-domain end-to-end, but the loss needs its *spectral* output
    before the iSTFT — otherwise `com_g` would be the STFT of `audio_g` and the
    consistency term would be identically zero by construction. So we drive the
    three stages by hand rather than going through `model.train_step`.
    """
    cf = float(h.get("compress_factor", 0.3) or 0.3)
    n_fft, hop, win = h.n_fft, h.hop_size, h.win_size

    stft_noisy = model.preproc(noisy_audio)                  # (B, 1, F, T) complex
    stft_pred = model(stft_noisy)                            # (B, 1, F, T) complex
    audio_g = model.postproc(stft_pred)                      # (B, 1, samples)

    audio_g, clean = model._crop_signals_if_needed(audio_g, clean_audio)
    audio_g, clean = audio_g.squeeze(1), clean.squeeze(1)

    # The network's direct spectral output.
    mag_g, _, com_g = mag_pha_from_complex(stft_pred.squeeze(1), cf)
    # The clean target, and the iSTFT -> STFT round trip of our own waveform.
    mag_r, _, com_r = mag_pha_stft(clean, n_fft, hop, win, cf)
    mag_g_hat, _, com_g_hat = mag_pha_stft(audio_g, n_fft, hop, win, cf)

    # A frame or two can differ when the segment length isn't a whole number of
    # hops; the terms are elementwise, so trim rather than let broadcasting lie.
    t = min(com_g.shape[-2], com_g_hat.shape[-2], com_r.shape[-2])
    mag_g, com_g = mag_g[..., :t], com_g[..., :t, :]
    mag_r, com_r = mag_r[..., :t], com_r[..., :t, :]
    mag_g_hat, com_g_hat = mag_g_hat[..., :t], com_g_hat[..., :t, :]

    return dict(
        mag_r=mag_r, com_r=com_r, mag_g=mag_g, com_g=com_g,
        mag_g_hat=mag_g_hat, com_g_hat=com_g_hat,
        audio_g=audio_g, audio_r=clean,
    )


def train(a, h):
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = build_causal_model(h).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Total Parameters: {n_params / 1e6:.3f}M")
    os.makedirs(a.checkpoint_path, exist_ok=True)
    os.makedirs(os.path.join(a.checkpoint_path, "logs"), exist_ok=True)
    print(f"checkpoints directory : {a.checkpoint_path}")

    # ----- resume from latest rolling checkpoint if present ------------------
    # Resume order: rolling checkpoint in this dir (exact continuation) >
    # --init_from (warm-start weights, fresh optim/sched/step counter).
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

    # h.scheduler selects exponential (default — back-compat with the original
    # config) vs cosine annealing. Cosine: lr decays from h.learning_rate to
    # h.lr_min over a.training_epochs, giving proper fine-tuning at the tail.
    sched_kind = h.get("scheduler", "exponential")
    if sched_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=a.training_epochs,
            eta_min=float(h.get("lr_min", 1e-5)),
            last_epoch=last_epoch,
        )
    elif sched_kind == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optim, gamma=h.lr_decay, last_epoch=last_epoch
        )
    else:
        raise ValueError(f"unknown scheduler {sched_kind!r}; use 'exponential' or 'cosine'")

    # ----- loss + optional metric-GAN discriminator --------------------------
    weights = loss_weights(h)
    weights.pop("phase", None)              # mask-only model — no phase to train
    pesq_fn = pesq_target_from_config(h)

    gan_cfg = h.get("gan", {}) or {}
    use_gan = bool(gan_cfg.get("enabled", False))
    if use_gan:
        # `gan.metric_loss_lambda` stays the knob for the metric term's weight;
        # it is just MP-SENet's 0.05 under another name.
        weights["metric"] = float(gan_cfg.get("metric_loss_lambda", weights["metric"]))
    else:
        weights.pop("metric", None)
    # The discriminator now sees the same compressed magnitude the loss uses
    # (h.compress_factor), so the old `gan.disc_compress_factor` override is gone.
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
        print(f"metric-GAN enabled: lambda={weights['metric']}")

    # Resume the LR schedule exactly. Reconstructing with last_epoch advances
    # the schedule one step past the saved position (and clobbers the lr that
    # optim.load_state_dict just restored), so for checkpoints that carry the
    # scheduler state we restore it and push get_last_lr() back into the
    # optimizer (load_state_dict alone does not update the param-group lr).
    # Older checkpoints lack these keys and keep the last_epoch behaviour.
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
    # Restore best_pesq on resume so a resumed run can't overwrite g_best with
    # an inferior model (the checkpoint persists it; default 0.0 for fresh runs).
    best_pesq = float(state["best_pesq"]) if state is not None and "best_pesq" in state else 0.0

    # ----- training loop -----------------------------------------------------
    for epoch in range(max(0, last_epoch), a.training_epochs):
        model.train()
        start = time.time()
        print(f"Epoch: {epoch + 1}")

        for i, batch in enumerate(train_loader):
            start_b = time.time()
            clean_audio, noisy_audio = batch
            clean_audio = _to_dev(clean_audio, device).unsqueeze(1)   # (B, 1, samples)
            noisy_audio = _to_dev(noisy_audio, device).unsqueeze(1)

            if use_gan:
                metrics = _gan_step(
                    model, discriminator, optim, optim_d,
                    noisy_audio, clean_audio, h, weights, device, pesq_fn,
                )
            else:
                optim.zero_grad()
                s = forward_spectra(model, noisy_audio, clean_audio, h)
                terms = generator_terms(
                    mag_g=s["mag_g"], com_g=s["com_g"], mag_r=s["mag_r"],
                    com_r=s["com_r"], com_g_hat=s["com_g_hat"],
                    audio_g=s["audio_g"], audio_r=s["audio_r"],
                )
                loss = generator_loss(terms, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optim.step()
                metrics = {"loss": float(loss.item()), **{
                    k: float(terms[k].item()) for k in _LOGGED_TERMS if k in terms
                }}

            if steps % a.stdout_interval == 0:
                extra = ""
                if use_gan:
                    extra = (f", Metric: {metrics['metric']:.4f}"
                             f", Disc: {metrics['loss_disc']:.4f}")
                print(
                    f"Steps : {steps:d}, Loss: {metrics['loss']:.4f}{extra}, "
                    f"Mag: {metrics['magnitude']:.4f}, "
                    f"s/b : {time.time() - start_b:.3f}"
                )
            if steps % a.summary_interval == 0:
                sw.add_scalar("Training/Loss", metrics["loss"], steps)
                for k in _LOGGED_TERMS:
                    if k in metrics:
                        sw.add_scalar(f"Training/{k}", metrics[k], steps)
                if use_gan:
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


def _gan_step(model, discriminator, optim, optim_d,
              noisy_audio, clean_audio, h, weights, device, pesq_fn=None):
    """One MetricGAN training step against the MP-SENet objective.

    The discriminator is trained to predict normalized PESQ: 1 for clean-vs-clean,
    the PESQ target for pred-vs-clean. The generator's metric term pulls
    disc(clean_mag, mag_g_hat) toward 1 (i.e. "sound like perfect PESQ").

    Both the discriminator and the metric term are shown the *round-tripped*
    magnitude, per upstream.

    Returns a dict of scalar metrics for logging.
    """
    pesq_fn = pesq_fn or pesq_target_from_config(h)
    s = forward_spectra(model, noisy_audio, clean_audio, h)

    # ----- discriminator step ----------------------------------------------
    pesq_target = pesq_fn(s["audio_r"].detach(), s["audio_g"].detach())
    optim_d.zero_grad()
    loss_disc = discriminator_loss(discriminator, s["mag_r"], s["mag_g_hat"],
                                   pesq_target, device)
    loss_disc.backward()
    optim_d.step()

    # ----- generator step ---------------------------------------------------
    optim.zero_grad()
    metric_g = discriminator(s["mag_r"], s["mag_g_hat"])       # keeps generator grad
    terms = generator_terms(
        mag_g=s["mag_g"], com_g=s["com_g"], mag_r=s["mag_r"], com_r=s["com_r"],
        com_g_hat=s["com_g_hat"], audio_g=s["audio_g"], audio_r=s["audio_r"],
        metric_g=metric_g,
    )
    loss_gen = generator_loss(terms, weights)
    loss_gen.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optim.step()

    metrics = {"loss": float(loss_gen.item()), "loss_disc": float(loss_disc.item())}
    metrics.update({k: float(terms[k].item()) for k in _LOGGED_TERMS if k in terms})
    return metrics


@torch.no_grad()
def _validate(model, validation_loader, device, h) -> float:
    """Enhance the validation split; report the MP-SENet loss and return mean PESQ.

    The reported loss is the GAN-free objective (no discriminator at validation),
    so it is comparable across runs but not to the training loss, which includes
    the metric term.
    """
    model.eval()
    weights = loss_weights(h)
    for k in ("phase", "metric"):
        weights.pop(k, None)

    audios_r, audios_g = [], []
    val_loss_sum = 0.0
    n = 0
    for batch in validation_loader:
        clean_audio, noisy_audio = batch
        clean_audio = _to_dev(clean_audio, device).unsqueeze(1)
        noisy_audio = _to_dev(noisy_audio, device).unsqueeze(1)
        noisy_p, clean_p = model._pad_signals_if_needed(noisy_audio, clean_audio)
        s = forward_spectra(model, noisy_p, clean_p, h)
        terms = generator_terms(
            mag_g=s["mag_g"], com_g=s["com_g"], mag_r=s["mag_r"], com_r=s["com_r"],
            com_g_hat=s["com_g_hat"], audio_g=s["audio_g"], audio_r=s["audio_r"],
        )
        audios_r.append(s["audio_r"].unsqueeze(1))
        audios_g.append(s["audio_g"].unsqueeze(1))
        val_loss_sum += float(generator_loss(terms, weights).item())
        n += 1
    val_loss = val_loss_sum / max(n, 1)
    val_pesq = pesq_score(audios_r, audios_g, h).item()
    print(f"  validation loss: {val_loss:.4f}")
    return val_pesq


def main():
    print("Initializing ConvFSENet Training Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/convfsenet.json")
    parser.add_argument("--checkpoint_path", default="cp_convfsenet")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--training_epochs", default=200, type=int)
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--checkpoint_interval", default=2000, type=int)
    parser.add_argument("--summary_interval", default=50, type=int)
    parser.add_argument("--validation_interval", default=2000, type=int)
    parser.add_argument("--best_checkpoint_start_epoch", default=5, type=int)
    parser.add_argument("--init_from", default=None,
                        help="Optional path to a g_best-style checkpoint "
                             "({'generator': state_dict}). Loads weights, "
                             "starts with a fresh optimizer + step counter. "
                             "Ignored if --checkpoint_path already contains a "
                             "rolling checkpoint (resume takes priority).")
    a = parser.parse_args()

    with open(a.config) as f:
        h = AttrDict(json.load(f))
    build_env(a.config, "config.json", a.checkpoint_path)

    train(a, h)


if __name__ == "__main__":
    main()
