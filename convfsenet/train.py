"""Training loop for ConvFSENet_QuantFriendly_TD (causal=True).

Single-process, single-GPU/CPU. Mirrors train.py's CLI surface and
checkpoint conventions (g_<steps> rolling + g_best PESQ-tracked) so the
existing run_sweep.sh / utils.scan_checkpoint plumbing works untouched —
this script can be slotted into the same sweep pattern.

Training is end-to-end time-domain: audio -> Spectrogram -> ConvFSENet
mask + complex multiply -> InverseSpectrogram -> audio_pred. The base loss
is the model's PreProcLoss(DynCompMSE) computed on the time-domain pair.

Optional metric-GAN (config block `gan.enabled`): a MetricDiscriminator
(common/discriminator.py) is trained to predict normalized PESQ, and a
metric loss MSE(disc(clean, pred), 1) is added to the generator objective
with weight `gan.metric_loss_lambda`. This mirrors the MP-SENet / NSNet2
recipe. The discriminator is TRAINING-ONLY — it never touches the
streaming / ONNX / int8 inference path.

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
from common.discriminator import MetricDiscriminator, batch_pesq
from common.metrics import pesq_score
from common.utils import load_checkpoint, save_checkpoint, scan_checkpoint
from nsnet2.sparsity import SparsityController

warnings.simplefilter(action="ignore", category=FutureWarning)


def _to_dev(t, device):
    return t.to(device, non_blocking=True)


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

    # Fixed structured-sparsity masks (h.sparsity). Built after the weight load
    # so magnitude selection sees trained weights. Only pointwise (1x1) convs
    # are eligible — the depthwise k=3 dconv lowers through im2col, not as a
    # GEMM, so an N:M group along C_in would not mean anything to a sparse
    # kernel. That still covers 98% of the parameters.
    sparsity = SparsityController.from_config(model, h.get("sparsity", None))
    if sparsity is not None:
        sparsity.apply()
        print(sparsity)
        for row in sparsity.report():
            print('  {name:<28} {shape} sparsity={sparsity:.3f} '
                  'tail={tail_elements}'.format(**row))

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

    # ----- optional metric-GAN discriminator --------------------------------
    gan_cfg = h.get("gan", {}) or {}
    use_gan = bool(gan_cfg.get("enabled", False))
    metric_lambda = float(gan_cfg.get("metric_loss_lambda", 0.05))
    # Discriminator sees compressed magnitudes (power 0.3) — matches the
    # MP-SENet / NSNet2 recipe; squashes dynamic range for a stabler
    # discriminator regardless of the generator's own compress_factor.
    disc_compress = float(gan_cfg.get("disc_compress_factor", 0.3))
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
        print(f"metric-GAN enabled: lambda={metric_lambda}, "
              f"disc_compress_factor={disc_compress}")

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
                    noisy_audio, clean_audio, h, metric_lambda,
                    disc_compress, device, sparsity,
                )
            else:
                optim.zero_grad()
                loss_dict = model.train_step(noisy_audio, clean_audio)
                loss = loss_dict["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                if sparsity is not None:
                    sparsity.mask_grads()
                optim.step()
                if sparsity is not None:
                    sparsity.apply()
                metrics = {"loss": float(loss.item())}

            if steps % a.stdout_interval == 0:
                extra = ""
                if use_gan:
                    extra = (f", Metric: {metrics['loss_metric']:.4f}"
                             f", Disc: {metrics['loss_disc']:.4f}")
                print(
                    f"Steps : {steps:d}, Loss: {metrics['loss']:.4f}{extra}, "
                    f"s/b : {time.time() - start_b:.3f}"
                )
            if steps % a.summary_interval == 0:
                sw.add_scalar("Training/Loss", metrics["loss"], steps)
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


def _gan_step(model, discriminator, optim, optim_d,
              noisy_audio, clean_audio, h,
              metric_lambda, disc_compress, device, sparsity=None):
    """One metric-GAN training step.

    Generator forward is end-to-end (audio -> audio_pred). The discriminator
    is trained to predict normalized PESQ: score 1 for clean-vs-clean, score
    batch_pesq() for pred-vs-clean. The generator gets a metric loss pulling
    disc(clean, pred) toward 1 (i.e. "sound like perfect PESQ").

    Returns a dict of scalar metrics for logging.
    """
    # ----- generator forward (shared by both phases) ------------------------
    x_pred, loss_dict = model.process_data(noisy_audio, clean_audio, crop_signals=True)
    base_loss = loss_dict["loss"]
    min_len = x_pred.shape[-1]
    clean_c = clean_audio[..., :min_len]                          # crop to match x_pred

    # Compressed magnitudes for the discriminator — (B, F, T).
    clean_mag, _, _ = mag_pha_stft(
        clean_c.squeeze(1), h.n_fft, h.hop_size, h.win_size, disc_compress)
    pred_mag, _, _ = mag_pha_stft(
        x_pred.squeeze(1), h.n_fft, h.hop_size, h.win_size, disc_compress)
    one_labels = torch.ones(x_pred.shape[0], device=device)

    # ----- discriminator step ----------------------------------------------
    # Ground-truth normalized PESQ targets for the pred-vs-clean branch.
    pesq_target = batch_pesq(
        list(clean_c.squeeze(1).detach().cpu().numpy()),
        list(x_pred.squeeze(1).detach().cpu().numpy()),
    )
    optim_d.zero_grad()
    metric_r = discriminator(clean_mag, clean_mag)
    metric_g = discriminator(clean_mag, pred_mag.detach())
    loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
    if pesq_target is not None:
        loss_disc_g = F.mse_loss(pesq_target.to(device), metric_g.flatten())
    else:
        # batch_pesq returns None when any utterance's PESQ failed (silent
        # segment etc.) — skip the fake branch for this batch.
        loss_disc_g = torch.zeros((), device=device)
    loss_disc = loss_disc_r + loss_disc_g
    loss_disc.backward()
    optim_d.step()

    # ----- generator step ---------------------------------------------------
    optim.zero_grad()
    metric_g = discriminator(clean_mag, pred_mag)                  # pred_mag keeps grad
    loss_metric = F.mse_loss(metric_g.flatten(), one_labels)
    loss_gen = base_loss + metric_lambda * loss_metric
    loss_gen.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    if sparsity is not None:
        sparsity.mask_grads()
    optim.step()
    if sparsity is not None:
        # Re-project onto the mask: AdamW would otherwise drift pruned weights
        # off exactly zero.
        sparsity.apply()

    return {
        "loss": float(loss_gen.item()),
        "base_loss": float(base_loss.item()),
        "loss_metric": float(loss_metric.item()),
        "loss_disc": float(loss_disc.item()),
    }


@torch.no_grad()
def _validate(model, validation_loader, device, h) -> float:
    """Run model.valid_step on the validation split and return mean PESQ."""
    model.eval()
    audios_r, audios_g = [], []
    val_loss_sum = 0.0
    n = 0
    for batch in validation_loader:
        clean_audio, noisy_audio = batch
        clean_audio = _to_dev(clean_audio, device).unsqueeze(1)
        noisy_audio = _to_dev(noisy_audio, device).unsqueeze(1)
        x_pred, x_clean_p, x_noisy_p, loss_dict = model.valid_step(noisy_audio, clean_audio)
        audios_r.append(x_clean_p)
        audios_g.append(x_pred)
        val_loss_sum += float(loss_dict["loss"].item())
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
