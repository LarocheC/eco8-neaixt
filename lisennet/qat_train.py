"""QAT fine-tuning for LiSenNet — load a hardened FP32 g_best, install fake-quant,
fine-tune so the weights become robust to int8 rounding.

Recovers the static-int8 PESQ drop (RESULTS_LISENNET reports ~0.12 for the conv
variant; the sibling NSNet2 / ConvFSENet QAT recovers ~0.5-0.8). Mirrors
``convfsenet/qat_train.py``: same ``common/quant_fake.py`` machinery (QSpec /
prepare_for_qat / STE fake-quant) and the same checkpoint discipline (strip the
parametrize wrappers + activation hooks before save so g_best loads into a plain
LiSenNet, then re-install for the next epoch).

Reconstruction loss only — complex + magnitude, NO MetricGAN discriminator (fast and
stable for a short fine-tune from a converged checkpoint). Validation runs with quant
active, so PESQ / g_best track the *quantized* model directly.

Intended for the deploy-hardened conv variant (``bottleneck="conv"``,
``norm="batchnorm"``, ``act="relu"``). Per-output-channel symmetric weight quant
commutes with the BatchNorm-into-Conv fold done at export, so QAT on the raw conv weight
matches the folded int8 weight (same argument as ConvFSENet).

Usage:

    python -m lisennet.qat_train \\
        --init_from cp_lisennet_conv_hardened/g_best \\
        --checkpoint_path cp_lisennet_conv_hardened_qat \\
        --epochs 30 --lr 1e-4

The model config is read from the sibling config.json next to --init_from.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.nn import Conv2d, ConvTranspose2d
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader

from lisennet.model import build_lisennet
from lisennet.train import _validate, forward_spectra
from common.dataset import Dataset, data_generator, load_voicebank_demand, seed_worker
from common.env import AttrDict, build_env
from common.losses import generator_loss, generator_terms, loss_weights
from common.quant_fake import (
    QSpec,
    _FakeQuantWeightParam,
    install_static_activation_fake_quant,
    prepare_for_qat,
)
from common.utils import load_checkpoint, save_checkpoint


def _recon_loss(model, noisy, clean, weights):
    """The MP-SENet objective minus the metric term — there is no discriminator
    during QAT. Magnitude + complex + STFT-consistency + time."""
    s = forward_spectra(model, noisy, clean)
    terms = generator_terms(
        mag_g=s["mag_g"], com_g=s["com_g"], mag_r=s["mag_r"], com_r=s["com_r"],
        com_g_hat=s["com_g_hat"], audio_g=s["audio_g"], audio_r=s["audio_r"],
    )
    return generator_loss(terms, weights)


def _load_fp32_into_model(model, init_from_path, device) -> None:
    """Strict-load a g_best checkpoint. MUST run before prepare_for_qat — once
    parametrize wraps the weights the state_dict keys change."""
    print(f"loading fp32 weights from {init_from_path}")
    sd = load_checkpoint(init_from_path, device)
    if "generator" not in sd:
        raise KeyError(f"{init_from_path} is not a g_best checkpoint (no 'generator' key)")
    model.load_state_dict(sd["generator"], strict=True)


def _save_qat_checkpoint(model, ckpt_path, w_bits, a_bits, source) -> None:
    """Save a QAT model as a plain ``{'generator': state_dict}`` that loads into a fresh
    LiSenNet (so ``lisennet/export_onnx.py`` works on it unchanged).

    Temporarily removes the weight parametrizations (baking the int8-robust fp32 weight
    back to a bare weight), clears the activation pre-forward hooks on the conv leaves,
    and drops the ``_act_fake_quant`` ModuleList; saves; then re-installs everything so
    the next epoch continues seamlessly.
    """
    # (1) snapshot + remove weight parametrizations.
    param_snapshots = []
    for mod in list(model.modules()):
        for attr in list(getattr(mod, "parametrizations", {}).keys()):
            spec = mod.parametrizations[attr][0].spec
            param_snapshots.append((mod, attr, spec))
            parametrize.remove_parametrizations(mod, attr, leave_parametrized=False)

    # (2) clear activation hooks on the conv leaves + detach the observer ModuleList so
    #     the saved state_dict is a plain LiSenNet (no _act_fake_quant.*).
    for mod in model.modules():
        if isinstance(mod, (Conv2d, ConvTranspose2d)) and mod._forward_pre_hooks:
            mod._forward_pre_hooks.clear()
            if hasattr(mod, "_forward_pre_hooks_with_kwargs"):
                mod._forward_pre_hooks_with_kwargs.clear()
    act_fq = getattr(model, "_act_fake_quant", None)
    if act_fq is not None:
        del model._act_fake_quant

    save_checkpoint(ckpt_path, {"generator": model.state_dict()})
    with open(os.path.join(os.path.dirname(ckpt_path), "qat_recipe.json"), "w") as f:
        json.dump({"w_bits": w_bits, "a_bits": a_bits, "source": source}, f, indent=2)

    # re-install for continued training — re-attach the SAME observer modules
    # (accumulated ranges preserved), re-register parametrizations + hooks.
    for mod, attr, spec in param_snapshots:
        parametrize.register_parametrization(mod, attr, _FakeQuantWeightParam(spec))
    if act_fq is not None:
        model.add_module("_act_fake_quant", act_fq)
    install_static_activation_fake_quant(model, a_bits)


def train_qat(a, h):
    torch.manual_seed(h.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_lisennet(h).to(device)
    _load_fp32_into_model(model, a.init_from, device)

    nw, na = prepare_for_qat(
        model,
        weight=QSpec(bits=a.w_bits, axis=None),       # axis=None -> per-output-channel default
        activation=QSpec(bits=a.a_bits),
        static_activation=True,                        # observer-based — matches static int8 PTQ
    )
    print(f"QAT scaffold: weight params fake-quantized={nw}, "
          f"static activation quant points={na} (w{a.w_bits}/a{a.a_bits})")
    model.to(device)                                    # move the fresh observers to device

    os.makedirs(a.checkpoint_path, exist_ok=True)
    os.makedirs(os.path.join(a.checkpoint_path, "logs"), exist_ok=True)

    weights = loss_weights(h)
    for k in ("phase", "metric"):               # no phase decoder; no discriminator in QAT
        weights.pop(k, None)
    optim = torch.optim.AdamW(model.parameters(), a.lr, betas=[h.adam_b1, h.adam_b2])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=a.epochs, eta_min=a.lr_min)

    hf = load_voicebank_demand(cache_dir=a.hf_cache_dir)
    trainset = Dataset(hf["train"], h.segment_size, h.sampling_rate, split=True, shuffle=True, seed=h.seed)
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True,
                              worker_init_fn=seed_worker, generator=data_generator(h.seed))
    validset = Dataset(hf["test"], h.segment_size, h.sampling_rate, split=False, shuffle=False, seed=h.seed)
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False, batch_size=1,
                                   pin_memory=True, drop_last=True)

    # Seed the static activation observers with a short forward-only pass (fills the
    # running_min/max) so the pre-QAT eval doesn't run on uninitialized ranges.
    model.train()
    with torch.no_grad():
        for i, (clean_audio, noisy_audio) in enumerate(train_loader):
            if i >= a.calib_batches:
                break
            model(noisy_audio.to(device, non_blocking=True), clean_audio.to(device, non_blocking=True))
    pre_pesq = _validate(model, validation_loader, device, h)
    print(f"pre-QAT static-quant-active PESQ: {pre_pesq:.4f}")

    best_pesq = 0.0
    step = 0
    for epoch in range(a.epochs):
        model.train()
        start = time.time()
        print(f"Epoch {epoch + 1}/{a.epochs}  (lr={optim.param_groups[0]['lr']:.2e})")
        for clean_audio, noisy_audio in train_loader:
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)

            optim.zero_grad()
            loss = _recon_loss(model, noisy_audio, clean_audio, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(h.get("gradient_clip", 5.0)))
            optim.step()

            if step % 50 == 0:
                print(f"  step {step:5d}  recon_loss={loss.item():.4f}")
            step += 1

        scheduler.step()
        val_pesq = _validate(model, validation_loader, device, h)
        print(f"  epoch {epoch + 1}: quant-active PESQ={val_pesq:.4f} (best {best_pesq:.4f}), "
              f"{int(time.time() - start)}s")

        if val_pesq > best_pesq:
            best_pesq = val_pesq
            _save_qat_checkpoint(model, os.path.join(a.checkpoint_path, "g_best"),
                                 a.w_bits, a.a_bits, a.init_from)
            print(f"  saved g_best (PESQ {best_pesq:.4f})")

    print(f"\nDone. pre-QAT={pre_pesq:.4f}  best QAT={best_pesq:.4f}  delta={best_pesq - pre_pesq:+.4f}")
    return best_pesq


def main():
    p = argparse.ArgumentParser(description="QAT fine-tuning for LiSenNet (deploy-hardened conv).")
    p.add_argument("--init_from", required=True, help="FP32 g_best checkpoint to fine-tune from.")
    p.add_argument("--checkpoint_path", required=True, help="Output dir for the QAT g_best + qat_recipe.json.")
    p.add_argument("--config", default=None, help="config.json path. Default: sibling of --init_from.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--w_bits", type=int, default=8, help="Weight bit-width (8 = match int8 deployment).")
    p.add_argument("--a_bits", type=int, default=8, help="Activation bit-width.")
    p.add_argument("--calib_batches", type=int, default=50,
                   help="Forward-only batches to seed the activation observers before the pre-QAT measurement.")
    p.add_argument("--hf_cache_dir", default=None)
    a = p.parse_args()

    config_path = a.config or os.path.join(os.path.dirname(a.init_from), "config.json")
    with open(config_path) as f:
        h = AttrDict(json.load(f))
    build_env(config_path, "config.json", a.checkpoint_path)
    train_qat(a, h)


if __name__ == "__main__":
    main()
