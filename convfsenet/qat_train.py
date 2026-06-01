"""QAT fine-tuning for ConvFSENet — load an FP32 g_best, install fake-quant,
fine-tune so the weights become robust to int8 rounding.

Mirrors qat_train.py (the NSNet QAT path) for consistency: same quant_fake.py
machinery (QSpec / prepare_for_qat / STE fake-quant), same checkpoint
discipline (strip the parametrize wrappers before save so g_best loads into a
plain model, write a qat_recipe.json sidecar, re-install for the next epoch).

Why this works for ConvFSENet's BN fold: prepare_for_qat fake-quants the raw
conv weights with per-output-channel symmetric quant. ConvFSENet folds BN into
the conv at export — a per-output-channel rescale. Per-channel symmetric
quantization is invariant to per-channel rescaling (the scale rescales with
the weight, indices unchanged), so QAT on the raw weight is equivalent to QAT
on the folded weight. No special BN handling needed.

Reconstruction loss only (the model's PreProcLoss/DynCompMSE) — no GAN, like
qat_train.py: fast and stable for a short fine-tune from an already-converged
checkpoint. Validation runs with quant active, so PESQ / g_best track the
*quantized* model directly.

Usage:

    python -m convfsenet.qat_train \\
        --init_from cp_convfsenet/g_best \\
        --checkpoint_path cp_convfsenet_qat \\
        --w_bits 8 --a_bits 8 --epochs 30 --lr 1e-4

The model config is read from the sibling config.json next to --init_from.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.nn import Conv1d
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader

from convfsenet.model import build_causal_model
from common.dataset import Dataset, load_voicebank_demand
from common.env import AttrDict, build_env
from common.quant_fake import (
    QSpec,
    _FakeQuantWeightParam,
    install_static_activation_fake_quant,
    prepare_for_qat,
)
from convfsenet.train import _validate
from common.utils import load_checkpoint, save_checkpoint


def _load_fp32_into_model(model, init_from_path, device) -> None:
    """Strict-load a g_best checkpoint. MUST run before prepare_for_qat — once
    parametrize wraps the weights the state_dict keys change."""
    print(f"loading fp32 weights from {init_from_path}")
    sd = load_checkpoint(init_from_path, device)
    if "generator" not in sd:
        raise KeyError(
            f"{init_from_path} is not a g_best checkpoint (no 'generator' key)"
        )
    model.load_state_dict(sd["generator"], strict=True)


def _save_qat_checkpoint(model, ckpt_path, w_bits, a_bits, source) -> None:
    """Save a QAT model as a plain {'generator': state_dict} that loads into a
    fresh ConvFSENet (so convfsenet/export_onnx.py works on it unchanged).

    Temporarily: (1) remove every weight parametrization with
    leave_parametrized=False — the underlying FP32 nn.Parameter (trained to be
    int8-robust) becomes the bare weight again; (2) clear the activation
    pre-forward hooks on the Conv1d modules. Save. Write qat_recipe.json. Then
    re-install both so the next epoch continues seamlessly.
    """
    # (1) snapshot + remove weight parametrizations.
    param_snapshots = []
    for mod in list(model.modules()):
        for attr in list(getattr(mod, "parametrizations", {}).keys()):
            spec = mod.parametrizations[attr][0].spec
            param_snapshots.append((mod, attr, spec))
            parametrize.remove_parametrizations(mod, attr, leave_parametrized=False)

    # (2) clear activation hooks + detach the StaticActFakeQuant ModuleList so
    #     the saved state_dict is a plain ConvFSENet (no _act_fake_quant.*).
    for mod in model.modules():
        if isinstance(mod, Conv1d) and mod._forward_pre_hooks:
            mod._forward_pre_hooks.clear()
            if hasattr(mod, "_forward_pre_hooks_with_kwargs"):
                mod._forward_pre_hooks_with_kwargs.clear()
    act_fq = getattr(model, "_act_fake_quant", None)
    if act_fq is not None:
        del model._act_fake_quant

    save_checkpoint(ckpt_path, {"generator": model.state_dict()})

    recipe = {"w_bits": w_bits, "a_bits": a_bits, "source": source}
    with open(os.path.join(os.path.dirname(ckpt_path), "qat_recipe.json"), "w") as f:
        json.dump(recipe, f, indent=2)

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

    model = build_causal_model(h).to(device)
    _load_fp32_into_model(model, a.init_from, device)

    nw, na = prepare_for_qat(
        model,
        weight=QSpec(bits=a.w_bits, axis=None),       # axis=None -> per-channel (walker default 0)
        activation=QSpec(bits=a.a_bits),
        static_activation=True,                        # observer-based — matches static int8 PTQ
    )
    print(f"QAT scaffold: weight params fake-quantized={nw}, "
          f"static activation quant points={na} (w{a.w_bits}/a{a.a_bits})")
    # prepare_for_qat added the StaticActFakeQuant observers fresh on CPU —
    # move the whole model (incl. the new _act_fake_quant submodules) to device.
    model.to(device)

    os.makedirs(a.checkpoint_path, exist_ok=True)
    os.makedirs(os.path.join(a.checkpoint_path, "logs"), exist_ok=True)

    optim = torch.optim.AdamW(model.parameters(), a.lr, betas=[h.adam_b1, h.adam_b2])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=a.epochs, eta_min=a.lr_min,
    )

    hf = load_voicebank_demand(cache_dir=a.hf_cache_dir)
    trainset = Dataset(hf["train"], h.segment_size, h.sampling_rate,
                       split=True, shuffle=True, seed=h.seed)
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True)
    validset = Dataset(hf["test"], h.segment_size, h.sampling_rate,
                       split=False, shuffle=False, seed=h.seed)
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                   batch_size=1, pin_memory=True, drop_last=True)

    # Seed the static activation observers with a short forward-only pass in
    # train mode (EMA-fills running_min/max). Without it the pre-QAT eval would
    # run on uninitialized ranges.
    model.train()
    with torch.no_grad():
        for i, (clean_audio, noisy_audio) in enumerate(train_loader):
            if i >= a.calib_batches:
                break
            model.train_step(
                noisy_audio.to(device, non_blocking=True).unsqueeze(1),
                clean_audio.to(device, non_blocking=True).unsqueeze(1),
            )
    # PESQ with the static fake-quant active — this should track the deployed
    # static int8 (the verification that the fake-quant matches quantize_static).
    pre_pesq = _validate(model, validation_loader, device, h)
    print(f"pre-QAT static-quant-active PESQ: {pre_pesq:.4f}")

    best_pesq = 0.0
    step = 0
    for epoch in range(a.epochs):
        model.train()
        start = time.time()
        print(f"Epoch {epoch + 1}/{a.epochs}  (lr={optim.param_groups[0]['lr']:.2e})")

        for clean_audio, noisy_audio in train_loader:
            clean_audio = clean_audio.to(device, non_blocking=True).unsqueeze(1)
            noisy_audio = noisy_audio.to(device, non_blocking=True).unsqueeze(1)

            optim.zero_grad()
            loss_dict = model.train_step(noisy_audio, clean_audio)
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

            if step % 50 == 0:
                print(f"  step {step:5d}  loss={loss.item():.4f}")
            step += 1

        scheduler.step()
        val_pesq = _validate(model, validation_loader, device, h)
        print(f"  epoch {epoch + 1}: quant-active PESQ={val_pesq:.4f} "
              f"(best {best_pesq:.4f}), {int(time.time() - start)}s")

        if val_pesq > best_pesq:
            best_pesq = val_pesq
            _save_qat_checkpoint(
                model, os.path.join(a.checkpoint_path, "g_best"),
                a.w_bits, a.a_bits, a.init_from,
            )
            print(f"  saved g_best (PESQ {best_pesq:.4f})")

    print(f"\nDone. pre-QAT={pre_pesq:.4f}  best QAT={best_pesq:.4f}  "
          f"delta={best_pesq - pre_pesq:+.4f}")
    return best_pesq


def main():
    p = argparse.ArgumentParser(description="QAT fine-tuning for ConvFSENet.")
    p.add_argument("--init_from", required=True,
                   help="FP32 g_best checkpoint to fine-tune from.")
    p.add_argument("--checkpoint_path", required=True,
                   help="Output dir for the QAT g_best + qat_recipe.json.")
    p.add_argument("--config", default=None,
                   help="config.json path. Default: sibling of --init_from.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--w_bits", type=int, default=8,
                   help="Weight bit-width (8 = match the int8 deployment).")
    p.add_argument("--a_bits", type=int, default=8, help="Activation bit-width.")
    p.add_argument("--calib_batches", type=int, default=100,
                   help="Forward-only batches to seed the activation observers "
                        "before the pre-QAT measurement.")
    p.add_argument("--hf_cache_dir", default=None)
    a = p.parse_args()

    config_path = a.config or os.path.join(os.path.dirname(a.init_from), "config.json")
    with open(config_path) as f:
        h = AttrDict(json.load(f))
    build_env(config_path, "config.json", a.checkpoint_path)

    train_qat(a, h)


if __name__ == "__main__":
    main()
