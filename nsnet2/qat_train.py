"""QAT fine-tuning: load an fp32 checkpoint, prepare for quant, fine-tune.

Reconstruction-loss only (no discriminator) — fast and stable for the few
epochs we want. The model uses the same NSNet2 config as the source
checkpoint; only the quant scaffold + LR + epoch count differ.

Usage:

    python -m nsnet2.qat_train \\
        --init_from cp_tr_monarch_8/g_best \\
        --w_bits 4 --a_bits 8 \\
        --epochs 5 --lr 3e-4 \\
        --checkpoint_path cp_qat_w4a8_monarch_8

Note this trains on the magnitude-domain reconstruction losses only:
0.9 * mag + 0.1 * com + 0.1 * stft + 0.2 * time. The original GAN
discriminator is not part of this loop — start with what already
exists in cp_<run> if you need to resume with the full setup later.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.dataset import Dataset, load_voicebank_demand, mag_pha_istft, mag_pha_stft
from common.env import AttrDict, build_env
from nsnet2.model import NSNet2
from common.metrics import pesq_score
from common.quant_fake import QSpec, prepare_for_qat
from common.utils import load_checkpoint, save_checkpoint


def _load_fp32_into_model(model: NSNet2, init_from_path: str, device) -> None:
    """Load a g_best checkpoint into ``model``. Strict load.

    Must be called BEFORE prepare_for_qat — once parametrize wraps the
    weights, the state_dict keys change (module.weight -> module.parametrizations.weight.original).
    """
    print(f"loading fp32 weights from {init_from_path}")
    sd = load_checkpoint(init_from_path, device)
    if "generator" not in sd:
        raise KeyError(
            f"{init_from_path} does not look like a g_best ckpt "
            f"(no 'generator' key); got keys: {list(sd)}"
        )
    model.load_state_dict(sd["generator"], strict=True)


def _save_qat_checkpoint(model: NSNet2, ckpt_path: str, a) -> None:
    """Save a QAT-trained model in a format eval_quant.py can read.

    Three things temporarily get reverted on the model so the saved
    state_dict matches a fresh ``NSNet2(h)`` exactly:

    1. ``parametrize.remove_parametrizations(..., leave_parametrized=False)``
       on every wrapped weight — the underlying fp32 ``nn.Parameter``
       (trained to be robust to fake-quant) becomes the bare weight again.
    2. Every ``GRUCellQuant.quant_x / quant_h_in / quant_h_out`` that
       was swapped to ``_FakeQuantSwap`` gets reverted to a freshly-built
       ``gru_qat.Identity`` (which carries the scale/zero_point/running_*
       buffers a strict load expects).
    3. Activation pre-forward hooks on Linear / BlockdiagLinear /
       Butterfly are removed (they have no state but they'd fire again
       on the next forward of the eval-side model if we left them).

    After saving, both the parametrize wrappers and the _FakeQuantSwap
    slots are re-installed identically so the next epoch's training is
    a no-op continuation of the prior epoch.

    A sidecar ``qat_recipe.json`` records (w_bits, a_bits, axis,
    group_size) so eval_quant.py can re-apply the same PTQ pattern.
    """
    from torch.nn.utils import parametrize

    from gru_qat.quantizers import Identity, QuantizerConfig
    from common.quant_fake import (
        _FakeQuantSwap,
        _FakeQuantWeightParam,
        _GRU_QAT_CELL_ACT_SLOTS,
        _is_gru_qat_cell,
    )

    # Snapshot parametrizations: (mod, attr, spec).
    param_snapshots = []
    for mod in list(model.modules()):
        attrs = list(getattr(mod, "parametrizations", {}).keys())
        for attr in attrs:
            spec = mod.parametrizations[attr][0].spec
            param_snapshots.append((mod, attr, spec))
            parametrize.remove_parametrizations(mod, attr, leave_parametrized=False)

    # Snapshot swapped activation slots: (cell, slot, spec).
    swap_snapshots = []
    for mod in model.modules():
        if not _is_gru_qat_cell(mod):
            continue
        for slot in _GRU_QAT_CELL_ACT_SLOTS:
            cur = getattr(mod, slot, None)
            if isinstance(cur, _FakeQuantSwap):
                swap_snapshots.append((mod, slot, cur.spec))
                # Restore a fresh Identity (matches what NSNet2(h) builds).
                setattr(mod, slot, Identity(QuantizerConfig(name=slot)))

    # Strip the activation pre-forward hooks we installed. They have no
    # state (just closure-captured spec) so we just clear them. PyTorch
    # tracks hooks in ``_forward_pre_hooks`` (and ``_forward_pre_hooks_with_kwargs``
    # on >=2.0 for with_kwargs hooks).
    hook_targets_act_spec = QSpec(  # captured for re-install
        bits=a.a_bits, axis=None, group_size=None,
    )
    hook_target_modules = []
    for mod in model.modules():
        if mod._forward_pre_hooks:
            # Only clear if we recognise the module type as one we hook
            # (Linear / BlockdiagLinear / Butterfly / GRU).
            from common.quant_fake import _is_blockdiag_linear, _is_butterfly
            if isinstance(mod, (torch.nn.Linear, torch.nn.GRU)) or \
               _is_blockdiag_linear(mod) or _is_butterfly(mod):
                mod._forward_pre_hooks.clear()
                if hasattr(mod, "_forward_pre_hooks_with_kwargs"):
                    mod._forward_pre_hooks_with_kwargs.clear()
                hook_target_modules.append(mod)

    save_checkpoint(ckpt_path, {"generator": model.state_dict()})

    recipe = {
        "w_bits": a.w_bits, "a_bits": a.a_bits,
        "w_axis": a.w_axis, "w_group_size": a.w_group_size,
        "source": a.init_from,
    }
    recipe_path = os.path.join(os.path.dirname(ckpt_path), "qat_recipe.json")
    with open(recipe_path, "w") as f:
        json.dump(recipe, f, indent=2)

    # ---- Re-install everything for continued training -------------------
    for mod, attr, spec in param_snapshots:
        parametrize.register_parametrization(mod, attr, _FakeQuantWeightParam(spec))
    for cell, slot, spec in swap_snapshots:
        setattr(cell, slot, _FakeQuantSwap(spec))
    # Re-stamp the activation pre-hooks. install_activation_fake_quant is
    # idempotent on the cell side (it always swaps) but it'll re-add hooks
    # on the Linear/BlockdiagLinear/Butterfly modules we just cleared.
    from common.quant_fake import install_activation_fake_quant
    install_activation_fake_quant(model, hook_targets_act_spec)


def train_qat(a, h):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(h.seed)

    generator = NSNet2(h).to(device)
    _load_fp32_into_model(generator, a.init_from, device)

    # Flip use_triton=False on any TritonGRU layers so the per-step Python
    # body runs (the persistent Triton kernel doesn't see our parametrize/
    # hooks and can't be gradient-traced through them).
    n_triton_flipped = 0
    for sub in generator.modules():
        if hasattr(sub, "use_triton") and isinstance(getattr(sub, "use_triton"), bool):
            sub.use_triton = False
            n_triton_flipped += 1
    if n_triton_flipped:
        print(f"flipped use_triton=False on {n_triton_flipped} layer(s)")

    nw, na = prepare_for_qat(
        generator,
        weight=QSpec(bits=a.w_bits, axis=a.w_axis, group_size=a.w_group_size),
        activation=QSpec(bits=a.a_bits),
    )
    print(f"QAT scaffold: weights_parametrized={nw}, activation_quant_points={na}")
    total = sum(p.numel() for p in generator.parameters())
    print(f"trainable parameters: {total / 1e6:.3f}M")

    os.makedirs(a.checkpoint_path, exist_ok=True)
    optim_g = torch.optim.AdamW(generator.parameters(), a.lr,
                                betas=[h.adam_b1, h.adam_b2])
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay)

    hf = load_voicebank_demand(cache_dir=a.hf_cache_dir)
    trainset = Dataset(hf["train"], h.segment_size, h.sampling_rate,
                       split=True, shuffle=True, seed=h.seed)
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True)
    validset = Dataset(hf["test"], h.segment_size, h.sampling_rate,
                       split=False, shuffle=False, seed=h.seed)
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                   batch_size=1, pin_memory=True, drop_last=True)

    best_pesq = 0.0
    step = 0
    for epoch in range(a.epochs):
        generator.train()
        start = time.time()
        print(f"Epoch {epoch + 1}/{a.epochs}")

        for i, batch in enumerate(train_loader):
            clean_audio, noisy_audio = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)

            clean_mag, _, clean_com = mag_pha_stft(
                clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            noisy_mag, noisy_pha, _ = mag_pha_stft(
                noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )

            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)
            audio_g = mag_pha_istft(
                mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            _, _, com_g_hat = mag_pha_stft(
                audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )

            loss_mag = F.mse_loss(clean_mag, mag_g)
            loss_com = F.mse_loss(clean_com, com_g) * 2
            loss_stft = F.mse_loss(com_g, com_g_hat) * 2
            loss_time = F.l1_loss(clean_audio, audio_g)

            loss = (0.9 * loss_mag + 0.1 * loss_com + 0.1 * loss_stft + 0.2 * loss_time)

            optim_g.zero_grad()
            loss.backward()
            optim_g.step()

            if step % 50 == 0:
                print(
                    f"  step {step:5d}  loss={loss.item():.4f}  "
                    f"mag={loss_mag.item():.4f}  time={loss_time.item():.4f}"
                )
            step += 1

        scheduler_g.step()
        print(f"  epoch {epoch + 1} time: {int(time.time() - start)}s")

        # Validate with quant active (the parametrize wrappers + hooks
        # are part of the eval forward).
        generator.eval()
        torch.cuda.empty_cache()
        audios_r, audios_g = [], []
        with torch.no_grad():
            for batch in validation_loader:
                ca, na_ = batch
                ca = ca.to(device, non_blocking=True)
                na_ = na_.to(device, non_blocking=True)
                cm, cp, _ = mag_pha_stft(ca, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                nm, np_, _ = mag_pha_stft(na_, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                mg, pg, _ = generator(nm, np_)
                ag = mag_pha_istft(mg, pg, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                audios_r += torch.split(ca, 1, dim=0)
                audios_g += torch.split(ag, 1, dim=0)
        val_pesq = pesq_score(audios_r, audios_g, h).item()
        print(f"  PESQ: {val_pesq:.4f}  (best so far: {best_pesq:.4f})")

        if val_pesq > best_pesq:
            best_pesq = val_pesq
            ckpt_path = os.path.join(a.checkpoint_path, "g_best")
            _save_qat_checkpoint(generator, ckpt_path, a)
            print(f"  saved -> {ckpt_path}")

    print(f"\nDone. Best PESQ = {best_pesq:.4f}")
    return best_pesq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init_from", required=True,
                   help="Path to fp32 g_best checkpoint to fine-tune from.")
    p.add_argument("--checkpoint_path", required=True,
                   help="Output directory for the QAT-trained checkpoint.")
    p.add_argument("--config", default=None,
                   help="Path to config.json. Default: alongside --init_from.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--w_bits", type=int, default=4)
    p.add_argument("--a_bits", type=int, default=8)
    p.add_argument("--w_axis", type=int, default=None,
                   help="Override per-module weight quant axis. Default: per-module sensible default.")
    p.add_argument("--w_group_size", type=int, default=None,
                   help="Per-group weight quant. Default: per-channel.")
    p.add_argument("--hf_cache_dir", default=None)
    a = p.parse_args()

    # Resolve config: same convention as inference.py / eval_torch.py.
    config_path = a.config or os.path.join(os.path.dirname(a.init_from), "config.json")
    with open(config_path) as f:
        h = AttrDict(json.load(f))
    if torch.cuda.is_available():
        h.num_gpus = 1
        h.batch_size = int(h.batch_size / max(h.num_gpus, 1))

    build_env(config_path, "config.json", a.checkpoint_path)
    train_qat(a, h)


if __name__ == "__main__":
    main()
