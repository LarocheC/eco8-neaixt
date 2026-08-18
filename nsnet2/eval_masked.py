"""Offline PESQ for a checkpoint, optionally with a sparsity mask applied.

Mirrors the validation loop in ``nsnet2/train.py`` exactly — same offline
(whole-utterance) forward, same STFT/iSTFT, same ``pesq_score`` over the full
VBD test split — so the number is directly comparable to the ``PESQ Score``
lines in a training log and to the ``g_best`` selection metric. (It is *not*
the streaming eval; use ``nsnet2/eval_torch.py`` for that.)

With ``--pattern`` it magnitude-prunes the loaded checkpoint before evaluating,
which gives the "pruned, no fine-tuning" reference point:

    python -m nsnet2.eval_masked --config cp_nsnet2/config.json \\
        --checkpoint cp_nsnet2/g_best --pattern 2:4
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.dataset import Dataset, load_voicebank_demand, mag_pha_istft, mag_pha_stft
from common.env import AttrDict
from common.metrics import pesq_score
from common.utils import load_checkpoint
from nsnet2.model import NSNet2
from nsnet2.sparsity import SparsityController


@torch.no_grad()
def evaluate(model, h, split, device):
    loader = DataLoader(Dataset(split, h.segment_size, h.sampling_rate,
                                split=False, shuffle=False, seed=h.seed),
                        num_workers=1, shuffle=False, sampler=None,
                        batch_size=1, pin_memory=True, drop_last=True)
    model.eval()
    audios_r, audios_g = [], []
    mag_err = 0.0
    for j, (clean_audio, noisy_audio) in enumerate(loader):
        clean_audio = clean_audio.to(device, non_blocking=True)
        noisy_audio = noisy_audio.to(device, non_blocking=True)
        clean_mag, _, _ = mag_pha_stft(clean_audio, h.n_fft, h.hop_size,
                                       h.win_size, h.compress_factor)
        noisy_mag, noisy_pha, _ = mag_pha_stft(noisy_audio, h.n_fft, h.hop_size,
                                               h.win_size, h.compress_factor)
        mag_g, pha_g, _ = model(noisy_mag, noisy_pha)
        audio_g = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size,
                                h.win_size, h.compress_factor)
        audios_r += torch.split(clean_audio, 1, dim=0)
        audios_g += torch.split(audio_g, 1, dim=0)
        mag_err += F.mse_loss(clean_mag, mag_g).item()
    return pesq_score(audios_r, audios_g, h).item(), mag_err / (j + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pattern", default="",
                    help="Magnitude-prune to this pattern before evaluating "
                         "(e.g. '2:4'). Default: evaluate as-is.")
    ap.add_argument("--axis", default="in", choices=("in", "out"))
    ap.add_argument("--scope", default="matrix", choices=("matrix", "row"))
    ap.add_argument("--hf_cache_dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    with open(a.config) as f:
        h = AttrDict(json.load(f))
    device = torch.device(a.device)

    model = NSNet2(h).to(device)
    model.load_state_dict(load_checkpoint(a.checkpoint, device)["generator"])

    label = "as-is"
    if a.pattern:
        ctrl = SparsityController(model, a.pattern, axis=a.axis, scope=a.scope)
        ctrl.apply()
        print(ctrl)
        label = a.pattern

    hf = load_voicebank_demand(cache_dir=a.hf_cache_dir)
    pesq, mag_err = evaluate(model, h, hf["test"], device)
    print(f"{a.checkpoint}  [{label}]  PESQ {pesq:.3f}  mag_mse {mag_err:.4f}")


if __name__ == "__main__":
    main()
