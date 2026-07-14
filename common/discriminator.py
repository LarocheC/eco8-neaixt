import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pesq import pesq
from torch.nn.utils import spectral_norm
from joblib import Parallel, delayed
from common.utils import *


def cal_pesq(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=-1)(delayed(cal_pesq)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score)


# ---------------------------------------------------------------------------
# MetricGAN target: the fast GPU path
# ---------------------------------------------------------------------------
#
# `batch_pesq` above runs the reference ITU-T P.862 implementation on the CPU,
# once per utterance, on EVERY training step. Measured on a batch of 16 x 2 s at
# 16 kHz it costs ~400 ms/step even with all 32 cores free — about 4.8 min of a
# ~5 min LiSenNet epoch, and it gets worse under concurrency because each
# trainer's joblib pool competes for the same cores.
#
# torch-pesq (https://github.com/audiolabs/torch-pesq) computes the same score on
# the GPU, on tensors that are already there. Measured: ~36 ms/step, an 11x
# speedup on the same batch.
#
# It is NOT the reference implementation — it skips time alignment and does level
# alignment with IIR filtering rather than frequency weighting. That is fine HERE
# and only here: this value is the discriminator's regression *target*, and
# MetricGAN only needs a signal that tracks quality monotonically. Measured
# against ITU on 200 real VBD pairs spanning noisy->clean: Pearson 0.999,
# Spearman 0.997, mean |err| 0.043, identical range.
#
# The number that gets REPORTED is a different thing entirely and must stay the
# reference implementation — `common.metrics.pesq_score` is untouched. It gates
# g_best and is what RESULTS_*.md compares against published baselines, so
# swapping it would silently redefine the metric.

_TORCH_PESQ = {}


def _torch_pesq_model(device, sample_rate):
    """One PesqLoss per (device, rate) — it builds filterbanks, so don't rebuild
    it 700 times an epoch."""
    key = (str(device), int(sample_rate))
    if key not in _TORCH_PESQ:
        from torch_pesq import PesqLoss
        _TORCH_PESQ[key] = PesqLoss(0.5, sample_rate=int(sample_rate)).to(device).eval()
    return _TORCH_PESQ[key]


@torch.no_grad()
def batch_pesq_torch(clean, deg, sample_rate=16000):
    """GPU MetricGAN target. Same contract as ``batch_pesq``: a ``(B,)`` tensor of
    normalised ``(pesq - 1) / 3.5``, or ``None`` if any utterance failed.

    Takes waveform TENSORS, not numpy lists — staying on the GPU is half the win.

    The None path is load-bearing and is why this is not a one-line swap. Where
    the reference implementation returns -1 (silence, too-short), torch-pesq
    instead returns **NaN** — or, for a too-short input, **raises**. Handing that
    NaN to the discriminator's MSE would NaN training outright. Reproducing
    batch_pesq's all-or-nothing None keeps the trainers' existing "skip the fake
    branch this step" handling correct and makes this a true drop-in.
    """
    try:
        mos = _torch_pesq_model(clean.device, sample_rate).mos(clean.float(), deg.float())
    except RuntimeError:
        return None                      # too short for the filterbank
    if not torch.isfinite(mos).all():
        return None                      # silence -> NaN (ITU returns -1 here)
    return ((mos - 1.0) / 3.5).float()


def make_pesq_target(backend="torch", sample_rate=16000):
    """Pick the MetricGAN target function. ``fn(clean_wav, deg_wav) -> (B,) | None``.

    backend="torch": torch-pesq on the GPU (default; ~11x faster).
    backend="itu"  : the reference CPU implementation — bit-identical to what the
                     published checkpoints were trained against. Use this to
                     reproduce them exactly, or to rule the approximation out if
                     a run looks off.
    """
    if backend == "torch":
        return lambda c, d: batch_pesq_torch(c, d, sample_rate)
    if backend == "itu":
        return lambda c, d: batch_pesq(list(c.detach().cpu().numpy()),
                                       list(d.detach().cpu().numpy()))
    raise ValueError(f"unknown pesq backend {backend!r} (want 'torch' or 'itu')")


def pesq_target_from_config(h):
    """Build the MetricGAN target fn from a config's ``gan.pesq_backend``
    (default "torch"). Only affects the discriminator's target — never the
    reported PESQ."""
    gan = (h.get("gan", {}) or {}) if hasattr(h, "get") else {}
    backend = gan.get("pesq_backend", "torch")
    return make_pesq_target(backend, int(h.get("sampling_rate", 16000)))


def metric_loss(metric_ref, metrics_gen):
    loss = 0
    for metric_gen in metrics_gen:
        term = F.mse_loss(metric_ref, metric_gen.flatten())
        loss += term

    return loss


class MetricDiscriminator(nn.Module):
    def __init__(self, dim=16, in_channel=2):
        super(MetricDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim*2, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*2, affine=True),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Conv2d(dim*2, dim*4, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*4, affine=True),
            nn.PReLU(dim*4),
            nn.utils.spectral_norm(nn.Conv2d(dim*4, dim*8, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*8, affine=True),
            nn.PReLU(dim*8),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim*8, dim*4)),
            nn.Dropout(0.3),
            nn.PReLU(dim*4),
            nn.utils.spectral_norm(nn.Linear(dim*4, 1)),
            LearnableSigmoid1d(1)
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)
