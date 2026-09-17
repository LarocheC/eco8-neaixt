"""The MP-SENet training loss, ported from https://github.com/yxlu-0102/MP-SENet.

This is the one canonical implementation in the repo — BASENet, ConvFSENet,
LiSenNet and NSNet2 all train against it, so a change here changes every model.

Upstream reference (MP-SENet ``train.py``, generator step)::

    loss_mag    = F.mse_loss(clean_mag, mag_g)
    loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g)
    loss_pha    = loss_ip + loss_gd + loss_iaf
    loss_com    = F.mse_loss(clean_com, com_g) * 2
    loss_stft   = F.mse_loss(com_g, com_g_hat) * 2
    loss_time   = F.l1_loss(clean_audio, audio_g)
    metric_g    = discriminator(clean_mag, mag_g_hat)
    loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

    loss_gen_all = (loss_mag * 0.9 + loss_pha * 0.3 + loss_com * 0.1
                    + loss_stft * 0.1 + loss_metric * 0.05 + loss_time * 0.2)

Three conventions from upstream are easy to get wrong, so they are pinned here:

1. The ``* 2`` on the complex and consistency terms is *inside* the term, and is
   applied on top of the 0.1 weight — the effective coefficient of each is 0.2.
2. ``com_g_hat`` is the spectrogram of the model's own output waveform, i.e. the
   result of an iSTFT -> STFT round trip. The consistency term is what stops the
   network emitting a (magnitude, phase) pair that no real waveform can produce.
3. The discriminator is shown ``mag_g_hat`` — the *round-tripped* magnitude — not
   the network's raw magnitude output. Both the discriminator step and the
   generator's metric term use it.

All magnitude/complex tensors are power-law compressed (``compress_factor``,
0.3 upstream); ``common.dataset.mag_pha_stft`` produces that representation and
``mag_pha_from_complex`` below converts an already-computed complex STFT into it
using the identical epsilon conventions.

Layout: upstream tensors are ``(B, F, T)``. Every term here except ``phase`` is
an elementwise mean, so it is layout-agnostic; ``phase_losses`` is not, and takes
explicit ``freq_dim`` / ``time_dim`` arguments.

Phase loss and mask-only models
-------------------------------
The anti-wrapping phase loss only trains a model that *predicts* phase. Of the
four models here only BASENet has a phase decoder — ConvFSENet, LiSenNet and
NSNet2 all reuse the noisy phase (LiSenNet refines it with Griffin-Lim seeded
from a detached magnitude, which carries no gradient either). For those, pass
``pha_g=None`` and the term is dropped rather than silently contributing a
constant with no gradient.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_TWO_PI = 2.0 * math.pi

# Upstream MP-SENet's generator weights (train.py, `loss_gen_all`).
MPSENET_WEIGHTS = {
    "magnitude": 0.9,
    "phase": 0.3,
    "complex": 0.1,
    "consistency": 0.1,
    "time": 0.2,
    "metric": 0.05,
}


def loss_weights(h, defaults=None):
    """Read a config's ``loss`` block, falling back to the MP-SENet weights.

    Unknown keys are an error, not a silent no-op: the pre-port configs used
    ``mag``/``adv`` for what are now ``magnitude``/``metric``, and quietly
    ignoring those would train a model at the default weights while the config
    on disk claimed otherwise.
    """
    cfg = (h.get("loss", {}) or {}) if hasattr(h, "get") else {}
    base = defaults or MPSENET_WEIGHTS
    unknown = set(cfg) - set(base)
    if unknown:
        raise ValueError(
            f"unknown loss weight(s) {sorted(unknown)} in config; "
            f"expected any of {sorted(base)}"
        )
    return {k: float(cfg.get(k, v)) for k, v in base.items()}


# ---------------------------------------------------------------------------
# Spectral representation
# ---------------------------------------------------------------------------


def mag_pha_from_complex(spec, compress_factor=1.0):
    """Compressed (mag, pha, com) from an already-computed complex STFT.

    The magnitude/phase epsilons match ``common.dataset.mag_pha_stft`` exactly
    (1e-9 under the sqrt; 1e-10 / 1e-5 in the atan2), so a model that hands over
    its own complex spectrum lands on the same numbers as one that re-runs the
    STFT from the waveform. Use this rather than ``spec.abs()`` / ``spec.angle()``
    when the value feeds the loss.
    """
    spec = torch.view_as_real(spec) if spec.is_complex() else spec
    mag = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    pha = torch.atan2(spec[..., 1] + 1e-10, spec[..., 0] + 1e-5)
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


# ---------------------------------------------------------------------------
# Phase loss (BASENet only — see module docstring)
# ---------------------------------------------------------------------------


def anti_wrapping_function(x):
    """``|x - round(x / 2pi) * 2pi|`` — the error to the nearest equivalent angle.

    Without this a phase error of ``2pi + eps`` would be penalised as a huge
    error rather than as ``eps``, and the network would be pushed to match a
    branch cut instead of an angle.
    """
    return torch.abs(x - torch.round(x / _TWO_PI) * _TWO_PI)


def phase_losses(phase_r, phase_g, freq_dim=1, time_dim=2):
    """Anti-wrapping phase loss, split into MP-SENet's three terms.

    IP  — instantaneous phase: the anti-wrapped angle error itself.
    GD  — group delay: the error in the derivative along frequency.
    IAF — instantaneous angular frequency: the error in the derivative along time.

    GD and IAF are what make the loss care about phase *structure* (formant
    alignment, frame-to-frame continuity) rather than only pointwise angle.
    """
    ip = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd = torch.mean(anti_wrapping_function(
        torch.diff(phase_r, dim=freq_dim) - torch.diff(phase_g, dim=freq_dim)))
    iaf = torch.mean(anti_wrapping_function(
        torch.diff(phase_r, dim=time_dim) - torch.diff(phase_g, dim=time_dim)))
    return ip, gd, iaf


# ---------------------------------------------------------------------------
# Generator loss
# ---------------------------------------------------------------------------


def generator_terms(mag_g, com_g, mag_r, com_r, com_g_hat, audio_g, audio_r,
                    pha_g=None, pha_r=None, metric_g=None,
                    freq_dim=1, time_dim=2):
    """The unweighted MP-SENet loss terms, as tensors with grad.

    Args:
        mag_g, com_g: the network's *direct* compressed magnitude / complex
            output — not the round trip. ``com_g`` is ``(..., 2)`` real.
        mag_r, com_r: the same for the clean target.
        com_g_hat: compressed complex spectrum of ``audio_g`` (the iSTFT ->
            STFT round trip). This is what makes ``consistency`` meaningful.
        audio_g, audio_r: predicted / clean waveforms, already length-matched.
        pha_g, pha_r: predicted / clean phase. Pass ``pha_g=None`` for a model
            with no phase decoder and the phase term is omitted entirely.
        metric_g: the discriminator's output on ``(mag_r, mag_g_hat)``, or None
            to train without the MetricGAN term.

    Returns a dict of scalar tensors. ``complex`` and ``consistency`` already
    carry upstream's ``* 2``; the caller applies :data:`MPSENET_WEIGHTS`.
    """
    terms = {
        "magnitude": F.mse_loss(mag_r, mag_g),
        "complex": F.mse_loss(com_r, com_g) * 2.0,
        "consistency": F.mse_loss(com_g, com_g_hat) * 2.0,
        "time": F.l1_loss(audio_r, audio_g),
    }

    if pha_g is not None:
        ip, gd, iaf = phase_losses(pha_r, pha_g, freq_dim=freq_dim, time_dim=time_dim)
        terms["phase"] = ip + gd + iaf
        terms["ip"], terms["gd"], terms["iaf"] = ip, gd, iaf

    if metric_g is not None:
        metric_g = metric_g.flatten()
        terms["metric"] = F.mse_loss(metric_g, torch.ones_like(metric_g))

    return terms


def generator_loss(terms, weights=None):
    """Weighted sum of whichever of the MP-SENet terms are present.

    Sub-terms of the phase loss (``ip``/``gd``/``iaf``) are logging-only and are
    not summed again — ``phase`` already holds them.
    """
    weights = weights or MPSENET_WEIGHTS
    total = None
    for name, weight in weights.items():
        term = terms.get(name)
        if term is None or weight == 0.0:
            continue
        contrib = weight * term
        total = contrib if total is None else total + contrib
    if total is None:
        raise ValueError("no loss terms to sum — check the config's `loss` block")
    return total


# ---------------------------------------------------------------------------
# MetricGAN discriminator loss
# ---------------------------------------------------------------------------


def discriminator_loss(discriminator, mag_r, mag_g_hat, pesq_target, device):
    """MetricGAN: map (clean, x) -> normalised PESQ. Clean-vs-clean is 1.

    ``pesq_target`` comes from ``common.discriminator.batch_pesq`` — normalised
    ``(pesq - 1) / 3.5`` — and is ``None`` when PESQ failed on any utterance in
    the batch (silence, too short). Upstream drops the fake branch for that
    batch rather than the whole step, so we do too.

    ``mag_g_hat`` must be the round-tripped magnitude, matching upstream.
    """
    metric_r = discriminator(mag_r, mag_r).flatten()
    loss_r = F.mse_loss(torch.ones_like(metric_r), metric_r)

    if pesq_target is None:
        loss_g = torch.zeros((), device=device)
    else:
        metric_g = discriminator(mag_r, mag_g_hat.detach()).flatten()
        loss_g = F.mse_loss(pesq_target.to(device), metric_g)

    return loss_r + loss_g
