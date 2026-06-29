"""MP-SENet training losses for BASENet.

The BASENet paper (arXiv:2606.12662, Sec. 2.1) states it "follow[s] the same
training procedure and loss functions as MP-SENet" (Lu et al., 2023). Those are:

  * **magnitude** loss  — L2 on the power-law-compressed magnitude,
  * **phase** loss      — anti-wrapping, decomposed into instantaneous-phase (IP),
                          group-delay (GD) and instantaneous-angular-frequency
                          (IAF) terms so equivalent angles 2*pi*k apart are not
                          penalised,
  * **complex** loss    — L2 on the (real, imag) compressed spectrogram (x2),

combined in the trainer with a time-domain L1 loss and a MetricGAN term.

Tensors are ``(B, F, T)`` with F = frequency bins (dim 1) and T = time frames
(dim 2), matching ``common.dataset.mag_pha_stft``.
"""

import math

import torch
import torch.nn.functional as F

_TWO_PI = 2.0 * math.pi


def anti_wrapping_function(x):
    """Wrap a phase difference into (-pi, pi] before taking its magnitude.

    ``|x - round(x / 2pi) * 2pi|`` — so a difference of, say, ``2pi + 0.1`` costs
    ``0.1`` rather than ``2pi + 0.1``. This is what makes the phase loss respect
    the circular topology of angles.
    """
    return torch.abs(x - torch.round(x / _TWO_PI) * _TWO_PI)


def phase_losses(phase_r, phase_g):
    """Anti-wrapping phase loss split into IP / GD / IAF terms.

    IP  — instantaneous phase: the direct (anti-wrapped) angle difference.
    GD  — group delay: difference of the freq-axis derivative (``diff`` over F).
    IAF — instantaneous angular frequency: difference of the time-axis
          derivative (``diff`` over T).

    Returns the three scalar losses; the trainer sums them into the phase loss.
    """
    ip = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd = torch.mean(anti_wrapping_function(
        torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf = torch.mean(anti_wrapping_function(
        torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))
    return ip, gd, iaf


def spectral_losses(mag_g, pha_g, com_g, mag_r, pha_r, com_r):
    """Magnitude (L2), phase (anti-wrapping, summed) and complex (2x L2) losses.

    All magnitude/complex tensors are expected power-law compressed (the
    representation ``mag_pha_stft`` returns). Returns a dict with the three
    aggregate losses plus the phase sub-terms for logging. The factor 2 on the
    complex term mirrors MP-SENet's weighting.
    """
    loss_mag = F.mse_loss(mag_g, mag_r)
    ip, gd, iaf = phase_losses(pha_r, pha_g)
    loss_pha = ip + gd + iaf
    loss_com = F.mse_loss(com_g, com_r) * 2.0
    return {
        "magnitude": loss_mag,
        "phase": loss_pha,
        "complex": loss_com,
        "ip": ip, "gd": gd, "iaf": iaf,
    }
