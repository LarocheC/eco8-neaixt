"""Parity: common/losses.py vs. a literal transcription of upstream MP-SENet.

The reference functions below are copy-pasted from
https://github.com/yxlu-0102/MP-SENet (``train.py`` generator step,
``models/model.py::phase_losses``, ``dataset.py::mag_pha_stft``) with only the
renaming needed to make them standalone. If a test here fails, our loss has
drifted from the paper's — not the other way round.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from common.dataset import mag_pha_stft
from common.losses import (
    MPSENET_WEIGHTS,
    anti_wrapping_function,
    discriminator_loss,
    generator_loss,
    generator_terms,
    mag_pha_from_complex,
    phase_losses,
)

TOL = dict(rtol=0, atol=1e-6)


# --------------------------------------------------------------------------
# Upstream reference (verbatim)
# --------------------------------------------------------------------------


def ref_anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def ref_phase_losses(phase_r, phase_g):
    ip_loss = torch.mean(ref_anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(ref_anti_wrapping_function(
        torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(ref_anti_wrapping_function(
        torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))
    return ip_loss, gd_loss, iaf_loss


def ref_mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
                           window=hann_window, center=center, pad_mode='reflect',
                           normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + (1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1] + (1e-10), stft_spec[:, :, :, 0] + (1e-5))
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def ref_generator_loss(clean_mag, clean_pha, clean_com, mag_g, pha_g, com_g,
                       com_g_hat, clean_audio, audio_g, metric_g, one_labels):
    """MP-SENet train.py, generator step — verbatim."""
    loss_mag = F.mse_loss(clean_mag, mag_g)
    loss_ip, loss_gd, loss_iaf = ref_phase_losses(clean_pha, pha_g)
    loss_pha = loss_ip + loss_gd + loss_iaf
    loss_com = F.mse_loss(clean_com, com_g) * 2
    loss_stft = F.mse_loss(com_g, com_g_hat) * 2
    loss_time = F.l1_loss(clean_audio, audio_g)
    loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

    loss_gen_all = (loss_mag * 0.9 + loss_pha * 0.3 + loss_com * 0.1
                    + loss_stft * 0.1 + loss_metric * 0.05 + loss_time * 0.2)
    return loss_gen_all


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

N_FFT, HOP, WIN, CF = 400, 100, 400, 0.3
B, SAMPLES = 3, 8000


@pytest.fixture
def batch():
    torch.manual_seed(0)
    clean_audio = torch.randn(B, SAMPLES) * 0.1
    # A plausible "prediction": the clean signal perturbed, so no term is zero.
    audio_g = clean_audio + torch.randn(B, SAMPLES) * 0.02
    return clean_audio, audio_g


# --------------------------------------------------------------------------
# Term-by-term parity
# --------------------------------------------------------------------------


def test_anti_wrapping_matches_upstream():
    x = torch.linspace(-20, 20, 1000)
    torch.testing.assert_close(anti_wrapping_function(x), ref_anti_wrapping_function(x), **TOL)


def test_anti_wrapping_is_periodic():
    """The whole point: an error of exactly 2*pi*k must cost nothing."""
    for k in (-2, -1, 0, 1, 2):
        err = anti_wrapping_function(torch.tensor(2 * np.pi * k))
        torch.testing.assert_close(err, torch.tensor(0.0), rtol=0, atol=1e-5)


def test_phase_losses_match_upstream():
    torch.manual_seed(1)
    pha_r = (torch.rand(B, 201, 80) - 0.5) * 4 * np.pi
    pha_g = (torch.rand(B, 201, 80) - 0.5) * 4 * np.pi
    got = phase_losses(pha_r, pha_g)
    want = ref_phase_losses(pha_r, pha_g)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, **TOL)


def test_mag_pha_from_complex_matches_stft_path(batch):
    """A model handing over its own complex spectrum must land on the same
    numbers as one that re-runs mag_pha_stft from the waveform."""
    clean_audio, _ = batch
    spec = torch.stft(clean_audio, N_FFT, hop_length=HOP, win_length=WIN,
                      window=torch.hann_window(WIN), center=True,
                      pad_mode="reflect", normalized=False, return_complex=True)
    mag_a, pha_a, com_a = mag_pha_from_complex(spec, CF)
    mag_b, pha_b, com_b = mag_pha_stft(clean_audio, N_FFT, HOP, WIN, CF)
    torch.testing.assert_close(mag_a, mag_b, **TOL)
    torch.testing.assert_close(pha_a, pha_b, **TOL)
    torch.testing.assert_close(com_a, com_b, **TOL)


def test_repo_mag_pha_stft_matches_upstream(batch):
    clean_audio, _ = batch
    got = mag_pha_stft(clean_audio, N_FFT, HOP, WIN, CF)
    want = ref_mag_pha_stft(clean_audio, N_FFT, HOP, WIN, CF)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, **TOL)


# --------------------------------------------------------------------------
# Whole-loss parity
# --------------------------------------------------------------------------


def test_generator_loss_matches_upstream_exactly(batch):
    """The full six-term objective, against upstream's verbatim expression."""
    clean_audio, audio_g = batch
    clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, N_FFT, HOP, WIN, CF)

    # Stand in for the network's direct spectral output (not the round trip).
    torch.manual_seed(2)
    mag_g = clean_mag + torch.randn_like(clean_mag) * 0.01
    pha_g = clean_pha + torch.randn_like(clean_pha) * 0.01
    com_g = torch.stack((mag_g * torch.cos(pha_g), mag_g * torch.sin(pha_g)), dim=-1)

    _, _, com_g_hat = mag_pha_stft(audio_g, N_FFT, HOP, WIN, CF)
    metric_g = torch.rand(B, 1)
    one_labels = torch.ones(B)

    terms = generator_terms(
        mag_g=mag_g, com_g=com_g, mag_r=clean_mag, com_r=clean_com,
        com_g_hat=com_g_hat, audio_g=audio_g, audio_r=clean_audio,
        pha_g=pha_g, pha_r=clean_pha, metric_g=metric_g,
    )
    got = generator_loss(terms, MPSENET_WEIGHTS)
    want = ref_generator_loss(clean_mag, clean_pha, clean_com, mag_g, pha_g, com_g,
                              com_g_hat, clean_audio, audio_g, metric_g, one_labels)
    torch.testing.assert_close(got, want, **TOL)


def test_complex_and_consistency_carry_the_factor_two(batch):
    """Upstream's `* 2` lives inside the term, on top of the 0.1 weight —
    effective coefficient 0.2. Easy to drop; assert it explicitly."""
    clean_audio, audio_g = batch
    mag_r, pha_r, com_r = mag_pha_stft(clean_audio, N_FFT, HOP, WIN, CF)
    mag_g, _, com_g = mag_pha_stft(audio_g, N_FFT, HOP, WIN, CF)
    _, _, com_g_hat = mag_pha_stft(audio_g, N_FFT, HOP, WIN, CF)

    terms = generator_terms(mag_g=mag_g, com_g=com_g, mag_r=mag_r, com_r=com_r,
                            com_g_hat=com_g_hat, audio_g=audio_g, audio_r=clean_audio)
    torch.testing.assert_close(terms["complex"], F.mse_loss(com_r, com_g) * 2, **TOL)
    torch.testing.assert_close(terms["consistency"], F.mse_loss(com_g, com_g_hat) * 2, **TOL)


def test_phase_term_omitted_when_no_phase_decoder(batch):
    """Mask-only models pass pha_g=None: the term must vanish, not contribute a
    gradient-free constant."""
    clean_audio, audio_g = batch
    mag_r, _, com_r = mag_pha_stft(clean_audio, N_FFT, HOP, WIN, CF)
    mag_g, _, com_g = mag_pha_stft(audio_g, N_FFT, HOP, WIN, CF)

    terms = generator_terms(mag_g=mag_g, com_g=com_g, mag_r=mag_r, com_r=com_r,
                            com_g_hat=com_g, audio_g=audio_g, audio_r=clean_audio)
    assert "phase" not in terms
    assert "metric" not in terms  # metric_g=None too

    total = generator_loss(terms, MPSENET_WEIGHTS)
    want = (0.9 * terms["magnitude"] + 0.1 * terms["complex"]
            + 0.1 * terms["consistency"] + 0.2 * terms["time"])
    torch.testing.assert_close(total, want, **TOL)


def test_consistency_is_zero_for_a_round_tripped_spectrum(batch):
    """Sanity check on what the consistency term means: a spectrum that already
    is the STFT of its own waveform is perfectly consistent."""
    _, audio_g = batch
    _, _, com_g = mag_pha_stft(audio_g, N_FFT, HOP, WIN, CF)
    terms = generator_terms(mag_g=com_g[..., 0], com_g=com_g, mag_r=com_g[..., 0],
                            com_r=com_g, com_g_hat=com_g, audio_g=audio_g, audio_r=audio_g)
    torch.testing.assert_close(terms["consistency"], torch.tensor(0.0), rtol=0, atol=1e-9)


def test_weights_are_the_paper_values():
    assert MPSENET_WEIGHTS == {
        "magnitude": 0.9, "phase": 0.3, "complex": 0.1,
        "consistency": 0.1, "time": 0.2, "metric": 0.05,
    }


# --------------------------------------------------------------------------
# Discriminator
# --------------------------------------------------------------------------


def test_discriminator_loss_matches_upstream():
    from common.discriminator import MetricDiscriminator

    torch.manual_seed(3)
    # eval(): the discriminator has Dropout and spectral-norm power iteration, so
    # in train mode two forward passes over the same input don't agree and there
    # is nothing to compare against.
    disc = MetricDiscriminator(dim=16).eval()
    mag_r = torch.rand(B, 201, 80)
    mag_g_hat = torch.rand(B, 201, 80)
    pesq_target = torch.rand(B)

    got = discriminator_loss(disc, mag_r, mag_g_hat, pesq_target, torch.device("cpu"))

    one_labels = torch.ones(B)
    metric_r = disc(mag_r, mag_r)
    metric_g = disc(mag_r, mag_g_hat.detach())
    want = (F.mse_loss(one_labels, metric_r.flatten())
            + F.mse_loss(pesq_target, metric_g.flatten()))
    torch.testing.assert_close(got, want, **TOL)


def test_discriminator_drops_fake_branch_when_pesq_failed():
    """batch_pesq returns None if PESQ failed on any utterance; upstream keeps
    the real branch and zeroes the fake one rather than skipping the step."""
    from common.discriminator import MetricDiscriminator

    torch.manual_seed(4)
    disc = MetricDiscriminator(dim=16).eval()
    mag_r = torch.rand(B, 201, 80)
    mag_g_hat = torch.rand(B, 201, 80)

    got = discriminator_loss(disc, mag_r, mag_g_hat, None, torch.device("cpu"))
    metric_r = disc(mag_r, mag_r)
    want = F.mse_loss(torch.ones(B), metric_r.flatten())
    torch.testing.assert_close(got, want, **TOL)
