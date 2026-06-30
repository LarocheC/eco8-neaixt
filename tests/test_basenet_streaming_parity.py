"""Frame-by-frame streaming parity for the causal BASENet.

The load-bearing test: feeding the model one STFT frame at a time (with the
TFConv ring buffers + GRU hidden state) reproduces the offline whole-utterance
forward to floating-point tolerance. That is what makes the model genuinely
deployable for real-time streaming. Also checks that toggling streaming leaves
the offline path untouched and that reset() makes runs independent.
"""

from __future__ import annotations

import math

import pytest
import torch

from basenet.model import BASENet
from basenet.streaming import (
    BASENetStreamer, disable_streaming, enable_streaming, stream_utterance,
)


SMALL = dict(base_channels=16, crn_hidden=48, crn_freq=8, dec_depth=1)


def _spec(b=1, f=201, n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    mag = torch.rand(b, f, n, generator=g)
    pha = (torch.rand(b, f, n, generator=g) * 2 - 1) * math.pi
    return mag, pha


def _complex(mag, pha):
    return torch.stack([mag * torch.cos(pha), mag * torch.sin(pha)], dim=-1)


def test_streaming_matches_offline():
    torch.manual_seed(0)
    model = BASENet(causal=True, **SMALL).eval()
    mag, pha = _spec(1, model.n_features, 40, seed=1)

    with torch.no_grad():
        off_mag, off_pha, _ = model(mag, pha)
    s_mag, s_pha, _ = stream_utterance(model, mag, pha)

    # The TFConv ring buffers are bit-exact; the only residual is nn.GRU using
    # different kernels for a full sequence vs single-step recurrence (~1e-6),
    # which atan2 amplifies a little in the phase. The complex spectrogram (what
    # gets iSTFT'd) stays within 1e-4 — far below audibility.
    mag_err = (off_mag - s_mag).abs().max().item()
    assert mag_err < 1e-4, f"streaming magnitude diverged: {mag_err:.3e}"
    cplx_err = (_complex(off_mag, off_pha) - _complex(s_mag, s_pha)).abs().max().item()
    assert cplx_err < 1e-4, f"streaming complex spectrogram diverged: {cplx_err:.3e}"


def test_streaming_matches_offline_batched():
    """Parity must hold for B>1 too (independent streams in a batch)."""
    torch.manual_seed(0)
    model = BASENet(causal=True, **SMALL).eval()
    mag, pha = _spec(3, model.n_features, 25, seed=2)
    with torch.no_grad():
        off_mag, off_pha, _ = model(mag, pha)
    s_mag, s_pha, _ = stream_utterance(model, mag, pha)
    assert torch.allclose(off_mag, s_mag, atol=1e-4)
    assert torch.allclose(_complex(off_mag, off_pha), _complex(s_mag, s_pha), atol=1e-4)


def test_toggling_streaming_leaves_offline_unchanged():
    torch.manual_seed(0)
    model = BASENet(causal=True, **SMALL).eval()
    mag, pha = _spec(1, model.n_features, 20, seed=3)
    with torch.no_grad():
        before, _, _ = model(mag, pha)
    enable_streaming(model)
    disable_streaming(model)
    with torch.no_grad():
        after, _, _ = model(mag, pha)
    assert torch.allclose(before, after), "offline forward changed after streaming toggle"


def test_reset_makes_runs_independent():
    torch.manual_seed(0)
    model = BASENet(causal=True, **SMALL).eval()
    mag, pha = _spec(1, model.n_features, 16, seed=4)
    streamer = BASENetStreamer(model)

    def run():
        streamer.reset()
        return torch.stack(
            [streamer.step(mag[:, :, t], pha[:, :, t])[0] for t in range(mag.shape[-1])],
            dim=-1,
        )

    out1, out2 = run(), run()
    assert torch.allclose(out1, out2), "reset() did not clear streaming state"


def test_step_frame_shapes():
    torch.manual_seed(0)
    model = BASENet(causal=True, **SMALL).eval()
    streamer = BASENetStreamer(model)
    streamer.reset()
    f = model.n_features
    mag_g, pha_g = streamer.step(torch.rand(2, f), torch.rand(2, f))
    assert mag_g.shape == (2, f) and pha_g.shape == (2, f)
    # (B, F, 1) input is accepted too and preserved.
    mag_g3, _ = streamer.step(torch.rand(2, f, 1), torch.rand(2, f, 1))
    assert mag_g3.shape == (2, f, 1)
