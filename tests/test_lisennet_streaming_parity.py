"""Frame-by-frame streaming parity for LiSenNet.

The load-bearing test: feeding the mask sub-network one STFT frame at a time
(with the causal-conv ring buffers + the DPR inter-time GRU hidden state, plus
the one-frame phase state for the IFD feature) reproduces the offline
whole-utterance ``predict_mask`` + ``apply_mask`` to floating-point tolerance.
That is what makes LiSenNet genuinely deployable for real-time streaming. Also
checks that toggling streaming leaves the offline path untouched and that
``reset()`` makes runs independent.
"""

from __future__ import annotations

import torch

from lisennet.model import build_lisennet
from lisennet.streaming import (
    LiSenNetStreamer, disable_streaming, enable_streaming, stream_est_mag,
)


SMALL = dict(num_channels=16, n_blocks=2)


def _spec(b=1, n=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    model_freqs = 257
    mag = torch.rand(b, n, model_freqs, generator=g)
    pha = (torch.rand(b, n, model_freqs, generator=g) * 2 - 1) * torch.pi
    return mag, pha


def _offline_est_mag(model, mag, pha):
    feat = model.build_features(mag, pha)
    with torch.no_grad():
        return model.apply_mask(model.predict_mask(feat), mag)


def test_streaming_matches_offline():
    torch.manual_seed(0)
    model = build_lisennet(SMALL).eval()
    mag, pha = _spec(1, 40, seed=1)

    off = _offline_est_mag(model, mag, pha)
    s = stream_est_mag(model, mag, pha)

    # Causal-conv ring buffers + the IFD phase state are bit-exact; the only
    # residual is nn.GRU using different kernels for a full sequence vs single-
    # step recurrence (~1e-6). Far below audibility.
    err = (off - s).abs().max().item()
    assert err < 1e-4, f"streaming est_mag diverged: {err:.3e}"


def test_streaming_matches_offline_batched():
    """Parity must hold for B>1 too (independent streams in a batch)."""
    torch.manual_seed(0)
    model = build_lisennet(SMALL).eval()
    mag, pha = _spec(3, 25, seed=2)
    off = _offline_est_mag(model, mag, pha)
    s = stream_est_mag(model, mag, pha)
    assert torch.allclose(off, s, atol=1e-4)


def test_toggling_streaming_leaves_offline_unchanged():
    torch.manual_seed(0)
    model = build_lisennet(SMALL).eval()
    mag, pha = _spec(1, 20, seed=3)
    before = _offline_est_mag(model, mag, pha)
    enable_streaming(model)
    disable_streaming(model)
    after = _offline_est_mag(model, mag, pha)
    assert torch.allclose(before, after), "offline forward changed after streaming toggle"


def test_reset_makes_runs_independent():
    torch.manual_seed(0)
    model = build_lisennet(SMALL).eval()
    mag, pha = _spec(1, 16, seed=4)
    streamer = LiSenNetStreamer(model)

    def run():
        streamer.reset()
        return torch.stack(
            [streamer.step(mag[:, t], pha[:, t]) for t in range(mag.shape[1])], dim=1)

    out1, out2 = run(), run()
    assert torch.allclose(out1, out2), "reset() did not clear streaming state"


def test_step_frame_shapes():
    torch.manual_seed(0)
    model = build_lisennet(SMALL).eval()
    streamer = LiSenNetStreamer(model)
    streamer.reset()
    f = model.n_freqs
    est = streamer.step(torch.rand(2, f), torch.rand(2, f))
    assert est.shape == (2, f)
    # (B, 1, F) input is accepted too and preserved.
    est3 = streamer.step(torch.rand(2, 1, f), torch.rand(2, 1, f))
    assert est3.shape == (2, 1, f)
