"""Frame-by-frame streaming parity for FSPEN.

The load-bearing test: feeding the enhancer one STFT frame at a time (carrying
only the DPE inter-GRU hidden states) reproduces the offline whole-utterance
``enhance_spectrum`` to floating-point tolerance. FSPEN is streaming-native —
everything else in the model is per-frame — so this is what makes it genuinely
deployable for real-time use. Also checks that ``reset()`` makes runs
independent and that carried state actually matters.
"""

from __future__ import annotations

import torch

from fspen.model import build_fspen
from fspen.streaming import FSPENStreamer, stream_enhance_spectrum


def _spec(b=1, n=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, 2, 257, generator=g)


def _offline(model, spec):
    amp = torch.sqrt(spec[:, :, 0:1] ** 2 + spec[:, :, 1:2] ** 2)
    with torch.no_grad():
        est, _ = model.enhance_spectrum(spec, amp)
    return est


def test_streaming_matches_offline():
    torch.manual_seed(0)
    model = build_fspen().eval()
    spec = _spec(1, 40, seed=1)

    off = _offline(model, spec)
    s = stream_enhance_spectrum(model, spec)

    # The only residual is nn.GRU using different kernels for a full sequence
    # vs single-step recurrence (~1e-6). Far below audibility.
    err = (off - s).abs().max().item()
    assert err < 1e-4, f"streaming est_spec diverged: {err:.3e}"


def test_streaming_matches_offline_batched():
    """Parity must hold for B>1 too (independent streams in a batch)."""
    torch.manual_seed(0)
    model = build_fspen().eval()
    spec = _spec(3, 25, seed=2)
    off = _offline(model, spec)
    s = stream_enhance_spectrum(model, spec)
    assert torch.allclose(off, s, atol=1e-4)


def test_reset_makes_runs_independent():
    torch.manual_seed(0)
    model = build_fspen().eval()
    spec = _spec(1, 16, seed=3)
    streamer = FSPENStreamer(model)

    def run():
        streamer.reset()
        return torch.stack(
            [streamer.step(spec[:, t]) for t in range(spec.shape[1])], dim=1)

    out1, out2 = run(), run()
    assert torch.allclose(out1, out2), "reset() did not clear streaming state"


def test_state_carry_matters():
    """Propagated inter-GRU state must change later frames (vs resetting each frame)."""
    torch.manual_seed(0)
    model = build_fspen().eval()
    spec = _spec(1, 12, seed=4)
    streamer = FSPENStreamer(model)

    streamer.reset()
    carried = [streamer.step(spec[:, t]) for t in range(spec.shape[1])]

    stateless = []
    for t in range(spec.shape[1]):
        streamer.reset()                                     # zero state every frame
        stateless.append(streamer.step(spec[:, t]))

    assert torch.allclose(carried[0], stateless[0]), "frame 0 must not depend on state"
    later_diff = max((c - s).abs().max().item()
                     for c, s in zip(carried[1:], stateless[1:]))
    assert later_diff > 1e-6, "carried state had no effect — inter GRUs not stateful?"


def test_step_frame_shapes():
    torch.manual_seed(0)
    model = build_fspen().eval()
    streamer = FSPENStreamer(model)
    streamer.reset()
    f = model.n_freqs
    est = streamer.step(torch.randn(2, 2, f))
    assert est.shape == (2, 2, f)
    # (B, 1, 2, F) input is accepted too and preserved.
    est4 = streamer.step(torch.randn(2, 1, 2, f))
    assert est4.shape == (2, 1, 2, f)
