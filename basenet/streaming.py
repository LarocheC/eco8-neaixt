"""Frame-by-frame (streaming) inference for the causal BASENet.

Real-time deployment processes one STFT frame at a time with bounded state
instead of a whole utterance. BASENet is built for this: every operation is
per-frame *except* the causal time convolutions and the CRN's GRU. Turning on
streaming makes each ``TFConv`` keep a ``(kt-1)``-frame ring buffer and the CRN
carry its GRU hidden state, so feeding the model one frame at a time reproduces
the offline whole-utterance forward exactly (verified in
``tests/test_basenet_streaming_parity.py``).

Everything else — cross-band attention, frequency-only normalisation, SE,
FreqResample, GLU, the 1x1 projections — is already per-frame, so the entire
offline ``forward`` is reused unchanged; only the two stateful primitives flip
into streaming mode.

Usage::

    streamer = BASENetStreamer(model)        # model: BASENet, causal
    streamer.reset()
    for t in range(n_frames):
        mag_g_t, pha_g_t = streamer.step(mag[:, :, t], pha[:, :, t])

This is the PyTorch streaming reference; an ONNX graph with explicit state I/O
(for edge runtimes) is the natural follow-up.
"""

from __future__ import annotations

import torch

from basenet.model import CRN, BASENet, TFConv

# The only modules that carry temporal state.
_STREAM_MODULES = (TFConv, CRN)


def enable_streaming(model: BASENet) -> None:
    """Flip the causal time-conv buffers and the GRU into streaming mode (and reset)."""
    assert getattr(model, "causal", False), "streaming requires a causal model"
    for m in model.modules():
        if isinstance(m, _STREAM_MODULES):
            m.streaming = True
    reset_streaming(model)


def disable_streaming(model: BASENet) -> None:
    """Return the model to offline (whole-utterance) behaviour."""
    for m in model.modules():
        if isinstance(m, _STREAM_MODULES):
            m.streaming = False


def reset_streaming(model: BASENet) -> None:
    """Clear all per-stream state so the next frame starts a fresh utterance."""
    for m in model.modules():
        if isinstance(m, _STREAM_MODULES):
            m.stream_reset()


class BASENetStreamer:
    """Online wrapper: ``reset()`` once, then ``step()`` one STFT frame at a time."""

    def __init__(self, model: BASENet):
        assert model.causal, "BASENetStreamer requires a causal BASENet"
        self.model = model.eval()
        enable_streaming(self.model)

    def reset(self) -> None:
        reset_streaming(self.model)

    @torch.no_grad()
    def step(self, mag_frame, pha_frame):
        """One frame in -> one frame out.

        Accepts ``(B, F)`` or ``(B, F, 1)`` for the current STFT frame's
        compressed magnitude and phase; returns the enhanced ``(mag_g, pha_g)``
        in the same (squeezed) layout.
        """
        squeeze = mag_frame.dim() == 2
        if squeeze:
            mag_frame = mag_frame.unsqueeze(-1)
            pha_frame = pha_frame.unsqueeze(-1)
        mag_g, pha_g, _ = self.model(mag_frame, pha_frame)      # (B, F, 1)
        if squeeze:
            return mag_g.squeeze(-1), pha_g.squeeze(-1)
        return mag_g, pha_g

    def close(self) -> None:
        """Restore offline behaviour (e.g. before training / export)."""
        disable_streaming(self.model)


@torch.no_grad()
def stream_utterance(model: BASENet, mag, pha):
    """Run a full ``(B, F, N)`` utterance frame-by-frame.

    Returns ``(mag_g, pha_g, com_g)`` like ``BASENet.forward``. Mainly for parity
    checks against the offline forward; real-time callers use ``BASENetStreamer``.
    """
    streamer = BASENetStreamer(model)
    streamer.reset()
    mags, phas = [], []
    for t in range(mag.shape[-1]):
        mag_g_t, pha_g_t = streamer.step(mag[:, :, t], pha[:, :, t])
        mags.append(mag_g_t)
        phas.append(pha_g_t)
    streamer.close()
    mag_g = torch.stack(mags, dim=-1)
    pha_g = torch.stack(phas, dim=-1)
    com_g = torch.stack((mag_g * torch.cos(pha_g), mag_g * torch.sin(pha_g)), dim=-1)
    return mag_g, pha_g, com_g
