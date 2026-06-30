"""Frame-by-frame (streaming) inference for LiSenNet.

Real-time deployment processes one STFT frame at a time with bounded state
instead of a whole utterance. LiSenNet's mask sub-network (encoder -> DPR blocks
-> decoder) is causal in time, so streaming reproduces the offline
``predict_mask`` exactly: each causal time-conv keeps a ``(kt-1)``-frame ring
buffer and the DPR inter-time GRU carries its hidden state (see
``lisennet/model.py``). Everything else — the sub-band convs, layer norms, the
bidirectional *frequency* GRU, the GLU — is already per-frame.

Two pieces of LiSenNet's pipeline are intentionally **outside** the streamer, and
are handled exactly as the offline model does but with bounded state:

  * **feature extraction** — the network input is
    ``[compressed_mag, group_delay/pi, instantaneous_freq_diff/pi]``. Group delay
    is a per-frame difference along frequency; the IFD is a difference along
    *time*, so it needs the previous frame's phase (one frame of state, kept here).
  * **phase recovery** — the offline model refines phase with a 2-iteration
    Griffin-Lim, which is non-causal (its iSTFT/STFT span the whole utterance and
    it needs the *enhanced-magnitude* of future frames). For real-time use the
    streamer returns the enhanced magnitude and the caller reuses the **noisy
    phase** (the very seed Griffin-Lim starts from) for a causal iSTFT. The small
    quality cost of dropping the 2 GL iterations is measured in the deploy eval.

Usage::

    streamer = LiSenNetStreamer(model)       # model: LiSenNet
    streamer.reset()
    for t in range(n_frames):
        est_mag_t = streamer.step(src_mag[:, t], src_pha[:, t])   # (B, F) -> (B, F)

This is the PyTorch streaming reference; ``lisennet/export_onnx.py`` exports the
same ``predict_mask`` sub-network to ONNX for edge runtimes.
"""

from __future__ import annotations

import torch

from lisennet.model import (
    ConvolutionalGLU, DSConv, DualPathRNN, LiSenNet, MaskDecoder,
)

# The modules that carry temporal state in streaming mode.
_STREAM_MODULES = (DSConv, ConvolutionalGLU, DualPathRNN, MaskDecoder)


def enable_streaming(model: LiSenNet) -> None:
    """Flip the causal time-conv buffers and the DPR GRU into streaming mode (and reset)."""
    for m in model.modules():
        if isinstance(m, _STREAM_MODULES):
            m.streaming = True
    reset_streaming(model)


def disable_streaming(model: LiSenNet) -> None:
    """Return the model to offline (whole-utterance) behaviour."""
    for m in model.modules():
        if isinstance(m, _STREAM_MODULES):
            m.streaming = False


def reset_streaming(model: LiSenNet) -> None:
    """Clear all per-stream state so the next frame starts a fresh utterance."""
    for m in model.modules():
        if isinstance(m, _STREAM_MODULES):
            m.stream_reset()


class LiSenNetStreamer:
    """Online wrapper: ``reset()`` once, then ``step()`` one STFT frame at a time.

    ``step`` consumes the noisy frame's compressed magnitude and phase and returns
    the enhanced magnitude for that frame. The caller pairs it with the noisy
    phase and runs a streaming iSTFT (overlap-add) to recover audio.
    """

    def __init__(self, model: LiSenNet):
        self.model = model.eval()
        self.hop_length = model.hop_length
        self.n_fft = model.n_fft
        enable_streaming(self.model)
        self._prev_pha = None
        self._ifd_bias = None

    def reset(self) -> None:
        reset_streaming(self.model)
        self._prev_pha = None

    def _features(self, mag, pha):
        """Build the 3-channel input for one frame. mag/pha: (B, 1, F)."""
        gd = LiSenNet.cal_gd(pha)                            # per-frame, along frequency
        if self._prev_pha is None:
            self._prev_pha = torch.zeros_like(pha)
        if self._ifd_bias is None:
            f = pha.shape[-1]
            self._ifd_bias = (2 * torch.pi * (self.hop_length / self.n_fft)
                              * torch.arange(f, device=pha.device))[None, None, :]
        x_if = pha - self._prev_pha                          # difference along time
        x_ifd = x_if - self._ifd_bias
        ifd = torch.atan2(x_ifd.sin(), x_ifd.cos())
        self._prev_pha = pha
        return torch.stack([mag, gd / torch.pi, ifd / torch.pi], dim=1)   # (B, 3, 1, F)

    @torch.no_grad()
    def step(self, mag_frame, pha_frame):
        """One frame in -> enhanced magnitude out.

        Accepts ``(B, F)`` or ``(B, 1, F)`` for the current STFT frame's
        compressed magnitude and phase; returns the enhanced magnitude in the
        same (squeezed) layout.
        """
        squeeze = mag_frame.dim() == 2
        if squeeze:
            mag_frame = mag_frame.unsqueeze(1)              # (B, 1, F)
            pha_frame = pha_frame.unsqueeze(1)
        feat = self._features(mag_frame, pha_frame)         # (B, 3, 1, F)
        mask = self.model.predict_mask(feat)                # (B, 2, 1, F)
        est_mag = self.model.apply_mask(mask, mag_frame)    # (B, 1, F)
        return est_mag.squeeze(1) if squeeze else est_mag

    def close(self) -> None:
        """Restore offline behaviour (e.g. before training / export)."""
        disable_streaming(self.model)


@torch.no_grad()
def stream_est_mag(model: LiSenNet, src_mag, src_pha):
    """Run a full ``(B, T, F)`` utterance frame-by-frame, returning ``est_mag``.

    Mainly for parity checks against the offline ``predict_mask`` + ``apply_mask``;
    real-time callers use ``LiSenNetStreamer`` directly.
    """
    streamer = LiSenNetStreamer(model)
    streamer.reset()
    mags = [streamer.step(src_mag[:, t], src_pha[:, t]) for t in range(src_mag.shape[1])]
    streamer.close()
    return torch.stack(mags, dim=1)                         # (B, T, F)
