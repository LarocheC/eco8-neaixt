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
from torch import nn

from lisennet.model import (
    ConvolutionalGLU, DSConv, DualPathRNN, LiSenNet, MaskDecoder, _CausalTimeConv,
)

# The modules that carry temporal state in streaming mode. ``DualPathRNN`` (the
# RNN bottleneck) carries a GRU hidden state; ``_CausalTimeConv`` (the conv
# bottleneck's inter-time layer) carries a FIFO ring buffer — only one of the two
# is present depending on the model's ``bottleneck``.
_STREAM_MODULES = (DSConv, ConvolutionalGLU, DualPathRNN, MaskDecoder, _CausalTimeConv)


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


# =============================================================================
# ONNX-exportable streaming view — explicit flat state I/O (à la ConvFSENet).
# =============================================================================


class LiSenNetStreamingONNX(nn.Module):
    """ONNX-friendly single-frame view of the mask sub-network's causal state.

    Every causal time-conv in the model keeps a FIFO ring buffer (the encoder
    ``DSConv``\\s, the ``ConvolutionalGLU`` depthwise conv, the conv-bottleneck's
    ``_CausalTimeConv`` layers and the ``MaskDecoder`` mask conv). The offline
    streamer carries those as internal ``_buf`` attributes; for ONNX they must be
    graph inputs/outputs, so this wrapper exposes them as flat positional
    ``state_i_in`` args and returns ``state_i_out`` — exactly ConvFSENet's
    ``ConvFSENetStreamingONNX`` contract (``export_onnx.export_streaming_fp32``).

    forward(feat, *states_in) -> (est_mag, *states_out) where
      * ``feat``      : ``(B, 3, 1, F)`` one frame of the 3-channel network input
        (compressed mag + group-delay/pi + IFD/pi); STFT / feature extraction and
        the phase recovery stay host-side, same as the whole-utterance export.
      * ``est_mag``   : ``(B, 1, F)`` enhanced compressed magnitude for the frame.
      * each state    : the FIFO buffer of one causal conv, a *static* shape (only
        the batch axis is dynamic) — Neural-ART wants fixed state shapes.

    This works with the conv bottleneck (``bottleneck="conv"``); the RNN
    bottleneck carries a GRU hidden state instead of conv FIFOs and is not the
    NPU target, so it is rejected here.
    """

    # Modules and the buffer attribute(s) each one carries, in a fixed order.
    _SLOT_ATTRS = {
        DSConv: ("_buf_low", "_buf_high"),
        ConvolutionalGLU: ("_buf",),
        _CausalTimeConv: ("_buf",),
        MaskDecoder: ("_buf",),
    }

    def __init__(self, model: LiSenNet):
        super().__init__()
        if getattr(model, "bottleneck", "rnn") != "conv":
            raise ValueError(
                "LiSenNetStreamingONNX requires bottleneck='conv' (the NPU variant); "
                f"got {getattr(model, 'bottleneck', 'rnn')!r}. The RNN bottleneck's GRU "
                "hidden state is not a conv FIFO and does not map to the NPU."
            )
        if any(isinstance(m, DualPathRNN) for m in model.modules()):
            raise ValueError("model still contains a DualPathRNN — not an NPU streaming target")
        self.model = model.eval()
        enable_streaming(self.model)
        # Ordered (module, attr) state slots, in module-registration order.
        self._slots = []
        for m in self.model.modules():
            for cls, attrs in self._SLOT_ATTRS.items():
                if isinstance(m, cls):
                    self._slots += [(m, a) for a in attrs]
                    break
        self.n_states = len(self._slots)
        self.n_freqs = int(self.model.n_freqs)

    @property
    def state_input_names(self):
        return [f"state_{i}_in" for i in range(self.n_states)]

    @property
    def state_output_names(self):
        return [f"state_{i}_out" for i in range(self.n_states)]

    @torch.no_grad()
    def init_states(self, batch_size: int, device="cpu", dtype=torch.float32):
        """Zero FIFO buffers of the correct static shape for each slot.

        Shapes are discovered by running one zero frame through the streaming
        model (each ``_causal_conv_stream`` allocates its buffer on first use),
        then zeroing what it allocated. Buffers depend on channel width and the
        sub-band frequency size at each stage, so they are not all the same shape.
        """
        reset_streaming(self.model)
        dummy = torch.zeros(batch_size, 3, 1, self.n_freqs, device=device, dtype=dtype)
        _ = self.model.predict_mask(dummy)
        states = [torch.zeros_like(getattr(mod, attr)) for mod, attr in self._slots]
        reset_streaming(self.model)
        return states

    def forward(self, feat, *states_in):
        """feat: (B, 3, 1, F); states_in: n_states positional FIFO buffers."""
        for (mod, attr), s in zip(self._slots, states_in):
            setattr(mod, attr, s)                          # seed FIFOs from graph inputs
        mask = self.model.predict_mask(feat)               # (B, 2, 1, F); mutates the FIFOs
        est_mag = self.model.apply_mask(mask, feat[:, 0])  # (B, 1, F)
        states_out = [getattr(mod, attr) for mod, attr in self._slots]
        return (est_mag, *states_out)
