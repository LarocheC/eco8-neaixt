"""Frame-by-frame (streaming) inference for FSPEN.

FSPEN is streaming-native: every conv/Linear operates along *frequency* within
one frame, the DPE intra GRU is bidirectional only over the 32 frequency bands
(fully available per frame), and the only time-recurrent modules are the DPE
inter GRUs. Real-time inference therefore just carries the inter-GRU hidden
states — ``dpe_modules`` tensors of shape ``(groups, B*bands, width)`` from
``FSPEN.init_hidden`` — and is otherwise the unchanged offline forward, one
frame at a time (BatchNorms use running stats in eval mode, so per-frame
statistics are identical to whole-utterance ones).

Unlike LiSenNet there is no phase question: FSPEN predicts a complex mask, so
the streamed output is the enhanced complex spectrum and a causal overlap-add
iSTFT recovers audio directly.

Usage::

    streamer = FSPENStreamer(model)          # model: FSPEN (eval mode)
    streamer.reset()
    for t in range(n_frames):
        est_t = streamer.step(spec[:, t])    # (B, 2, F) -> (B, 2, F)

This is the PyTorch streaming reference; ``fspen/export_onnx.py`` exports the
same core with explicit state I/O for edge runtimes.
"""

from __future__ import annotations

import torch
from torch import nn

from fspen.model import FSPEN


def _amplitude(spec):
    """(B, T, 2, F) stacked re/im -> (B, T, 1, F) magnitude (ONNX-safe ops)."""
    return torch.sqrt(spec[:, :, 0:1] ** 2 + spec[:, :, 1:2] ** 2)


class FSPENStreamer:
    """Online wrapper: ``reset()`` once, then ``step()`` one STFT frame at a time.

    ``step`` consumes one noisy complex-spectrum frame (stacked re/im) and
    returns the enhanced frame; the inter-GRU hidden states are carried
    internally across calls.
    """

    def __init__(self, model: FSPEN):
        self.model = model.eval()
        self._hidden = None

    def reset(self) -> None:
        self._hidden = None

    @torch.no_grad()
    def step(self, spec_frame):
        """One frame in -> enhanced frame out.

        Accepts ``(B, 2, F)`` or ``(B, 1, 2, F)`` for the current noisy frame;
        returns the enhanced complex spectrum in the same (squeezed) layout.
        """
        squeeze = spec_frame.dim() == 3
        if squeeze:
            spec_frame = spec_frame.unsqueeze(1)             # (B, 1, 2, F)
        if self._hidden is None:
            self._hidden = self.model.init_hidden(
                spec_frame.shape[0], device=spec_frame.device, dtype=spec_frame.dtype)
        est, self._hidden = self.model.enhance_spectrum(
            spec_frame, _amplitude(spec_frame), self._hidden)
        return est.squeeze(1) if squeeze else est


@torch.no_grad()
def stream_enhance_spectrum(model: FSPEN, spec):
    """Run a full ``(B, T, 2, F)`` utterance frame-by-frame, returning the estimate.

    Mainly for parity checks against the offline ``enhance_spectrum``; real-time
    callers use ``FSPENStreamer`` directly.
    """
    streamer = FSPENStreamer(model)
    streamer.reset()
    ests = [streamer.step(spec[:, t]) for t in range(spec.shape[1])]
    return torch.stack(ests, dim=1)                          # (B, T, 2, F)


# =============================================================================
# ONNX-exportable streaming view — explicit flat state I/O (à la ConvFSENet).
# =============================================================================


class FSPENStreamingONNX(nn.Module):
    """ONNX-friendly single-frame view with the inter-GRU states as graph I/O.

    forward(spec, *states_in) -> (est_spec, *states_out) where
      * ``spec``     : ``(B, 1, 2, F)`` one noisy frame as stacked re/im; the
        magnitude input is derived in-graph (`sqrt(re^2 + im^2)`), so consumers
        feed the spectrum frame only.
      * ``est_spec`` : ``(B, 1, 2, F)`` the enhanced frame.
      * each state   : one DPE block's stacked inter-GRU hidden state,
        ``(groups, B*bands, width)`` — the ``init_hidden`` layout. States are
        flat positional args so the exported graph gets clean ``state_i_in`` /
        ``state_i_out`` tensor names (nested lists would collapse into one
        opaque input).
    """

    def __init__(self, model: FSPEN):
        super().__init__()
        self.model = model.eval()
        self.n_states = len(model.dpe_blocks)
        self.n_freqs = int(model.n_freqs)

    @property
    def state_input_names(self):
        return [f"state_{i}_in" for i in range(self.n_states)]

    @property
    def state_output_names(self):
        return [f"state_{i}_out" for i in range(self.n_states)]

    def init_states(self, batch_size: int, device="cpu", dtype=torch.float32):
        return self.model.init_hidden(batch_size, device=device, dtype=dtype)

    def forward(self, spec, *states_in):
        """spec: (B, 1, 2, F); states_in: n_states stacked inter-GRU hidden tensors."""
        est, states_out = self.model.enhance_spectrum(
            spec, _amplitude(spec), list(states_in))
        return (est, *states_out)
