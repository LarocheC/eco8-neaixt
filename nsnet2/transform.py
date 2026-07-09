"""Learned butterfly analysis / synthesis transform (STFT replacement).

Drop-in replacement for the fixed ``torch.stft`` / ``torch.istft`` front-/
back-end in the NSNet2 pipeline. Instead of a DFT the time->feature map is a
learnable ``torch_structured.Butterfly`` — the same O(N log N) factorization the
FFT itself uses (Cooley-Tukey), so it *can* represent an STFT-like transform but
is free to learn a different basis end-to-end. Crucially, unlike ``torch.stft``,
a butterfly lowers to plain structured MatMuls and therefore lives *inside* the
exported ONNX graph, enabling a single waveform->waveform enhancement model.

Design (TasNet-style, confirmed with the user):
  * framing / overlap-add are fixed identity Conv1d / ConvTranspose1d (kept in
    the graph, ONNX-native, non-trainable),
  * the per-frame window is learnable (init = sqrt-Hann so analysis*synthesis
    starts near COLA at 50% hop),
  * analysis and synthesis butterflies are *independent* learned transforms
    (not constrained to be exact inverses) — the end-to-end loss shapes both.

The butterfly is always applied on a 2D ``(B*T, N)`` reshape. That matches the
``_butterfly_export_forward`` contract in ``nsnet2/export_onnx.py`` (which
collapses leading dims to ``-1``), so eager and exported forward are identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsnet2.layers import Butterfly, HAVE_BUTTERFLY


def _identity_frame_kernel(win: int) -> torch.Tensor:
    """(win, 1, win) identity kernel: Conv1d gathers overlapping frames,
    ConvTranspose1d with the same kernel performs the overlap-add."""
    return torch.eye(win).unsqueeze(1)


def _init_window(win: int, kind: str) -> torch.Tensor:
    if kind == "sqrt_hann":
        return torch.hann_window(win, periodic=True).clamp_min(0).sqrt()
    if kind == "hann":
        return torch.hann_window(win, periodic=True)
    if kind == "ones":
        return torch.ones(win)
    raise ValueError(f"Unknown window init: {kind!r}")


def _register_window(module: nn.Module, win: int, learnable: bool, kind: str) -> None:
    """Register ``module.window`` as a trainable Parameter or a fixed buffer."""
    w = _init_window(win, kind)
    if learnable:
        module.window = nn.Parameter(w)
    else:
        module.register_buffer("window", w)


class ButterflyAnalysis(nn.Module):
    """waveform ``(B, L)`` -> coefficients ``(B, T, N)``.

    ``L`` must satisfy ``(L - win) % hop == 0`` (the caller pads); ``T`` is then
    ``(L - win) // hop + 1``.
    """

    def __init__(self, win: int = 512, hop: int = 256, n_coeffs: int | None = None,
                 learnable_window: bool = True, window_init: str = "sqrt_hann",
                 nblocks: int = 1, init: str = "randn"):
        super().__init__()
        if not HAVE_BUTTERFLY:
            raise ImportError("torch_structured.Butterfly unavailable; rebuild torch-butterfly.")
        self.win = win
        self.hop = hop
        self.n_coeffs = n_coeffs or win
        self.register_buffer("frame_kernel", _identity_frame_kernel(win))
        _register_window(self, win, learnable_window, window_init)
        self.butterfly = Butterfly(in_size=win, out_size=self.n_coeffs, bias=False,
                                   nblocks=nblocks, init=init)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # frames: (B, win, T) via a fixed strided identity conv
        frames = F.conv1d(wav.unsqueeze(1), self.frame_kernel, stride=self.hop)
        frames = frames.transpose(1, 2)                          # (B, T, win)
        frames = frames * self.window                            # learnable window
        # Collapse (B, T) with a single -1 so the traced Reshape keeps T dynamic
        # (a literal B*T would bake the trace-time length in). The butterfly is
        # applied on 2D — matching the _butterfly_export_forward contract.
        w = self.butterfly(frames.reshape(-1, self.win))         # (B*T, N)
        return w.reshape(frames.shape[0], -1, self.n_coeffs)     # (B, T, N)


class ButterflySynthesis(nn.Module):
    """coefficients ``(B, T, N)`` -> waveform ``(B, L)`` via inverse butterfly +
    windowed overlap-add. ``L = (T - 1) * hop + win``."""

    def __init__(self, win: int = 512, hop: int = 256, n_coeffs: int | None = None,
                 learnable_window: bool = True, window_init: str = "sqrt_hann",
                 nblocks: int = 1, init: str = "randn"):
        super().__init__()
        if not HAVE_BUTTERFLY:
            raise ImportError("torch_structured.Butterfly unavailable; rebuild torch-butterfly.")
        self.win = win
        self.hop = hop
        self.n_coeffs = n_coeffs or win
        self.register_buffer("oa_kernel", _identity_frame_kernel(win))
        _register_window(self, win, learnable_window, window_init)
        self.butterfly = Butterfly(in_size=self.n_coeffs, out_size=win, bias=False,
                                   nblocks=nblocks, init=init)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # Single -1 collapse keeps T dynamic in the traced graph (see analysis).
        frames = self.butterfly(w.reshape(-1, self.n_coeffs))    # (B*T, win)
        frames = frames.reshape(w.shape[0], -1, self.win)        # (B, T, win)
        frames = frames * self.window
        frames = frames.transpose(1, 2)                          # (B, win, T)
        wav = F.conv_transpose1d(frames, self.oa_kernel, stride=self.hop)
        return wav.squeeze(1)                                    # (B, L)
