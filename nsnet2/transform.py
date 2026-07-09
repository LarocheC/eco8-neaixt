"""Learned butterfly analysis / synthesis transform (STFT replacement).

Drop-in replacement for the fixed ``torch.stft`` / ``torch.istft`` front-/
back-end in the NSNet2 pipeline. Instead of a DFT the time->feature map is a
learnable ``torch_structured.Butterfly`` — the same O(N log N) factorization the
FFT itself uses (Cooley-Tukey), so it *can* represent an STFT-like transform but
is free to learn a different basis end-to-end. Crucially, unlike ``torch.stft``,
a butterfly lowers to plain structured MatMuls and therefore lives *inside* the
exported ONNX graph, enabling a single waveform->waveform enhancement model.

Design:
  * ONE shared butterfly. Analysis runs it forward; synthesis runs it
    TRANSPOSED (``transpose=True``) — the exact inverse when the twiddles are
    orthogonal. This mirrors how the real STFT/iSTFT share one transform
    (DFT / iDFT), gives near-perfect reconstruction by construction (~+29 dB at
    init with ``init='ortho'`` vs ~-38 dB for an independent random synthesis),
    and halves the transform parameters. The orthogonality penalty
    (``nsnet2.layers.butterfly_ortho_penalty``, applied by the trainer) keeps
    ``transpose == inverse`` valid as the transform learns.
  * framing / overlap-add are fixed identity Conv1d / ConvTranspose1d (kept in
    the graph, ONNX-native, non-trainable),
  * separate learnable analysis / synthesis windows (init = sqrt-Hann so the
    windowed overlap-add starts near COLA at 50% hop).

The butterfly is applied on a 2D ``(B*T, N)`` reshape (single ``-1`` collapse so
the traced graph keeps T dynamic) — matching the ``_butterfly_export_forward``
contract in ``nsnet2/export_onnx.py``, so eager and exported forward agree.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsnet2.layers import Butterfly, HAVE_BUTTERFLY

# Backend: torch_structured>=1.2.5 fixes the Triton butterfly backward kernel,
# so we use the fast Triton path on CUDA. The library's dispatch delegator
# auto-routes CPU tensors to the pure-PyTorch kernel regardless of backend, so
# CPU-only runs (e.g. the test suite) still work. Set TORCH_STRUCTURED_BACKEND=
# torch to force the pure-PyTorch path everywhere (e.g. debugging / older wheels
# with the Triton backward bug).
if HAVE_BUTTERFLY and os.environ.get("TORCH_STRUCTURED_BACKEND", "").lower() == "torch":
    import torch_structured as _ts
    _ts.set_backend("torch")


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


def _register_window(module: nn.Module, name: str, win: int, learnable: bool, kind: str) -> None:
    """Register ``module.<name>`` as a trainable Parameter or a fixed buffer."""
    w = _init_window(win, kind)
    if learnable:
        setattr(module, name, nn.Parameter(w))
    else:
        module.register_buffer(name, w)


class ButterflyTransform(nn.Module):
    """Shared learned butterfly used forward for analysis and transposed for
    synthesis.

    ``analyze``:  waveform ``(B, L)`` -> coefficients ``(B, T, N)`` with
    ``N == win`` and ``T = (L - win) // hop + 1`` (caller ensures ``L`` is
    frame-aligned).
    ``synthesize``: coefficients ``(B, T, N)`` -> waveform ``(B, L)`` with
    ``L = (T - 1) * hop + win``.
    """

    def __init__(self, win: int = 512, hop: int = 256,
                 learnable_window: bool = True, window_init: str = "sqrt_hann",
                 nblocks: int = 1, init: str = "ortho"):
        super().__init__()
        if not HAVE_BUTTERFLY:
            raise ImportError("torch_structured.Butterfly unavailable; rebuild torch-butterfly.")
        self.win = win
        self.hop = hop
        self.n_coeffs = win
        self.register_buffer("frame_kernel", _identity_frame_kernel(win))
        self.register_buffer("oa_kernel", _identity_frame_kernel(win))
        _register_window(self, "ana_window", win, learnable_window, window_init)
        _register_window(self, "syn_window", win, learnable_window, window_init)
        self.butterfly = Butterfly(in_size=win, out_size=win, bias=False,
                                   nblocks=nblocks, init=init)

    def analyze(self, wav: torch.Tensor) -> torch.Tensor:
        frames = F.conv1d(wav.unsqueeze(1), self.frame_kernel, stride=self.hop)
        frames = frames.transpose(1, 2) * self.ana_window        # (B, T, win)
        w = self.butterfly(frames.reshape(-1, self.win))         # (B*T, N)
        return w.reshape(frames.shape[0], -1, self.n_coeffs)     # (B, T, N)

    def synthesize(self, w: torch.Tensor) -> torch.Tensor:
        # transpose=True runs the shared butterfly as its (orthogonal) inverse.
        frames = self.butterfly(w.reshape(-1, self.n_coeffs), transpose=True)
        frames = frames.reshape(w.shape[0], -1, self.win) * self.syn_window
        wav = F.conv_transpose1d(frames.transpose(1, 2), self.oa_kernel, stride=self.hop)
        return wav.squeeze(1)                                    # (B, L)
