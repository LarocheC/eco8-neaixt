"""Learned butterfly STFT (FFT-warm-started) replacing torch.stft / torch.istft.

The analysis/synthesis transforms are a **complex butterfly pair initialised to
the exact FFT / iFFT** (`fft_no_br` / `ifft_no_br` — the butterfly IS the
Cooley-Tukey factorisation of the DFT). So at init the front-/back-end is a real
STFT (frequency-selective basis, magnitude/phase interface, perfect
reconstruction), and training is free to let it deviate. Unlike `torch.stft`, a
butterfly lowers to (real-decomposable) structured MatMuls, so it can live
inside the exported ONNX graph — a single waveform->waveform enhancer.

Why not a random real butterfly (the first attempt): a random orthogonal basis
stays broadband (spectral concentration ~0.04), and elementwise masking on a
broadband basis cannot separate speech from noise — it improves SI-SNR by
removing energy but smears distortion across the spectrum (PESQ drops). A
frequency-selective basis is what makes magnitude masking work; the FFT init
provides it from step 0.

Config note: `fft_no_br(increasing_stride=True)` + `ifft_no_br(increasing_stride
=False)` is the pair that reconstructs exactly (bit-reversal cancels). Windows
are sqrt-Hann on both sides so their product is Hann → COLA at 50% hop → the
windowed overlap-add reconstructs.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsnet2.layers import Butterfly, HAVE_BUTTERFLY

# Backend: torch_structured>=1.2.5 fixes the Triton butterfly backward kernel,
# so CUDA uses the fast Triton path; the dispatch delegator auto-routes CPU
# tensors to pure-PyTorch (CPU test suite still works). TORCH_STRUCTURED_BACKEND=
# torch forces pure-PyTorch everywhere.
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
    w = _init_window(win, kind)
    if learnable:
        setattr(module, name, nn.Parameter(w))
    else:
        module.register_buffer(name, w)


class ButterflyTransform(nn.Module):
    """FFT-initialised learnable butterfly STFT.

    ``analyze(wav) -> (mag, X)`` returns the power-law-compressed magnitude
    ``(B, T, N)`` (the network's input) and the complex coefficients ``X``
    (``N == win``). Magnitude masking is a real gain on ``X`` (phase preserved),
    so ``synthesize(X_masked) -> wav`` takes the already-masked complex — no
    atan2/cos/sin, which keeps the ONNX export a pure real-valued graph.
    """

    def __init__(self, win: int = 512, hop: int = 256, compress_factor: float = 0.3,
                 learnable_window: bool = True, window_init: str = "sqrt_hann"):
        super().__init__()
        if not HAVE_BUTTERFLY:
            raise ImportError("torch_structured.Butterfly unavailable; rebuild torch-butterfly.")
        self.win = win
        self.hop = hop
        self.n_coeffs = win
        self.compress_factor = compress_factor
        self.register_buffer("frame_kernel", _identity_frame_kernel(win))
        self.register_buffer("oa_kernel", _identity_frame_kernel(win))
        _register_window(self, "ana_window", win, learnable_window, window_init)
        _register_window(self, "syn_window", win, learnable_window, window_init)
        # FFT / iFFT butterfly pair. This stride combination reconstructs exactly.
        self.analysis = Butterfly(win, win, bias=False, complex=True,
                                  init="fft_no_br", increasing_stride=True)
        self.synthesis = Butterfly(win, win, bias=False, complex=True,
                                   init="ifft_no_br", increasing_stride=False)

    def analyze(self, wav: torch.Tensor):
        frames = F.conv1d(wav.unsqueeze(1), self.frame_kernel, stride=self.hop)
        frames = frames.transpose(1, 2) * self.ana_window       # (B, T, win)
        X = self.analysis(frames.reshape(-1, self.win).to(torch.complex64))  # (B*T, N) complex
        X = X.reshape(frames.shape[0], -1, self.n_coeffs)       # (B, T, N)
        mag = (X.real.pow(2) + X.imag.pow(2) + 1e-9).sqrt().pow(self.compress_factor)
        return mag, X

    def gain_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Compressed-domain mask [0,1] -> linear-domain gain applied to X.
        new_mag^c = mag^c * mask  =>  linear gain = mask^(1/compress)."""
        return mask.pow(1.0 / self.compress_factor)

    def synthesize(self, X: torch.Tensor) -> torch.Tensor:
        frames = self.synthesis(X.reshape(-1, self.n_coeffs)).real      # (B*T, win)
        frames = frames.reshape(X.shape[0], -1, self.win) * self.syn_window
        wav = F.conv_transpose1d(frames.transpose(1, 2), self.oa_kernel, stride=self.hop)
        return wav.squeeze(1)                                   # (B, L)
