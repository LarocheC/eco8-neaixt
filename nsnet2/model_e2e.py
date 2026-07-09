"""End-to-end NSNet2 with a learned butterfly transform in place of the STFT.

``NSNet2E2E`` is waveform-in / waveform-out:

    wav (B, L)
      -> ButterflyAnalysis      -> coeffs w (B, T, N)
      -> NSNet2.predict_mask(w) -> mask   (B, T, N)   [sigmoid, TasNet-style]
      -> w_hat = w * mask
      -> ButterflySynthesis     -> wav_hat (B, L)

The NSNet2 core (dense FC + 2-layer GRU) is reused byte-for-byte via
``NSNet2.predict_mask`` so the *only* changed variable vs the STFT baseline is
the transform. ``n_freq`` is driven to ``n_coeffs`` (default = ``win``) instead
of the STFT's ``n_fft // 2 + 1``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsnet2.model import NSNet2
from nsnet2.transform import ButterflyTransform


class NSNet2E2E(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.win = getattr(h, "win_size", 512)
        self.hop = getattr(h, "hop_size", 256)
        self.n_coeffs = self.win        # square orthogonal butterfly transform

        tcfg = getattr(h, "transform", None) or {}
        learnable_window = tcfg.get("learnable_window", True)
        window_init = tcfg.get("window_init", "sqrt_hann")

        self.transform = ButterflyTransform(
            self.win, self.hop, compress_factor=getattr(h, "compress_factor", 0.3),
            learnable_window=learnable_window, window_init=window_init,
        )
        # Reuse the exact NSNet2 core; feed it n_coeffs "frequency" bins.
        self.core = NSNet2(_CoreDims(h, self.n_coeffs))

    def _pad_to_valid(self, wav: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Right-pad so ``(L - win) % hop == 0`` and ``L >= win``. Returns the
        padded waveform and the original length for cropping the output."""
        L = wav.shape[-1]
        if L < self.win:
            pad = self.win - L
        else:
            rem = (L - self.win) % self.hop
            pad = 0 if rem == 0 else self.hop - rem
        if pad:
            wav = F.pad(wav, (0, pad))
        return wav, L

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav_p, L = self._pad_to_valid(wav)
        mag, X = self.transform.analyze(wav_p)       # compressed mag (B,T,N), complex X
        mask = self.core.predict_mask(mag)           # (B, T, N) in [0, 1]
        X_masked = X * self.transform.gain_from_mask(mask)   # real gain, phase preserved
        wav_hat = self.transform.synthesize(X_masked)
        return wav_hat[..., :L]


class _CoreDims:
    """Thin view over the config that overrides ``n_fft`` so ``NSNet2`` derives
    ``n_freq = n_coeffs``. NSNet2 only reads ``n_fft`` (for ``n_fft//2+1``) plus
    the width/backend knobs, all of which pass through to ``h``."""

    def __init__(self, h, n_coeffs: int):
        self._h = h
        # NSNet2 computes n_freq = n_fft // 2 + 1; invert so it lands on n_coeffs.
        self.n_fft = 2 * (n_coeffs - 1)

    def __getattr__(self, name):
        return getattr(self._h, name)
