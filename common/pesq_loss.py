"""Differentiable PESQ as an auxiliary training loss.

Backed by ``torch_pesq`` from https://github.com/LarocheC/triton-pesq (a Triton
backend for audiolabs/torch-pesq; the import path is still ``torch_pesq``):

    uv pip install "triton-pesq @ git+https://github.com/LarocheC/triton-pesq"

Two backends with the same interface:

    triton  ``PesqLossTriton`` -- every stage forward and backward is a Triton
            kernel. CUDA only; ~60x faster forward, up to ~180x fwd+bwd.
    torch   ``PesqLoss`` -- the original PyTorch implementation. Runs anywhere,
            but its IIR level alignment is a Python loop over samples, so it is
            only practical for tests and tiny runs.

``auto`` picks triton on CUDA and torch otherwise. The loss is
``0.1 * d_symm + 0.0309 * d_asymm`` (the MOS model before its range compression),
so lower is better and its gradient pushes the estimate toward a higher PESQ.
PESQ alone is a poor objective for noise suppression (the triton-pesq README
says as much), so it is meant to be *added* to a model's existing losses.

Digital silence: the PESQ model's backward is NaN for an utterance containing
exactly-zero samples (the whole row, measured with both signals zero-padded --
which every training segment shorter than ``segment_size`` is). ``pesq_loss``
therefore adds a -100 dB dither (1e-5 on RMS-normalised audio) to both signals.
That keeps the gradient finite and leaves the loss unchanged on ordinary audio
(no change measured at 1e-5 or 1e-4; 1e-3 starts to move it).

Config (both families read the same top-level key; weight 0 = off, the default):

    "pesq_loss": {"weight": 0.2, "backend": "auto"}
"""

from __future__ import annotations

import warnings

import torch

BACKENDS = ("auto", "triton", "torch")
INSTALL_HINT = 'uv pip install "triton-pesq @ git+https://github.com/LarocheC/triton-pesq"'


def pesq_loss_config(h) -> tuple[float, str]:
    """(weight, backend) from a config's ``pesq_loss`` block; weight 0.0 when absent."""
    cfg = (h.get("pesq_loss") if hasattr(h, "get") else getattr(h, "pesq_loss", None)) or {}
    weight = float(cfg.get("weight", 0.0))
    backend = str(cfg.get("backend", "auto"))
    if backend not in BACKENDS:
        raise ValueError(f"pesq_loss.backend must be one of {BACKENDS}, got {backend!r}")
    if weight < 0:
        raise ValueError(f"pesq_loss.weight must be >= 0, got {weight}")
    return weight, backend


def build_pesq_loss(sample_rate: int, device, backend: str = "auto") -> torch.nn.Module:
    """A frozen PESQ loss module on ``device`` (see the module docstring for backends)."""
    try:
        import torch_pesq
    except ImportError as err:
        raise ImportError(f"the PESQ loss needs triton-pesq: {INSTALL_HINT}") from err

    device = torch.device(device)
    if backend == "auto":
        backend = "triton" if device.type == "cuda" else "torch"
    if backend == "triton":
        if device.type != "cuda":
            raise ValueError("the triton PESQ backend needs a CUDA device; use backend='torch'")
        loss = torch_pesq.PesqLossTriton(1.0, sample_rate=sample_rate)
    elif backend == "torch":
        if device.type == "cuda":
            warnings.warn("PESQ loss on CUDA with the torch backend is ~60x slower than 'triton'")
        loss = torch_pesq.PesqLoss(1.0, sample_rate=sample_rate)
    else:
        raise ValueError(f"unknown PESQ backend {backend!r}; choose from {BACKENDS}")
    return loss.to(device).eval().requires_grad_(False)


def pesq_loss(loss_fn: torch.nn.Module, clean: torch.Tensor, est: torch.Tensor,
              dither: float = 1e-5) -> torch.Tensor:
    """Batch-mean PESQ loss of ``est`` against ``clean``.

    Accepts ``(B, T)`` or ``(B, 1, T)`` waveforms and trims both to the shorter
    length (an iSTFT may return a few samples fewer than it was given). ``dither``
    is the std of the noise added to both signals; see the module docstring.
    """
    clean = clean.reshape(clean.shape[0], -1).float()
    est = est.reshape(est.shape[0], -1).float()
    n = min(clean.shape[-1], est.shape[-1])
    clean, est = clean[..., :n], est[..., :n]
    if dither > 0:
        clean = clean + dither * torch.randn_like(clean)
        est = est + dither * torch.randn_like(est)
    return loss_fn(clean.contiguous(), est.contiguous()).mean()
