"""The MetricGAN target: torch-pesq (GPU) must be a true drop-in for the CPU ITU one.

`batch_pesq` (ITU-T P.862, CPU) runs on every training step and dominates epoch
time. `batch_pesq_torch` replaces it with torch-pesq on the GPU. Two things have
to hold for that to be safe, and both are easy to get wrong:

1. **The failure contract.** Where ITU returns -1 on silence / too-short audio
   (so `batch_pesq` returns None and the trainer skips the discriminator's fake
   branch), torch-pesq returns **NaN** or **raises**. Feeding that NaN to the
   discriminator's MSE would NaN training outright. `batch_pesq_torch` must
   return None in exactly those cases.

2. **Agreement.** torch-pesq is not the reference implementation (no time
   alignment; IIR level alignment). It is only acceptable because it is a
   *target* for a discriminator that needs a monotone quality signal — so the
   rank correlation with ITU is what actually matters.

The REPORTED PESQ (`common.metrics.pesq_score`) is deliberately not covered here
because it is deliberately not changed: it stays the reference implementation.
"""

import numpy as np
import pytest
import torch

from common.discriminator import (
    batch_pesq, batch_pesq_torch, cal_pesq, make_pesq_target, pesq_target_from_config,
)
from common.env import AttrDict

SR = 16000


def _speechlike(n, seed, sr=SR):
    """Voiced-ish audio — PESQ needs speech-like structure to score at all."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    f0 = 110 + 40 * (seed % 5)
    sig = sum(0.4 / k * np.sin(2 * np.pi * f0 * k * t) for k in (1, 2, 3, 4))
    sig *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t)          # syllabic envelope
    return (sig + 0.01 * rng.standard_normal(n)).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. The failure contract — this is what stops it NaN-ing training
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,ref,deg", [
    ("both digitally silent", torch.zeros(32000), torch.zeros(32000)),
    ("silent reference",      torch.zeros(32000), torch.randn(32000) * 0.05),
    ("silent degraded",       torch.from_numpy(_speechlike(32000, 0)), torch.zeros(32000)),
    ("too short",             torch.randn(1600) * 0.1, torch.randn(1600) * 0.1),
])
def test_returns_none_exactly_where_itu_fails(name, ref, deg):
    """torch-pesq gives NaN (or raises) on these; ITU gives -1. Both must surface
    as None so the trainer skips the discriminator's fake branch."""
    itu = batch_pesq([ref.numpy()], [deg.numpy()])
    assert itu is None, f"precondition: ITU should fail on {name!r}"

    got = batch_pesq_torch(ref[None], deg[None])
    assert got is None, (
        f"batch_pesq_torch must return None on {name!r} — it returns NaN/raises "
        f"there, and a NaN target NaNs the discriminator loss. Got: {got}"
    )


def _degraded(ref, seed):
    rng = np.random.default_rng(seed)
    noise = torch.from_numpy(
        (0.05 * rng.standard_normal(ref.shape)).astype(np.float32))
    return ref + noise


def test_never_returns_a_non_finite_target():
    """The whole batch is dropped if ANY utterance fails (upstream's semantics),
    so a returned tensor is always safe to hand to F.mse_loss."""
    ref = torch.stack([torch.from_numpy(_speechlike(32000, i)) for i in range(4)])
    deg = _degraded(ref, 1)
    ref = ref.clone()
    ref[2] = 0.0                                   # one bad utterance poisons the batch
    assert batch_pesq_torch(ref, deg) is None

    ref = torch.stack([torch.from_numpy(_speechlike(32000, i)) for i in range(4)])
    deg = _degraded(ref, 2)
    out = batch_pesq_torch(ref, deg)
    assert out is not None and torch.isfinite(out).all()
    assert out.shape == (4,)


# ---------------------------------------------------------------------------
# 2. Agreement with the reference implementation
# ---------------------------------------------------------------------------

def test_normalisation_matches_batch_pesq():
    """Both must emit the same normalised (pesq - 1) / 3.5 target, not raw MOS —
    the discriminator regresses onto this scale."""
    ref = torch.stack([torch.from_numpy(_speechlike(32000, i)) for i in range(3)])
    deg = _degraded(ref, 3)

    itu = batch_pesq(list(ref.numpy()), list(deg.numpy()))
    tp = batch_pesq_torch(ref, deg)
    assert itu is not None and tp is not None

    # Same scale, and it inverts back to a plausible wideband MOS (1.0 .. 4.64).
    for v in (itu, tp):
        assert (v >= -0.1).all() and (v <= 1.1).all()
    mos_back = tp * 3.5 + 1.0
    assert (mos_back >= 1.0).all() and (mos_back <= 4.7).all()

    # Correlated: this is an approximation, so bound the error rather than equate.
    assert torch.abs(itu - tp).max() < 0.35, (
        f"torch-pesq target drifted too far from ITU: itu={itu}, torch={tp}"
    )


def test_rank_correlation_with_itu_is_high():
    """MetricGAN only needs the target to ORDER quality correctly — rank
    correlation with the reference is the property that actually licenses the
    swap, not absolute agreement.

    On 200 real VoiceBank-DEMAND pairs spanning noisy->clean this measures
    Spearman 0.997 (Pearson 0.999, mean |err| 0.043). Here we use an SNR sweep
    on synthetic speech to keep the test dataset-free, which is a coarser probe:
    it reaches ~0.987, so 0.95 is the floor with margin.

    The sweep must span a WIDE quality range. A narrow one bunches the PESQ
    scores together, and the resulting rank ties depress Spearman for reasons
    that have nothing to do with torch-pesq's fidelity.
    """
    from scipy.stats import spearmanr

    # Seed locally: torch.randn draws from the GLOBAL RNG, so using it here made
    # the audio — and hence the correlation — depend on which tests ran first.
    rng = np.random.default_rng(1234)
    refs, degs = [], []
    for i in range(8):
        clean = torch.from_numpy(_speechlike(32000, i))
        noise = torch.from_numpy(rng.standard_normal(32000).astype(np.float32))
        for snr_db in (-5, 0, 5, 10, 15, 20, 25, 30):
            gain = float(10 ** (-snr_db / 20)) * clean.abs().mean() / noise.abs().mean()
            refs.append(clean)
            degs.append(clean + gain * noise)
    ref, deg = torch.stack(refs), torch.stack(degs)

    itu = np.array([cal_pesq(r.numpy(), d.numpy()) for r, d in zip(ref, deg)])
    keep = itu > 0
    assert keep.sum() >= 40, "not enough scorable pairs to judge correlation"
    assert itu[keep].max() - itu[keep].min() > 2.0, (
        "the degradation sweep is too narrow to measure ranking meaningfully"
    )

    tp = (batch_pesq_torch(ref, deg) * 3.5 + 1.0).numpy()
    rho = spearmanr(itu[keep], tp[keep]).statistic
    assert rho >= 0.95, f"torch-pesq does not track ITU's ordering (Spearman {rho:.3f})"


# ---------------------------------------------------------------------------
# 3. Backend selection
# ---------------------------------------------------------------------------

def test_config_selects_backend_and_defaults_to_torch():
    assert pesq_target_from_config(AttrDict({})) is not None            # default: torch
    itu_fn = pesq_target_from_config(AttrDict({"gan": {"pesq_backend": "itu"}}))

    ref = torch.stack([torch.from_numpy(_speechlike(32000, i)) for i in range(2)])
    deg = _degraded(ref, 4)
    out = itu_fn(ref, deg)                          # must accept TENSORS, like the torch path
    assert out is not None and out.shape == (2,)

    with pytest.raises(ValueError, match="unknown pesq backend"):
        make_pesq_target("bogus")
