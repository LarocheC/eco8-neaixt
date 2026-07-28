#!/usr/bin/env python
"""Real, state-propagating calibration data for the streaming int8 exports.

WHY THIS EXISTS
---------------
``export_tflite_multi.py`` used to build calibration samples by jittering only the
feature input and leaving every *state* input at its ``init_states()`` zeros. The
quantizer therefore saw an all-zero tensor on every state port, collapsed its range,
and pinned the scale at the calibrator floor ``3.9215683651783184e-12``.

An int8 tensor at that scale spans about +/-5e-10, so every real state value
saturates: the recurrence stops carrying information and the model runs as if it
were **stateless**. Measured damage: ConvFSENet 9/9 states floored, NSNet2 1/1,
LiSenNet 0/25 (clean — because the LiSenNet-only exporter already used a
state-propagating reader, ``lisennet/quant_onnx.VBDStreamingCalibrationReader``).

Nothing caught it. The requant loop is data-independent so cycle counts are
identical either way, and the output checksum looks normal. This module
generalises the LiSenNet approach to every architecture: run the torch streaming
view frame-by-frame over real VoiceBank-DEMAND speech and record the **actual
propagated state** at each step.

FEATURE CONTRACTS (traced from each model's own inference code — get this wrong and
the ranges are plausible but meaningless)
-----------------------------------------------------------------------------------
Both pipelines RMS-normalise the waveform before the STFT; that normalisation sets
the activation scale, so it is part of the contract, not a detail.

``convfsenet``          convfsenet/inference_onnx.py:133-152
    raw |STFT| magnitude. The graph's own extractor (``mag_compressed``) applies
    ``(mag + 1e-9) ** compress_factor`` internally.
``convfsenet_sliced``   the TFLM-legal deploy builder
    Sets ``extractor_type="mag"``, i.e. the Pow is REMOVED from the graph because
    TFLite-Micro has no kernel for it. To remain the same function, the host must
    feed the already-compressed magnitude. That host-side hoist is exact
    (max|d| = 0 on mask and all states); simply *dropping* the compression is not
    (max|dmask| = 0.999979).
``nsnet2``              nsnet2/inference.py:27-31
    ``mag_pha_stft(...)[0]`` — magnitude ``sqrt(re^2+im^2+1e-9) ** compress_factor``.
    Note this is a different eps placement from ``compress_magnitude``; each arch
    gets its own model's transform, never a shared guess.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The value ai-edge-quantizer pins a collapsed range to. Any state scale at or
#: near this means that port was never exercised during calibration.
CALIB_FLOOR = 3.9215683651783184e-12


class CalibrationUnavailable(RuntimeError):
    """Raised when real calibration data cannot be produced.

    Deliberately fatal. The predecessor of this module caught every exception and
    silently substituted zero-state frames, which is exactly how the dead-state
    bug shipped while printing a success message.
    """


# --------------------------------------------------------------------------
# per-architecture feature extraction: waveform -> torch tensor [T, *feat_dims]
# --------------------------------------------------------------------------

def _rms_normalize(wav: torch.Tensor) -> torch.Tensor:
    """The normalisation both inference paths apply before the STFT."""
    nf = torch.sqrt(wav.numel() / (torch.sum(wav ** 2.0) + 1e-8))
    return wav * nf


def _stft_mag(wav: torch.Tensor, h) -> torch.Tensor:
    """|STFT| exactly as convfsenet/inference_onnx.py computes it -> [F, T]."""
    x = _rms_normalize(wav).unsqueeze(0)
    spec = torch.stft(
        x, int(h["n_fft"]), hop_length=int(h["hop_size"]),
        win_length=int(h["win_size"]), window=torch.hann_window(int(h["win_size"])),
        center=True, normalized=False, return_complex=True,
    )
    return spec.abs()[0]


def _feats_convfsenet_raw(wav, h):
    return _stft_mag(wav, h).transpose(0, 1)                      # [T, F]


def _feats_convfsenet_compressed(wav, h):
    """Host-side hoist: feed (mag + eps)^c, matching convfsenet.model.compress_magnitude."""
    from convfsenet.model import _MAG_EPS
    mag = _stft_mag(wav, h)
    return (mag + _MAG_EPS).pow(float(h["compress_factor"])).transpose(0, 1)


def _feats_nsnet2(wav, h):
    """nsnet2/inference.py: mag_pha_stft(...)[0] -> compressed magnitude."""
    from common.dataset import mag_pha_stft
    x = _rms_normalize(wav).unsqueeze(0)
    mag, _pha, _com = mag_pha_stft(x, int(h["n_fft"]), int(h["hop_size"]),
                                   int(h["win_size"]), float(h["compress_factor"]))
    return mag[0].transpose(0, 1)                                 # [T, F]


FEATURE_BUILDERS = {
    "convfsenet": _feats_convfsenet_raw,
    "convfsenet_sliced": _feats_convfsenet_compressed,
    "nsnet2": _feats_nsnet2,
}


# --------------------------------------------------------------------------
# the calibrator
# --------------------------------------------------------------------------

def _load_utterances(n_utts, seed, split, cache_dir):
    """Deterministically sample noisy utterances from VoiceBank-DEMAND."""
    try:
        from common.dataset import load_voicebank_demand
        ds = load_voicebank_demand(cache_dir=cache_dir)
    except Exception as exc:                       # noqa: BLE001 - re-raised, not swallowed
        raise CalibrationUnavailable(
            "could not load VoiceBank-DEMAND (looked in %s). Real calibration data is "
            "required; pass --calib flowtest to deliberately produce a NON-quality-valid "
            "artifact instead. Underlying error: %s: %s"
            % (cache_dir or "the default HF datasets cache", type(exc).__name__, exc)
        ) from exc

    if split not in ds:
        raise CalibrationUnavailable(
            "split %r not in dataset (have %s)" % (split, list(ds)))
    d = ds[split]
    if len(d) == 0:
        raise CalibrationUnavailable("split %r is empty" % split)

    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(d), size=min(n_utts, len(d)), replace=False).tolist())
    out = []
    for i in idx:
        row = d[int(i)]
        # Calibrate on what the model actually sees at inference: the NOISY signal.
        wav = np.asarray(row["noisy"]["array"], dtype=np.float32)
        out.append((row.get("id", str(i)), wav))
    return out, idx


def build_calibration_samples(arch, checkpoint_file, view, args, in_names, patch_ctx,
                              *, n_utts=8, max_frames=256, warmup=0, seed=0,
                              split="train", cache_dir=None):
    """Run the streaming view over real speech, recording propagated state.

    Returns ``(samples, stats)`` where ``samples`` is the list of dicts
    ai_edge_quantizer consumes (keyed ``args_0``..``args_N``) and ``stats`` is
    provenance + per-state activation statistics.

    ``patch_ctx`` MUST be held open around the run: the ``convfsenet_sliced``
    builder monkeypatches ``forward_step``, so calibrating outside it would
    calibrate a *different graph* than the one being exported.
    """
    if arch not in FEATURE_BUILDERS:
        raise CalibrationUnavailable(
            "no feature contract for arch %r (have %s). Add one to FEATURE_BUILDERS "
            "-- guessing the feature semantics produces plausible but wrong ranges."
            % (arch, ", ".join(sorted(FEATURE_BUILDERS))))

    cfg_path = Path(checkpoint_file).parent / "config.json"
    if not cfg_path.exists():
        raise CalibrationUnavailable("no config.json beside %s" % checkpoint_file)
    with open(cfg_path) as f:
        h = json.load(f)

    utts, idx = _load_utterances(n_utts, seed, split, cache_dir)
    feat_of = FEATURE_BUILDERS[arch]

    init_states = [t.clone() for t in args[1:]]
    samples = []
    state_acc = [[] for _ in init_states]
    n_frames_used = 0
    digest = hashlib.sha256()

    with patch_ctx():
        for _uid, wav in utts:
            if len(samples) >= max_frames:
                break
            feats = feat_of(torch.from_numpy(wav), h)             # [T, *feat_dims]
            states = [t.clone() for t in init_states]
            for t in range(feats.shape[0]):
                if len(samples) >= max_frames:
                    break
                feat = feats[t].unsqueeze(0)                      # add batch axis
                if t >= warmup:
                    smp = {in_names[0]: feat.detach().numpy().astype(np.float32)}
                    for nm, s in zip(in_names[1:], states):
                        smp[nm] = s.detach().numpy().astype(np.float32)
                    for k in sorted(smp):
                        digest.update(smp[k].tobytes())
                    samples.append(smp)
                    for j, s in enumerate(states):
                        state_acc[j].append(s.detach().numpy().astype(np.float32))
                with torch.no_grad():
                    out = view(feat, *states)
                out = out if isinstance(out, (tuple, list)) else (out,)
                states = [o.detach() for o in out[1:]]
                n_frames_used += 1

    if not samples:
        raise CalibrationUnavailable("produced zero calibration frames")

    per_state = []
    for j, chunks in enumerate(state_acc):
        a = np.concatenate([c.reshape(-1) for c in chunks]) if chunks else np.zeros(1, np.float32)
        per_state.append({
            "input": in_names[j + 1],
            "min": float(a.min()), "max": float(a.max()),
            "std": float(a.std()), "frac_nonzero": float((a != 0).mean()),
        })

    dead = [s for s in per_state if s["max"] == 0.0 and s["min"] == 0.0]
    if dead:
        raise CalibrationUnavailable(
            "%d/%d state ports are still all-zero across %d calibration frames (%s). "
            "The quantizer would floor their scales and the recurrence would be dead. "
            "This is the exact defect this module exists to prevent."
            % (len(dead), len(per_state), len(samples),
               ", ".join(s["input"] for s in dead)))

    stats = {
        "mode": "vbd",
        "quality_valid": True,
        "arch": arch,
        "checkpoint": str(checkpoint_file),
        "dataset": "JacobLinCool/VoiceBank-DEMAND-16k",
        "split": split,
        "utterance_indices": idx,
        "n_utts": len(utts),
        "n_frames": len(samples),
        "warmup": warmup,
        "seed": seed,
        "stft": {k: h.get(k) for k in ("n_fft", "hop_size", "win_size", "compress_factor")},
        "calib_sha256": digest.hexdigest(),
        "per_state": per_state,
    }
    return samples, stats


def build_flowtest_samples(view, args, in_names, patch_ctx, *, n_frames=64, seed=0):
    """Synthetic calibration. NOT quality-valid — states are still propagated, but the
    input is noise, so activation ranges do not reflect speech. Explicit opt-in only."""
    rng = np.random.default_rng(seed)
    states = [t.clone() for t in args[1:]]
    samples = []
    with patch_ctx():
        for _ in range(n_frames):
            feat = torch.from_numpy(
                (rng.standard_normal(tuple(args[0].shape)) * 0.5).astype(np.float32))
            smp = {in_names[0]: feat.numpy()}
            for nm, s in zip(in_names[1:], states):
                smp[nm] = s.detach().numpy().astype(np.float32)
            samples.append(smp)
            with torch.no_grad():
                out = view(feat, *states)
            out = out if isinstance(out, (tuple, list)) else (out,)
            states = [o.detach() for o in out[1:]]
    return samples, {"mode": "flowtest", "quality_valid": False, "n_frames": len(samples),
                     "seed": seed,
                     "warning": "synthetic noise input; activation ranges are not speech"}
