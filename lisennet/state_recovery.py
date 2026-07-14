"""State-recovery: how long does a streaming model remember a *wrong* state?

The deploy question the latency table cannot answer. Both bottlenecks stream with
bounded state, but they forget a corrupted one on completely different terms:

* **conv + FIFO (TCM)** — memory *is* the last ``RF`` input frames. Feed two streams
  the same input from different states and after exactly ``RF`` frames the FIFOs hold
  identical data, so the outputs are **bit-identical**. Corruption is not damped, it is
  *evicted*, in bounded time — a dead-beat guarantee that holds independent of weights,
  data and quantization.
* **GRU** — memory is a recurrent state. It cannot blow up (``h' = (1-z)*n + z*h`` is a
  convex combination of ``h`` and ``tanh in (-1,1)``, so ``|h| <= 1`` for all time by
  induction, whatever the weights). But forgetting is *asymptotic* and data-dependent:
  an update gate driven toward 1 holds a stale state, and nothing in the architecture
  bounds for how long.

So the trade is not "crash vs. no crash" — the GRU cannot diverge — it is **guaranteed
forgetting in bounded time vs. asymptotic forgetting with no bound**. This script
measures it: stream an utterance twice with identical input, corrupt every state tensor
of the second run at frame ``t0``, and track when its output re-converges to the first.

Corruptions (all applied to the *whole* streaming state, which is the realistic device
event — a glitch, a mid-stream reset, a scene change leaves everything stale):

  * ``foreign`` — state carried over from a *different* utterance. The realistic worst
    case, and the only one that is in-distribution: every value is one the model
    actually produces, so it cannot be dismissed as an unreachable input.
  * ``zeros``   — a mid-stream reset (what a dropped frame or a re-sync does).
  * ``noise``   — random state at the same scale as the real one (a bit-flip / glitch).

Usage::

    python -m lisennet.state_recovery --checkpoints cp_lisennet cp_lisennet_conv_hardened_nc24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.env import AttrDict
from lisennet.model import (
    ConvolutionalGLU, DSConv, DualPathRNN, LiSenNet, MaskDecoder, TimeGRU, _CausalTimeConv,
)
from lisennet.streaming import enable_streaming, reset_streaming

# Every module that carries temporal state, and the attribute(s) it keeps it in.
_SLOTS = {
    DSConv: ("_buf_low", "_buf_high"),
    ConvolutionalGLU: ("_buf",),
    _CausalTimeConv: ("_buf",),
    TimeGRU: ("_h",),
    DualPathRNN: ("_h",),
    MaskDecoder: ("_buf",),
}


def _state_slots(model):
    slots = []
    for m in model.modules():
        for cls, attrs in _SLOTS.items():
            if isinstance(m, cls):
                slots += [(m, a) for a in attrs]
                break
    return slots


def _get_state(model):
    return [(m, a, getattr(m, a).clone() if getattr(m, a) is not None else None)
            for m, a in _state_slots(model)]


def _set_state(model, state):
    for m, a, v in state:
        setattr(m, a, None if v is None else v.clone())


@torch.no_grad()
def _stream(model, feat, corrupt_at=None, corrupt_state=None):
    """Stream ``feat`` (1, 3, T, F) frame by frame; optionally overwrite the entire
    state at frame ``corrupt_at``. Returns est_mag (T, F)."""
    reset_streaming(model)
    outs = []
    for t in range(feat.shape[2]):
        if corrupt_at is not None and t == corrupt_at:
            _set_state(model, corrupt_state)
        x = feat[:, :, t:t + 1, :]
        mask = model.predict_mask(x)
        outs.append(model.apply_mask(mask, x[:, 0])[0, 0])
    return torch.stack(outs)                                   # (T, F)


@torch.no_grad()
def _state_after(model, feat, n_frames):
    """The model's own state after streaming ``n_frames`` of ``feat`` — an in-distribution
    state vector, just the wrong one."""
    reset_streaming(model)
    for t in range(n_frames):
        model.predict_mask(feat[:, :, t:t + 1, :])
    return _get_state(model)


def _corrupt(kind, ref_state, foreign_state, gen):
    if kind == "foreign":
        return foreign_state
    out = []
    for m, a, v in ref_state:
        if v is None:
            out.append((m, a, None))
        elif kind == "zeros":
            out.append((m, a, torch.zeros_like(v)))
        elif kind == "noise":
            s = v.std().clamp(min=1e-3)
            out.append((m, a, torch.randn(v.shape, generator=gen) * s))
        else:
            raise ValueError(kind)
    return out


def _load(ckpt_dir) -> LiSenNet:
    d = Path(ckpt_dir)
    h = AttrDict(json.load(open(d / "config.json")))
    model = build = __import__("lisennet.model", fromlist=["build_lisennet"]).build_lisennet(h)
    ck = torch.load(d / "g_best", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["generator"], strict=True)
    return model.eval()


def main():
    p = argparse.ArgumentParser(description="How long does a streaming model remember a wrong state?")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--t0", type=int, default=150, help="frame at which the state is corrupted")
    p.add_argument("--n_frames", type=int, default=600)
    p.add_argument("--n_utts", type=int, default=8, help="utterances averaged over")
    p.add_argument("--out", default=None, help="write per-frame curves to this .json")
    a = p.parse_args()

    from common.dataset import Dataset, load_voicebank_demand
    hf = load_voicebank_demand()

    curves = {}
    for ck in a.checkpoints:
        model = _load(ck)
        enable_streaming(model)
        h = AttrDict(json.load(open(Path(ck) / "config.json")))
        ds = Dataset(hf["test"], h.segment_size, h.sampling_rate, split=False, shuffle=False)
        gen = torch.Generator().manual_seed(0)

        feats = []
        for i in range(a.n_utts + 1):
            _, noisy = ds[i]
            spec = model.power_compress(model.apply_stft(noisy.unsqueeze(0)))
            f = model.build_features(spec.abs(), spec.angle())
            if f.shape[2] < a.n_frames:                       # pad short utts by repetition
                reps = -(-a.n_frames // f.shape[2])
                f = f.repeat(1, 1, reps, 1)
            feats.append(f[:, :, :a.n_frames, :])

        for kind in ("foreign", "zeros", "noise"):
            errs = []
            for i in range(a.n_utts):
                feat = feats[i]
                ref = _stream(model, feat)                                     # clean state
                foreign = _state_after(model, feats[i + 1], a.t0)              # someone else's state
                bad = _corrupt(kind, _state_after(model, feat, a.t0), foreign, gen)
                per = _stream(model, feat, corrupt_at=a.t0, corrupt_state=bad)
                d = (per - ref).abs().amax(dim=1)[a.t0:]                       # max|Δ| per frame
                errs.append(d)
            e = torch.stack(errs).mean(0)
            curves[f"{Path(ck).name}|{kind}"] = e.tolist()

            scale = ref.abs().mean().item()
            exact = (e == 0).nonzero()
            n_exact = int(exact[0]) if len(exact) else None
            def first_below(thr):
                m = (e < thr * scale).nonzero()
                # require it to STAY below (recovery, not a zero-crossing)
                for idx in m.flatten().tolist():
                    if bool((e[idx:] < thr * scale).all()):
                        return idx
                return None
            print(f"{Path(ck).name:<32} {kind:<8} "
                  f"exact0={str(n_exact):>6}  <1%={str(first_below(0.01)):>6}  "
                  f"<0.1%={str(first_below(0.001)):>6}  (frames after corruption)")

    if a.out:
        json.dump(curves, open(a.out, "w"))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
