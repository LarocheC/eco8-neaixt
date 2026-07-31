"""Do-no-harm gating: blend the mask toward unity where SR members disagree.

The UQ study showed per-frame member disagreement D tracks the removable
quantization noise (Spearman 0.79 vs int8-vs-FP32 error). This experiment asks
the deployment question: if the ensemble's members disagree at a frame, should
the device *suppress less* there — insurance against confident-but-wrong
suppression? On-device cost is one multiply-add per bin on quantities the
member-correct fold already emits (est_mag (B,K,1,F) -> mean and variance).

Gate: est' = (1 - a)·est_avg + a·src_mag, with a from D:
  frame-lin    a_t  = a_max·min(1, D_t / theta)           (D_t = var_K mean_F)
  frame-thr    a_t  = a_max·1[D_t > theta]
  bin-lin      a_tf = a_max·min(1, D_tf / theta_bin)      (per-(t,f) variance)
  bin-thr      a_tf = a_max·1[D_tf > theta_bin]
Thetas are FIXED from the pooled ID D distribution (a deployment constant),
taken at a percentile; a_max in {0.25, 0.5, 1.0}.

Expectation set by the study: mean PESQ should NOT reward this (residual
damage concentrates where D is LOW — the shared, non-averageable error), so
the primary read-outs are tails and counts, vs BOTH baselines:
  * worst-percentile paired deltas vs the ungated ensemble (p1/p5/p10);
  * do-no-harm accounting vs noisy: #utterances where enhancement HURTS
    (PESQ < noisy) before/after gating, and the worst (PESQ - noisy) tail;
  * DNSMOS (SIG/BAK/OVRL, P808) on noisy / ungated / selected variants.

Runs on the deployment artifact itself: the member-correct MinMax K=4 fold
(`ens_memb_mm_k4.onnx`, one session, all member est_mags per frame). Member
stacks are cached on the first pass so the gate sweep and metrics are offline
re-runs.

The smoke run showed the ID pool has nobody to insure (enhancement beats noisy
on every utterance, worst case +0.38 PESQ), so the gate is also evaluated on
the UQ study's OOD conditions — thetas stay FROZEN from ID (they are deployment
constants): `white5db` (unseen noise, D fires, AUROC 1.0), `gain-12` (D provably
blind — the honest negative control), `gain+12`. PESQ reference is always the
clean signal; "noisy" is the corrupted input itself.

usage: HF_HUB_OFFLINE=1 ./.venv/bin/python3 sr_gating.py [--n 824] [--jobs 24]
       [--n_ood 200] [--dnsmos {none,subset,full}]
Results -> sr_gating_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(REPO))

from sr_stream_ensemble import CKPT, CKPT_DIR, ENS_DIR  # noqa: E402

FOLD = ENS_DIR / "ens_memb_mm_k4.onnx"
OUT = REPO / "sr_gating_results.json"
CACHE = Path(os.environ.get("SR_GATING_CACHE",
                            Path("/tmp") / "sr_gating_cache"))
K = 4

VARIANTS = [("none", None, None, None)] + [
    (f"{form}_p{pct}_a{amax}", form, pct, amax)
    for form in ("frame-lin", "frame-thr", "bin-lin", "bin-thr")
    for pct in (75, 90)
    for amax in (0.25, 0.5, 1.0)
]
# OOD passes evaluate the ungated ensemble + a gentle-to-aggressive gate ladder
OOD_VARIANTS = ["none", "bin-thr_p90_a0.25", "frame-thr_p90_a0.25",
                "frame-thr_p90_a0.5", "frame-thr_p90_a1.0"]
CONDS = ("id", "white5db", "gain-12", "gain+12")


def corrupt(clean, noisy, cond, idx):
    """The UQ study's OOD recipes (white noise @5 dB SNR vs clean; +/-12 dB)."""
    import torch

    if cond == "id":
        return noisy
    if cond == "gain+12":
        return noisy * (10 ** (12 / 20))
    if cond == "gain-12":
        return noisy * (10 ** (-12 / 20))
    rng = np.random.default_rng(1000 + idx)   # per-utt seed (parallel-safe)
    w = torch.from_numpy(rng.standard_normal(clean.shape[-1]).astype(np.float32))
    w = w * clean.pow(2).mean().sqrt() / w.pow(2).mean().sqrt() * (10 ** (-5 / 20))
    return clean + w


# ------------------------------------------------------------------ pass 1
_CTX = {}


def _ctx():
    if "sess" in _CTX:
        return _CTX
    import onnxruntime as ort
    import torch

    torch.set_num_threads(1)
    from common.dataset import Dataset, load_voicebank_demand
    from common.env import AttrDict
    from lisennet.export_onnx import _load_from_checkpoint

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    _CTX["sess"] = ort.InferenceSession(str(FOLD), sess_options=so,
                                        providers=["CPUExecutionProvider"])
    _CTX["state_in"] = [i.name for i in _CTX["sess"].get_inputs() if i.name != "feat"]
    _CTX["zeros"] = {i.name: np.zeros([1 if isinstance(d, str) else d for d in i.shape],
                                      np.float32)
                     for i in _CTX["sess"].get_inputs() if i.name != "feat"}
    _CTX["model"] = _load_from_checkpoint(CKPT)
    with open(CKPT_DIR / "config.json") as f:
        h = AttrDict(json.load(f))
    _CTX["ds"] = Dataset(load_voicebank_demand()["test"], h.segment_size,
                         h.sampling_rate, split=False, shuffle=False, seed=h.seed)
    return _CTX


def collect_one(idx, cond="id"):
    """Stream the K=4 fold over utterance idx (under `cond`); cache the stack."""
    import torch

    c = _ctx()
    p = CACHE / f"u{idx:04d}_{cond}.npz"
    if p.exists():
        return str(p)
    model, sess, ds = c["model"], c["sess"], c["ds"]
    clean, noisy = ds[idx]
    noisy = corrupt(clean, noisy, cond, idx)
    with torch.no_grad():
        spec = model.power_compress(model.apply_stft(noisy.unsqueeze(0)))
        mag, pha = spec.abs(), spec.angle()
        feat = model.build_features(mag, pha).numpy().astype(np.float32)
    T = feat.shape[2]
    states = dict(c["zeros"])
    ems = np.empty((K, T, feat.shape[3]), np.float32)
    for t in range(T):
        res = sess.run(None, {"feat": feat[:, :, t:t + 1, :], **states})
        ems[:, t] = res[0][0, :, 0]
        states = dict(zip(c["state_in"], res[1:]))
    np.savez_compressed(
        p, ems=ems, mag=mag.numpy().astype(np.float32)[0],
        pha=pha.numpy().astype(np.float32)[0],
        clean=clean.numpy().astype(np.float32),
        noisy=noisy.numpy().astype(np.float32))
    return str(p)


# ------------------------------------------------------------------ pass 2
def gate_alpha(form, D_t, D_tf, theta_t, theta_tf, amax):
    if form == "frame-lin":
        a = amax * np.minimum(1.0, D_t / theta_t)
        return a[:, None]
    if form == "frame-thr":
        return amax * (D_t > theta_t).astype(np.float32)[:, None]
    if form == "bin-lin":
        return amax * np.minimum(1.0, D_tf / theta_tf)
    if form == "bin-thr":
        return amax * (D_tf > theta_tf).astype(np.float32)
    raise ValueError(form)


def eval_one(idx, thetas, cond="id", names=None):
    """Gate variants for one cached utterance -> per-variant PESQ (+noisy)."""
    import torch

    from lisennet.eval_metrics_ext import _pesq_one

    c = _ctx()
    model = c["model"]
    z = np.load(CACHE / f"u{idx:04d}_{cond}.npz")
    ems, mag, pha = z["ems"], z["mag"], z["pha"]           # (K,T,F), (T,F), (T,F)
    clean, noisy = z["clean"], z["noisy"]
    length = len(noisy)
    avg = ems.mean(axis=0)
    D_tf = ems.var(axis=0)
    D_t = D_tf.mean(axis=-1)

    out = {}
    with torch.no_grad():
        pha_t = torch.from_numpy(pha).unsqueeze(0)
        for name, form, pct, amax in VARIANTS:
            if names is not None and name not in names:
                continue
            if form is None:
                est = avg
            else:
                a = gate_alpha(form, D_t, D_tf, thetas[f"t_p{pct}"],
                               thetas[f"tf_p{pct}"], amax)
                est = (1.0 - a) * avg + a * mag
            em = torch.from_numpy(est).unsqueeze(0)
            spec = torch.complex(em * pha_t.cos(), em * pha_t.sin())
            wav = model.apply_istft(model.power_uncompress(spec), length=length)
            out[name] = float(_pesq_one(clean, wav.squeeze().numpy()))
    out["noisy_pesq"] = float(_pesq_one(clean, noisy))
    out["D_t_mean"] = float(D_t.mean())
    return out


def wav_for(idx, variant, thetas, cond="id"):
    """Reconstruct one variant's waveform (for DNSMOS / listening exports)."""
    import torch

    c = _ctx()
    model = c["model"]
    z = np.load(CACHE / f"u{idx:04d}_{cond}.npz")
    ems, mag, pha, noisy = z["ems"], z["mag"], z["pha"], z["noisy"]
    if variant == "noisy":
        return noisy
    avg = ems.mean(axis=0)
    D_tf = ems.var(axis=0)
    D_t = D_tf.mean(axis=-1)
    name, form, pct, amax = next(v for v in VARIANTS if v[0] == variant)
    if form is None:
        est = avg
    else:
        a = gate_alpha(form, D_t, D_tf, thetas[f"t_p{pct}"],
                       thetas[f"tf_p{pct}"], amax)
        est = (1.0 - a) * avg + a * mag
    with torch.no_grad():
        em = torch.from_numpy(est).unsqueeze(0)
        pha_t = torch.from_numpy(pha).unsqueeze(0)
        spec = torch.complex(em * pha_t.cos(), em * pha_t.sin())
        return model.apply_istft(model.power_uncompress(spec),
                                 length=len(noisy)).squeeze().numpy()


def tail_stats(deltas):
    d = np.asarray(deltas)
    d = d[np.isfinite(d)]
    return {"mean": float(d.mean()),
            "p1": float(np.percentile(d, 1)), "p5": float(np.percentile(d, 5)),
            "p10": float(np.percentile(d, 10)),
            "n_neg": int((d < 0).sum()), "n": int(len(d))}


def _report(res, tag, n):
    print(f"\n[{tag}] ungated: PESQ {res['ungated_pesq']:.4f}, noisy "
          f"{res['noisy_pesq']:.4f}  (vs noisy: harmed "
          f"{res['ungated_vs_noisy']['n_neg']}/{n}, "
          f"p1 {res['ungated_vs_noisy']['p1']:+.3f})", flush=True)
    for name in sorted(res["variants"],
                       key=lambda k: -res["variants"][k]["vs_noisy"]["p1"]):
        r = res["variants"][name]
        if name == "none":
            continue
        print(f"  {name:>18}: mean {r['pesq']:.4f} ({r['vs_ungated']['mean']:+.4f})  "
              f"vs-noisy p1 {r['vs_noisy']['p1']:+.3f} p5 {r['vs_noisy']['p5']:+.3f} "
              f"harmed {r['vs_noisy']['n_neg']}/{n}", flush=True)


def _pool_stats(rows, names):
    base = np.array([r["none"] for r in rows])
    noisy = np.array([r["noisy_pesq"] for r in rows])
    out = {"ungated_pesq": float(np.nanmean(base)),
           "noisy_pesq": float(np.nanmean(noisy)),
           "mean_D_t": float(np.mean([r["D_t_mean"] for r in rows])),
           "ungated_vs_noisy": tail_stats(base - noisy),
           "variants": {}}
    for name in names:
        v = np.array([r[name] for r in rows])
        out["variants"][name] = {
            "pesq": float(np.nanmean(v)),
            "vs_ungated": tail_stats(v - base),
            "vs_noisy": tail_stats(v - noisy),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=824)
    ap.add_argument("--n_ood", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--dnsmos", choices=["none", "subset", "full"], default="full")
    args = ap.parse_args()

    from joblib import Parallel, delayed

    CACHE.mkdir(parents=True, exist_ok=True)
    from common.dataset import load_voicebank_demand

    n = min(args.n, len(load_voicebank_demand()["test"]))
    n_ood = min(args.n_ood, n)

    t0 = time.time()
    Parallel(n_jobs=args.jobs)(delayed(collect_one)(i, "id") for i in range(n))
    print(f"ID member stacks cached ({time.time() - t0:.0f}s)", flush=True)

    # deployment-constant thetas from the pooled ID D distribution
    Dt_all, Dtf_samp = [], []
    rng = np.random.default_rng(0)
    for i in range(n):
        D_tf = np.load(CACHE / f"u{i:04d}_id.npz")["ems"].var(axis=0)
        Dt_all.append(D_tf.mean(axis=-1))
        flat = D_tf.ravel()
        Dtf_samp.append(flat[rng.integers(0, len(flat), 200)])
    Dt_all = np.concatenate(Dt_all)
    Dtf_samp = np.concatenate(Dtf_samp)
    thetas = {}
    for pct in (75, 90):
        thetas[f"t_p{pct}"] = float(np.percentile(Dt_all, pct))
        thetas[f"tf_p{pct}"] = float(np.percentile(Dtf_samp, pct))
    print("thetas (ID-frozen):", {k: f"{v:.3e}" for k, v in thetas.items()},
          flush=True)

    res = {"n": n, "n_ood": n_ood, "thetas": thetas, "fold": FOLD.name}

    t0 = time.time()
    rows = Parallel(n_jobs=args.jobs)(
        delayed(eval_one)(i, thetas, "id") for i in range(n))
    print(f"ID gate sweep + PESQ done ({time.time() - t0:.0f}s)", flush=True)
    res["id"] = _pool_stats(rows, [v[0] for v in VARIANTS])
    _report(res["id"], "id", n)

    for cond in ("white5db", "gain-12", "gain+12"):
        t0 = time.time()
        Parallel(n_jobs=args.jobs)(
            delayed(collect_one)(i, cond) for i in range(n_ood))
        rows = Parallel(n_jobs=args.jobs)(
            delayed(eval_one)(i, thetas, cond, set(OOD_VARIANTS))
            for i in range(n_ood))
        res[cond] = _pool_stats(rows, OOD_VARIANTS)
        res[cond]["gate_duty_note"] = "thetas ID-frozen"
        _report(res[cond], cond, n_ood)
        print(f"[{cond}] mean D_t {res[cond]['mean_D_t']:.2e} "
              f"(ID {res['id']['mean_D_t']:.2e})  ({time.time() - t0:.0f}s)",
              flush=True)

    json.dump(res, open(OUT, "w"), indent=1)

    # ---- DNSMOS: ID full + white5db subset, noisy/ungated/2 gates --------
    if args.dnsmos != "none":
        from lisennet.eval_metrics_ext import DNSMOS

        dns = DNSMOS()
        for cond, n_d in (("id", n if args.dnsmos == "full" else min(200, n)),
                          ("white5db", n_ood)):
            ranked = [k for k in sorted(
                res[cond]["variants"],
                key=lambda k: -res[cond]["variants"][k]["vs_noisy"]["p1"])
                if k != "none"]
            picks = ["noisy", "none"] + ranked[:2]
            for variant in picks:
                wavs = Parallel(n_jobs=args.jobs)(
                    delayed(wav_for)(i, variant, thetas, cond) for i in range(n_d))
                scores = Parallel(n_jobs=args.jobs, backend="threading")(
                    delayed(dns)(w) for w in wavs)
                agg = {k: float(np.mean([s[k] for s in scores]))
                       for k in ("SIG", "BAK", "OVRL", "P808")}
                res[cond].setdefault("dnsmos", {})[variant] = {"n": n_d, **agg}
                print(f"DNSMOS [{cond}] {variant:>18}: " +
                      "  ".join(f"{k}={v:.3f}" for k, v in agg.items()), flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
