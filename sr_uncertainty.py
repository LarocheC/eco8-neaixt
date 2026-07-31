"""Is SR-member disagreement a usable uncertainty / OOD signal?

Three questions, one pass (relu6-deep, Table-1 protocol features):

1. UQ validity: does per-frame member disagreement D predict where the int8
   ensemble deviates from FP32 (E = squared est_mag error vs FP32)? Frame-level
   pooled + per-utterance Spearman, plus an energy-controlled version (within
   energy deciles) so D doesn't win by just tracking loudness. Utterance-level:
   does D_utt predict the residual PESQ damage (pesq_fp32 - pesq_k8, from
   sr_decoder_ensemble_results.json)?

2. OOD: AUROC of D_utt (and input clip-rate) for ID vs gain-shifted (+/-12 dB)
   and unseen white noise @5 dB SNR conditions. Watch for variance collapse
   under saturation.

3. Adaptive-K: utterance-level policy sim "K=2 always, escalate to K=4 when
   D2 > theta", where D2 uses only the two members a K=2 deployment carries.
   PESQ-vs-cost curve from the existing per-utt PESQ lists; oracle for
   reference.

usage: ./.venv/bin/python3 sr_uncertainty.py [--n 824] [--n_ood 200]
Results -> sr_uncertainty_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

REPO = Path("/home/claroche/eco8-neaixt")
sys.path.insert(0, str(REPO))

from common.dataset import Dataset, load_voicebank_demand  # noqa: E402
from common.env import AttrDict  # noqa: E402
from lisennet.eval_metrics_ext import _session, full_graph_mask  # noqa: E402
from lisennet.export_onnx import _load_from_checkpoint  # noqa: E402

TMP = REPO / "paper" / "data" / "tmp_quant"
GRAPHS = TMP / "sr_graphs"  # built by sr_decoder_ensemble.py (run it first)
CKPT = REPO / "cp_lisennet_conv_hardened_deep_relu6" / "g_best"
ENS = REPO / "sr_decoder_ensemble_results.json"
OUT = REPO / "sr_uncertainty_results.json"
K = 8


def spearman(a, b):
    def rank(x):
        r = np.empty(len(x))
        r[np.argsort(x)] = np.arange(len(x))
        return r
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra, rb = rank(a[m]), rank(b[m])
    return float(np.corrcoef(ra, rb)[0, 1])


def auroc(pos, neg):
    """P(pos > neg) via rank statistic."""
    x = np.concatenate([pos, neg])
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(1, len(x) + 1)
    rp = r[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def input_qparams():
    m = onnx.load(TMP / "relu6deep_rt_int8_pc_signed.onnx")
    inits = {t.name: t for t in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0] == "feat":
            s = float(numpy_helper.to_array(inits[n.input[1]]))
            z = int(numpy_helper.to_array(inits[n.input[2]]))
            return s, z
    return None, None


@torch.no_grad()
def masks_and_stats(model, sessions, fp32_sess, noisy):
    """noisy (1,L) -> per-frame D8, D2, E, energy + utt clip rate."""
    spec = model.power_compress(model.apply_stft(noisy))
    mag, pha = spec.abs(), spec.angle()
    feat = model.build_features(mag, pha).numpy().astype(np.float32)
    ems = np.stack([full_graph_mask(s, feat) for s in sessions])  # (K,1,T,F)
    avg = ems.mean(axis=0)
    d8 = ems.var(axis=0).mean(axis=-1).ravel()                    # (T,)
    d2 = (0.25 * (ems[0] - ems[1]) ** 2).mean(axis=-1).ravel()
    ef = en = None
    if fp32_sess is not None:
        fp = full_graph_mask(fp32_sess, feat)
        ef = ((avg - fp) ** 2).mean(axis=-1).ravel()
        en = (fp ** 2).mean(axis=-1).ravel()
    return d8, d2, ef, en, feat


def clip_rate(feat, s, z):
    q = feat / s + z
    return float(((q < -128) | (q > 127)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=824)
    ap.add_argument("--n_ood", type=int, default=200)
    args = ap.parse_args()

    with open(REPO / "cp_lisennet" / "config.json") as f:
        h = AttrDict(json.load(f))
    model = _load_from_checkpoint(CKPT)
    ds = Dataset(load_voicebank_demand()["test"], h.segment_size, h.sampling_rate,
                 split=False, shuffle=False, seed=h.seed)
    n = min(args.n, len(ds))
    n_ood = min(args.n_ood, n)

    sessions = [_session(GRAPHS / f"all_sr_m{m}.onnx") for m in range(K)]
    fp32_sess = _session(TMP / "relu6deep_rt_int8_fp32.onnx")
    s_in, z_in = input_qparams()
    ens = json.load(open(ENS))
    p_fp32 = np.array(ens["fp32@824"]["pesq"])[:n]
    p_k2 = np.array(ens["all_sr_k2@824"]["pesq"])[:n]
    p_k4 = np.array(ens["all_sr_k4@824"]["pesq"])[:n]
    p_k8 = np.array(ens["all_sr_k8@824"]["pesq"])[:n]

    # ---- Phase A: in-distribution pass -----------------------------------
    fD, fE, fEn = [], [], []
    D8u, D2u, clipu = [], [], []
    t0 = time.time()
    for i in range(n):
        clean, noisy = ds[i]
        d8, d2, ef, en, feat = masks_and_stats(
            model, sessions, fp32_sess, noisy.unsqueeze(0))
        fD.append(d8)
        fE.append(ef)
        fEn.append(en)
        D8u.append(float(d8.mean()))
        D2u.append(float(d2.mean()))
        clipu.append(clip_rate(feat, s_in, z_in))
        if (i + 1) % 200 == 0:
            print(f"  ID pass {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    D8u, D2u, clipu = map(np.array, (D8u, D2u, clipu))
    Df, Ef, Enf = map(np.concatenate, (fD, fE, fEn))

    res = {"n": n, "frames": int(len(Df))}
    res["frame_spearman_D_vs_E"] = spearman(Df, Ef)
    per_utt = [spearman(d, e) for d, e in zip(fD, fE) if len(d) > 10]
    res["frame_spearman_per_utt_median"] = float(np.nanmedian(per_utt))
    dec = np.quantile(Enf, np.linspace(0, 1, 11))
    within = [spearman(Df[m], Ef[m]) for lo, hi in zip(dec[:-1], dec[1:])
              if (m := (Enf >= lo) & (Enf < hi)).sum() > 100]
    res["frame_spearman_energy_controlled"] = float(np.nanmean(within))
    res["utt_spearman_D8_vs_residual_damage"] = spearman(D8u, p_fp32 - p_k8)
    res["utt_spearman_D2_vs_escalation_gain"] = spearman(D2u, p_k4 - p_k2)
    print(f"UQ: frame Spearman(D,E)={res['frame_spearman_D_vs_E']:.3f}  "
          f"per-utt median={res['frame_spearman_per_utt_median']:.3f}  "
          f"energy-controlled={res['frame_spearman_energy_controlled']:.3f}",
          flush=True)
    print(f"UQ utt: D8 vs (pesq_fp32-pesq_k8)="
          f"{res['utt_spearman_D8_vs_residual_damage']:.3f}   "
          f"D2 vs (pesq_k4-pesq_k2)="
          f"{res['utt_spearman_D2_vs_escalation_gain']:.3f}", flush=True)

    # ---- Phase B: OOD conditions -----------------------------------------
    rng = np.random.default_rng(0)
    conds = {}
    for name in ("gain+12", "gain-12", "white5db"):
        du, cu = [], []
        for i in range(n_ood):
            clean, noisy = ds[i]
            if name == "gain+12":
                x = noisy * (10 ** (12 / 20))
            elif name == "gain-12":
                x = noisy * (10 ** (-12 / 20))
            else:
                w = torch.from_numpy(
                    rng.standard_normal(clean.shape[-1]).astype(np.float32))
                w *= clean.pow(2).mean().sqrt() / w.pow(2).mean().sqrt() \
                    * (10 ** (-5 / 20))
                x = clean + w
            d8, d2, _, _, feat = masks_and_stats(
                model, sessions, None, x.unsqueeze(0))
            du.append(float(d8.mean()))
            cu.append(clip_rate(feat, s_in, z_in))
        du, cu = np.array(du), np.array(cu)
        conds[name] = {
            "mean_D8": float(du.mean()), "mean_clip": float(cu.mean()),
            "auroc_D8": auroc(du, D8u[:n_ood]),
            "auroc_clip": auroc(cu, clipu[:n_ood]),
        }
        print(f"OOD {name}: mean_D8={du.mean():.2e} (ID {D8u[:n_ood].mean():.2e})  "
              f"AUROC_D8={conds[name]['auroc_D8']:.3f}  "
              f"AUROC_clip={conds[name]['auroc_clip']:.3f}", flush=True)
    res["ood"] = conds
    res["id_mean_D8"] = float(D8u.mean())
    res["id_mean_clip"] = float(clipu.mean())

    # ---- Phase C: adaptive-K policy sim (utterance granularity) ----------
    order = np.argsort(-D2u)  # escalate highest-D2 first
    curve = []
    for frac in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        k_hi = order[: int(round(frac * n))]
        mix = p_k2.copy()
        mix[k_hi] = p_k4[k_hi]
        curve.append({"escalate_frac": frac,
                      "mean_members": 2 + 2 * frac,
                      "pesq": float(np.nanmean(mix))})
    oracle = p_k2.copy()
    gain = p_k4 - p_k2
    oracle[gain > 0] = p_k4[gain > 0]
    res["adaptive_curve"] = curve
    res["adaptive_oracle"] = {"pesq": float(np.nanmean(oracle)),
                              "escalate_frac": float((gain > 0).mean())}
    res["anchors"] = {"k2": float(np.nanmean(p_k2)), "k4": float(np.nanmean(p_k4))}
    for c in curve:
        print(f"adaptive: escalate {c['escalate_frac']:.0%} -> "
              f"{c['mean_members']:.1f} members avg, PESQ {c['pesq']:.4f}",
              flush=True)
    print(f"adaptive oracle: PESQ {res['adaptive_oracle']['pesq']:.4f} "
          f"@ {res['adaptive_oracle']['escalate_frac']:.0%}", flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
