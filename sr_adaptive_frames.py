"""Frame-level adaptive-K ceiling: escalate only high-disagreement frames.

Per utterance: K=2 mask (members 0,1), K=4 mask (members 0-3), D2 per frame
from the deployed pair. Mix masks frame-by-frame — K=4 on the top-X% D2 frames
of the utterance, K=2 elsewhere — synthesize, PESQ. `rand` rows escalate the
same fraction of random frames (selectivity control). Offline full-utterance
graphs, so this is the *optimistic ceiling*: a streaming deployment must keep
escalated members' FIFO states warm (e.g. state-copy from the base pair).

usage: ./.venv/bin/python3 sr_adaptive_frames.py [--n 824]
Results -> sr_adaptive_frames_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from common.dataset import Dataset, load_voicebank_demand  # noqa: E402
from common.env import AttrDict  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from lisennet.eval_metrics_ext import _pesq_one, _session, full_graph_mask  # noqa: E402
from lisennet.export_onnx import _load_from_checkpoint  # noqa: E402

GRAPHS = REPO / "paper" / "data" / "tmp_quant" / "sr_graphs"  # built by sr_decoder_ensemble.py
for _name in ("cp_lisennet_conv_hardened_deep_relu6",
              "cp_lisennet_conv_hardened_nc24_deep_relu6"):   # laptop / training box
    if (REPO / _name / "g_best").exists():
        CKPT = REPO / _name / "g_best"
        break
OUT = REPO / "sr_adaptive_frames_results.json"

FRACS = (0.0, 0.10, 0.25, 0.50, 1.0)
RAND_FRACS = (0.10, 0.25, 0.50)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=824)
    args = ap.parse_args()

    with open(REPO / "cp_lisennet" / "config.json") as f:
        h = AttrDict(json.load(f))
    model = _load_from_checkpoint(CKPT)
    ds = Dataset(load_voicebank_demand()["test"], h.segment_size, h.sampling_rate,
                 split=False, shuffle=False, seed=h.seed)
    n = min(args.n, len(ds))
    sessions = [_session(GRAPHS / f"all_sr_m{m}.onnx") for m in range(4)]
    rng = np.random.default_rng(7)

    policies = [("d2", f) for f in FRACS] + [("rand", f) for f in RAND_FRACS]
    pesqs = {p: [] for p in policies}
    t0 = time.time()
    for i in range(n):
        clean, noisy = ds[i]
        clean, noisy = clean.unsqueeze(0), noisy.unsqueeze(0)
        length = noisy.shape[-1]
        spec = model.power_compress(model.apply_stft(noisy))
        mag, pha = spec.abs(), spec.angle()
        feat = model.build_features(mag, pha).numpy().astype(np.float32)
        ems = np.stack([full_graph_mask(s, feat) for s in sessions])  # (4,1,T,F)
        em2, em4 = ems[:2].mean(axis=0), ems.mean(axis=0)             # (1,T,F)
        d2 = ((ems[0] - ems[1]) ** 2).mean(axis=-1).ravel()           # (T,)
        t_frames = len(d2)

        jobs = []
        for kind, frac in policies:
            if frac == 0.0:
                em = em2
            elif frac == 1.0:
                em = em4
            else:
                k = int(round(frac * t_frames))
                if kind == "d2":
                    hi = np.argsort(-d2)[:k]
                else:
                    hi = rng.choice(t_frames, size=k, replace=False)
                em = em2.copy()
                em[:, hi, :] = em4[:, hi, :]
            emt = torch.from_numpy(em.astype(np.float32))
            est = torch.complex(emt * pha.cos(), emt * pha.sin())
            wav = model.apply_istft(model.power_uncompress(est), length=length)
            jobs.append(((kind, frac), clean.squeeze().numpy(),
                         wav.squeeze().numpy()))
        scores = Parallel(n_jobs=8)(delayed(_pesq_one)(r, e)
                                    for _, r, e in jobs)
        for (key, _r, _e), s in zip(jobs, scores):
            pesqs[key].append(float(s))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)

    res = {"n": n}
    base = np.nanmean(pesqs[("d2", 0.0)])
    for key in policies:
        kind, frac = key
        m = float(np.nanmean(pesqs[key]))
        res[f"{kind}@{frac}"] = {"pesq": m, "members_avg": 2 + 2 * frac,
                                 "delta_vs_k2": round(m - base, 4)}
        print(f"{kind:>4} escalate {frac:>5.0%}: PESQ {m:.4f} "
              f"({m - base:+.4f} vs K=2, {2 + 2 * frac:.1f} members avg)",
              flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
