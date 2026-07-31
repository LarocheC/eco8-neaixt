"""SR-ensemble: does averaging K stochastic-rounding weight draws recover int8 PESQ?

relu6-deep, Table-1 protocol (full-utterance static-int8 graph + noisy phase,
`paper/data/eval_conv_rt_int8.py` harness; its pc_signed anchor: PESQ 3.0268).

Members share the deployed pc_signed quantization grid — every activation Q/DQ
and every scale/zero-point stays fixed; only the *rounding realization* of the
selected int8 weight tensors differs per member, and members' est_mag
(compressed magnitude) outputs are averaged. `*_w_exact` bypasses the weight
DQ with the FP32 weights: the K→inf cap of SR ensembling, i.e. the
weight-rounding share of the gap with activation quantization untouched.

Configs:
  fp32          FP32 graph (upper anchor)
  rtn           deployed pc_signed QDQ (round-to-nearest baseline)
  dec_w_exact   decoder weights exact          (cap for dec_sr_*)
  all_w_exact   all 144 weight tensors exact   (cap for all_sr_*)
  dec_sr_kN     N SR draws of decoder weights, est_mag averaged  (N in 1,2,4,8)
  all_sr_kN     N SR draws of all weights

usage: ./.venv/bin/python3 sr_decoder_ensemble.py --n 200 --configs rtn dec_w_exact
Results merge into sr_decoder_ensemble_results.json under "<config>@<n_utts>",
with per-utterance PESQ kept for paired deltas vs rtn.
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

REPO = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(REPO))

from common.dataset import Dataset, load_voicebank_demand  # noqa: E402
from common.env import AttrDict  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from lisennet.eval_metrics_ext import _pesq_one, _session, full_graph_mask  # noqa: E402
from lisennet.export_onnx import _load_from_checkpoint  # noqa: E402

TMP = REPO / "paper" / "data" / "tmp_quant"
QDQ = TMP / "relu6deep_rt_int8_pc_signed.onnx"
FP32 = TMP / "relu6deep_rt_int8_fp32.onnx"
for _name in ("cp_lisennet_conv_hardened_deep_relu6",
              "cp_lisennet_conv_hardened_nc24_deep_relu6"):   # laptop / training box
    if (REPO / _name / "g_best").exists():
        CKPT = REPO / _name / "g_best"
        break
OUT = REPO / "sr_decoder_ensemble_results.json"
GRAPH_DIR = TMP / "sr_graphs"  # members regenerate deterministically (CRC-seeded)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def weight_targets(model, scope):
    """[(int8_init_name, fp32_name, scale_arr, axis, dq_output)] for the scope."""
    inits = {t.name: t for t in model.graph.initializer}
    out = []
    for n in model.graph.node:
        if n.op_type != "DequantizeLinear":
            continue
        wname = n.input[0]
        t = inits.get(wname)
        if t is None or t.data_type != onnx.TensorProto.INT8:
            continue
        if not wname.endswith("_quantized"):
            continue
        base = wname[: -len("_quantized")]
        if scope == "dec" and "decoder" not in base:
            continue
        axis = next((a.i for a in n.attribute if a.name == "axis"), 1)
        scale = numpy_helper.to_array(inits[n.input[1]])
        out.append((wname, base, scale, axis, n.output[0]))
    return out


def fp32_weights():
    m = onnx.load(FP32)
    return {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}


def build_exact(scope, dst):
    """Bypass the weight DQ for `scope` with exact FP32 weights."""
    m = onnx.load(QDQ)
    fw = fp32_weights()
    targets = weight_targets(m, scope)
    rewires = {}
    for wname, base, _s, _ax, dq_out in targets:
        exact = base + "_exact"
        m.graph.initializer.append(
            numpy_helper.from_array(fw[base].astype(np.float32), exact))
        rewires[dq_out] = exact
    n_rw = 0
    for n in m.graph.node:
        for i, x in enumerate(n.input):
            if x in rewires:
                n.input[i] = rewires[x]
                n_rw += 1
    onnx.save(m, dst)
    return len(targets), n_rw


def build_sr(scope, seed, dst):
    """Redraw the scope's int8 weights with stochastic rounding on fixed scales."""
    m = onnx.load(QDQ)
    fw = fp32_weights()
    inits = {t.name: t for t in m.graph.initializer}
    targets = weight_targets(m, scope)
    for wname, base, scale, axis, _dq in targets:
        w = fw[base].astype(np.float64)
        s = scale.astype(np.float64)
        if s.size == 1:
            q = w / float(s)
        else:
            ax = axis if axis < w.ndim and w.shape[axis] == s.size else \
                next(i for i, d in enumerate(w.shape) if d == s.size)
            shape = [1] * w.ndim
            shape[ax] = -1
            q = w / s.reshape(shape)
        lo = np.floor(q)
        frac = q - lo
        rng = np.random.default_rng((zlib.crc32(base.encode()) << 8) ^ seed)
        draw = lo + (rng.random(q.shape) < frac)
        draw = np.clip(draw, -127, 127).astype(np.int8)
        inits[wname].CopyFrom(numpy_helper.from_array(draw, wname))
    onnx.save(m, dst)
    return len(targets)


def member_paths(config):
    if config == "fp32":
        return [FP32]
    if config == "rtn":
        return [QDQ]
    if config.endswith("_w_exact"):
        scope = config.split("_")[0]
        p = GRAPH_DIR / f"{config}.onnx"
        if not p.exists():
            nt, nr = build_exact(scope, p)
            print(f"  built {config}: {nt} weights exact, {nr} inputs rewired")
        return [p]
    scope, _, k = config.partition("_sr_k")
    paths = []
    for mem in range(int(k)):
        p = GRAPH_DIR / f"{scope}_sr_m{mem}.onnx"
        if not p.exists():
            nt = build_sr(scope, mem, p)
            print(f"  built {p.name}: {nt} weights redrawn (seed {mem})")
        paths.append(p)
    return paths


@torch.no_grad()
def eval_config(config, model, ds, n):
    sessions = [_session(p) for p in member_paths(config)]
    refs, wavs = [], []
    for i in range(n):
        clean, noisy = ds[i]
        clean, noisy = clean.unsqueeze(0), noisy.unsqueeze(0)
        length = noisy.shape[-1]
        spec = model.power_compress(model.apply_stft(noisy))
        mag, pha = spec.abs(), spec.angle()
        feat = model.build_features(mag, pha).numpy().astype(np.float32)
        ems = [full_graph_mask(s, feat) for s in sessions]
        em = torch.from_numpy(np.mean(ems, axis=0).astype(np.float32))
        est = torch.complex(em * pha.cos(), em * pha.sin())
        wav = model.apply_istft(model.power_uncompress(est), length=length)
        refs.append(clean.squeeze().numpy())
        wavs.append(wav.squeeze().numpy())
    pesqs = Parallel(n_jobs=24)(delayed(_pesq_one)(r, e) for r, e in zip(refs, wavs))
    return [float(p) for p in pesqs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=824)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with open(REPO / "cp_lisennet" / "config.json") as f:
        h = AttrDict(json.load(f))
    model = _load_from_checkpoint(CKPT)
    ds = Dataset(load_voicebank_demand()["test"], h.segment_size, h.sampling_rate,
                 split=False, shuffle=False, seed=h.seed)
    n = min(args.n, len(ds))

    results = json.load(open(OUT)) if OUT.exists() else {}
    for config in args.configs:
        key = f"{config}@{n}"
        if key in results and not args.force:
            print(f"{key}: cached PESQ {results[key]['pesq_mean']:.4f}")
            continue
        import time
        t0 = time.time()
        pesqs = eval_config(config, model, ds, n)
        valid = [p for p in pesqs if np.isfinite(p)]
        results[key] = {"pesq_mean": float(np.mean(valid)), "n_valid": len(valid),
                        "members": len(member_paths(config)), "pesq": pesqs,
                        "secs": round(time.time() - t0, 1)}
        json.dump(results, open(OUT, "w"), indent=1)
        print(f"{key}: PESQ {results[key]['pesq_mean']:.4f}  "
              f"({len(valid)}/{n} valid, {results[key]['secs']}s)", flush=True)

    rtn_key = f"rtn@{n}"
    if rtn_key in results:
        base = np.array(results[rtn_key]["pesq"])
        print(f"\npaired deltas vs {rtn_key} (mean ± 1.96·se):")
        for key, r in sorted(results.items()):
            if not key.endswith(f"@{n}") or key == rtn_key:
                continue
            d = np.array(r["pesq"]) - base
            d = d[np.isfinite(d)]
            se = d.std(ddof=1) / np.sqrt(len(d))
            print(f"  {key:>16}: {d.mean():+.4f} ± {1.96 * se:.4f}")


if __name__ == "__main__":
    main()
