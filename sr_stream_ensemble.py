"""Streamed SR-ensemble eval: the FIFO-streaming graph, frame by frame, on VBD.

The offline study (`sr_decoder_ensemble.py`) measured SR-draw ensembling on the
full-utterance relu6-deep graph. Deployment runs the *streaming* graph
(`g_best_streaming_int8_static.onnx`: feat + 25 FIFO states -> est_mag + states),
which carries its own activation calibration (state-threaded, percentile). This
harness answers whether SR ensembling survives that grid: members are SR draws
of the streaming graph's int8 weights on the *fixed* deployed scales, each member
streams with its own FIFO state, and the members' per-frame est_mag are averaged.

Anchor: `paper/data/stream_pesq.json` recorded the streamed RTN PESQ 3.0132
("r6"); the rtn config here must reproduce it.

Configs:
  rtn           deployed streaming QDQ (round-to-nearest baseline)
  fp32          streaming FP32 graph (upper anchor)
  all_w_exact   all int8 weight tensors bypassed with exact FP32 weights
  all_sr_kN     N SR draws of all weights, est_mag averaged  (N in 1,2,4,8)
  memb_fold_kN  the member-correct folded graph ens_memb_k<N>.onnx (one session
                computing all N members; est_mag (B,K,1,F) averaged over K) —
                must match all_sr_kN to float noise, see fold_members.py

usage: ./.venv/bin/python3 sr_stream_ensemble.py --n 824 --configs rtn fp32 all_sr_k2 all_sr_k4
Results merge into sr_stream_ensemble_results.json under "<config>@<n_utts>",
with per-utterance PESQ kept for paired deltas vs rtn.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(REPO))

# Checkpoint dir differs between the training box and the laptop copy.
for _name in ("cp_lisennet_conv_hardened_nc24_deep_relu6", "cp_lisennet_conv_hardened_deep_relu6"):
    CKPT_DIR = REPO / _name
    if (CKPT_DIR / "g_best").exists():
        break
CKPT = CKPT_DIR / "g_best"
TMP = REPO / "paper" / "data" / "tmp_quant"


def _graph(local_name, tracked_name):
    """Prefer the fresh export next to the checkpoint; fall back to the tracked copy."""
    p = CKPT_DIR / local_name
    return p if p.exists() else TMP / tracked_name


QDQ = _graph("g_best_streaming_int8_static.onnx", "relu6deep_streaming_int8_signed.onnx")
FP32 = _graph("g_best_streaming_fp32.onnx", "relu6deep_streaming_fp32.onnx")
OUT = REPO / "sr_stream_ensemble_results.json"
GRAPH_DIR = REPO / "paper" / "data" / "tmp_quant" / "sr_stream_graphs"
ENS_DIR = REPO / "paper" / "data" / "tmp_quant" / "ens_graphs"  # fold_members.py output


# --------------------------------------------------------------------- members
def weight_targets(model):
    """[(int8_init_name, fp32_name, scale_arr, axis)] for every int8 weight DQ."""
    import onnx
    from onnx import numpy_helper

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
        axis = next((a.i for a in n.attribute if a.name == "axis"), 1)
        scale = numpy_helper.to_array(inits[n.input[1]])
        out.append((wname, base, scale, axis, n.output[0]))
    return out


def fp32_weights():
    import onnx
    from onnx import numpy_helper

    m = onnx.load(FP32)
    return {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}


def build_sr(seed, dst, qdq=None):
    """Redraw all int8 weights with stochastic rounding on the fixed scales.

    Same draw scheme as the offline study (CRC-per-tensor ^ seed), so member i
    here is the streaming-graph analogue of offline member i.
    """
    import onnx
    from onnx import numpy_helper

    m = onnx.load(qdq or QDQ)
    fw = fp32_weights()
    inits = {t.name: t for t in m.graph.initializer}
    targets = weight_targets(m)
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


def build_exact(dst, qdq=None):
    """Bypass every int8 weight DQ with the exact FP32 weights (acts stay int8)."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(qdq or QDQ)
    fw = fp32_weights()
    targets = weight_targets(m)
    rewires = {}
    for wname, base, _s, _ax, dq_out in targets:
        exact = base + "_exact"
        m.graph.initializer.append(
            numpy_helper.from_array(fw[base].astype(np.float32), exact))
        rewires[dq_out] = exact
    for n in m.graph.node:
        for i, x in enumerate(n.input):
            if x in rewires:
                n.input[i] = rewires[x]
    onnx.save(m, dst)
    return len(targets)


def member_paths(config):
    """Resolve a config to the list of ONNX graphs to stream (1 per member).

    An `mm_` prefix evaluates the same config on the MinMax-calibrated grid
    (g_best_streaming_int8_minmax.onnx) instead of the deployed percentile one —
    the attribution experiment for the streamed-vs-offline recovery difference.
    """
    qdq, gdir = QDQ, GRAPH_DIR
    if config.startswith("mm_"):
        config = config[3:]
        qdq = CKPT_DIR / "g_best_streaming_int8_minmax.onnx"
        gdir = GRAPH_DIR.with_name(GRAPH_DIR.name + "_mm")
        assert qdq.exists(), f"{qdq} missing — build the MinMax quantization first"
    if config == "fp32":
        return [FP32]
    if config == "rtn":
        return [qdq]
    if config == "all_w_exact":
        p = gdir / "all_w_exact.onnx"
        if not p.exists():
            gdir.mkdir(parents=True, exist_ok=True)
            nt = build_exact(p, qdq)
            print(f"  built {p.name}: {nt} weights exact")
        return [p]
    if config.startswith("memb_fold_k"):
        p = ENS_DIR / f"ens_memb_k{int(config.rsplit('k', 1)[1])}.onnx"
        assert p.exists(), f"{p} missing — run fold_members.py first"
        return [p]
    assert config.startswith("all_sr_k"), f"unknown config {config!r}"
    k = int(config.rsplit("k", 1)[1])
    paths = []
    for mem in range(k):
        p = gdir / f"all_sr_stream_m{mem}.onnx"
        if not p.exists():
            gdir.mkdir(parents=True, exist_ok=True)
            nt = build_sr(mem, p, qdq)
            print(f"  built {p.name}: {nt} weights redrawn (seed {mem})")
        paths.append(p)
    return paths


# --------------------------------------------------------------------- workers
_CTX = {}  # per-process cache: model + sessions per config


def _worker_ctx(config):
    if config in _CTX:
        return _CTX[config]
    import onnxruntime as ort
    import torch

    torch.set_num_threads(1)
    sys.path.insert(0, str(REPO))
    from lisennet.export_onnx import _load_from_checkpoint

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    sessions = [ort.InferenceSession(str(p), sess_options=so,
                                     providers=["CPUExecutionProvider"])
                for p in member_paths(config)]
    model = _load_from_checkpoint(CKPT)
    # state wiring: graph outputs are (est_mag, state_0_out, ...) in fixed order
    s0 = sessions[0]
    state_in = [i.name for i in s0.get_inputs() if i.name != "feat"]
    out_names = [o.name for o in s0.get_outputs()]
    assert out_names[0] == "est_mag" and \
        out_names[1:] == [n.replace("_in", "_out") for n in state_in]
    zero_states = []
    for s in sessions:
        zs = {i.name: np.zeros([1 if isinstance(d, str) or d is None else d
                                for d in i.shape], np.float32)
              for i in s.get_inputs() if i.name != "feat"}
        zero_states.append(zs)
    _CTX[config] = (model, sessions, state_in, zero_states)
    return _CTX[config]


def stream_members(sessions, state_in_names, zero_states, feat_np):
    """Stream every member over feat (1,3,T,F); return member-avg est_mag (T,F).

    A folded member-correct graph is a single 'member' whose est_mag carries K
    member channels (1,K,1,F) — averaged over them; single-member graphs emit
    (1,1,F). Either way the frame estimate is the member mean.
    """
    T = feat_np.shape[2]
    ests = np.zeros((T, feat_np.shape[3]), np.float32)
    for sess, zs in zip(sessions, zero_states):
        states = dict(zs)
        acc = []
        for t in range(T):
            feed = {"feat": feat_np[:, :, t:t + 1, :]}
            feed.update(states)
            res = sess.run(None, feed)
            em = res[0]
            if em.ndim == 4:                      # folded: (1,K,1,F) -> member mean
                em = em.mean(axis=1)              # (1,1,F)
            acc.append(em[0, 0])
            states = dict(zip(state_in_names, res[1:]))
        ests += np.stack(acc)
    return ests / len(sessions)


def enhance_one(config, idx):
    """Worker: stream utterance `idx` under `config`; return (ref, est) waveforms."""
    import torch

    from common.dataset import Dataset, load_voicebank_demand
    from common.env import AttrDict

    model, sessions, state_in, zero_states = _worker_ctx(config)
    if "ds" not in _CTX:
        with open(CKPT_DIR / "config.json") as f:
            h = AttrDict(json.load(f))
        hf = load_voicebank_demand()
        _CTX["ds"] = Dataset(hf["test"], h.segment_size, h.sampling_rate,
                             split=False, shuffle=False, seed=h.seed)
    ds = _CTX["ds"]
    clean, noisy = ds[idx]
    clean, noisy = clean.unsqueeze(0), noisy.unsqueeze(0)
    length = noisy.shape[-1]
    with torch.no_grad():
        spec = model.power_compress(model.apply_stft(noisy))
        mag, pha = spec.abs(), spec.angle()
        feat = model.build_features(mag, pha).numpy().astype(np.float32)
        em = stream_members(sessions, state_in, zero_states, feat)   # (T,F)
        em = torch.from_numpy(em).unsqueeze(0)
        est = torch.complex(em * pha.cos(), em * pha.sin())
        wav = model.apply_istft(model.power_uncompress(est), length=length)
    return clean.squeeze().numpy(), wav.squeeze().numpy()


def eval_config(config, n, n_jobs):
    from joblib import Parallel, delayed

    from lisennet.eval_metrics_ext import _pesq_one

    member_paths(config)                       # build member graphs up front (once)
    pairs = Parallel(n_jobs=n_jobs)(
        delayed(enhance_one)(config, i) for i in range(n))
    pesqs = Parallel(n_jobs=n_jobs)(
        delayed(_pesq_one)(r, e) for r, e in pairs)
    return [float(p) for p in pesqs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=824)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from common.dataset import load_voicebank_demand

    n = min(args.n, len(load_voicebank_demand()["test"]))

    results = json.load(open(OUT)) if OUT.exists() else {}
    for config in args.configs:
        key = f"{config}@{n}"
        if key in results and not args.force:
            print(f"{key}: cached PESQ {results[key]['pesq_mean']:.4f}")
            continue
        t0 = time.time()
        pesqs = eval_config(config, n, args.jobs)
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
