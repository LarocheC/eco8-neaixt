#!/usr/bin/env python
"""Generalized model_io_layout.h generator (multi-model companion to gen_io_layout.py).

Same value-matching approach as gen_io_layout.py — run the torch streaming view and the
tflite on identical random (feat, states) input, match each interpreter output *position*
to its semantic role (mask / state k) by value, and emit the C header — but the streaming
view + example input come from export_tflite_multi.BUILDERS so it works for any model
whose builder is registered there (convfsenet, nsnet2, ...), not just LiSenNet.

Adds MODEL_FEATURE_TOTAL (product of ALL feat dims) so the driver feeds the right element
count for single-channel models (LiSenNet feat is 3-channel; convfsenet/nsnet2 are 1).

    ~/.venvs/rt595-export/bin/python deploy/rt595/host/gen_io_layout_multi.py \
        --model nsnet2 --checkpoint_file cp_nsnet2_plain_synth/g_best \
        --tflite deploy/rt595/host_out/nsnet2_plain_streaming.tflite \
        --output <builddir>/model_io_layout.h
"""
import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint_file", required=True)
    ap.add_argument("--tflite", required=True, help="the int8 streaming .tflite")
    ap.add_argument("--output", required=True)
    ap.add_argument("--repo", default=str(REPO_ROOT))
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    sys.path.insert(0, str(Path(a.repo) / "deploy/rt595/host"))
    import numpy as np
    import torch
    from ai_edge_litert.interpreter import Interpreter
    from export_tflite_multi import BUILDERS

    view, args, in_names, patch_ctx = BUILDERS[a.model](a.checkpoint_file)

    # Randomize every input so each interpreter output is discriminable by value.
    torch.manual_seed(0)
    rand_in = [torch.randn_like(t) for t in args]
    with patch_ctx():
        with torch.no_grad():
            out = view(*rand_in)
    torch_out = [o.numpy() for o in (out if isinstance(out, (tuple, list)) else [out])]

    it = Interpreter(model_path=a.tflite)
    it.allocate_tensors()
    in_details, out_details = it.get_input_details(), it.get_output_details()
    n_states = len(in_details) - 1
    assert len(out_details) == len(in_details), "in/out count mismatch"
    assert len(torch_out) == len(out_details), \
        f"torch {len(torch_out)} vs tflite {len(out_details)} outputs"

    in_pos_by_n = {}
    for pos, d in enumerate(in_details):
        in_pos_by_n[int(re.search(r"args_(\d+)", d["name"]).group(1))] = pos

    def quant(x, d):
        s, z = d["quantization"]
        if s == 0:
            return x.astype(np.dtype(d["dtype"]))
        q = np.round(x / s) + z
        info = np.iinfo(np.dtype(d["dtype"]))
        return np.clip(q, info.min, info.max).astype(np.dtype(d["dtype"]))

    def dequant(q, d):
        s, z = d["quantization"]
        return (q.astype(np.float32) - z) * s if s != 0 else q.astype(np.float32)

    # Feed identical input into tflite (semantic slot n -> args_n).
    for n, t in enumerate(rand_in):
        d = in_details[in_pos_by_n[n]]
        it.set_tensor(d["index"], quant(t.numpy(), d))
    it.invoke()
    interp_out = [dequant(it.get_tensor(d["index"]), d) for d in out_details]

    out_pos_by_j, used = {}, set()
    for j, ref in enumerate(torch_out):
        cands = [(float(np.mean((interp_out[p] - ref) ** 2)), p)
                 for p in range(len(out_details))
                 if p not in used and interp_out[p].shape == ref.shape]
        assert cands, f"no shape-{ref.shape} interpreter output left for semantic slot {j}"
        mse, p = min(cands)
        out_pos_by_j[j] = (p, mse)
        used.add(p)
    worst = max(m for _, m in out_pos_by_j.values())

    feat_pos = in_pos_by_n[0]
    feat = in_details[feat_pos]
    mask_pos = out_pos_by_j[0][0]
    mask = out_details[mask_pos]
    feat_shape = [int(x) for x in feat["shape"]]
    feat_len = feat_shape[-1]
    feat_total = 1
    for x in feat_shape:
        feat_total *= x

    def qp(d):
        s, z = d["quantization"]
        return float(s), int(z)

    def nbytes(d):
        n = 1
        for x in d["shape"]:
            n *= int(x)
        return n * np.dtype(d["dtype"]).itemsize

    feat_s, feat_z = qp(feat)
    mask_s, mask_z = qp(mask)

    states, all_memcpy_safe = [], True
    for k in range(n_states):
        in_pos = in_pos_by_n[k + 1]
        out_pos = out_pos_by_j[k + 1][0]
        sin, sout = in_details[in_pos], out_details[out_pos]
        assert list(sin["shape"]) == list(sout["shape"]), f"state {k} shape mismatch"
        sin_s, sin_z = qp(sin)
        sout_s, sout_z = qp(sout)
        safe = (abs(sin_s - sout_s) < 1e-12) and (sin_z == sout_z) \
            and np.dtype(sin["dtype"]) == np.dtype(sout["dtype"])
        all_memcpy_safe &= safe
        states.append(dict(k=k, in_pos=in_pos, out_pos=out_pos, bytes=nbytes(sin),
                           dtype=np.dtype(sin["dtype"]).name, in_s=sin_s, in_z=sin_z,
                           out_s=sout_s, out_z=sout_z, safe=safe))

    def cf(x):
        return repr(float(x)) + "f"

    src = Path(a.tflite).resolve()
    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_int8 = feat["dtype"].__name__ == "int8"

    L = ["/* AUTO-GENERATED by deploy/rt595/host/gen_io_layout_multi.py — DO NOT EDIT. */",
         f"/* model: {a.model}   source: {src} */",
         "#ifndef MODEL_IO_LAYOUT_H", "#define MODEL_IO_LAYOUT_H", "",
         f"#define MODEL_N_IO            ({len(in_details)})   /* inputs == outputs */",
         f"#define MODEL_N_STATES        ({n_states})", "",
         f"#define MODEL_FEATURE_IN_POS  ({feat_pos})",
         f"#define MODEL_FEATURE_LEN     ({feat_len})   /* last dim (freq bins) */",
         f"#define MODEL_FEATURE_TOTAL   ({feat_total})   /* product of all feat dims */",
         f"#define MODEL_FEATURE_IS_INT8 ({1 if is_int8 else 0})",
         f"#define MODEL_MASK_OUT_POS    ({mask_pos})",
         f"#define MODEL_MASK_LEN        ({int(list(mask['shape'])[-1])})", "",
         f"#define MODEL_FEATURE_SCALE   ({cf(feat_s)})",
         f"#define MODEL_FEATURE_ZEROPOINT ({feat_z})",
         f"#define MODEL_MASK_SCALE      ({cf(mask_s)})",
         f"#define MODEL_MASK_ZEROPOINT  ({mask_z})", "",
         f"#define MODEL_STATE_FEEDBACK_MEMCPY_SAFE ({1 if all_memcpy_safe else 0})", "",
         "typedef struct {",
         "  int   in_pos, out_pos;",
         "  int   count;",
         "  float in_scale;  int in_zp;",
         "  float out_scale; int out_zp;",
         "  int   memcpy_safe;",
         "} model_state_map_t;",
         "static const model_state_map_t MODEL_STATE_MAP[MODEL_N_STATES] = {"]
    for s in states:
        cnt = s["bytes"]
        L.append(f"  {{ {s['in_pos']:2d}, {s['out_pos']:2d}, {cnt:6d}, "
                 f"{cf(s['in_s'])}, {s['in_z']:4d}, {cf(s['out_s'])}, {s['out_z']:4d}, "
                 f"{1 if s['safe'] else 0} }},  /* state {s['k']:2d} {s['dtype']} */")
    L += ["};", "", "#endif /* MODEL_IO_LAYOUT_H */"]
    out_path.write_text("\n".join(L) + "\n")

    print(f"Wrote {out_path}")
    print(f"  {len(in_details)} IO, {n_states} states; feat pos {feat_pos} "
          f"len {feat_len} total {feat_total} ({'int8' if is_int8 else feat['dtype'].__name__}); "
          f"mask pos {mask_pos} len {int(list(mask['shape'])[-1])}")
    print(f"  state feedback memcpy-safe: {all_memcpy_safe} "
          f"({sum(s['safe'] for s in states)}/{n_states})")
    print(f"  worst output-match MSE: {worst:.3e}")


if __name__ == "__main__":
    main()
