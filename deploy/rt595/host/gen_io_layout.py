#!/usr/bin/env python
"""Emit the firmware I/O contract header from a LiSenNet streaming .tflite.

The RT595 analog of ``deploy/stm32n6/host/gen_io_layout.py``. The TFLM interpreter
addresses inputs/outputs by *position* (``interpreter->input(i)`` / ``->output(j)``).
ai-edge-torch preserves input roles in the tensor names (``serving_default_args_N:0``,
N=0 is ``feat``, N=1..K the K FIFO/GRU states) but *shuffles* output positions and
gives their signature tensors different quant params than the primary outputs the
firmware reads — so name/index/metadata can't bridge outputs to positions.

Ground truth instead: run the PyTorch streaming module and the tflite on the *same*
random (feat, states) input and match each interpreter output position to its semantic
role (est_mag / state k) by value. That also fixes the ``state_in[k] <-> state_out[k]``
feedback pairing. Then emit ``model_io_layout.h``.

    ~/.venvs/rt595-export/bin/python deploy/rt595/host/gen_io_layout.py \
        --model host_out/..._int8.tflite --checkpoint_file cp_lisennet_hybrid_nc24/g_best \
        --output deploy/rt595/app/model_io_layout.h
"""
import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="streaming .tflite (int8)")
    ap.add_argument("--checkpoint_file", required=True, help="torch ground truth (g_best + config.json)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--repo", default=str(REPO_ROOT))
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    import numpy as np
    import torch
    from ai_edge_litert.interpreter import Interpreter
    from export_tflite import build_streaming_view      # reuse the loader/wrapper

    model, view = build_streaming_view(a.checkpoint_file)

    it = Interpreter(model_path=a.model)
    it.allocate_tensors()
    in_details, out_details = it.get_input_details(), it.get_output_details()
    n_states = len(in_details) - 1
    assert len(out_details) == len(in_details), "in/out count mismatch"

    # Input positions by semantic slot N, parsed from the stable raw names.
    in_pos_by_n = {}
    for pos, d in enumerate(in_details):
        in_pos_by_n[int(re.search(r"args_(\d+)", d["name"]).group(1))] = pos

    def quant(x, d):
        s, z = d["quantization"]
        if s == 0:                                       # not quantized
            return x.astype(np.dtype(d["dtype"]))
        q = np.round(x / s) + z
        info = np.iinfo(np.dtype(d["dtype"]))
        return np.clip(q, info.min, info.max).astype(np.dtype(d["dtype"]))

    def dequant(q, d):
        s, z = d["quantization"]
        return (q.astype(np.float32) - z) * s if s != 0 else q.astype(np.float32)

    # Random torch input (random states make every output discriminable by value).
    torch.manual_seed(0)
    feat_t = torch.randn(1, 3, 1, model.n_freqs)
    zero_states = view.init_states(1)
    states_t = [torch.randn_like(s) for s in zero_states]
    with torch.no_grad():
        torch_out = [o.numpy() for o in view(feat_t, *states_t)]   # [est_mag, s0..s_{K-1}]

    # Feed the SAME input into the tflite (feat -> args_0, state k -> args_(k+1)).
    torch_in = [feat_t.numpy()] + [s.numpy() for s in states_t]
    for n, arr in enumerate(torch_in):
        d = in_details[in_pos_by_n[n]]
        it.set_tensor(d["index"], quant(arr, d))
    it.invoke()
    interp_out = [dequant(it.get_tensor(d["index"]), d) for d in out_details]

    # Match each semantic output (torch slot j: 0=est_mag, k+1=state k) to the interpreter
    # output position with the same shape and smallest error (int8 => approximate).
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
    feat_len = int(list(feat["shape"])[-1])

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

    def cf(x):   # valid C float literal (repr keeps a '.' or exponent, e.g. 0.0->'0.0f')
        return repr(float(x)) + "f"

    src = Path(a.model).resolve()
    out_path = Path(a.output) if a.output else src.with_name("model_io_layout.h")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_int8 = feat["dtype"].__name__ == "int8"
    L = ["/* AUTO-GENERATED by deploy/rt595/host/gen_io_layout.py — DO NOT EDIT. */",
         f"/* source: {src} */",
         "#ifndef MODEL_IO_LAYOUT_H", "#define MODEL_IO_LAYOUT_H", "",
         f"#define MODEL_N_IO            ({len(in_details)})   /* inputs == outputs */",
         f"#define MODEL_N_STATES        ({n_states})", "",
         "/* feat: interpreter->input(MODEL_FEATURE_IN_POS); est_mag: ->output(MODEL_MASK_OUT_POS). */",
         f"#define MODEL_FEATURE_IN_POS  ({feat_pos})",
         f"#define MODEL_FEATURE_LEN     ({feat_len})   /* freq bins */",
         f"#define MODEL_FEATURE_IS_INT8 ({1 if is_int8 else 0})",
         f"#define MODEL_MASK_OUT_POS    ({mask_pos})",
         f"#define MODEL_MASK_LEN        ({int(list(mask['shape'])[-1])})", "",
         "/* On-device quant: q = round(x/scale)+zp; dequant: x = (q-zp)*scale. */",
         f"#define MODEL_FEATURE_SCALE   ({cf(feat_s)})",
         f"#define MODEL_FEATURE_ZEROPOINT ({feat_z})",
         f"#define MODEL_MASK_SCALE      ({cf(mask_s)})",
         f"#define MODEL_MASK_ZEROPOINT  ({mask_z})", "",
         f"/* 1 = every state_in[k]/state_out[k] shares scale+zp -> feedback is a raw memcpy. */",
         f"#define MODEL_STATE_FEEDBACK_MEMCPY_SAFE ({1 if all_memcpy_safe else 0})", "",
         "/* Per-state feedback map. Each frame the driver copies output(out_pos) into",
         "   input(in_pos): raw memcpy if memcpy_safe, else dequant(out)*->requant(in). */",
         "typedef struct {",
         "  int   in_pos, out_pos;   /* interpreter->input(in_pos) / ->output(out_pos) */",
         "  int   count;             /* element count (int8: == byte size)             */",
         "  float in_scale;  int in_zp;",
         "  float out_scale; int out_zp;",
         "  int   memcpy_safe;",
         "} model_state_map_t;",
         "static const model_state_map_t MODEL_STATE_MAP[MODEL_N_STATES] = {"]
    for s in states:
        cnt = s["bytes"]  # int8 => elements == bytes
        L.append(f"  {{ {s['in_pos']:2d}, {s['out_pos']:2d}, {cnt:6d}, "
                 f"{cf(s['in_s'])}, {s['in_z']:4d}, {cf(s['out_s'])}, {s['out_z']:4d}, "
                 f"{1 if s['safe'] else 0} }},  /* state {s['k']:2d} {s['dtype']} */")
    L += ["};", "", "#endif /* MODEL_IO_LAYOUT_H */"]
    out_path.write_text("\n".join(L) + "\n")

    print(f"Wrote {out_path}")
    print(f"  {len(in_details)} IO, {n_states} states; feat pos {feat_pos} len {feat_len} "
          f"({'int8' if is_int8 else feat['dtype'].__name__}); mask pos {mask_pos}")
    print(f"  state feedback memcpy-safe: {all_memcpy_safe} "
          f"({sum(s['safe'] for s in states)}/{n_states})")
    print(f"  worst output-match MSE: {worst:.3e} (int8 rounding; should be small & well-separated)")


if __name__ == "__main__":
    main()
