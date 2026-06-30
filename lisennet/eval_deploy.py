"""Deploy eval for LiSenNet: ONNX / int8 PESQ parity + real-time RTF.

Measures, on the VoiceBank-DEMAND test split, the wideband PESQ of the enhanced
speech under every deployment variant, so the cost of each deployment step is
isolated:

  mask backend  x  phase method
  ------------     ------------
  torch            Griffin-Lim (2 iter)   <- the trained offline recipe
  fp32 ONNX        noisy phase            <- the causal, real-time phase
  int8 dynamic
  int8 static

Two orthogonal questions:
  1. Does exporting + quantizing the mask sub-network cost PESQ?
     (hold phase = Griffin-Lim, vary the mask backend)
  2. What does dropping the non-causal Griffin-Lim for real-time cost?
     (hold backend = torch, vary the phase)
The headline real-time config — int8 mask + noisy phase — is reported too.

RTF is measured on the frame-by-frame streamer (the real-time path): per-frame
mask compute vs the frame's wall-clock duration (hop / sample_rate).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from common.dataset import Dataset, load_voicebank_demand
from common.env import AttrDict
from common.metrics import pesq_score
from lisennet.export_onnx import _load_from_checkpoint, export_fp32
from lisennet.quant_onnx import VBDCalibrationReader, quantize_static_int8
from lisennet.streaming import LiSenNetStreamer


def _session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def _reconstruct(model, est_mag, src_pha, length, use_gl):
    """Compressed est_mag (B,T,F) + a phase choice -> waveform (B, length)."""
    est_pha = model.griffinlim(est_mag, src_pha, length) if use_gl else src_pha
    est_spec = torch.complex(est_mag * est_pha.cos(), est_mag * est_pha.sin())
    return model.apply_istft(model.power_uncompress(est_spec), length=length)


@torch.no_grad()
def evaluate(model, sessions, h, n_utts, device="cpu"):
    hf = load_voicebank_demand()
    ds = Dataset(hf["test"], h.segment_size, h.sampling_rate, split=False, shuffle=False, seed=h.seed)
    n_utts = min(n_utts, len(ds))

    # variant -> (mask backend key, use_gl). Static int8 is the embedded-
    # deployable quantization (dynamic weight-only int8 is not supported by the
    # target edge runtimes), so it is the int8 path reported here.
    variants = {
        "torch + GL (offline recipe)": ("torch", True),
        "torch + noisy phase (real-time phase)": ("torch", False),
        "fp32 ONNX + GL": ("fp32", True),
        "int8-static ONNX + GL": ("int8_static", True),
        "int8-static ONNX + noisy phase (full real-time)": ("int8_static", False),
    }
    refs = []
    ests = {name: [] for name in variants}

    for i in range(n_utts):
        clean, noisy = ds[i]
        clean = clean.unsqueeze(0).to(device)
        noisy = noisy.unsqueeze(0).to(device)
        length = noisy.shape[-1]

        spec = model.power_compress(model.apply_stft(noisy))
        src_mag, src_pha = spec.abs(), spec.angle()
        feat = model.build_features(src_mag, src_pha)            # (1,3,T,F)

        # mask backends -> compressed est_mag (1,T,F)
        masks = {}
        masks["torch"] = model.apply_mask(model.predict_mask(feat), src_mag)
        feat_np = feat.cpu().numpy()
        for key in ("fp32", "int8_static"):
            if key in sessions:
                em = sessions[key].run(["est_mag"], {"feat": feat_np})[0]
                masks[key] = torch.from_numpy(em).to(device)

        for name, (backend, use_gl) in variants.items():
            if backend not in masks:
                continue
            est = _reconstruct(model, masks[backend], src_pha, length, use_gl)
            n = min(est.shape[-1], clean.shape[-1])
            ests[name].append(est[..., :n])
        refs.append(clean[..., :clean.shape[-1]])

    # align ref lengths per utterance with each est (PESQ pairs index-wise)
    results = {}
    for name, est_list in ests.items():
        if not est_list:
            continue
        rr = [refs[j][..., :est_list[j].shape[-1]] for j in range(len(est_list))]
        results[name] = float(pesq_score(rr, est_list, h))
    return results, n_utts


@torch.no_grad()
def measure_rtf(model, h, n_frames=2000, device="cpu"):
    """Per-frame mask-compute real-time factor via the streaming path."""
    f = model.n_freqs
    streamer = LiSenNetStreamer(model.to(device))
    streamer.reset()
    g = torch.Generator().manual_seed(0)
    mag = torch.rand(1, n_frames, f, generator=g).to(device)
    pha = ((torch.rand(1, n_frames, f, generator=g) * 2 - 1) * torch.pi).to(device)
    # warmup
    for t in range(20):
        streamer.step(mag[:, t], pha[:, t])
    streamer.reset()
    start = time.perf_counter()
    for t in range(n_frames):
        streamer.step(mag[:, t], pha[:, t])
    elapsed = time.perf_counter() - start
    streamer.close()
    frame_dt = h.hop_size / h.sampling_rate
    audio_dt = n_frames * frame_dt
    rtf = elapsed / audio_dt
    return {
        "frames": n_frames, "compute_s": elapsed, "audio_s": audio_dt,
        "rtf": rtf, "ms_per_frame": 1e3 * elapsed / n_frames,
        "frame_budget_ms": 1e3 * frame_dt,
    }


def main():
    p = argparse.ArgumentParser(description="LiSenNet deploy eval: ONNX/int8 PESQ + RTF.")
    p.add_argument("--checkpoint_file", default="cp_lisennet/g_best")
    p.add_argument("--n_utts", type=int, default=824)
    p.add_argument("--calib_utts", type=int, default=24)
    p.add_argument("--workdir", default=None,
                   help="Where to write the exported ONNX graphs "
                        "(default: the checkpoint's run directory, next to g_best).")
    p.add_argument("--skip_static", action="store_true")
    a = p.parse_args()

    ckpt = Path(a.checkpoint_file)
    with open(ckpt.parent / "config.json") as f:
        h = AttrDict(json.load(f))
    model = _load_from_checkpoint(ckpt)

    work = Path(a.workdir) if a.workdir else ckpt.parent      # artifacts live in the run dir
    work.mkdir(parents=True, exist_ok=True)
    fp32 = export_fp32(model, work / "g_best_fp32.onnx")
    sessions = {"fp32": _session(fp32)}
    sizes = {"fp32": fp32.stat().st_size}
    if not a.skip_static:
        try:
            # per_channel=False: ORT's per-channel int32-bias scale adjustment
            # trips on this graph's biases; static int8 is the QAT-grade path
            # anyway, so we keep it per-tensor and tolerant.
            int8_static = quantize_static_int8(
                fp32, work / "g_best_int8_static.onnx",
                VBDCalibrationReader(h, a.calib_utts), per_channel=False)
            sessions["int8_static"] = _session(int8_static)
            sizes["int8_static"] = int8_static.stat().st_size
        except Exception as e:                          # noqa: BLE001 — characterisation, not deployment
            print(f"\n[static int8 skipped: {type(e).__name__}: {e}]")

    print(f"\nONNX graph sizes (KiB): "
          + ", ".join(f"{k}={v/1024:.1f}" for k, v in sizes.items()))

    rtf = measure_rtf(model, h)
    print(f"\nReal-time factor (frame-by-frame mask compute, 1 thread CPU):")
    print(f"  {rtf['ms_per_frame']:.3f} ms/frame vs {rtf['frame_budget_ms']:.2f} ms frame budget "
          f"-> RTF {rtf['rtf']:.4f}  ({1/rtf['rtf']:.1f}x faster than real time)")

    print(f"\nEvaluating PESQ on {a.n_utts} VBD test utterances ...")
    results, n = evaluate(model, sessions, h, a.n_utts)
    print(f"\n=== LiSenNet deploy PESQ (n={n}) ===")
    width = max(len(k) for k in results)
    for name, score in results.items():
        print(f"  {name:<{width}} : {score:.4f}")


if __name__ == "__main__":
    main()
