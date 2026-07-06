"""Deploy eval for FSPEN: ONNX / int8 PESQ parity + real-time RTF.

Measures, on the VoiceBank-DEMAND test split, the wideband PESQ of the enhanced
speech under every deployment variant, so the cost of each deployment step is
isolated:

    torch (offline recipe)
    fp32 ONNX
    int8-dynamic ONNX (weight-only)
    int8-static ONNX  (the embedded-deployable quantization)

FSPEN predicts a complex mask, so unlike LiSenNet there is no phase axis to the
study — every backend reconstructs audio directly from its enhanced complex
spectrum via the (causal) iSTFT.

RTF is measured on the frame-by-frame streamer (the real-time path): per-frame
compute vs the frame's wall-clock duration (hop / sample_rate).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import onnxruntime as ort
import torch

from common.dataset import Dataset, load_voicebank_demand
from common.env import AttrDict
from common.metrics import pesq_score
from fspen.export_onnx import _load_from_checkpoint, export_fp32
from fspen.quant_onnx import VBDCalibrationReader, quantize_dynamic_int8, quantize_static_int8
from fspen.streaming import FSPENStreamer


def _session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def _reconstruct(model, est, length):
    """Enhanced stacked-re/im spectrum (B,T,2,F) -> waveform (B, length)."""
    est_spec = torch.complex(est[:, :, 0], est[:, :, 1])     # (B, T, F)
    return model.apply_istft(est_spec, length=length)


@torch.no_grad()
def evaluate(model, sessions, h, n_utts, device="cpu"):
    hf = load_voicebank_demand()
    ds = Dataset(hf["test"], h.segment_size, h.sampling_rate, split=False, shuffle=False, seed=h.seed)
    n_utts = min(n_utts, len(ds))

    variants = ["torch (offline recipe)"] + [f"{k} ONNX" for k in sessions]
    refs = []
    ests = {name: [] for name in variants}

    for i in range(n_utts):
        clean, noisy = ds[i]
        clean = clean.unsqueeze(0).to(device)
        noisy = noisy.unsqueeze(0).to(device)
        length = noisy.shape[-1]

        spec, amp = model.spec_features(model.apply_stft(noisy))   # (1,T,2,F), (1,T,1,F)

        outs = {}
        outs["torch (offline recipe)"], _ = model.enhance_spectrum(spec, amp)
        spec_np = spec.cpu().numpy()
        for key, sess in sessions.items():
            out = sess.run(["est_spec"], {"spec": spec_np})[0]
            outs[f"{key} ONNX"] = torch.from_numpy(out).to(device)

        for name, est in outs.items():
            wav = _reconstruct(model, est, length)
            n = min(wav.shape[-1], clean.shape[-1])
            ests[name].append(wav[..., :n])
        refs.append(clean)

    results = {}
    for name, est_list in ests.items():
        if not est_list:
            continue
        rr = [refs[j][..., :est_list[j].shape[-1]] for j in range(len(est_list))]
        results[name] = float(pesq_score(rr, est_list, h))
    return results, n_utts


@torch.no_grad()
def measure_rtf(model, h, n_frames=2000, device="cpu"):
    """Per-frame compute real-time factor via the streaming path.

    Pins torch to one thread (process-global, acceptable in this CLI) so the
    number matches the printed "1 thread CPU" methodology and the ORT
    sessions' ``intra_op_num_threads=1``.
    """
    torch.set_num_threads(1)
    f = model.n_freqs
    streamer = FSPENStreamer(model.to(device))
    streamer.reset()
    g = torch.Generator().manual_seed(0)
    spec = torch.randn(1, n_frames, 2, f, generator=g).to(device)
    # warmup
    for t in range(20):
        streamer.step(spec[:, t])
    streamer.reset()
    start = time.perf_counter()
    for t in range(n_frames):
        streamer.step(spec[:, t])
    elapsed = time.perf_counter() - start
    frame_dt = h.hop_size / h.sampling_rate
    audio_dt = n_frames * frame_dt
    rtf = elapsed / audio_dt
    return {
        "frames": n_frames, "compute_s": elapsed, "audio_s": audio_dt,
        "rtf": rtf, "ms_per_frame": 1e3 * elapsed / n_frames,
        "frame_budget_ms": 1e3 * frame_dt,
    }


def main():
    p = argparse.ArgumentParser(description="FSPEN deploy eval: ONNX/int8 PESQ + RTF.")
    p.add_argument("--checkpoint_file", default="cp_fspen/g_best")
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

    int8_dyn = quantize_dynamic_int8(fp32, work / "g_best_int8_dynamic.onnx")
    sessions["int8-dynamic"] = _session(int8_dyn)
    sizes["int8-dynamic"] = int8_dyn.stat().st_size

    if not a.skip_static:
        try:
            int8_static = quantize_static_int8(
                fp32, work / "g_best_int8_static.onnx",
                VBDCalibrationReader(h, a.calib_utts), per_channel=False)
            sessions["int8-static"] = _session(int8_static)
            sizes["int8-static"] = int8_static.stat().st_size
        except Exception as e:                          # noqa: BLE001 — characterisation, not deployment
            print(f"\n[static int8 skipped: {type(e).__name__}: {e}]")

    print(f"\nONNX graph sizes (KiB): "
          + ", ".join(f"{k}={v/1024:.1f}" for k, v in sizes.items()))

    rtf = measure_rtf(model, h)
    print(f"\nReal-time factor (frame-by-frame compute, 1 thread CPU):")
    print(f"  {rtf['ms_per_frame']:.3f} ms/frame vs {rtf['frame_budget_ms']:.2f} ms frame budget "
          f"-> RTF {rtf['rtf']:.4f}  ({1/rtf['rtf']:.1f}x faster than real time)")

    print(f"\nEvaluating PESQ on {a.n_utts} VBD test utterances ...")
    results, n = evaluate(model, sessions, h, a.n_utts)
    print(f"\n=== FSPEN deploy PESQ (n={n}) ===")
    width = max(len(k) for k in results)
    for name, score in results.items():
        print(f"  {name:<{width}} : {score:.4f}")


if __name__ == "__main__":
    main()
