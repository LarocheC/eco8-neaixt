"""ORT CPU inference for the stateless-windowed ConvFSENet ONNX (FP32 or int8).

The deploy counterpart of inference_onnx.py for Track 1. Instead of threading
per-block FIFO state across single frames, the host keeps a ring buffer of the
last L = sum_blocks (K-1)*D magnitude columns and feeds fixed windows

    noisy_mag_window  [B, n_freq, L+T]  ->  mask  [B, n_freq, T]

For a whole-utterance offline PESQ eval we left-pad the magnitude by L and stack
every output frame's window into the BATCH axis, so one session.run() masks the
entire utterance — the graph's batch axis is dynamic by design.

Cold start: the model was trained with a zero-ACTIVATION history before t=0
(offline causal left-padding), but a zero-MAGNITUDE ring buffer feeds the
frontend bias (frontend(0) != 0) -> an out-of-distribution start-of-clip
transient that costs ~0.045 PESQ on short utterances. The default `coldstart`
is therefore 'replicate' (repeat the first frame) — CAUSAL (no look-ahead, so
on-device deployable) and in-distribution, recovering that PESQ. See --coldstart.

256-bin (`drop_nyquist`) models drop the Nyquist magnitude bin before the graph
and pad the mask back to 257 after it (Nyquist mask := 0 by default).

RTF note: the windowed graph recomputes the full L+T receptive field per output
frame, so on a CPU (where the MAC array is irrelevant) its per-frame compute is
~(L+T)x the per-frame streaming model. The host RTF here is a throughput proxy
on batched windows, NOT the on-NPU per-frame latency the rework targets — that
is measured on-board (deploy/stm32n6/ONBOARD_MEASUREMENT.md). The mask values
(hence PESQ) are what this script certifies.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
from rich.progress import track

from common.dataset import load_voicebank_demand
from common.env import AttrDict
from common.metrics import eval_pesq
from convfsenet.inference_onnx import _make_session, _check_ort_version


def _graph_window_dims(onnx_path):
    """Read (n_freq, window_len=L+T) from the windowed ONNX input shape, and
    (L, T, drop_nyquist) from metadata when present.

    The input `noisy_mag_window` has static dims [B, n_freq, L+T]; L/T are also
    stamped into metadata by quant_windowed (FP32 exports have no metadata, so
    fall back to deriving T from window_len once L is known, or assume T=1)."""
    model = onnx.load(str(onnx_path))
    meta = {p.key: p.value for p in model.metadata_props}
    inp = next(i for i in model.graph.input if i.name == "noisy_mag_window")
    dims = inp.type.tensor_type.shape.dim
    n_freq = int(dims[1].dim_value)
    window_len = int(dims[2].dim_value)
    drop_nyquist = bool(int(meta.get("drop_nyquist", "1" if n_freq == 256 else "0")))
    if "windowed_L" in meta and "windowed_T" in meta:
        L = int(meta["windowed_L"]); T = int(meta["windowed_T"])
    else:
        T = 1
        L = window_len - T
    return n_freq, window_len, L, T, drop_nyquist


def _coldstart_pad(mag_np: np.ndarray, L: int, mode: str) -> np.ndarray:
    """Left-pad mag [F, T_total] by L columns for the ring-buffer cold start.

    mode:
      'zero'      — silence prefix (physically correct for a powered-on stream;
                    but the trained model expects zero-ACTIVATION not zero-
                    MAGNITUDE history, so the frontend bias leaks -> a start-of-
                    clip transient that costs PESQ on short utterances).
      'replicate' — repeat the first real column (edge replicate); keeps the
                    convs in-distribution at the boundary.
      'reflect'   — mirror the first L real columns (what center=True STFT does
                    at signal edges). Best in-distribution prefix for short clips.
    """
    F, T_total = mag_np.shape
    if mode == "zero" or L == 0:
        pad = np.zeros((F, L), dtype=mag_np.dtype)
    elif mode == "replicate":
        pad = np.repeat(mag_np[:, :1], L, axis=1)
    elif mode == "reflect":
        k = min(L, T_total)
        refl = mag_np[:, 1:k + 1][:, ::-1] if k > 1 else mag_np[:, :1]
        if refl.shape[1] < L:                          # very short clip: pad the reflection
            refl = np.concatenate([np.repeat(refl[:, :1], L - refl.shape[1], axis=1), refl], axis=1)
        pad = refl[:, -L:]
    else:
        raise ValueError(f"unknown coldstart mode {mode!r}")
    return np.concatenate([pad, mag_np], axis=1)


def _build_window_batch(mag_np: np.ndarray, L: int, T: int, coldstart: str = "replicate"):
    """mag_np [F, T_total] -> (batch [n_calls, F, L+T], offsets [n_calls]).

    Left-pads by L (per `coldstart`). Window k emits T mask columns for absolute
    output frames [offsets[k], offsets[k]+T). For full windows offsets[k]=k*T;
    the final window is clamped so it ends exactly at T_total (offset=T_total-T)
    when T does not divide T_total — so emit-frame placement stays correct (the
    blind reshape that assumed offset=k*T corrupted the trailing T_total%T
    frames). n_calls = ceil(T_total / T)."""
    F, T_total = mag_np.shape
    W = L + T
    padded = _coldstart_pad(mag_np, L, coldstart)        # [F, L+T_total]; padded[j] = real[j-L]
    n_calls = (T_total + T - 1) // T
    batch = np.empty((n_calls, F, W), dtype=np.float32)
    offsets = np.empty(n_calls, dtype=np.int64)
    for k in range(n_calls):
        start = k * T                                    # window start in padded coords
        if start + W > padded.shape[1]:                  # tail: clamp to end at T_total
            start = padded.shape[1] - W
        batch[k] = padded[:, start:start + W]
        # window padded[start:start+W] emits real frames [start, start+T) (the
        # window's emit offset in output coords equals its padded start index).
        offsets[k] = start
    return batch, offsets


def _run_windowed(session, mag_np, L, T, n_freq_full, drop_nyquist, measure_rtf=False, coldstart="replicate"):
    """mag_np [F_full, T_total] -> mask [F_full, T_total], rtf_seconds.

    Drops the Nyquist bin before the graph if drop_nyquist; pads the mask back
    to F_full (Nyquist mask := 0) after. One batched session.run."""
    F_full, T_total = mag_np.shape
    F_graph = n_freq_full - (1 if drop_nyquist else 0)
    graph_mag = mag_np[:F_graph, :] if drop_nyquist else mag_np
    batch, offsets = _build_window_batch(graph_mag, L, T, coldstart=coldstart)   # [n_calls,F_graph,L+T]
    rtf_s = 0.0
    if measure_rtf:
        t0 = time.perf_counter()
    out = session.run(["mask"], {"noisy_mag_window": batch})[0]   # [n_calls, F_graph, T]
    if measure_rtf:
        rtf_s = time.perf_counter() - t0
    # Scatter each window's T columns to its true absolute emit offset (handles
    # the clamped tail window; a blind reshape would misplace the last frames).
    mask_graph = np.zeros((out.shape[1], T_total), dtype=np.float32)
    for k in range(out.shape[0]):
        off = int(offsets[k])
        mask_graph[:, off:off + T] = out[k]
    if drop_nyquist:
        mask = np.zeros((F_full, T_total), dtype=np.float32)       # Nyquist row stays 0
        mask[:F_full - 1, :] = mask_graph
    else:
        mask = mask_graph
    return mask, rtf_s


def enhance_one_utterance_windowed(
    noisy_wav, primary_session, h, L, T, n_freq_full, drop_nyquist,
    fp32_session=None, fp32_dims=None, coldstart="replicate",
):
    """STFT once -> batched windowed mask -> iSTFT (per session)."""
    noisy_t = torch.from_numpy(np.asarray(noisy_wav, dtype=np.float32))
    norm_factor = torch.sqrt(len(noisy_t) / (torch.sum(noisy_t ** 2.0) + 1e-8))
    noisy_t = (noisy_t * norm_factor).unsqueeze(0)
    L_samples = noisy_t.shape[-1]

    window = torch.hann_window(int(h.win_size))
    stft = torch.stft(
        noisy_t, int(h.n_fft), hop_length=int(h.hop_size), win_length=int(h.win_size),
        window=window, center=True, normalized=False, return_complex=True,
    )                                                   # (1, F, T) complex
    mag_np = stft.abs().squeeze(0).numpy().astype(np.float32)   # [F_full, T_total]

    primary_mask, rtf_s = _run_windowed(
        primary_session, mag_np, L, T, n_freq_full, drop_nyquist, measure_rtf=True,
        coldstart=coldstart,
    )
    duration_s = L_samples / float(h.sampling_rate)
    rtf_primary = rtf_s / duration_s

    def _istft_from_mask(mask_np):
        masked = stft * torch.from_numpy(mask_np).unsqueeze(0)
        wav = torch.istft(
            masked, int(h.n_fft), hop_length=int(h.hop_size), win_length=int(h.win_size),
            window=window, center=True, normalized=False, length=L_samples, return_complex=False,
        )
        return (wav / norm_factor).squeeze(0).numpy()

    primary_audio = _istft_from_mask(primary_mask)

    fp32_audio = None
    if fp32_session is not None:
        fL, fT, fnf, fdrop = fp32_dims
        fp32_mask, _ = _run_windowed(fp32_session, mag_np, fL, fT, fnf, fdrop,
                                     measure_rtf=False, coldstart=coldstart)
        fp32_audio = _istft_from_mask(fp32_mask)

    return primary_audio, fp32_audio, rtf_primary


def main():
    print("Initializing windowed ConvFSENet ORT Inference..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_file", required=True,
                        help="Primary windowed ONNX (typically the int8 deploy artifact).")
    parser.add_argument("--fp32_checkpoint_file", default=None,
                        help="Optional FP32 windowed ONNX sidecar for PESQ delta.")
    parser.add_argument("--input_noisy_wavs_dir", default=None)
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--output_dir", default="generated_files_convfsenet_win")
    parser.add_argument("--max_utterances", type=int, default=None)
    parser.add_argument("--coldstart", default="replicate", choices=["zero", "replicate", "reflect"],
                        help="ring-buffer cold-start padding for the first L frames. Default "
                             "'replicate' (repeat the first frame): CAUSAL (no look-ahead), "
                             "deployable, and recovers ~+0.045 PESQ over 'zero' by avoiding the "
                             "frontend-bias out-of-distribution start-of-clip transient. 'reflect' "
                             "is marginally better but needs look-ahead (non-causal); 'zero' is "
                             "silence (physically correct for a powered-on stream but OOD here).")
    a = parser.parse_args()

    parent_dir = Path(a.checkpoint_file).parent
    config_file = parent_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"sibling config.json missing at {config_file}")
    with open(config_file) as f:
        h = AttrDict(json.load(f))
    torch.manual_seed(int(h.seed))

    primary_path = Path(a.checkpoint_file)
    if not primary_path.exists():
        raise FileNotFoundError(f"windowed ONNX missing at {primary_path}")
    _check_ort_version(primary_path)
    n_freq, window_len, L, T, drop_nyquist = _graph_window_dims(primary_path)
    n_freq_full = int(h.n_features)
    primary_session = _make_session(primary_path)

    fp32_session = fp32_dims = None
    if a.fp32_checkpoint_file:
        fp32_path = Path(a.fp32_checkpoint_file)
        if not fp32_path.exists():
            raise FileNotFoundError(f"--fp32_checkpoint_file {fp32_path} not found")
        fnf, fwl, fL, fT, fdrop = _graph_window_dims(fp32_path)
        fp32_session = _make_session(fp32_path)
        fp32_dims = (fL, fT, n_freq_full, fdrop)
    dual = fp32_session is not None
    print(f"Loaded primary: {primary_path} (n_freq={n_freq}, L={L}, T={T}, "
          f"drop_nyquist={drop_nyquist})" + (f"; fp32 sidecar: {a.fp32_checkpoint_file}" if dual else ""))

    os.makedirs(a.output_dir, exist_ok=True)
    pesq_primary, pesq_fp32, deltas, rtfs = [], [], [], []

    hf = load_voicebank_demand(cache_dir=a.hf_cache_dir)
    test_split = hf["test"]
    if a.max_utterances is not None:
        test_split = test_split.select(range(min(a.max_utterances, len(test_split))))
    N = len(test_split)
    for i, item in enumerate(track(test_split)):
        noisy_wav = np.asarray(item["noisy"]["array"], dtype=np.float32)
        clean_wav = np.asarray(item["clean"]["array"], dtype=np.float32)
        primary_audio, fp32_audio, rtf = enhance_one_utterance_windowed(
            noisy_wav, primary_session, h, L, T, n_freq_full, drop_nyquist,
            fp32_session, fp32_dims, coldstart=a.coldstart,
        )
        p_pri = eval_pesq(clean_wav, primary_audio, int(h.sampling_rate))
        line = f"[{i+1}/{N}] {item['id']}: PESQ primary={p_pri:.3f}; RTF={rtf:.3f}"
        if dual:
            p_fp32 = eval_pesq(clean_wav, fp32_audio, int(h.sampling_rate))
            if p_pri != -1 and p_fp32 != -1:
                delta = p_fp32 - p_pri
                line = (f"[{i+1}/{N}] {item['id']}: PESQ FP32={p_fp32:.3f}, "
                        f"primary={p_pri:.3f}, delta={delta:.3f}; RTF={rtf:.3f}")
                deltas.append(delta)
            else:
                deltas.append(float("nan"))
            pesq_fp32.append(p_fp32)
        print(line)
        pesq_primary.append(p_pri)
        rtfs.append(rtf)
        sf.write(os.path.join(a.output_dir, item["id"] + ".wav"),
                 primary_audio, int(h.sampling_rate), "PCM_16")

    pri_clean = [s for s in pesq_primary if s != -1]
    pri_mean = float(np.mean(pri_clean)) if pri_clean else float("nan")
    rtf_mean = float(np.mean(rtfs)) if rtfs else float("nan")
    n_fail = len(pesq_primary) - len(pri_clean)
    summary = f"Mean: PESQ primary={pri_mean:.3f}; RTF(batched proxy)={rtf_mean:.3f}; PESQ failures={n_fail}"
    if dual:
        fp32_clean = [s for s in pesq_fp32 if s != -1]
        fp32_mean = float(np.mean(fp32_clean)) if fp32_clean else float("nan")
        delta_mean = float(np.nanmean(deltas)) if deltas else float("nan")
        summary = (f"Mean: PESQ FP32={fp32_mean:.3f}, primary={pri_mean:.3f}, "
                   f"delta={delta_mean:.3f}; RTF(batched proxy)={rtf_mean:.3f}; PESQ failures={n_fail}")
    print(summary)


if __name__ == "__main__":
    main()
