"""ORT CPU inference for ConvFSENet streaming ONNX (FP32 or int8).

Mirrors inference_onnx.py (NSNet2's) shape with ConvFSENet adaptations:

  - 9 per-block FIFO state tensors threaded across frames (vs NSNet2's
    single (L, B, H) GRU state).
  - Real-valued magnitude in, real-valued mask out; the complex multiply
    `stft_t * mask` and the STFT / iSTFT live in this Python wrapper.
  - State shapes derived from the sibling config.json (n_channels_conv,
    kernel_size, n_blocks, n_stacks). Per-block dilation D follows the
    TCM construction order: 2^n cycling, n_blocks per stack.

Two operating modes:

  --checkpoint_file <onnx>                           single-session
  --checkpoint_file <int8.onnx> --fp32_checkpoint_file <fp32.onnx>
                                                     dual; PESQ comparison
                                                     + delta; RTF measured
                                                     on int8 only.

When --fp32_checkpoint_file is omitted, the script auto-derives a sibling
g_best_fp32.onnx in the same directory and uses it if present (matches the
inference_onnx.py convention). To force single-session, point both args
at the same file.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
from rich.progress import track

from common.dataset import load_voicebank_demand
from common.env import AttrDict
from common.metrics import eval_pesq
from common.quality import QualitySuite, resolve_metrics


# ----- session + state plumbing ---------------------------------------------


def _make_session(onnx_path) -> ort.InferenceSession:
    """4-knob determinism config — reruns are byte-identical."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"],
    )


def state_shapes_from_h(h) -> List[Tuple[int, int]]:
    """Derive per-block state shape (channels, buffer_size) from the config.

    Buffer size = (K-1) * D, where D = 2^n for the n-th block in each stack
    (matches TCM construction in convfsenet/model.py).
    """
    K = int(h.kernel_size)
    C = int(h.n_channels_conv)
    n_blocks = int(h.n_blocks)
    n_stacks = int(h.n_stacks)
    return [(C, (K - 1) * (2 ** n)) for _ in range(n_stacks) for n in range(n_blocks)]


def _state_io_names(n: int) -> Tuple[List[str], List[str]]:
    return (
        [f"state_{i}_in" for i in range(n)],
        [f"state_{i}_out" for i in range(n)],
    )


def _run_session_frame_by_frame(
    session: ort.InferenceSession,
    noisy_mag_np: np.ndarray,                 # (B, F, T)
    state_shapes: List[Tuple[int, int]],
    measure_rtf: bool = False,
):
    """Per-utterance ORT frame loop with per-utterance state reset.

    Returns (mask_np (B, F, T), rtf_total_seconds).
    """
    B, F, T = noisy_mag_np.shape
    state_in_names, state_out_names = _state_io_names(len(state_shapes))
    states = [
        np.zeros((B, C, buf), dtype=np.float32) for (C, buf) in state_shapes
    ]
    masks = []
    rtf_total = 0.0
    for t in range(T):
        feeds = {"noisy_mag": noisy_mag_np[:, :, t]}
        for name, s in zip(state_in_names, states):
            feeds[name] = s
        if measure_rtf:
            t0 = time.perf_counter()
        outs = session.run(["mask", *state_out_names], feeds)
        if measure_rtf:
            rtf_total += time.perf_counter() - t0
        masks.append(outs[0])
        states = outs[1:]
    mask_np = np.stack(masks, axis=2)         # (B, F, T)
    return mask_np, rtf_total


# ----- per-utterance enhancement --------------------------------------------


def enhance_one_utterance(
    noisy_wav: np.ndarray,
    primary_session: ort.InferenceSession,
    h,
    state_shapes: List[Tuple[int, int]],
    fp32_session: Optional[ort.InferenceSession] = None,
):
    """STFT once → frame loop → iSTFT (per session).

    Returns (primary_audio (1-D float32), fp32_audio or None, rtf_primary).
    RTF is measured around session.run() of `primary_session` only (consistent
    with inference_onnx.py's int8-only RTF convention).
    """
    # 1. RMS normalize (matches dataset.py and NSNet2 inference.py).
    noisy_t = torch.from_numpy(np.asarray(noisy_wav, dtype=np.float32))
    norm_factor = torch.sqrt(len(noisy_t) / (torch.sum(noisy_t ** 2.0) + 1e-8))
    noisy_t = (noisy_t * norm_factor).unsqueeze(0)             # (1, L_samples)
    L_samples = noisy_t.shape[-1]

    # 2. STFT once. Use complex output so we can multiply by the mask later.
    window = torch.hann_window(int(h.win_size))
    stft = torch.stft(
        noisy_t,
        int(h.n_fft),
        hop_length=int(h.hop_size),
        win_length=int(h.win_size),
        window=window,
        center=True,
        normalized=False,
        return_complex=True,
    )                                                          # (1, F, T) complex
    noisy_mag_np = stft.abs().numpy().astype(np.float32)       # (1, F, T) real

    # 3. Primary session (RTF measured here).
    primary_mask_np, rtf_s = _run_session_frame_by_frame(
        primary_session, noisy_mag_np, state_shapes, measure_rtf=True,
    )
    duration_s = L_samples / float(h.sampling_rate)
    rtf_primary = rtf_s / duration_s

    # 4. Optional FP32 sidecar (no RTF).
    fp32_audio = None
    if fp32_session is not None:
        fp32_mask_np, _ = _run_session_frame_by_frame(
            fp32_session, noisy_mag_np, state_shapes, measure_rtf=False,
        )
        fp32_stft = stft * torch.from_numpy(fp32_mask_np)
        fp32_wav = torch.istft(
            fp32_stft,
            int(h.n_fft),
            hop_length=int(h.hop_size),
            win_length=int(h.win_size),
            window=window,
            center=True,
            normalized=False,
            length=L_samples,
            return_complex=False,
        )
        fp32_audio = (fp32_wav / norm_factor).squeeze(0).numpy()

    # 5. Primary mask → iSTFT → unscale.
    primary_stft = stft * torch.from_numpy(primary_mask_np)
    primary_wav = torch.istft(
        primary_stft,
        int(h.n_fft),
        hop_length=int(h.hop_size),
        win_length=int(h.win_size),
        window=window,
        center=True,
        normalized=False,
        length=L_samples,
        return_complex=False,
    )
    primary_audio = (primary_wav / norm_factor).squeeze(0).numpy()

    return primary_audio, fp32_audio, rtf_primary


# ----- ORT version stamp check ----------------------------------------------


def _check_ort_version(int8_path):
    """Warn-not-crash on quantize-time / runtime ORT version skew."""
    model = onnx.load(str(int8_path))
    stamped = next(
        (p.value for p in model.metadata_props if p.key == "ort_quantize_version"),
        None,
    )
    if stamped is None:
        return
    if stamped != ort.__version__:
        print(
            f"WARNING: int8 ONNX was quantized with ORT {stamped} but runtime "
            f"is ORT {ort.__version__}; may produce different results."
        )


# ----- main -----------------------------------------------------------------


def _validate_paths(int8_path: Path, fp32_override: Optional[str]) -> Tuple[Path, Optional[Path]]:
    """Resolve --checkpoint_file + optional FP32 sidecar.

    The sidecar is opportunistic: if --fp32_checkpoint_file is not given and
    no sibling g_best_fp32.onnx exists, we fall back to single-session mode.
    """
    if not int8_path.exists():
        raise FileNotFoundError(
            f"ONNX missing at {int8_path}; run: "
            f"python -m convfsenet.quant --checkpoint_dir {int8_path.parent} "
            f"--num_utterances 200"
        )
    if fp32_override:
        fp32_path = Path(fp32_override)
        if not fp32_path.exists():
            raise FileNotFoundError(f"--fp32_checkpoint_file {fp32_path} not found")
        return int8_path, fp32_path
    sibling = int8_path.parent / "g_best_fp32.onnx"
    return int8_path, sibling if sibling.exists() and sibling != int8_path else None


def main():
    print("Initializing ConvFSENet ORT Inference..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_file", required=True,
                        help="Primary ONNX (typically the int8 deploy artifact).")
    parser.add_argument("--fp32_checkpoint_file", default=None,
                        help="Optional FP32 ONNX sidecar; defaults to "
                             "<dir>/g_best_fp32.onnx when present.")
    parser.add_argument("--input_noisy_wavs_dir", default=None,
                        help="Directory of noisy wavs. If unset, uses HF VBD test split.")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--output_dir", default="generated_files_convfsenet")
    parser.add_argument("--log_dir", default=None,
                        help="TB log directory; if unset, no TB writes.")
    parser.add_argument("--max_utterances", type=int, default=None,
                        help="Optional slice for fast smoke iteration.")
    parser.add_argument("--metrics", default="none",
                        help="Extra perceptual metrics on top of PESQ: "
                             "'all' | 'none' | e.g. 'dnsmos,nisqa,scoreq'. "
                             "Off by default -- DNSMOS alone costs ~4 min on the full split.")
    parser.add_argument("--metrics_json", default=None,
                        help="Write the extra metrics' means to this JSON path")
    a = parser.parse_args()
    extra_metrics = resolve_metrics(a.metrics)

    # Sibling config.json
    parent_dir = Path(a.checkpoint_file).parent
    config_file = parent_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"sibling config.json missing at {config_file}; checkpoint dir "
            f"is corrupted or non-standard."
        )
    with open(config_file) as f:
        h = AttrDict(json.load(f))
    torch.manual_seed(int(h.seed))

    primary_path, fp32_path = _validate_paths(Path(a.checkpoint_file), a.fp32_checkpoint_file)
    _check_ort_version(primary_path)

    primary_session = _make_session(primary_path)
    fp32_session = _make_session(fp32_path) if fp32_path is not None else None
    dual = fp32_session is not None
    state_shapes = state_shapes_from_h(h)
    print(
        f"Loaded primary: {primary_path}"
        + (f", fp32 sidecar: {fp32_path}" if dual else " (single-session mode)")
    )
    print(f"State shapes ({len(state_shapes)} blocks): "
          f"{[buf for _, buf in state_shapes]} buf widths")

    os.makedirs(a.output_dir, exist_ok=True)
    pesq_primary, pesq_fp32, deltas, rtfs = [], [], [], []
    # Only retained when --metrics is on: the learned MOS metrics are far cheaper
    # scored in one batched pass at the end than per-utterance inside this loop.
    refs, ests_primary, ests_fp32 = [], [], []

    if a.input_noisy_wavs_dir:
        import librosa
        files = sorted(os.listdir(a.input_noisy_wavs_dir))
        if a.max_utterances is not None:
            files = files[:a.max_utterances]
        for name in track(files):
            wav, _ = librosa.load(
                os.path.join(a.input_noisy_wavs_dir, name), sr=int(h.sampling_rate),
            )
            # WAV-dir mode has no PESQ comparison, so don't run the FP32
            # sidecar session just to discard its output.
            primary_audio, _, rtf = enhance_one_utterance(
                wav, primary_session, h, state_shapes, None,
            )
            sf.write(
                os.path.join(a.output_dir, name),
                primary_audio, int(h.sampling_rate), "PCM_16",
            )
            rtfs.append(rtf)
            print(f"{name}: RTF={rtf:.3f} (no PESQ — WAV-dir mode)")
    else:
        hf = load_voicebank_demand(cache_dir=a.hf_cache_dir)
        test_split = hf["test"]
        if a.max_utterances is not None:
            test_split = test_split.select(range(min(a.max_utterances, len(test_split))))
        N = len(test_split)
        for i, item in enumerate(track(test_split)):
            noisy_wav = np.asarray(item["noisy"]["array"], dtype=np.float32)
            clean_wav = np.asarray(item["clean"]["array"], dtype=np.float32)
            primary_audio, fp32_audio, rtf = enhance_one_utterance(
                noisy_wav, primary_session, h, state_shapes, fp32_session,
            )
            p_pri = eval_pesq(clean_wav, primary_audio, int(h.sampling_rate))
            line = f"[{i+1}/{N}] {item['id']}: PESQ primary={p_pri:.3f}; RTF={rtf:.3f}"
            if dual:
                p_fp32 = eval_pesq(clean_wav, fp32_audio, int(h.sampling_rate))
                if p_pri != -1 and p_fp32 != -1:
                    delta = p_fp32 - p_pri
                    line = (
                        f"[{i+1}/{N}] {item['id']}: PESQ FP32={p_fp32:.3f}, "
                        f"primary={p_pri:.3f}, delta={delta:.3f}; RTF={rtf:.3f}"
                    )
                    deltas.append(delta)
                else:
                    deltas.append(float("nan"))
                pesq_fp32.append(p_fp32)
            print(line)
            pesq_primary.append(p_pri)
            rtfs.append(rtf)
            if extra_metrics:
                refs.append(clean_wav)
                ests_primary.append(primary_audio)
                if dual:
                    ests_fp32.append(fp32_audio)
            sf.write(
                os.path.join(a.output_dir, item["id"] + ".wav"),
                primary_audio, int(h.sampling_rate), "PCM_16",
            )

    pri_clean = [s for s in pesq_primary if s != -1]
    pri_mean = float(np.mean(pri_clean)) if pri_clean else float("nan")
    rtf_mean = float(np.mean(rtfs)) if rtfs else float("nan")
    n_fail = len(pesq_primary) - len(pri_clean)
    summary = f"Mean: PESQ primary={pri_mean:.3f}; RTF={rtf_mean:.3f}; PESQ failures={n_fail}"
    if dual:
        fp32_clean = [s for s in pesq_fp32 if s != -1]
        fp32_mean = float(np.mean(fp32_clean)) if fp32_clean else float("nan")
        delta_mean = float(np.nanmean(deltas)) if deltas else float("nan")
        summary = (
            f"Mean: PESQ FP32={fp32_mean:.3f}, primary={pri_mean:.3f}, "
            f"delta={delta_mean:.3f}; RTF primary={rtf_mean:.3f}; "
            f"PESQ failures={n_fail}"
        )
    print(summary)

    # Extra perceptual metrics (opt-in; the PESQ summary above is unchanged).
    if extra_metrics and refs:
        suite = QualitySuite(metrics=extra_metrics, sr=int(h.sampling_rate))
        conds = [("primary", ests_primary)] + ([("fp32", ests_fp32)] if dual else [])
        means_by_cond = {}
        for cond, ests in conds:
            means = QualitySuite.summarize(suite.score(ests, refs))
            means_by_cond[cond] = means
            print(f"Mean [{cond}]: "
                  + ", ".join(f"{k}={v:.3f}" for k, v in means.items()))
        if a.metrics_json:
            with open(a.metrics_json, "w") as f:
                json.dump({"n_utts": len(refs), "metrics": list(extra_metrics),
                           "means": means_by_cond}, f, indent=2)
            print(f"wrote {a.metrics_json}")

    if a.log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(a.log_dir)
        writer.add_scalar("Inference/PESQ Score (primary)", pri_mean, 0)
        writer.add_scalar("Inference/RTF (primary)", rtf_mean, 0)
        if dual:
            writer.add_scalar("Inference/PESQ Score (FP32)", fp32_mean, 0)
            writer.add_scalar("Inference/PESQ Delta (FP32-primary)", delta_mean, 0)
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
