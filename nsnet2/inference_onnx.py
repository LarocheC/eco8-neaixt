"""inference_onnx.py - ORT CPU x86 inference mirror of inference.py for Phase 4's int8 ONNX.

Loads BOTH the int8 ONNX (cp_<run>/g_best.onnx; Phase 4 output) AND the FP32
sibling (cp_<run>/g_best_fp32.onnx; Phase 2 output), runs each frame-by-frame on
CPUExecutionProvider with per-utterance state reset (Pitfall 8 closure), and
logs paired PESQ scores + RTF.

Per-utterance: STFT once -> dual frame-loop (FP32 first, int8 second with RTF
measurement) -> dual iSTFT -> two PESQ scores -> int8 WAV write. After the
loop, mean values are printed to stdout and (when --log_dir is set) written as
TB scalars 'Inference/PESQ Score (FP32)', 'Inference/PESQ Score (int8)',
'Inference/PESQ Delta (FP32-int8)', 'Inference/RTF (int8)'.

INF-04 (Pitfall 9 runtime side): runtime ORT version compared against the
ort_quantize_version metadata stamp written by Phase 4; mismatch and missing
stamp both emit a stdout WARNING and never raise (D-14, D-15).

See .planning/phases/05-ort-inference-script/05-CONTEXT.md (D-01..D-21) for
locked behavior. Mirrors inference.py CLI shape + quant.py helper+main shape.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
from rich.progress import track

from common.env import AttrDict
from common.dataset import mag_pha_stft, mag_pha_istft, load_voicebank_demand
from common.metrics import eval_pesq


def _make_session(onnx_path):
    """4-knob determinism recipe (D-08) - identical config for FP32 and int8 sessions.

    Required by INF-02 byte-identical reruns. Verified by Plan 02-03 at max_abs_err=5.96e-08.
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1                                            # determinism
    so.inter_op_num_threads = 1                                            # determinism
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL                   # determinism
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )


def _check_ort_version(int8_path):
    """INF-04 (Pitfall 9 runtime side): warn-not-crash on version mismatch / missing stamp.

    Reads model.metadata_props for 'ort_quantize_version' (written by quant.py:_add_metadata).
    Reads ort.__version__ LIVE (not cached) so test monkeypatching fires.
    String equality (D-14), not semver.
    """
    model = onnx.load(str(int8_path))
    stamped = next(
        (p.value for p in model.metadata_props if p.key == "ort_quantize_version"),
        None,
    )
    if stamped is None:                                                    # D-15
        print(
            f"WARNING: int8 ONNX has no ort_quantize_version stamp; "
            f"cannot check Pitfall 9 runtime-side."
        )
    elif stamped != ort.__version__:                                       # D-14
        print(
            f"WARNING: int8 ONNX was quantized with ORT {stamped} but runtime is "
            f"ORT {ort.__version__}; may produce different results "
            f"(Pitfall 9 runtime-side)."
        )
    # match -> silent, proceed normally


def _validate_paths(int8_path, fp32_override):
    """D-09 default FP32 derivation + D-11/D-12 fail-fast with verbatim fix-hint.

    int8_path     : Path to int8 ONNX (cp_<run>/g_best.onnx).
    fp32_override : Optional Path; if None, auto-derived as int8_path.parent/"g_best_fp32.onnx".

    Returns (int8_path, fp32_path) both as resolved pathlib.Path objects.

    Raises FileNotFoundError with verbatim fix-hint substrings - the substrings
    'uv run python -m nsnet2.quant --checkpoint_dir' and 'uv run python -m nsnet2.export_onnx --checkpoint_file'
    are load-bearing for the test gate (Plan 05-02 tests 4 and 5).
    """
    int8_path = Path(int8_path)
    if not int8_path.exists():                                             # D-11
        raise FileNotFoundError(
            f"int8 ONNX missing at {int8_path}; run: "
            f"uv run python -m nsnet2.quant --checkpoint_dir {int8_path.parent} "
            f"--num_utterances 200"
        )
    fp32_path = (
        Path(fp32_override) if fp32_override
        else int8_path.parent / "g_best_fp32.onnx"                         # D-09
    )
    if not fp32_path.exists():                                             # D-12
        raise FileNotFoundError(
            f"FP32 ONNX missing at {fp32_path}; run: "
            f"uv run python -m nsnet2.export_onnx --checkpoint_file {int8_path.parent}/g_best"
        )
    return int8_path, fp32_path


def _run_session_frame_by_frame(session, noisy_mag_np, h, measure_rtf=False):
    """Per-utterance ORT frame loop with state reset (Pitfall 8 closure / D-10).

    noisy_mag_np : (B=1, F, T) float32 numpy.
    measure_rtf  : True only for the int8 session (D-02 -- RTF measures int8 cost only).

    Returns (enhanced_mag_np, rtf_total) where rtf_total is the cumulative session.run() time in
    seconds (0.0 when measure_rtf=False).
    """
    B, F, T = noisy_mag_np.shape                                           # (B, F, T)
    L = getattr(h, "num_gru_layers", 2)                                    # match NSNet2 defaults
    H = getattr(h, "hidden_dim", 400)                                      # match NSNet2 defaults
    states_in = np.zeros((L, B, H), dtype=np.float32)                      # (L, B, H) -- Pitfall 8 closure
    masks = []
    rtf_total = 0.0
    for t in range(T):
        frame_in = noisy_mag_np[:, :, t]                                   # (B, F) -- trailing T axis collapses
        if measure_rtf:
            t0 = time.perf_counter()                                       # D-02: around session.run() ONLY
        out = session.run(
            ["mask", "states_out"],
            {"frame_in": frame_in, "states_in": states_in},
        )
        if measure_rtf:
            rtf_total += time.perf_counter() - t0
        masks.append(out[0])                                               # mask: (B, F)
        states_in = out[1]                                                 # thread states_out into next frame
    enhanced_mag_np = np.stack(masks, axis=2)                              # (B, F, T)
    return enhanced_mag_np, rtf_total


def enhance_one_utterance(noisy_wav, fp32_session, int8_session, h):
    """STFT once -> dual frame-loop -> dual iSTFT -> (fp32_audio, int8_audio, rtf_int8).

    noisy_wav : 1-D numpy float32 (samples,) at h.sampling_rate.
    Returns (fp32_audio_np, int8_audio_np, rtf_int8_seconds_per_audio_second).

    Phase 6's training hook will reimport this helper directly (CONTEXT 'Integration Points').
    Mirrors inference.py:28-35 enhance() shape: RMS norm, STFT, denoise, iSTFT, unscale.
    """
    # 1. RMS normalize per inference.py:30-31 (load-bearing -- CONVENTIONS.md)
    noisy_t = torch.from_numpy(np.asarray(noisy_wav, dtype=np.float32))    # (L_samples,)
    norm_factor = torch.sqrt(len(noisy_t) / (torch.sum(noisy_t ** 2.0) + 1e-8))
    noisy_t = (noisy_t * norm_factor).unsqueeze(0)                         # (B=1, L_samples)

    # 2. STFT once
    noisy_mag, noisy_pha, _ = mag_pha_stft(
        noisy_t, h.n_fft, h.hop_size, h.win_size, h.compress_factor,
    )                                                                      # (B=1, F, T)

    # 3. Dual frame-loop on numpy view of noisy_mag
    noisy_mag_np = noisy_mag.numpy().astype(np.float32)                    # (B=1, F, T) float32 -- Pitfall 3
    fp32_mask_np, _      = _run_session_frame_by_frame(fp32_session, noisy_mag_np, h, measure_rtf=False)
    int8_mask_np, rtf_s  = _run_session_frame_by_frame(int8_session, noisy_mag_np, h, measure_rtf=True)

    # 4. Dual iSTFT -- mask * noisy_mag, reuse noisy_pha (NSNet2 phase passthrough)
    fp32_mag = torch.from_numpy(fp32_mask_np) * noisy_mag                  # (B=1, F, T)
    int8_mag = torch.from_numpy(int8_mask_np) * noisy_mag                  # (B=1, F, T)
    fp32_audio = mag_pha_istft(fp32_mag, noisy_pha, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    int8_audio = mag_pha_istft(int8_mag, noisy_pha, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

    # 5. Unscale by norm_factor (inference.py:35 convention)
    fp32_audio = (fp32_audio / norm_factor).squeeze().numpy()
    int8_audio = (int8_audio / norm_factor).squeeze().numpy()

    # 6. RTF = session.run-time / utterance-duration in audio-seconds
    duration_s = len(noisy_wav) / h.sampling_rate
    rtf_int8 = rtf_s / duration_s

    return fp32_audio, int8_audio, rtf_int8


def main():
    print("Initializing Inference Process..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_file", required=True,
                        help="Path to int8 ONNX (cp_<run>/g_best.onnx)")
    parser.add_argument("--fp32_checkpoint_file", default=None,
                        help="Optional FP32 ONNX override; default = <ckpt_parent>/g_best_fp32.onnx (D-09)")
    parser.add_argument("--input_noisy_wavs_dir", default=None,
                        help="Directory of noisy wavs. If unset, uses HuggingFace test split (D-17 default).")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--output_dir", default="generated_files")
    parser.add_argument("--log_dir", default=None,
                        help="TB log directory; if unset, no TB writes (D-01)")
    parser.add_argument("--max_utterances", type=int, default=None,
                        help="Optional slice for fast smoke iteration. None = all.")
    a = parser.parse_args()

    # --- 1. Sibling config.json (D-13 defensive fail-fast) -----------------
    parent_dir = os.path.split(a.checkpoint_file)[0]
    config_file = os.path.join(parent_dir, "config.json")
    if not os.path.exists(config_file):                                    # D-13 + D-11 fix-hint (UAT gap closure)
        raise FileNotFoundError(
            f"sibling config.json missing at {config_file}; "
            f"checkpoint dir is corrupted or non-standard. "
            f"If the int8 ONNX is also missing, run: "
            f"uv run python -m nsnet2.quant --checkpoint_dir {parent_dir} "
            f"--num_utterances 200"
        )
    with open(config_file) as f:
        h = AttrDict(json.loads(f.read()))                                 # mirrors inference.py:80-82
    torch.manual_seed(h.seed)                                              # mirrors inference.py:84

    # --- 2. Validate paths + INF-04 metadata check -------------------------
    int8_path, fp32_path = _validate_paths(a.checkpoint_file, a.fp32_checkpoint_file)
    _check_ort_version(int8_path)                                          # D-14, D-15 -- never raises

    # --- 3. Build BOTH ORT sessions ONCE (D-07 -- not per-utterance) -------
    fp32_session = _make_session(fp32_path)
    int8_session = _make_session(int8_path)

    os.makedirs(a.output_dir, exist_ok=True)                               # mirrors inference.py:44

    # --- 4. State accumulators for after-loop aggregate --------------------
    pesq_fp32_scores, pesq_int8_scores, deltas, rtfs = [], [], [], []

    # --- 5. Mode dispatch --------------------------------------------------
    if a.input_noisy_wavs_dir:                                             # D-16: WAV-dir mode (no PESQ)
        import librosa                                                     # D-20: lazy-import inside branch ONLY
        test_indexes = sorted(os.listdir(a.input_noisy_wavs_dir))
        if a.max_utterances is not None:
            test_indexes = test_indexes[:a.max_utterances]
        for index in track(test_indexes):
            noisy_wav, _ = librosa.load(
                os.path.join(a.input_noisy_wavs_dir, index), sr=h.sampling_rate,
            )
            _, int8_audio, rtf_int8 = enhance_one_utterance(
                noisy_wav, fp32_session, int8_session, h,
            )
            sf.write(                                                      # D-21
                os.path.join(a.output_dir, index),
                int8_audio, h.sampling_rate, "PCM_16",
            )
            rtfs.append(rtf_int8)
            print(f"{index}: RTF int8={rtf_int8:.3f} (no PESQ - WAV-dir mode)")
    else:                                                                  # D-17: HF-test mode (default; PESQ + WAV)
        hf = load_voicebank_demand(cache_dir=a.hf_cache_dir)               # mirrors inference.py:56
        test_split = hf["test"]
        if a.max_utterances is not None:
            test_split = test_split.select(range(min(a.max_utterances, len(test_split))))
        N = len(test_split)
        for i, item in enumerate(track(test_split)):
            noisy_wav = np.asarray(item["noisy"]["array"], dtype=np.float32)   # mirrors inference.py:59
            clean_wav = np.asarray(item["clean"]["array"], dtype=np.float32)
            fp32_audio, int8_audio, rtf_int8 = enhance_one_utterance(
                noisy_wav, fp32_session, int8_session, h,
            )
            pesq_fp32 = eval_pesq(clean_wav, fp32_audio, h.sampling_rate)  # D-05/D-06
            pesq_int8 = eval_pesq(clean_wav, int8_audio, h.sampling_rate)
            if pesq_fp32 == -1 or pesq_int8 == -1:                         # D-06 partial-failure handling
                delta = float("nan")
                print(f"[{i+1}/{N}] {item['id']}: PESQ failed (skipped from mean)")
            else:
                delta = pesq_fp32 - pesq_int8
                print(
                    f"[{i+1}/{N}] {item['id']}: PESQ FP32={pesq_fp32:.3f}, "
                    f"int8={pesq_int8:.3f}, delta={delta:.3f}; RTF int8={rtf_int8:.3f}"
                )
            pesq_fp32_scores.append(pesq_fp32)
            pesq_int8_scores.append(pesq_int8)
            deltas.append(delta)
            rtfs.append(rtf_int8)
            sf.write(                                                      # D-21
                os.path.join(a.output_dir, item["id"] + ".wav"),
                int8_audio, h.sampling_rate, "PCM_16",
            )

    # --- 6. After-loop aggregate (D-04 PESQ-primary scope shift) -----------
    fp32_clean = [s for s in pesq_fp32_scores if s != -1]                  # D-06 -1-filter
    int8_clean = [s for s in pesq_int8_scores if s != -1]
    fp32_mean = float(np.mean(fp32_clean)) if fp32_clean else float("nan")
    int8_mean = float(np.mean(int8_clean)) if int8_clean else float("nan")
    delta_mean = float(np.nanmean(deltas)) if deltas else float("nan")
    rtf_mean = float(np.mean(rtfs)) if rtfs else float("nan")
    n_failures = (
        (len(pesq_fp32_scores) - len(fp32_clean))
        + (len(pesq_int8_scores) - len(int8_clean))
    )
    print(
        f"Mean: PESQ FP32={fp32_mean:.3f}, int8={int8_mean:.3f}, "
        f"delta={delta_mean:.3f}; RTF int8={rtf_mean:.3f}; PESQ failures={n_failures}"
    )

    # --- 7. TB scalars (D-04 + D-01: only when --log_dir set) --------------
    if a.log_dir:                                                          # D-01
        from torch.utils.tensorboard import SummaryWriter                  # lazy-import per D-01
        writer = SummaryWriter(a.log_dir)
        writer.add_scalar("Inference/PESQ Score (FP32)",      fp32_mean,  0)
        writer.add_scalar("Inference/PESQ Score (int8)",      int8_mean,  0)
        writer.add_scalar("Inference/PESQ Delta (FP32-int8)", delta_mean, 0)
        writer.add_scalar("Inference/RTF (int8)",             rtf_mean,   0)
        writer.flush()                                                     # Pitfall 5: mandatory pre-exit
        writer.close()


if __name__ == "__main__":
    main()
