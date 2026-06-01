"""quant_hook.py - In-training int8 quant-eval hook for train.py's validation block.

Builds a per-cycle scratch artifact set in <ckpt_dir>/.quant_cache/ (FP32 ONNX +
int8 ONNX), runs onnxruntime int8 inference frame-by-frame on the FULL VBD test
split, and emits two new TensorBoard scalars (TRN-02) on the existing
SummaryWriter at the same `steps` x-axis as the existing FP32 PESQ. The FP32
PESQ is reused from train.py's existing val_pesq_score (line 233) -- the hook
runs ONLY the int8 ORT pass (~50% wall-clock saved vs dual-session).

Reuses Phase 5's _make_session, _check_ort_version, _run_session_frame_by_frame
(Pitfall 8 + Pitfall 9 closures inherited transitively).

Phase 6 boundary (D-02): this module is loaded ONLY when h.quant.enabled=True
(lazy-import gate inside train.py's validation block). When disabled, train.py
startup imports nothing from quant.py / onnxruntime.quantization (TRN-05).

See .planning/phases/06-in-training-quant-eval-hook/06-CONTEXT.md (D-01..D-04).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from nsnet2.quant import quantize_checkpoint
from nsnet2.export_onnx import export_streaming
from nsnet2.inference_onnx import _make_session, _check_ort_version, _run_session_frame_by_frame
from common.dataset import mag_pha_stft, mag_pha_istft
from common.metrics import eval_pesq


def run_quant_eval(generator, h, hf, sw, steps, ckpt_dir, *,
                   fp32_pesq, log_breakdown=False, num_gpus=0):
    """In-training int8 quant-eval hook.

    Per cycle:
      1. DDP-unwrap the live generator + write state_dict to .quant_cache/g_best
         (filename HARDCODED by quant.py:92 -- quantize_checkpoint reads cp_dir/g_best).
      2. FP32-export to .quant_cache/g_best_fp32.onnx (name HARDCODED by quant.py:93).
      3. quantize_checkpoint(.quant_cache, n_calib_utts, out_path=.quant_cache/g_train.onnx).
      4. Build int8 ORT session; for each item in hf['test'], run int8 frame-loop
         and accumulate eval_pesq(clean, int8_audio, h.sampling_rate).
      5. Write the two new TB scalars (TRN-02 verbatim names) on the existing SummaryWriter.
      6. If log_breakdown: print verbatim TRN-03 stdout format.

    Returns dict with int8_pesq, fp32_pesq, delta, t_export, t_calibrate, t_ort_validate.
    """
    # Defensive: re-wrap h.quant as AttrDict in case h was JSON-loaded (env.py AttrDict is flat).
    from common.env import AttrDict
    if not isinstance(h.get("quant"), AttrDict):
        h = AttrDict({**dict(h), "quant": AttrDict(h.get("quant", {}))})

    # --- 1. Scratch dir + DDP unwrap + write ckpt + sibling config.json ---
    # PyTorch ckpt filename HARDCODED to "g_best" by quant.py:92's read path
    # (quantize_checkpoint internally calls NSNet2Streaming.from_checkpoint(cp_dir / "g_best")).
    # The int8 ONNX OUTPUT filename ("g_train.onnx") at line 83 IS configurable via out_path=.
    scratch_dir = Path(ckpt_dir) / ".quant_cache"
    scratch_dir.mkdir(exist_ok=True)
    ckpt_path = scratch_dir / "g_best"
    gen_module = generator.module if num_gpus > 1 else generator
    torch.save({"generator": gen_module.state_dict()}, str(ckpt_path))

    # quant.quantize_checkpoint reads <scratch_dir>/config.json via from_checkpoint.
    # AttrDict subclasses dict (env.py:4); dict(h) round-trips through json fine,
    # but nested AttrDict blocks (h.quant, h.linear, h.gru, h.dist_config) need
    # explicit dict() conversion to be JSON-serializable.
    h_serial = {k: (dict(v) if isinstance(v, dict) else v) for k, v in dict(h).items()}
    # Pin h.calibration.seed = h.seed AND h.calibration.num_utterances = h.quant.n_calib_utts
    # so VBDCalibrationReader's deterministic rng.sample(...) produces identical indices
    # across cycles (TRN-04 cache-once via determinism per RESEARCH Synthesis #4).
    cal_block = dict(h_serial.get("calibration", {}))
    cal_block["seed"] = h.seed
    cal_block["num_utterances"] = h.quant.n_calib_utts
    h_serial["calibration"] = cal_block
    with open(scratch_dir / "config.json", "w") as f:
        json.dump(h_serial, f)

    # --- 2. FP32 export. quant.py:93 HARDCODES 'g_best_fp32.onnx' as input name (D-03 refined). ---
    fp32_onnx_path = scratch_dir / "g_best_fp32.onnx"
    t0 = time.perf_counter()
    export_streaming(str(ckpt_path), output_path=str(fp32_onnx_path))
    t_export = time.perf_counter() - t0

    # --- 3. Int8 quantize. KWARG IS 'out_path=' (NOT 'output_path='; different from export_streaming). ---
    int8_onnx_path = scratch_dir / "g_train.onnx"
    t0 = time.perf_counter()
    quantize_checkpoint(scratch_dir, n_calib_utts=h.quant.n_calib_utts, out_path=int8_onnx_path)
    t_calibrate = time.perf_counter() - t0

    # --- 4. Build int8 ORT session ONCE (NOT per-utterance). FP32 session NOT built (Open Q1). ---
    int8_session = _make_session(int8_onnx_path)
    _check_ort_version(int8_onnx_path)                       # never raises (Pitfall 9 / D-14, D-15)

    # --- 5. Iterate hf['test'] DIRECTLY (NOT the train.py val DataLoader -- avoids RMS double-norm
    #        per RESEARCH B4 Option A). Mirrors inference_onnx.py:252-282 verbatim. ---
    int8_scores = []
    t0 = time.perf_counter()
    for item in hf["test"]:
        noisy_wav = np.asarray(item["noisy"]["array"], dtype=np.float32)
        clean_wav = np.asarray(item["clean"]["array"], dtype=np.float32)

        # RMS-normalize once (mirrors inference_onnx.py:156-159 / inference.py:30-31).
        noisy_t = torch.from_numpy(noisy_wav)
        norm_factor = torch.sqrt(len(noisy_t) / (torch.sum(noisy_t ** 2.0) + 1e-8))
        noisy_t = (noisy_t * norm_factor).unsqueeze(0)

        # STFT once.
        noisy_mag, noisy_pha, _ = mag_pha_stft(
            noisy_t, h.n_fft, h.hop_size, h.win_size, h.compress_factor,
        )

        # Int8 frame-by-frame ORT (state reset INSIDE the helper -- Pitfall 8).
        noisy_mag_np = noisy_mag.numpy().astype(np.float32)
        int8_mask_np, _ = _run_session_frame_by_frame(int8_session, noisy_mag_np, h, measure_rtf=False)

        # iSTFT + unscale (mirrors inference_onnx.py:172-179).
        int8_mag = torch.from_numpy(int8_mask_np) * noisy_mag
        int8_audio = mag_pha_istft(int8_mag, noisy_pha, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
        int8_audio = (int8_audio / norm_factor).squeeze().numpy()

        int8_scores.append(eval_pesq(clean_wav, int8_audio, h.sampling_rate))
    t_ort_validate = time.perf_counter() - t0

    # --- 6. Aggregate (-1 filter pattern from inference_onnx.py:285-289). ---
    int8_clean = [s for s in int8_scores if s != -1]
    int8_mean = float(np.mean(int8_clean)) if int8_clean else float("nan")
    delta = fp32_pesq - int8_mean

    # --- 7. TB scalars -- TRN-02 verbatim names. ---
    sw.add_scalar("Validation/PESQ Score (int8)", int8_mean, steps)
    sw.add_scalar("Validation/PESQ Delta (FP32-int8)", delta, steps)

    # --- 8. Cycle-1 stdout breakdown -- TRN-03 verbatim format. ---
    if log_breakdown:
        print("Quant eval: export={:.2f}s, calibrate={:.2f}s, ort_validate={:.2f}s".format(
            t_export, t_calibrate, t_ort_validate,
        ))

    return {
        "int8_pesq":      int8_mean,
        "fp32_pesq":      fp32_pesq,
        "delta":          delta,
        "t_export":       t_export,
        "t_calibrate":    t_calibrate,
        "t_ort_validate": t_ort_validate,
    }
