"""Stage 1: published checkpoint -> enhanced audio on disk.

One adapter per family. Each adapter *calls the family's own* ``enhance_one_utterance``
rather than reimplementing it: the streaming loops thread recurrent/FIFO state
per frame and are exactly the kind of code that breaks silently when duplicated.

    python -m benchmarks.enhance                       # every published model
    python -m benchmarks.enhance --slug nsnet2__baseline
    python -m benchmarks.enhance --jobs 8 --max_utterances 50

Output layout (float32 WAV -- see the level convention note below):

    <out>/audio/<family>__<name>/<condition>/<utt_id>.wav
    <out>/audio/noisy/<utt_id>.wav
    <out>/audio/clean/<utt_id>.wav

Level convention
----------------
All three families RMS-normalise the noisy input before the STFT. NSNet2 and
ConvFSENet divide that factor back out after the iSTFT, returning audio at the
dataset's original level; LiSenNet's eval_deploy does not -- it scores in the
normalised domain, against a normalised reference. That was harmless when PESQ
was the only metric (PESQ level-aligns internally), but DNSMOS and NISQA are
level-sensitive MOS predictors, so leaving it would make LiSenNet's MOS numbers
incomparable with the other two families.

Everything written here is therefore in the **original dataset level**: LiSenNet's
output is un-normalised on the way out, and the reference is the raw clean signal.

Enhancement runs single-threaded ONNX Runtime (the determinism/RTF convention the
eval scripts use), so models are parallelised across *processes* instead.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from benchmarks.published import BASELINES, MODELS, Model, by_slug, fetch
from common.dataset import load_voicebank_demand
from common.env import AttrDict

# Lossless float32 so the cached audio is bit-for-bit what the model produced.
# PCM_16 would round at ~-90 dBFS -- inaudible, but enough to stop the cached
# PESQ from reproducing the published PESQ exactly, which is the check that tells
# us this harness is wired up correctly.
_SUBTYPE = "FLOAT"


def _write(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr, subtype=_SUBTYPE)


def _config(model_dir: Path) -> AttrDict:
    with open(model_dir / "config.json") as f:
        return AttrDict(json.load(f))


def _rms_norm(noisy: np.ndarray):
    """The normalisation every family applies. Returns (normalised, norm_factor)."""
    t = torch.from_numpy(np.asarray(noisy, dtype=np.float32))
    nf = torch.sqrt(len(t) / (torch.sum(t ** 2.0) + 1e-8))
    return t * nf, nf


# --- per-family adapters -----------------------------------------------------
# Each yields {condition: enhanced_1d_float32} for one utterance.


def _adapter_nsnet2(model_dir: Path, h):
    from nsnet2.inference_onnx import _make_session, enhance_one_utterance

    fp32_s = _make_session(model_dir / "g_best_fp32.onnx")
    int8_s = _make_session(model_dir / "g_best.onnx")

    def run(noisy):
        fp32_audio, int8_audio, _ = enhance_one_utterance(noisy, fp32_s, int8_s, h)
        return {"fp32": fp32_audio, "int8": int8_audio}

    return run


def _adapter_convfsenet(model_dir: Path, h):
    from convfsenet.inference_onnx import (
        _make_session,
        enhance_one_utterance,
        state_shapes_from_h,
    )

    shapes = state_shapes_from_h(h)
    int8_s = _make_session(model_dir / "g_best.onnx")
    fp32_s = _make_session(model_dir / "g_best_fp32.onnx")

    def run(noisy):
        # ConvFSENet's signature takes the int8 graph as `primary` (that's what
        # its RTF is measured on) and the FP32 graph as an optional sidecar.
        int8_audio, fp32_audio, _ = enhance_one_utterance(
            noisy, int8_s, h, shapes, fp32_session=fp32_s
        )
        return {"fp32": fp32_audio, "int8": int8_audio}

    return run


def _adapter_lisennet(model_dir: Path, h):
    from lisennet.eval_deploy import _reconstruct, _session
    from lisennet.export_onnx import _load_from_checkpoint

    model = _load_from_checkpoint(model_dir / "g_best").eval()
    sessions = {
        "fp32": _session(model_dir / "g_best_fp32.onnx"),
        "int8": _session(model_dir / "g_best_int8_static.onnx"),
    }

    @torch.no_grad()
    def run(noisy):
        noisy_t, nf = _rms_norm(noisy)
        noisy_t = noisy_t.unsqueeze(0)                       # (1, L)
        length = noisy_t.shape[-1]

        spec = model.power_compress(model.apply_stft(noisy_t))
        src_mag, src_pha = spec.abs(), spec.angle()
        feat_np = model.build_features(src_mag, src_pha).cpu().numpy()

        out = {}
        for key, sess in sessions.items():
            est_mag = torch.from_numpy(sess.run(["est_mag"], {"feat": feat_np})[0])
            # fp32/int8 hold the Griffin-Lim phase fixed so they isolate the cost
            # of quantising the mask; int8_rt swaps in the noisy phase, which is
            # the causal phase you can actually stream.
            est = _reconstruct(model, est_mag, src_pha, length, use_gl=True)
            out[key] = (est / nf).squeeze().numpy()
            if key == "int8":
                rt = _reconstruct(model, est_mag, src_pha, length, use_gl=False)
                out["int8_rt"] = (rt / nf).squeeze().numpy()
        return out

    return run


_ADAPTERS = {
    "nsnet2": _adapter_nsnet2,
    "convfsenet": _adapter_convfsenet,
    "lisennet": _adapter_lisennet,
}


def enhance_model(model: Model, hf_root: Path, out_root: Path, max_utterances=None) -> str:
    """Run one published model over the test split, writing every condition's audio."""
    # Keep ORT/torch single-threaded inside each worker: we get our parallelism
    # from running models concurrently, and oversubscribing would distort nothing
    # here but would slow everything down.
    torch.set_num_threads(1)

    model_dir = model.local_dir(hf_root)
    h = _config(model_dir)
    run = _ADAPTERS[model.family](model_dir, h)

    split = load_voicebank_demand()["test"]
    if max_utterances:
        split = split.select(range(min(max_utterances, len(split))))

    audio_root = Path(out_root) / "audio" / model.slug
    done = 0
    for item in split:
        noisy = np.asarray(item["noisy"]["array"], dtype=np.float32)
        for cond, audio in run(noisy).items():
            _write(audio_root / cond / f"{item['id']}.wav", audio, h.sampling_rate)
        done += 1
    return f"{model.slug}: {done} utts -> {audio_root}"


def dump_baselines(out_root: Path, sr=16000, max_utterances=None) -> str:
    """Write the unprocessed noisy input and the clean reference through the same path."""
    split = load_voicebank_demand()["test"]
    if max_utterances:
        split = split.select(range(min(max_utterances, len(split))))
    root = Path(out_root) / "audio"
    for item in split:
        for tag in BASELINES:
            key = "noisy" if tag == "noisy" else "clean"
            _write(root / tag / f"{item['id']}.wav",
                   np.asarray(item[key]["array"], dtype=np.float32), sr)
    return f"baselines: {len(split)} utts -> {root}/{{noisy,clean}}"


def main():
    p = argparse.ArgumentParser(description="Enhance the VBD test split with the published models.")
    p.add_argument("--out", default="benchmarks/out", help="Benchmark working root")
    p.add_argument("--hf_root", default=None, help="Where the HF snapshots live (default <out>/hf)")
    p.add_argument("--slug", action="append", default=None,
                   help="Only this model (repeatable), e.g. nsnet2__baseline")
    p.add_argument("--jobs", type=int, default=8,
                   help="Models enhanced concurrently; each is single-threaded ORT")
    p.add_argument("--max_utterances", type=int, default=None)
    p.add_argument("--skip_fetch", action="store_true")
    a = p.parse_args()

    out = Path(a.out)
    hf_root = Path(a.hf_root) if a.hf_root else out / "hf"
    models = [by_slug(s) for s in a.slug] if a.slug else MODELS

    if not a.skip_fetch:
        fetch(hf_root, repos=sorted({m.repo for m in models}))

    print(dump_baselines(out, max_utterances=a.max_utterances))

    # Bound BLAS threads in the children too -- numpy/torch each default to
    # spawning one thread per core, which with --jobs 8 means 256 threads.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futs = {
            pool.submit(enhance_model, m, hf_root, out, a.max_utterances): m
            for m in models
        }
        for i, fut in enumerate(as_completed(futs), 1):
            m = futs[fut]
            try:
                print(f"[{i}/{len(futs)}] {fut.result()}", flush=True)
            except Exception as e:  # one bad checkpoint shouldn't sink the sweep
                print(f"[{i}/{len(futs)}] FAILED {m.slug}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
