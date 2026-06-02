"""calibration.py - VBDCalibrationReader for static int8 PTQ calibration.

Yields per-frame (frame_in, states_in) dicts to onnxruntime.quantization.quantize_static
from the VoiceBank-DEMAND-16k train split with real-propagated GRU state per utterance.

Closes Pitfall 4 (calibration nondeterminism — D-12, D-13, D-14) via a content-derived
calibration_hash logged at construction, and Pitfall 5 (calibration-set leakage —
D-09, D-10, D-11) via sha256-of-audio-bytes disjoint-set assertion against the test split.

Single-use iterator. quantize_static consumes via a single while loop (no rewind).

`frames_per_utterance` semantics: when None (default), yield every frame of each
calibration utterance. When an int N, yield only the first N frames per utterance
(plain truncation, consistent across utterances).

HF cache_dir resolution: this module resolves the on-disk hash cache via
`datasets.config.HF_DATASETS_CACHE`. Callers passing a one-off `cache_dir` to
`load_voicebank_demand` without setting the `HF_DATASETS_CACHE` env var will see
the hash cache live in the default HF location — minor inefficiency, not a
correctness issue.
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import DatasetDict
from datasets import config as datasets_config
from onnxruntime.quantization import CalibrationDataReader

from common.env import AttrDict
from common.dataset import mag_pha_stft
from nsnet2.streaming import NSNet2Streaming


class VBDCalibrationReader(CalibrationDataReader):
    """Yields (frame_in, states_in) dicts from VBD train split with real GRU state.

    Constructor (D-05):
        VBDCalibrationReader(streaming: NSNet2Streaming, h: AttrDict, hf_dataset: DatasetDict)

    Reads h.calibration sub-block with getattr defaults (D-07):
        num_utterances=200, seed=h.seed, frames_per_utterance=None

    Single-use. quantize_static iterates once via while loop; subsequent get_next() returns None.
    """

    def __init__(self, streaming: NSNet2Streaming, h, hf_dataset: DatasetDict):
        # Variant guard lifted: VBDCalibrationReader runs streaming.forward_step
        # in PyTorch eager mode (not ONNX). Structured FC/GRU custom ops work
        # fine at the eager level — the only export-time blockers
        # (BlockdiagMultiply, butterfly_multiply C++ op) live in torch.onnx,
        # not in the eager forward path. Calibration frames produced here go
        # to ORT, which sees only the post-patch FP32 ONNX graph.
        # D-08: eval mode + later wrap forward pass in no_grad().
        self.streaming = streaming.eval()
        self.h = h
        self.hf = hf_dataset

        # D-07: getattr defaults so cp_*/config.json without h.calibration still works.
        cal_cfg = getattr(h, "calibration", AttrDict({}))
        self.num_utterances = getattr(cal_cfg, "num_utterances", 200)
        self.seed = getattr(cal_cfg, "seed", h.seed)
        self.frames_per_utterance = getattr(cal_cfg, "frames_per_utterance", None)

        # Seeded local RNG (matches dataset.py:54 idiom — never the global random module).
        n_train = len(hf_dataset["train"])
        if self.num_utterances > n_train:
            raise ValueError(
                f"num_utterances={self.num_utterances} exceeds train split size {n_train}"
            )
        rng = random.Random(self.seed)
        self.calib_indices = rng.sample(range(n_train), self.num_utterances)

        # Pitfall 5 closure (CAL-02 / D-09 / D-10 / D-11). Runs every construction.
        self._assert_train_test_disjoint()

        # Pitfall 4 closure (CAL-04 / D-12 / D-13 / D-14).
        self.calibration_hash = self._compute_calibration_hash()
        # D-13: log at construction. Format MUST match VALIDATION.md test_calibration_hash_logged.
        print(
            f"VBDCalibrationReader: calib_hash={self.calibration_hash} "
            f"(utts={self.num_utterances}, seed={self.seed}, "
            f"frames_per_utterance={self.frames_per_utterance})"
        )

        # D-01: per-utterance up-front cache. Initialized empty; populated lazily.
        self._utt_iter = iter(self.calib_indices)
        self._frame_cache: list[dict] = []

    # --- CalibrationDataReader contract -------------------------------------------------

    def get_next(self) -> Optional[dict]:
        """Return next frame dict, or None on exhaustion (per CalibrationDataReader contract).

        D-01: when cache empty, advance to next calib utterance, fill cache, then yield head.
        Returning None (not raising StopIteration; not returning {}) matches the documented
        ORT 1.25 quantize_static consumption pattern (verified in 03-RESEARCH.md).
        """
        while not self._frame_cache:
            idx = next(self._utt_iter, None)
            if idx is None:
                return None
            self._load_next_utterance(idx)
        return self._frame_cache.pop(0)

    # --- private helpers ----------------------------------------------------------------

    def _load_next_utterance(self, idx: int) -> None:
        """D-01 / D-02 / D-03 / D-04 / D-08: STFT + state-prop loop for one train utterance."""
        item = self.hf["train"][idx]
        # D-02: noisy audio (production input distribution); NOT clean.
        audio_np = np.asarray(item["noisy"]["array"], dtype=np.float32)  # (N,) float32
        audio = torch.from_numpy(audio_np).unsqueeze(0)                  # (1, N)
        # D-03: reuse existing dataset.mag_pha_stft. Magnitude only (ignore pha + com).
        mag, _, _ = mag_pha_stft(
            audio,
            self.h.n_fft,
            self.h.hop_size,
            self.h.win_size,
            self.h.compress_factor,
        )                                                                # (1, F, T)
        T = mag.shape[2]
        if self.frames_per_utterance is not None:
            T = min(T, self.frames_per_utterance)

        # D-04: zeros h0 per utterance — NOT carried across utterance boundaries.
        states = torch.zeros(
            self.streaming.num_layers,
            1,
            self.streaming.hidden_size,
            dtype=torch.float32,
        )                                                                # (L, 1, H)

        # D-08: no_grad wraps the forward-prop pass.
        with torch.no_grad():
            for t in range(T):
                frame = mag[:, :, t]                                     # (1, F)
                # Cache the (frame, current_state) pair BEFORE advancing state -
                # the calibrator sees the input distribution at the current frame, including h0 at t=0.
                self._frame_cache.append({
                    "frame_in":  frame.detach().cpu().contiguous().numpy().astype(np.float32),    # (1, F)
                    "states_in": states.detach().cpu().contiguous().numpy().astype(np.float32),   # (L, 1, H)
                })
                _, states = self.streaming.forward_step(frame, states)

    # --- Pitfall 5 closure (CAL-02 / D-09 / D-10 / D-11) --------------------------------

    @staticmethod
    def _hash_audio(arr: np.ndarray) -> str:
        """D-09: sha256 of audio bytes with explicit little-endian dtype.

        The explicit `'<f4'` cast pins byte order so two machines that share the same
        audio array (e.g., x86 dev box + ARM CI runner) compute identical hashes —
        the cache and the disjoint assertion stay portable.
        """
        # `'<f4'` == little-endian float32. copy=False avoids extra allocation when
        # the input is already contiguous little-endian f32 (the common case).
        arr_le = arr.astype('<f4', copy=False)
        return hashlib.sha256(arr_le.tobytes()).hexdigest()

    def _hash_audio_at(self, split, idx: int) -> str:
        item = split[int(idx)]
        arr = np.asarray(item["noisy"]["array"], dtype=np.float32)
        return self._hash_audio(arr)

    def _hash_cache_path(self, train_fp: str, test_fp: str) -> Path:
        # D-10: HF datasets cache root + project-namespaced subdir + per-fingerprint file.
        return Path(datasets_config.HF_DATASETS_CACHE) / "vbd-hashes" / f"{train_fp}_{test_fp}.json"

    def _load_or_build_hash_cache(self, train_fp: str, test_fp: str):
        """D-10: try on-disk cache; on miss/corruption/unwriteable, recompute (and try to persist)."""
        cache_path = self._hash_cache_path(train_fp, test_fp)
        # Read path
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cache = json.load(f)
                if (
                    cache.get("train_fingerprint") == train_fp
                    and cache.get("test_fingerprint") == test_fp
                    and isinstance(cache.get("train_hashes"), list)
                    and isinstance(cache.get("test_hashes"), list)
                    and len(cache["train_hashes"]) == len(self.hf["train"])
                    and len(cache["test_hashes"]) == len(self.hf["test"])
                ):
                    return cache["train_hashes"], cache["test_hashes"]
            except (json.JSONDecodeError, OSError, KeyError, TypeError):
                pass  # fall through to recompute

        # Recompute path
        train_split = self.hf["train"]
        test_split = self.hf["test"]
        train_hashes = [self._hash_audio_at(train_split, i) for i in range(len(train_split))]
        test_hashes = [self._hash_audio_at(test_split, i) for i in range(len(test_split))]

        # Best-effort persist; degrade silently to in-memory if cache_dir unwriteable.
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "train_fingerprint": train_fp,
                "test_fingerprint": test_fp,
                "train_hashes": train_hashes,
                "test_hashes": test_hashes,
                "computed_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(cache_path, "w") as f:
                json.dump(payload, f)
        except OSError as e:
            print(
                f"VBDCalibrationReader: hash cache unwriteable at {cache_path} ({e}); "
                f"computing in-memory."
            )
        return train_hashes, test_hashes

    def _assert_train_test_disjoint(self) -> None:
        train_fp = self.hf["train"]._fingerprint
        test_fp = self.hf["test"]._fingerprint
        train_hashes, test_hashes = self._load_or_build_hash_cache(train_fp, test_fp)
        # Subset only to indices we actually use for calibration:
        calib_hashes = {train_hashes[i] for i in self.calib_indices}
        test_hash_set = set(test_hashes)
        assert calib_hashes.isdisjoint(test_hash_set), (
            "Calibration set leakage detected: at least one train utt has the same audio "
            "bytes as a test utt. Pitfall 5 closure (D-09 / CAL-02). Overlap is not "
            "allowed; the calibration set must be disjoint from the validation set."
        )

    # --- Pitfall 4 closure (CAL-04 / D-12 / D-13 / D-14) --------------------------------

    def _compute_calibration_hash(self) -> str:
        """D-12: sha256 over a deterministic JSON serialization of the input tuple."""
        payload = json.dumps(
            {
                "num_utterances": self.num_utterances,
                "seed": self.seed,
                "frames_per_utterance": self.frames_per_utterance,
                "calib_indices": sorted(int(i) for i in self.calib_indices),
                "train_fingerprint": self.hf["train"]._fingerprint,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
