"""Calibration data reader for ConvFSENet streaming ONNX int8 PTQ.

Yields per-frame dicts with keys matching the ConvFSENetStreamingONNX graph:

    {
        "noisy_mag":  (B=1, F)             float32,
        "state_0_in": (B=1, C, buf_0)      float32,
        ...
        "state_8_in": (B=1, C, buf_8)      float32,
    }

Drives a PyTorch ConvFSENetStreamingFast through real VoiceBank-DEMAND
train-split noisy audio, propagating the per-block FIFO state per utterance
(h0 = zeros per utterance, never carried across).

Closes the same two production pitfalls VBDCalibrationReader closes for
NSNet2, slightly slimmer (no on-disk hash cache — calibration runs are
infrequent and the in-memory hash is fast enough on the 11k-utt split):

  - Calibration nondeterminism: seeded RNG over train indices + a content-
    derived calibration_hash logged at construction.
  - Calibration-set leakage: sha256-of-audio-bytes disjoint-set assertion
    against the test split.

Mirrors calibration.py:VBDCalibrationReader but adapts to ConvFSENet's
per-block state shape and the real-magnitude (not GRU-hidden-state) input
contract.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Optional

import numpy as np
import torch
from onnxruntime.quantization import CalibrationDataReader

from common.dataset import mag_pha_stft
from common.env import AttrDict
from convfsenet.streaming import (
    ConvFSENetStreamingFast,
    ConvFSENetStreamingONNX,
    ConvFSENetWindowedONNX,
)


class ConvFSENetCalibrationReader(CalibrationDataReader):
    """ORT CalibrationDataReader for ConvFSENet streaming ONNX.

    Constructor:
        ConvFSENetCalibrationReader(
            fast       = ConvFSENetStreamingFast (eval, causal=True),
            onnx_view  = ConvFSENetStreamingONNX wrapping `fast`,
            h          = AttrDict; reads n_fft / hop_size / win_size /
                         compress_factor + a `calibration` sub-block with
                         num_utterances / seed / frames_per_utterance.
            hf_dataset = HF DatasetDict-like with ["train"] / ["test"]
                         splits returning dicts with `noisy.array`.
        )

    Single-use: quantize_static consumes via a single get_next() loop.
    """

    def __init__(
        self,
        fast: ConvFSENetStreamingFast,
        onnx_view: ConvFSENetStreamingONNX,
        h: AttrDict,
        hf_dataset,
    ):
        self.fast = fast.eval()
        self.onnx_view = onnx_view
        self.h = h
        self.hf = hf_dataset

        cal_cfg = getattr(h, "calibration", AttrDict({}))
        self.num_utterances = int(getattr(cal_cfg, "num_utterances", 200))
        self.seed = int(getattr(cal_cfg, "seed", getattr(h, "seed", 0)))
        self.frames_per_utterance = getattr(cal_cfg, "frames_per_utterance", None)

        n_train = len(hf_dataset["train"])
        if self.num_utterances > n_train:
            raise ValueError(
                f"num_utterances={self.num_utterances} exceeds train split size {n_train}"
            )
        rng = random.Random(self.seed)
        self.calib_indices = rng.sample(range(n_train), self.num_utterances)

        # Pitfall closures.
        self._assert_train_test_disjoint()
        self.calibration_hash = self._compute_calibration_hash()
        # Format matches calibration.py's log line so external tooling that
        # greps for "calib_hash=" picks both up.
        print(
            f"ConvFSENetCalibrationReader: calib_hash={self.calibration_hash} "
            f"(utts={self.num_utterances}, seed={self.seed}, "
            f"frames_per_utterance={self.frames_per_utterance})"
        )

        # Lazy per-utterance materialization. _frame_cache holds the next
        # utterance's frame dicts; refilled on demand from _utt_iter.
        self._utt_iter = iter(self.calib_indices)
        self._frame_cache: list[dict] = []

    # ---- CalibrationDataReader contract ----------------------------------

    def get_next(self) -> Optional[dict]:
        while not self._frame_cache:
            idx = next(self._utt_iter, None)
            if idx is None:
                return None
            self._load_next_utterance(idx)
        return self._frame_cache.pop(0)

    # ---- per-utterance materialization -----------------------------------

    def _load_next_utterance(self, idx: int) -> None:
        """STFT + state-prop loop for one calibration utterance.

        Builds a list of per-frame feed dicts. The state for frame t is the
        state BEFORE that frame's computation — i.e., the per-block FIFO
        buffer accumulated from frames 0..t-1. Calibrator sees the activation
        distribution at the frame's input, not after it.
        """
        item = self.hf["train"][idx]
        audio_np = np.asarray(item["noisy"]["array"], dtype=np.float32)   # (N,)
        audio = torch.from_numpy(audio_np).unsqueeze(0)                   # (1, N)

        # Plain magnitude — compress_factor pinned to 1.0 here regardless of
        # the model's extractor. The ONNX `noisy_mag` input is always plain
        # |stft|; a 'mag_compressed' model applies its own Pow inside the
        # graph. Passing h.compress_factor here would double-compress.
        mag, _, _ = mag_pha_stft(
            audio,
            self.h.n_fft,
            self.h.hop_size,
            self.h.win_size,
            1.0,
        )                                                                 # (1, F, T)
        T = mag.shape[2]
        if self.frames_per_utterance is not None:
            T = min(T, int(self.frames_per_utterance))

        states = self.onnx_view.init_states(1, "cpu", dtype=torch.float32)
        state_in_names = self.onnx_view.state_input_names

        with torch.no_grad():
            for t in range(T):
                frame = mag[:, :, t].to(torch.float32)                    # (1, F)
                feed = {
                    "noisy_mag": frame.detach().cpu().contiguous().numpy().astype(np.float32),
                }
                for name, s in zip(state_in_names, states):
                    feed[name] = s.detach().cpu().contiguous().numpy().astype(np.float32)
                self._frame_cache.append(feed)
                # Advance state AFTER caching — the frame is calibrated at
                # the activations it sees on the way IN.
                _, states = self.fast.forward_mask_step(frame, states)

    # ---- audio hashing + disjoint check ----------------------------------

    @staticmethod
    def _hash_audio(arr: np.ndarray) -> str:
        # Little-endian f4 cast pins byte order — same as calibration.py:_hash_audio
        # so train/test hashes are interchangeable between the two readers.
        return hashlib.sha256(arr.astype("<f4", copy=False).tobytes()).hexdigest()

    def _hash_at(self, split, idx: int) -> str:
        return self._hash_audio(np.asarray(split[int(idx)]["noisy"]["array"], dtype=np.float32))

    def _assert_train_test_disjoint(self) -> None:
        """Calibration set must be disjoint from the evaluation set.

        Hashes only the calibration subset (cheap) and ALL of the test split
        (paid once per construction; ~3k items for VBD). Skipping the on-disk
        cache that VBDCalibrationReader maintains — calibration runs are
        infrequent and the cost is acceptable.
        """
        test_split = self.hf["test"]
        calib_hashes = {self._hash_at(self.hf["train"], i) for i in self.calib_indices}
        test_hashes = {self._hash_at(test_split, i) for i in range(len(test_split))}
        assert calib_hashes.isdisjoint(test_hashes), (
            "Calibration set leakage detected: a calibration train utt has the "
            "same audio bytes as a test utt. Overlap is not allowed; calibration "
            "must be disjoint from the validation set."
        )

    # ---- deterministic content hash --------------------------------------

    def _compute_calibration_hash(self) -> str:
        """sha256 over a deterministic JSON serialization of the input tuple.

        Two runs with the same num_utterances / seed / frames_per_utterance /
        calib_indices / train_fingerprint produce the same hash — embed it in
        the int8 ONNX metadata so a future inference run can detect calibration
        drift.
        """
        train_fp = getattr(self.hf["train"], "_fingerprint", "no-fingerprint")
        payload = json.dumps(
            {
                "num_utterances": self.num_utterances,
                "seed": self.seed,
                "frames_per_utterance": self.frames_per_utterance,
                "calib_indices": sorted(int(i) for i in self.calib_indices),
                "train_fingerprint": train_fp,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class ConvFSENetWindowedCalibrationReader(ConvFSENetCalibrationReader):
    """ORT CalibrationDataReader for the stateless-windowed ConvFSENet ONNX.

    Yields per-window dicts with a single key matching the windowed graph:

        {"noisy_mag_window": (B=1, n_freq, L+T) float32}

    There is NO per-block state (the rework's whole point), so calibration is
    just a slide of fixed L+T windows over each utterance's plain |stft|
    magnitude. The left of the stream is zero-padded by L columns, mirroring
    the host ring buffer at start-up (and the offline causal left-padding), so
    the calibrator sees the same start-of-stream window distribution the
    deployed graph will. Windows slide with hop = T (matching how the host
    advances), capped by `frames_per_utterance` output frames.

    Reuses ConvFSENetCalibrationReader's seeded index sampling, train/test
    disjoint check and calibration_hash verbatim — only the per-utterance
    materialization and the ONNX feed key differ.
    """

    def __init__(
        self,
        onnx_view: ConvFSENetWindowedONNX,
        h: AttrDict,
        hf_dataset,
    ):
        # Bypass the parent __init__ (it wants fast/onnx_view streaming
        # wrappers); reuse its sampling + pitfall closures directly.
        self.onnx_view = onnx_view
        self.h = h
        self.hf = hf_dataset
        self.L = int(onnx_view.L)
        self.T = int(onnx_view.T)
        self.drop_nyquist = bool(onnx_view.drop_nyquist)

        cal_cfg = getattr(h, "calibration", AttrDict({}))
        self.num_utterances = int(getattr(cal_cfg, "num_utterances", 200))
        self.seed = int(getattr(cal_cfg, "seed", getattr(h, "seed", 0)))
        self.frames_per_utterance = getattr(cal_cfg, "frames_per_utterance", None)

        n_train = len(hf_dataset["train"])
        if self.num_utterances > n_train:
            raise ValueError(
                f"num_utterances={self.num_utterances} exceeds train split size {n_train}"
            )
        rng = random.Random(self.seed)
        self.calib_indices = rng.sample(range(n_train), self.num_utterances)

        self._assert_train_test_disjoint()
        self.calibration_hash = self._compute_calibration_hash()
        print(
            f"ConvFSENetWindowedCalibrationReader: calib_hash={self.calibration_hash} "
            f"(utts={self.num_utterances}, seed={self.seed}, L={self.L}, T={self.T}, "
            f"drop_nyquist={self.drop_nyquist}, "
            f"frames_per_utterance={self.frames_per_utterance})"
        )

        self._utt_iter = iter(self.calib_indices)
        self._frame_cache: list[dict] = []

    def _load_next_utterance(self, idx: int) -> None:
        """Slide fixed L+T windows over one calibration utterance's magnitude."""
        item = self.hf["train"][idx]
        audio_np = np.asarray(item["noisy"]["array"], dtype=np.float32)
        audio = torch.from_numpy(audio_np).unsqueeze(0)                   # (1, N)

        # Plain |stft| — the graph applies its own compression Pow internally.
        mag, _, _ = mag_pha_stft(
            audio, self.h.n_fft, self.h.hop_size, self.h.win_size, 1.0,
        )                                                                 # (1, F, T_total)
        if self.drop_nyquist:
            mag = mag[:, : self.onnx_view.n_features, :]                  # drop Nyquist bin
        T_total = mag.shape[2]

        # Left-pad by L (start-of-stream zeros) so the first output frame's
        # window matches the host ring buffer at t=0.
        pad = torch.zeros(mag.shape[0], mag.shape[1], self.L, dtype=mag.dtype)
        mag_padded = torch.cat([pad, mag], dim=-1)                        # (1, F, L+T_total)

        # Full windows only, hop = T (how the host advances). A window emitting
        # output frames [k*T, k*T+T-1] spans mag_padded[:, :, k*T : k*T+L+T];
        # the last full window starts at T_total-T. Any partial tail is dropped
        # (calibration only needs representative distributions, not every frame).
        W = self.L + self.T
        max_start = (mag_padded.shape[-1] - W)                            # = T_total - T
        n_windows = max_start // self.T + 1 if max_start >= 0 else 0
        if self.frames_per_utterance is not None:
            n_windows = min(n_windows, int(self.frames_per_utterance))
        for k in range(n_windows):
            start = k * self.T
            window = mag_padded[:, :, start : start + W]
            self._frame_cache.append({
                "noisy_mag_window": window.detach().cpu().contiguous().numpy().astype(np.float32),
            })
