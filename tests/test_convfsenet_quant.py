"""Step 7: int8 QDQ quantization smoke + sanity tests for ConvFSENet streaming.

We avoid the HF VoiceBank-DEMAND download by injecting a synthetic dataset
that quacks like the HF DatasetDict — same .__getitem__ / .__len__ /
._fingerprint surface used by ConvFSENetCalibrationReader.

Tests:
  - test_calibration_reader_yields_frames : reader produces correctly-shaped
    feed dicts (mag + 9 named states), advances state across frames, stops
    cleanly on exhaustion.
  - test_calibration_reader_train_test_overlap_rejected : disjoint-set check
    fires when a calib utt has the same audio bytes as a test utt.
  - test_quantize_smoke_synthetic : end-to-end FP32 → int8 quant via
    quantize_fp32_onnx; resulting int8 ONNX is checker-clean, loads in ORT,
    has QuantizeLinear nodes, has no BN / RNN.
  - test_int8_mask_in_range : int8 session produces masks in (0, 1).
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

from convfsenet.calibration import ConvFSENetCalibrationReader
from convfsenet.model import ConvFSENet_QuantFriendly
from common.env import AttrDict
from convfsenet.export_onnx import export_streaming_fp32
from convfsenet.streaming import (
    ConvFSENetStreamingFast,
    ConvFSENetStreamingONNX,
)
from convfsenet.quant import quantize_fp32_onnx


# ---- synthetic HF-shaped dataset --------------------------------------------


class _Split:
    """Mimics an HF Dataset split: __len__, __getitem__ returning a dict,
    and a ._fingerprint attribute (used by the calibration hash)."""

    def __init__(self, n: int, sr: int, length_s: float, seed: int):
        rng = np.random.default_rng(seed)
        self._items = []
        for i in range(n):
            # Smooth noise + a couple of sines so the FFT magnitude is non-degenerate
            # and the calibration scales are meaningful.
            t = np.arange(int(sr * length_s)) / sr
            x = (
                0.05 * rng.standard_normal(len(t)).astype(np.float32)
                + 0.10 * np.sin(2 * np.pi * (200 + 50 * i) * t).astype(np.float32)
                + 0.05 * np.sin(2 * np.pi * (1200 + 80 * i) * t).astype(np.float32)
            )
            self._items.append({
                "id": f"utt_{seed}_{i:04d}",
                "noisy": {"array": x.astype(np.float32)},
                "clean": {"array": x.astype(np.float32) * 0.5},
            })
        self._fingerprint = f"synthetic_{seed}_{n}_{sr}_{length_s}"

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[int(i)]


class _SyntheticHFDataset:
    """Mimics HuggingFace DatasetDict shape: hf["train"], hf["test"]."""

    def __init__(self, n_train=4, n_test=2, sr=16000, length_s=0.5):
        # Different seeds so train ∩ test = ∅.
        self._train = _Split(n=n_train, sr=sr, length_s=length_s, seed=1)
        self._test = _Split(n=n_test, sr=sr, length_s=length_s, seed=2)

    def __getitem__(self, key):
        return {"train": self._train, "test": self._test}[key]


# ---- offline model fixture (shared with the other test files) --------------


def _fresh_offline() -> ConvFSENet_QuantFriendly:
    return ConvFSENet_QuantFriendly(
        n_fft=512, win_length=512, n_features=257,
        n_channels_res=128, n_channels_conv=256, kernel_size=3,
        n_blocks=3, n_stacks=3,
        extractor_type="mag", compress_factor=None, causal=True,
        preproc=None, postproc=None,
    )


def _randomize_bn(m: nn.Module) -> None:
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm1d):
                nn.init.normal_(mod.weight, 1.0, 0.1)
                nn.init.normal_(mod.bias, 0.0, 0.1)
                mod.running_mean.normal_(0.0, 0.1)
                mod.running_var.uniform_(0.5, 1.5)
                mod.num_batches_tracked.fill_(1)


@pytest.fixture
def offline_model() -> ConvFSENet_QuantFriendly:
    torch.manual_seed(0)
    m = _fresh_offline()
    _randomize_bn(m)
    m.eval()
    return m


@pytest.fixture
def fast_view(offline_model):
    return ConvFSENetStreamingFast(offline_model).eval()


@pytest.fixture
def onnx_view(fast_view):
    return ConvFSENetStreamingONNX(fast_view).eval()


@pytest.fixture
def hf_dataset():
    return _SyntheticHFDataset(n_train=4, n_test=2, sr=16000, length_s=0.5)


@pytest.fixture
def calib_h():
    return AttrDict({
        "n_fft": 512, "hop_size": 256, "win_size": 512, "compress_factor": 1.0,
        "seed": 0,
        "calibration": AttrDict({
            "num_utterances": 3,
            "seed": 0,
            "frames_per_utterance": 16,        # cap for fast tests
        }),
    })


@pytest.fixture
def fp32_onnx_path(offline_model, tmp_path):
    out = tmp_path / "g_best_fp32.onnx"
    export_streaming_fp32(offline_model, out)
    return out


# ---- calibration reader tests ---------------------------------------------


def test_calibration_reader_yields_frames(fast_view, onnx_view, calib_h, hf_dataset):
    reader = ConvFSENetCalibrationReader(fast_view, onnx_view, calib_h, hf_dataset)
    expected_keys = {"noisy_mag", *onnx_view.state_input_names}

    frames = []
    while True:
        feed = reader.get_next()
        if feed is None:
            break
        frames.append(feed)

    assert len(frames) > 0, "reader produced zero frames"
    # 3 calib utterances × 16 frames/utt cap = 48
    assert len(frames) == 3 * 16, f"expected 48 frames, got {len(frames)}"

    for feed in frames:
        assert set(feed.keys()) == expected_keys, (
            f"feed keys mismatch: {sorted(feed)} vs {sorted(expected_keys)}"
        )
        assert feed["noisy_mag"].shape == (1, onnx_view.n_features)
        assert feed["noisy_mag"].dtype == np.float32
        for i, name in enumerate(onnx_view.state_input_names):
            blk = fast_view.tcm_blocks[i]
            assert feed[name].shape == (1, blk.n_channels_conv, blk.buffer_size)
            assert feed[name].dtype == np.float32

    # First frame of an utterance has zero state; subsequent frames don't.
    first = frames[0]
    second = frames[1]
    assert np.all(first[onnx_view.state_input_names[0]] == 0.0), (
        "first frame of an utterance must have zero state (h0 reset)"
    )
    # The 0th block has buffer_size = (K-1)*D = 2 (D=1). After one step, the
    # state holds y_0, which is non-zero almost surely.
    assert not np.all(second[onnx_view.state_input_names[0]] == 0.0), (
        "second frame must see propagated state from frame 0"
    )


def test_calibration_reader_train_test_overlap_rejected(fast_view, onnx_view, calib_h):
    """Disjoint-set assertion fires when train ∩ test ≠ ∅."""
    overlapping = _SyntheticHFDataset(n_train=3, n_test=2)
    # Force a leak: copy a train item's audio into a test slot.
    overlapping._test._items[0]["noisy"]["array"] = (
        overlapping._train._items[0]["noisy"]["array"].copy()
    )
    with pytest.raises(AssertionError, match="leakage"):
        ConvFSENetCalibrationReader(fast_view, onnx_view, calib_h, overlapping)


def test_calibration_hash_deterministic(fast_view, onnx_view, calib_h, hf_dataset):
    """Two readers with identical inputs produce the same calibration_hash."""
    r1 = ConvFSENetCalibrationReader(fast_view, onnx_view, calib_h, hf_dataset)
    r2 = ConvFSENetCalibrationReader(fast_view, onnx_view, calib_h, hf_dataset)
    assert r1.calibration_hash == r2.calibration_hash


# ---- end-to-end quant smoke ------------------------------------------------


def _make_session(onnx_path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )


@pytest.fixture
def int8_onnx_path(
    fp32_onnx_path, fast_view, onnx_view, calib_h, hf_dataset, tmp_path
):
    int8_path = tmp_path / "g_best.onnx"
    quantize_fp32_onnx(
        fp32_onnx_path, int8_path,
        fast_view, onnx_view, calib_h, hf_dataset,
    )
    return int8_path


def test_quantize_smoke_synthetic(int8_onnx_path, fp32_onnx_path):
    """Quant pipeline ran end-to-end and produced a structurally sound int8 ONNX."""
    assert int8_onnx_path.exists()
    model = onnx.load(str(int8_onnx_path))
    onnx.checker.check_model(model)

    # Audits matching quant_convfsenet.py's post-quant checks.
    op_types = {n.op_type for n in model.graph.node}
    assert "BatchNormalization" not in op_types
    for opaque in ("GRU", "LSTM", "RNN"):
        assert opaque not in op_types
    op_hist = {n.op_type: 0 for n in model.graph.node}
    for n in model.graph.node:
        op_hist[n.op_type] = op_hist.get(n.op_type, 0) + 1
    assert op_hist.get("QuantizeLinear", 0) > 0, (
        "no QuantizeLinear in int8 graph — quant silently no-op'd"
    )

    # Metadata stamps survive the audit reload.
    meta = {p.key: p.value for p in model.metadata_props}
    for key in ("ort_quantize_version", "torch_version", "calibration_hash"):
        assert key in meta, f"missing metadata key {key!r}"
    assert meta["ort_quantize_version"] == ort.__version__
    assert meta["torch_version"] == torch.__version__


def test_int8_session_runs_and_mask_in_range(
    int8_onnx_path, fast_view, onnx_view
):
    """ORT loads the int8 ONNX, runs one frame, mask is in (0, 1)."""
    sess = _make_session(int8_onnx_path)
    B = 1
    feeds = {"noisy_mag": np.random.randn(B, onnx_view.n_features).astype(np.float32) * 0.5}
    for name, blk in zip(onnx_view.state_input_names, fast_view.tcm_blocks):
        feeds[name] = np.zeros(
            (B, blk.n_channels_conv, blk.buffer_size), dtype=np.float32
        )
    outs = sess.run(None, feeds)
    mask = outs[0]
    assert mask.shape == (B, onnx_view.n_features)
    # Sigmoid output: strict (0, 1). Allow tiny float slack at the edges.
    assert (mask >= 0.0).all() and (mask <= 1.0).all(), (
        f"int8 mask out of [0,1]: min={mask.min()}, max={mask.max()}"
    )


def test_int8_vs_fp32_within_sanity_bound(
    int8_onnx_path, fp32_onnx_path, fast_view, onnx_view
):
    """Int8 vs FP32 mask diff is bounded by sanity (< 0.5 mean abs diff).

    Not a parity test — quantization is lossy by design. We're just checking
    that int8 didn't catastrophically destroy the model (saturated to 0/1).
    With a random-init model + synthetic calib data the diff floor is loose;
    a real-trained model would land much tighter. NSNet2 monarch variants
    quantize loss-free (PESQ delta < 0.021).
    """
    fp32_sess = _make_session(fp32_onnx_path)
    int8_sess = _make_session(int8_onnx_path)
    B, F, T = 1, onnx_view.n_features, 20
    rng = np.random.default_rng(0)
    noisy_mag = np.abs(
        rng.standard_normal((B, F, T)).astype(np.float32)
        + 1j * rng.standard_normal((B, F, T)).astype(np.float32)
    ).astype(np.float32)

    def _run(sess):
        state_in = [
            np.zeros((B, blk.n_channels_conv, blk.buffer_size), dtype=np.float32)
            for blk in fast_view.tcm_blocks
        ]
        masks = []
        for t in range(T):
            feeds = {"noisy_mag": noisy_mag[..., t]}
            for name, s in zip(onnx_view.state_input_names, state_in):
                feeds[name] = s
            outs = sess.run(["mask", *onnx_view.state_output_names], feeds)
            masks.append(outs[0])
            state_in = outs[1:]
        return np.stack(masks, axis=-1)

    fp32_mask = _run(fp32_sess)
    int8_mask = _run(int8_sess)
    diff = np.abs(fp32_mask - int8_mask)
    print(f"  int8 vs fp32: max={diff.max():.3e}, mean={diff.mean():.3e}")
    assert diff.mean() < 0.5, (
        f"int8 mask is catastrophically different from fp32: "
        f"mean_abs_diff={diff.mean():.3f}"
    )
