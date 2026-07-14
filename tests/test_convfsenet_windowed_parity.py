"""Parity + int8 gates for the stateless-windowed ConvFSENet view (Track 1).

Mirrors test_convfsenet_streaming_parity.py / test_convfsenet_quant.py, but for
ConvFSENetWindowedONNX:

  - FP32 windowed parity: the emitted T mask columns are bit-exact (< 1e-5) to
    the offline causal ConvFSENet_QuantFriendly mask on those frames — the
    receptive-field algebra (valid convs + cropped residuals) the rework rests
    on. Parameterized over extractor (mag / mag_compressed) and T (1 / 4).
  - Window geometry: L = sum_blocks (K-1)*D, window_length = L + T.
  - drop_nyquist: 256-wide graph, valid mask range, weight slicing is small-norm.
  - Guards: non-causal base and train-mode base both hard-fail.
  - int8 gate: export windowed FP32 -> quantize_windowed_fp32_onnx with a
    synthetic dataset; the int8 graph is checker-clean with NO Gather / Pad /
    state / BN / RNN, and the int8 mask cosine vs FP32 is >= 0.99.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

from common.env import AttrDict
from convfsenet.model import ConvFSENet_QuantFriendly
from convfsenet.streaming import ConvFSENetWindowedONNX
from convfsenet.export_onnx import export_windowed_fp32
from convfsenet.quant_windowed import quantize_windowed_fp32_onnx


# ---- fixtures ---------------------------------------------------------------


def _fresh_offline(causal=True, extractor_type="mag", res=128, conv=256):
    return ConvFSENet_QuantFriendly(
        n_fft=512, win_length=512, n_features=257,
        n_channels_res=res, n_channels_conv=conv, kernel_size=3,
        n_blocks=3, n_stacks=3,
        extractor_type=extractor_type,
        compress_factor=0.3 if extractor_type == "mag_compressed" else None,
        causal=causal, preproc=None, postproc=None,
    )


def _randomize_bn(m: nn.Module) -> None:
    """Replace identity-affine BN with non-trivial stats so a time-axis or
    fold bug surfaces in the parity diff (same discipline as the streaming test)."""
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm1d):
                nn.init.normal_(mod.weight, 1.0, 0.1)
                nn.init.normal_(mod.bias, 0.0, 0.1)
                mod.running_mean.normal_(0.0, 0.1)
                mod.running_var.uniform_(0.5, 1.5)
                mod.num_batches_tracked.fill_(1)


@pytest.fixture(params=["mag", "mag_compressed"])
def offline_model(request):
    torch.manual_seed(0)
    m = _fresh_offline(causal=True, extractor_type=request.param)
    _randomize_bn(m)
    m.eval()
    return m


# ---- synthetic HF-shaped dataset (self-contained) ---------------------------


class _Split:
    def __init__(self, n, sr, length_s, seed):
        rng = np.random.default_rng(seed)
        self._items = []
        for i in range(n):
            t = np.arange(int(sr * length_s)) / sr
            x = (0.05 * rng.standard_normal(len(t)).astype(np.float32)
                 + 0.10 * np.sin(2 * np.pi * (200 + 50 * i) * t).astype(np.float32)
                 + 0.05 * np.sin(2 * np.pi * (1200 + 80 * i) * t).astype(np.float32))
            self._items.append({
                "id": f"utt_{seed}_{i:04d}",
                "noisy": {"array": x.astype(np.float32)},
                "clean": {"array": (x * 0.5).astype(np.float32)},
            })
        self._fingerprint = f"synthetic_{seed}_{n}_{sr}_{length_s}"

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[int(i)]


class _SyntheticHFDataset:
    def __init__(self, n_train=4, n_test=2, sr=16000, length_s=0.6):
        self._train = _Split(n_train, sr, length_s, seed=1)
        self._test = _Split(n_test, sr, length_s, seed=2)

    def __getitem__(self, key):
        return {"train": self._train, "test": self._test}[key]


def _make_session(onnx_path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CPUExecutionProvider"])


# ---- geometry --------------------------------------------------------------


def test_window_geometry(offline_model):
    """L = sum_blocks (K-1)*D = 3 stacks * (1+2+4) * 2 = 42; RF = L+1 = 43."""
    for T in (1, 4):
        w = ConvFSENetWindowedONNX(offline_model, T=T).eval()
        assert w.L == 42, f"L must be 42 (got {w.L})"
        assert w.window_length == 42 + T
        assert w.n_features == 257


# ---- the parity gate -------------------------------------------------------


@pytest.mark.parametrize("T", [1, 4])
def test_windowed_parity_vs_offline_causal(offline_model, T):
    """Emitted T mask columns are bit-exact (< 1e-5) to the offline causal
    model's last T mask columns, given full real left context."""
    torch.manual_seed(0)
    w = ConvFSENetWindowedONNX(offline_model, T=T).eval()
    B, F = 2, 257
    T_total = w.L + T + 8                      # comfortably longer than the window
    stft = torch.randn(B, F, T_total, dtype=torch.complex64)
    with torch.no_grad():
        feats = offline_model.features_extractor(stft)
        x = offline_model.frontend(feats)
        x = offline_model.tcm(x)
        offline_mask = offline_model.backend(x)            # [B, 257, T_total]
        window = stft.abs()[..., T_total - (w.L + T):]     # [B, 257, L+T]
        wmask = w.forward_mask_window(window)              # [B, 257, T]
        ref = offline_mask[..., T_total - T:]
    err = (wmask - ref).abs().max().item()
    assert err < 1e-5, (
        f"Windowed parity FAILED: max_abs_err = {err:.3e} (expected < 1e-5). "
        f"Bucket: 1e-4 suspect residual crop alignment (valid output[j] aligns "
        f"to causal position j+trim, so residual must be x[..., trim:]); "
        f"1e-3 suspect BN fold / eval mode; "
        f"any: suspect trim accumulation (sum of (K-1)*D per block must equal L)."
    )


def test_windowed_coldstart_transient_decays(offline_model):
    """The ring-buffer cold start is a *bounded, decaying transient*, NOT bit-exact.

    Key semantic difference from the FIFO streaming model: the windowed host
    zero-pads the INPUT MAGNITUDE for the missing past, whereas the offline /
    FIFO-streaming model zero-pads the internal block ACTIVATIONS (and the
    frontend bias is non-zero, so frontend(0) != 0). The two cold starts
    therefore differ for frames whose receptive field still reaches into the
    zero-padded prefix — the first < L frames — and the error decays to FP
    noise once the ring buffer fills with real data (frame >= L). This is the
    deployed behaviour: a brief start-of-stream transient over leading
    silence, then bit-exact. PESQ impact is measured on real audio, not here."""
    torch.manual_seed(1)
    w = ConvFSENetWindowedONNX(offline_model, T=1).eval()
    B, F = 1, 257
    T_total = w.L + 20
    stft = torch.randn(B, F, T_total, dtype=torch.complex64)
    with torch.no_grad():
        feats = offline_model.features_extractor(stft)
        x = offline_model.frontend(feats); x = offline_model.tcm(x)
        offline_mask = offline_model.backend(x)            # [B, 257, T_total]
        mag = stft.abs()
        pad = torch.zeros(B, F, w.L)
        mag_pad = torch.cat([pad, mag], dim=-1)
        errs = []
        for t in range(T_total):
            window = mag_pad[..., t:t + w.L + 1]
            m = w.forward_mask_window(window)
            errs.append((m[..., 0] - offline_mask[..., t]).abs().max().item())
    # Transient is bounded (mask is in (0,1), so error <= 1) and modest at worst.
    assert max(errs[: w.L]) < 0.1, f"cold-start transient too large: {max(errs[:w.L]):.3e}"
    # Bit-exact once the ring buffer is full of real data (frame >= L).
    assert max(errs[w.L:]) < 1e-5, (
        f"frames >= L must be bit-exact to offline (got {max(errs[w.L:]):.3e}); "
        f"suspect residual-crop alignment or trim accumulation."
    )
    # Error decays: the tail-half of the transient is smaller than the head.
    head = max(errs[: w.L // 2]) if w.L >= 2 else errs[0]
    tail = max(errs[w.L // 2 : w.L])
    assert tail <= head, f"cold-start error must decay (head={head:.3e}, tail={tail:.3e})"


# ---- 256-bin drop_nyquist --------------------------------------------------


def test_drop_nyquist_shape_and_range(offline_model):
    w = ConvFSENetWindowedONNX(offline_model, T=1, drop_nyquist=True).eval()
    assert w.n_features == 256
    window = torch.rand(2, 256, w.window_length)
    with torch.no_grad():
        mask = w.forward_mask_window(window)
    assert mask.shape == (2, 256, 1)
    assert torch.all((mask > 0) & (mask < 1)), "sigmoid mask must be in (0,1)"


# ---- guards ----------------------------------------------------------------


def test_non_causal_base_rejected():
    base = _fresh_offline(causal=False).eval()
    with pytest.raises(ValueError, match="causal=True"):
        ConvFSENetWindowedONNX(base)


def test_train_mode_base_rejected():
    base = _fresh_offline(causal=True)
    base.train()
    with pytest.raises(ValueError, match="eval"):
        ConvFSENetWindowedONNX(base)


# ---- int8 gate -------------------------------------------------------------


# ---- host-runner stitch (T>1 tail-window placement) ------------------------


class _TorchSession:
    """Fake ORT session: runs the torch windowed model over a [n,F,W] batch."""
    def __init__(self, win):
        self.win = win

    def run(self, _names, feeds):
        x = torch.from_numpy(feeds["noisy_mag_window"])
        with torch.no_grad():
            out = self.win.forward_mask_window(x)          # [n, F, T]
        return [out.numpy()]


@pytest.mark.parametrize("T", [2, 3, 4])
@pytest.mark.parametrize("rem", [0, 1, 2])
def test_host_stitch_tail_placement(offline_model, T, rem):
    """_run_windowed must place the clamped tail window's T columns at the true
    trailing frames when T_total % T != 0 (the blind reshape corrupted them)."""
    from convfsenet.inference_windowed_onnx import _run_windowed
    w = ConvFSENetWindowedONNX(offline_model, T=T).eval()
    torch.manual_seed(3)
    T_total = w.L + 10 + rem
    stft = torch.randn(1, 257, T_total, dtype=torch.complex64)
    with torch.no_grad():
        feats = offline_model.features_extractor(stft)
        x = offline_model.frontend(feats); x = offline_model.tcm(x)
        offline_mask = offline_model.backend(x)[0].numpy()          # [257, T_total]
    mag = stft.abs()[0].numpy()                                     # [257, T_total]
    sess = _TorchSession(w)
    # Use coldstart='zero' so frames >= L are bit-exact to offline (isolates the
    # stitch correctness from the cold-start transient).
    mask, _ = _run_windowed(sess, mag, w.L, T, 257, drop_nyquist=False, coldstart="zero")
    assert mask.shape == (257, T_total)
    # Frames with full real context must match offline exactly — especially the
    # trailing T_total%T frames the tail window covers.
    err_tail = np.abs(mask[:, w.L:] - offline_mask[:, w.L:]).max()
    assert err_tail < 1e-5, (
        f"T={T}, T_total%T={T_total % T}: trailing frames misplaced "
        f"(max_err={err_tail:.3e}); tail-window emit offset must be T_total-T."
    )


@pytest.mark.parametrize("drop_nyquist", [False, True])
def test_windowed_int8_clean_and_close(offline_model, tmp_path, drop_nyquist):
    """End-to-end FP32 -> int8: graph is checker-clean with no FIFO/state/Pad
    class, and int8 mask cosine vs FP32 >= 0.99 on random windows."""
    T = 1
    bins = 256 if drop_nyquist else 257
    fp32_path = tmp_path / f"win{bins}_fp32.onnx"
    int8_path = tmp_path / f"win{bins}_int8.onnx"
    export_windowed_fp32(offline_model, fp32_path, T=T, drop_nyquist=drop_nyquist)

    onnx_view = ConvFSENetWindowedONNX(offline_model, T=T, drop_nyquist=drop_nyquist).eval()
    h = AttrDict({
        "n_fft": 512, "hop_size": 256, "win_size": 512, "compress_factor": 1.0, "seed": 0,
        "calibration": AttrDict({"num_utterances": 3, "seed": 0, "frames_per_utterance": 16}),
    })
    hf = _SyntheticHFDataset(n_train=4, n_test=2)
    quantize_windowed_fp32_onnx(fp32_path, int8_path, onnx_view, h, hf)

    # Graph audit.
    model = onnx.load(str(int8_path))
    assert onnx.checker.check_model(model) is None
    ops = [n.op_type for n in model.graph.node]
    assert "QuantizeLinear" in ops
    assert ops.count("Gather") == 0
    assert ops.count("Pad") == 0
    assert ops.count("BatchNormalization") == 0
    assert not any(t in ("GRU", "LSTM", "RNN") for t in ops)
    assert sum(1 for n in model.graph.node if "state" in n.name.lower()) == 0

    # int8 vs FP32 cosine on random windows.
    W = onnx_view.window_length
    rng = np.random.default_rng(0)
    x = np.abs(rng.standard_normal((8, onnx_view.n_features, W)).astype(np.float32))
    fp32_sess = _make_session(fp32_path)
    int8_sess = _make_session(int8_path)
    m_fp32 = fp32_sess.run(["mask"], {"noisy_mag_window": x})[0].reshape(-1)
    m_int8 = int8_sess.run(["mask"], {"noisy_mag_window": x})[0].reshape(-1)
    cos = float(np.dot(m_fp32, m_int8) / (np.linalg.norm(m_fp32) * np.linalg.norm(m_int8) + 1e-12))
    assert cos >= 0.99, f"int8 mask cosine vs FP32 = {cos:.4f} (expected >= 0.99)"
