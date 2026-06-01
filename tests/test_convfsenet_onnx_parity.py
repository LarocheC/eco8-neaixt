"""Steps 5+6: FP32 ONNX export of ConvFSENetStreamingFast + parity gate.

Tests:
  - test_export_smoke      : export succeeds, ONNX checker passes, no opaque
                             RNN / BatchNormalization ops remain in the graph.
  - test_parity_pt_vs_onnx : ORT frame-by-frame agrees with PyTorch
                             ConvFSENetStreamingFast to max_abs_err < 1e-5.
  - test_parity_onnx_vs_offline : transitively, ORT agrees with the offline
                             whole-utterance forward to within 1e-5.

The ORT session uses the 4-knob determinism config from inference_onnx.py
so reruns are byte-identical.

Empirically the gap is on the order of 1e-7 (NSNet2's equivalent FP32 parity
runs at 5.96e-8).
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

from convfsenet.model import ConvFSENet_QuantFriendly
from convfsenet.export_onnx import export_streaming_fp32
from convfsenet.streaming import (
    ConvFSENetStreamingFast,
    ConvFSENetStreamingONNX,
)


# ---- fixtures (shared shape with test_convfsenet_streaming_parity.py) ------


def _fresh_offline(causal: bool = True, extractor_type: str = "mag") -> ConvFSENet_QuantFriendly:
    return ConvFSENet_QuantFriendly(
        n_fft=512, win_length=512, n_features=257,
        n_channels_res=128, n_channels_conv=256, kernel_size=3,
        n_blocks=3, n_stacks=3,
        extractor_type=extractor_type,
        compress_factor=0.3 if extractor_type == "mag_compressed" else None,
        causal=causal,
        loss=None, preproc=None, postproc=None,
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
    m = _fresh_offline(causal=True)
    _randomize_bn(m)
    m.eval()
    return m


@pytest.fixture
def onnx_path(offline_model, tmp_path):
    out = tmp_path / "convfsenet_fp32.onnx"
    export_streaming_fp32(offline_model, out)
    return out


def _make_session(onnx_path):
    """Determinism recipe from inference_onnx.py:42-56."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )


def test_mag_compressed_export_and_parity(tmp_path):
    """A mag_compressed model exports with the magnitude-compression Pow op
    baked into the graph (FP32, ahead of the quantized convs) and the ORT
    session matches PyTorch frame-by-frame."""
    torch.manual_seed(0)
    m = _fresh_offline(causal=True, extractor_type="mag_compressed")
    _randomize_bn(m)
    m.eval()

    out = tmp_path / "convfsenet_mc_fp32.onnx"
    export_streaming_fp32(m, out)
    graph = onnx.load(str(out))
    op_types = [n.op_type for n in graph.graph.node]
    assert "Pow" in op_types, (
        "mag_compressed export must contain a Pow op — the magnitude "
        "compression should be baked into the ONNX graph."
    )

    fast = ConvFSENetStreamingFast(m).eval()
    onnx_view = ConvFSENetStreamingONNX(fast)
    sess = _make_session(out)

    torch.manual_seed(1)
    B, F, T = 1, 257, 24
    stft = torch.randn(B, F, T, dtype=torch.complex64)
    noisy_mag = stft.abs().to(torch.float32)

    with torch.no_grad():
        states_pt = onnx_view.init_states(B, "cpu")
        pt_masks = []
        for t in range(T):
            outs = onnx_view(noisy_mag[..., t], *states_pt)
            pt_masks.append(outs[0])
            states_pt = list(outs[1:])
        pt_mask = torch.stack(pt_masks, dim=-1).numpy()

    states_np = [s.numpy() for s in onnx_view.init_states(B, "cpu")]
    ort_masks = []
    for t in range(T):
        feeds = {"noisy_mag": noisy_mag[..., t].numpy()}
        for name, s in zip(onnx_view.state_input_names, states_np):
            feeds[name] = s
        outs = sess.run(["mask", *onnx_view.state_output_names], feeds)
        ort_masks.append(outs[0])
        states_np = outs[1:]
    ort_mask = np.stack(ort_masks, axis=-1)

    err = float(np.abs(pt_mask - ort_mask).max())
    assert err < 1e-5, (
        f"mag_compressed ONNX vs PyTorch parity FAILED: max_abs_err={err:.3e}."
    )


# ---- step 5: export smoke --------------------------------------------------


def test_export_smoke(onnx_path, offline_model):
    """Export produces a checker-clean ONNX with the expected graph
    structure: no opaque RNN, no BatchNormalization (folded into Conv)."""
    assert onnx_path.exists(), "export did not create the output file"
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    op_types = {n.op_type for n in model.graph.node}
    assert "BatchNormalization" not in op_types, (
        "BatchNormalization survived export — BN fold did not happen. "
        "Check _fold_bn_into_conv() and ConvFSENetStreamingFast construction."
    )
    for opaque in ("GRU", "LSTM", "RNN"):
        assert opaque not in op_types, (
            f"{opaque} op in graph — ConvFSENet has no recurrent layers; "
            f"any RNN node is a tracing bug."
        )

    # IO names match what export_streaming_fp32 declared.
    input_names = [i.name for i in model.graph.input]
    output_names = [o.name for o in model.graph.output]
    fast = ConvFSENetStreamingFast(offline_model).eval()
    onnx_view = ConvFSENetStreamingONNX(fast)
    expected_inputs = ["noisy_mag"] + onnx_view.state_input_names
    expected_outputs = ["mask"] + onnx_view.state_output_names
    assert input_names == expected_inputs, (
        f"input names mismatch: {input_names} vs {expected_inputs}"
    )
    assert output_names == expected_outputs, (
        f"output names mismatch: {output_names} vs {expected_outputs}"
    )


def test_session_runs_one_frame(onnx_path, offline_model):
    """ORT loads the graph and runs one frame end-to-end."""
    fast = ConvFSENetStreamingFast(offline_model).eval()
    onnx_view = ConvFSENetStreamingONNX(fast)
    sess = _make_session(onnx_path)
    B = 1
    feeds = {"noisy_mag": np.zeros((B, onnx_view.n_features), dtype=np.float32)}
    for name, blk in zip(onnx_view.state_input_names, fast.tcm_blocks):
        feeds[name] = np.zeros(
            (B, blk.n_channels_conv, blk.buffer_size), dtype=np.float32,
        )
    outs = sess.run(None, feeds)
    assert len(outs) == 1 + onnx_view.n_blocks, "expected mask + N state tensors"
    assert outs[0].shape == (B, onnx_view.n_features)
    assert outs[0].dtype == np.float32


# ---- step 6: ORT vs PyTorch parity -----------------------------------------


def test_parity_pt_vs_onnx(offline_model, onnx_path):
    """ORT frame-by-frame agrees with PyTorch fast streaming.

    Drives both via the same ONNX wrapper signature (real magnitude in,
    mask out) so the comparison is op-by-op. Tolerance 1e-5; expect ~1e-7
    in practice (NSNet2's equivalent runs at 5.96e-8).
    """
    torch.manual_seed(0)
    B, F, T = 1, 257, 30
    stft = torch.randn(B, F, T, dtype=torch.complex64)
    noisy_mag = stft.abs().to(torch.float32)        # [B, F, T] real

    # --- PyTorch reference ---
    fast = ConvFSENetStreamingFast(offline_model).eval()
    onnx_view = ConvFSENetStreamingONNX(fast).eval()
    with torch.no_grad():
        states_pt = onnx_view.init_states(B, "cpu", dtype=torch.float32)
        pt_masks = []
        for t in range(T):
            outs = onnx_view(noisy_mag[..., t], *states_pt)
            pt_masks.append(outs[0])
            states_pt = list(outs[1:])
        pt_mask = torch.stack(pt_masks, dim=-1).numpy()     # [B, F, T]

    # --- ORT path ---
    sess = _make_session(onnx_path)
    state_in_names = onnx_view.state_input_names
    state_out_names = onnx_view.state_output_names
    states_np = [s.numpy() for s in onnx_view.init_states(B, "cpu")]
    ort_masks = []
    for t in range(T):
        feeds = {"noisy_mag": noisy_mag[..., t].numpy()}
        for name, s in zip(state_in_names, states_np):
            feeds[name] = s
        outs = sess.run(["mask", *state_out_names], feeds)
        ort_masks.append(outs[0])
        states_np = outs[1:]
    ort_mask = np.stack(ort_masks, axis=-1)                  # [B, F, T]

    err = float(np.abs(pt_mask - ort_mask).max())
    assert err < 1e-5, (
        f"ONNX FP32 vs PyTorch parity FAILED: max_abs_err = {err:.3e}. "
        f"Expected < 1e-5 (typical ~1e-7). Suspect: ORT op-fusion divergence "
        f"(disable graph optimization to localize), or trace-time op rewrite "
        f"that changed math (check torch.onnx warnings during export)."
    )


def test_parity_onnx_vs_offline(offline_model, onnx_path):
    """Transitive parity: ORT frame-by-frame agrees with offline whole-utterance.

    This is the user-facing gate — proves the exported ONNX produces the
    same mask trajectory the trained PyTorch model would, modulo float noise.
    """
    torch.manual_seed(0)
    B, F, T = 1, 257, 30
    stft = torch.randn(B, F, T, dtype=torch.complex64)

    # Offline mask: rebuild it from the offline complex output.
    # Avoid div-by-zero by adding a tiny eps where |stft_noisy| is ~0; the
    # offline model would output |stft_pred|≈0 in those cells too.
    with torch.no_grad():
        stft_pred = offline_model(stft.unsqueeze(1)).squeeze(1)    # [B, F, T] complex
    eps = 1e-12
    offline_mask = (stft_pred.abs() / (stft.abs() + eps)).numpy()  # [B, F, T] real

    # ORT mask trajectory (mirrors test_parity_pt_vs_onnx setup).
    fast = ConvFSENetStreamingFast(offline_model).eval()
    onnx_view = ConvFSENetStreamingONNX(fast)
    sess = _make_session(onnx_path)
    state_in_names = onnx_view.state_input_names
    state_out_names = onnx_view.state_output_names
    states_np = [s.numpy() for s in onnx_view.init_states(B, "cpu")]
    noisy_mag = stft.abs().to(torch.float32)
    ort_masks = []
    for t in range(T):
        feeds = {"noisy_mag": noisy_mag[..., t].numpy()}
        for name, s in zip(state_in_names, states_np):
            feeds[name] = s
        outs = sess.run(["mask", *state_out_names], feeds)
        ort_masks.append(outs[0])
        states_np = outs[1:]
    ort_mask = np.stack(ort_masks, axis=-1)

    err = float(np.abs(offline_mask - ort_mask).max())
    assert err < 1e-5, (
        f"ONNX FP32 vs offline parity FAILED: max_abs_err = {err:.3e}. "
        f"Expected < 1e-5. Streaming parity already passed at this tolerance "
        f"(see test_convfsenet_streaming_parity.py) so a regression here is "
        f"specifically an export-time issue, not a streaming-math one."
    )
