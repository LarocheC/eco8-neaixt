"""Streaming state-I/O ONNX export for the NPU conv variant — the deploy gate.

Mirrors ConvFSENet's streaming-export test: the frame-by-frame graph exposes
every causal time-conv's FIFO buffer as explicit ``state_i_in``/``state_i_out``
tensors (so the board carries state across frames instead of resetting), and
ONNX Runtime driven in a frame loop reproduces the offline whole-utterance mask.
Also pins that the two Neural-ART blockers stay out of the exported graph.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from lisennet.model import build_lisennet
from lisennet.streaming import LiSenNetStreamingONNX
from lisennet.export_onnx import export_streaming_fp32


CONV = dict(num_channels=16, n_blocks=2, bottleneck="conv",
            intra_kernel=7, inter_kernel=3, inter_dilations=(1, 2, 4))


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_lisennet(CONV).eval()


@pytest.fixture
def onnx_path(model, tmp_path):
    return export_streaming_fp32(model, tmp_path / "lisennet_stream.onnx")


def _session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def _offline_est_mag(model, mag, pha):
    with torch.no_grad():
        return model.apply_mask(model.predict_mask(model.build_features(mag, pha)), mag)


def test_export_smoke_and_io_names(model, onnx_path):
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    ops = {n.op_type for n in m.graph.node}
    assert "GRU" not in ops and "LSTM" not in ops and "RNN" not in ops, "recurrent op survived export"
    assert "LayerNormalization" not in ops, "2-axis LayerNorm blocker survived export"
    assert "Conv" in ops
    view = LiSenNetStreamingONNX(model)
    assert [i.name for i in m.graph.input] == ["feat"] + view.state_input_names
    assert [o.name for o in m.graph.output] == ["est_mag"] + view.state_output_names


def test_state_shapes_static_batch_dynamic(model, onnx_path):
    """Every FIFO state has a static (C, 1, F) shape; only batch is symbolic."""
    m = onnx.load(str(onnx_path))
    for inp in m.graph.input:
        if inp.name == "feat":
            continue
        dims = inp.type.tensor_type.shape.dim
        assert dims[0].dim_param, f"{inp.name} batch axis should be dynamic"
        assert all(d.dim_value > 0 for d in dims[1:]), f"{inp.name} non-batch dims must be static"


@pytest.mark.parametrize("B", [1, 2])
def test_ort_frame_loop_matches_offline(model, onnx_path, B):
    view = LiSenNetStreamingONNX(model)
    sess = _session(onnx_path)
    out_names = ["est_mag"] + view.state_output_names

    T, F = 24, model.n_freqs
    g = torch.Generator().manual_seed(B + 1)
    mag = torch.rand(B, T, F, generator=g)
    pha = (torch.rand(B, T, F, generator=g) * 2 - 1) * torch.pi
    feat_full = model.build_features(mag, pha)
    off = _offline_est_mag(model, mag, pha).numpy()

    states = [s.numpy() for s in view.init_states(B)]
    outs = []
    for t in range(T):
        feed = {"feat": feat_full[:, :, t:t + 1, :].numpy()}
        feed.update(dict(zip(view.state_input_names, states)))
        res = sess.run(out_names, feed)
        outs.append(res[0])
        states = res[1:]
    s = np.concatenate(outs, axis=1)
    err = float(np.abs(off - s).max())
    assert err < 1e-4, f"streaming ORT frame loop diverged from offline: {err:.3e}"


def test_rnn_bottleneck_rejected():
    """The RNN bottleneck is not an NPU streaming target (GRU hidden state)."""
    rnn = build_lisennet({"num_channels": 16, "n_blocks": 2}).eval()
    with pytest.raises(ValueError, match="conv"):
        LiSenNetStreamingONNX(rnn)
