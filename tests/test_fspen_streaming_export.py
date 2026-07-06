"""Streaming state-I/O ONNX export for FSPEN — the real-time deploy gate.

Mirrors the other families' streaming-export tests: the frame-by-frame graph
exposes each DPE block's stacked inter-GRU hidden state as explicit
``state_i_in``/``state_i_out`` tensors (so the runtime carries state across
frames instead of resetting), and ONNX Runtime driven in a frame loop
reproduces the offline whole-utterance enhancement.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from fspen.model import build_fspen
from fspen.streaming import FSPENStreamingONNX
from fspen.export_onnx import export_streaming_fp32


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_fspen().eval()


@pytest.fixture
def onnx_path(model, tmp_path):
    return export_streaming_fp32(model, tmp_path / "fspen_stream.onnx")


def _session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def _offline(model, spec):
    amp = torch.sqrt(spec[:, :, 0:1] ** 2 + spec[:, :, 1:2] ** 2)
    with torch.no_grad():
        est, _ = model.enhance_spectrum(spec, amp)
    return est


def test_export_smoke_and_io_names(model, onnx_path):
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    ops = {n.op_type for n in m.graph.node}
    assert "GRU" in ops, "DPE GRUs missing from the streaming graph"
    view = FSPENStreamingONNX(model)
    assert [i.name for i in m.graph.input] == ["spec"] + view.state_input_names
    assert [o.name for o in m.graph.output] == ["est_spec"] + view.state_output_names


def test_state_shapes(model, onnx_path):
    """States are (groups, B*bands, width): static except the batch-scaled axis."""
    m = onnx.load(str(onnx_path))
    for inp in m.graph.input:
        if inp.name == "spec":
            continue
        dims = inp.type.tensor_type.shape.dim
        assert dims[0].dim_value == model.dpe_groups
        assert dims[1].dim_param, f"{inp.name} batch-scaled axis should be dynamic"
        assert dims[2].dim_value == model.dpe_state_width


@pytest.mark.parametrize("B", [1, 2])
def test_ort_frame_loop_matches_offline(model, onnx_path, B):
    view = FSPENStreamingONNX(model)
    sess = _session(onnx_path)
    out_names = ["est_spec"] + view.state_output_names

    T = 24
    g = torch.Generator().manual_seed(B + 1)
    spec = torch.randn(B, T, 2, model.n_freqs, generator=g)
    off = _offline(model, spec).numpy()

    states = [s.numpy() for s in view.init_states(B)]
    outs = []
    for t in range(T):
        feed = {"spec": spec[:, t:t + 1].numpy()}
        feed.update(dict(zip(view.state_input_names, states)))
        res = sess.run(out_names, feed)
        outs.append(res[0])
        states = res[1:]
    s = np.concatenate(outs, axis=1)
    err = float(np.abs(off - s).max())
    assert err < 1e-4, f"streaming ORT frame loop diverged from offline: {err:.3e}"
