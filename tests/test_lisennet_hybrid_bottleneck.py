"""Tests for the HYBRID bottleneck (bottleneck="hybrid") — conv over frequency,
GRU over time.

The hybrid exists to answer one question the conv variant cannot: on the *time*
axis, is a dilated conv + FIFO the right call, or should the recurrence have
stayed? It is a single-variable swap against ``bottleneck="conv"`` — encoder,
decoder, norms, activations and the intra-frequency mixer are identical; only the
inter-time mixer changes (``_CausalTimeConv`` stack -> :class:`TimeGRU`). These
tests pin what makes that comparison meaningful, and what makes the recurrent
variant deployable at all:

  * the single-step GRU **cell** (1x1 convs over an explicit hidden state, the form
    that ships) is numerically ``nn.GRU`` (the fused kernel that trains) — the
    whole export story rests on this;
  * it streams frame-by-frame with parity to the offline whole-utterance model;
  * it is causal in time;
  * the exported streaming graph contains **no ``GRU`` op** and no ``LayerNorm``
    (both Neural-ART blockers) — recurrence reaches the NPU as Conv/Sigmoid/Tanh;
  * it carries *less* streaming state than the conv variant it replaces (a
    compressed hidden state, not an activation history), which is the trade;
  * the stateless windowed export is refused (unbounded lookback has no window).
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn

from lisennet.model import DualPathConv, DualPathHybrid, DualPathRNN, TimeGRU, build_lisennet
from lisennet.streaming import (
    LiSenNetStreamingONNX, disable_streaming, enable_streaming, stream_est_mag,
)
from lisennet.export_onnx import export_streaming_fp32, export_windowed_fp32


# The two variants under comparison. Identical but for the inter-time mixer; the
# hidden width (18) is chosen so the parameter counts match to <0.3%, so any quality
# difference is attributable to recurrence-vs-FIFO and not to capacity.
HARDENED = dict(num_channels=24, n_blocks=2, norm="batchnorm", act="relu",
                upsample="convtranspose", intra_kernel=11, inter_kernel=3,
                inter_dilations=(1, 2, 4, 8))
HYBRID = dict(HARDENED, bottleneck="hybrid", hidden_dim=18)
CONV = dict(HARDENED, bottleneck="conv")


def _spec(b=1, n=40, seed=0, f=257):
    g = torch.Generator().manual_seed(seed)
    mag = torch.rand(b, n, f, generator=g)
    pha = (torch.rand(b, n, f, generator=g) * 2 - 1) * torch.pi
    return mag, pha


def _offline_est_mag(model, mag, pha):
    with torch.no_grad():
        return model.apply_mask(model.predict_mask(model.build_features(mag, pha)), mag)


def _session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


# --- the load-bearing equivalence: the cell that ships == the kernel that trains ---


def test_gru_cell_matches_nn_gru():
    """The streaming 1x1-conv cell must reproduce ``nn.GRU`` step for step.

    Training uses the fused ``nn.GRU`` sequence kernel; deployment unrolls the same
    weights into convolutions. If these ever diverge, the board runs a different model
    than the one that was trained — so this is pinned tightly (fp32 round-off only).
    """
    torch.manual_seed(0)
    b, d, t, f, H = 2, 24, 12, 32, 18
    gru = TimeGRU(d, H).eval()
    x = torch.randn(b, d, t, f)

    with torch.no_grad():
        ref = gru(x)                                          # offline: nn.GRU over (b*f, t, d)
        enable_streaming(gru)                                 # not a LiSenNet, but the flag is the same
        gru.streaming, gru._h = True, None
        step = torch.cat([gru(x[:, :, i:i + 1, :]) for i in range(t)], dim=2)

    err = (ref - step).abs().max().item()
    assert err < 1e-5, f"GRU cell diverged from nn.GRU: {err:.3e}"


def test_gru_hidden_state_shape_and_reset():
    torch.manual_seed(0)
    gru = TimeGRU(24, 18).eval()
    gru.streaming = True
    gru.stream_reset()
    with torch.no_grad():
        gru(torch.randn(2, 24, 1, 32))
    assert gru._h.shape == (2, 18, 1, 32), "hidden state must be (B, H, 1, F)"
    gru.stream_reset()
    assert gru._h is None


def test_streaming_expects_single_frame():
    gru = TimeGRU(24, 18).eval()
    gru.streaming = True
    with pytest.raises(AssertionError):
        gru(torch.randn(1, 24, 4, 32))                        # t > 1 in streaming mode


# --- model-level: builds, streams, stays causal ---


def test_hybrid_builds_and_forwards():
    torch.manual_seed(0)
    model = build_lisennet(HYBRID).eval()
    assert model.bottleneck == "hybrid"
    assert any(isinstance(m, DualPathHybrid) for m in model.modules())
    assert not any(isinstance(m, (DualPathRNN, DualPathConv)) for m in model.modules())
    x = torch.randn(2, 32000)
    with torch.no_grad():
        r = model(x, x)
    assert r["est"].shape == (2, 32000)
    assert torch.isfinite(r["est"]).all()
    assert (r["est_mag"] >= 0).all()


def test_hybrid_param_matched_to_conv():
    """The comparison is only clean if capacity is held fixed — pin the match."""
    n_hyb = sum(p.numel() for p in build_lisennet(HYBRID).parameters())
    n_conv = sum(p.numel() for p in build_lisennet(CONV).parameters())
    rel = abs(n_hyb - n_conv) / n_conv
    assert rel < 0.01, f"hybrid {n_hyb:,} vs conv {n_conv:,} — {rel:.1%} apart, not a controlled swap"


def test_hybrid_keeps_the_conv_intra_path():
    """Only the *time* axis changes: the intra-frequency mixer stays a conv."""
    model = build_lisennet(HYBRID).eval()
    blk = model.blocks[0].dp_rnn_attn
    assert isinstance(blk.intra_dw, nn.Conv2d) and isinstance(blk.intra_pw, nn.Conv2d)
    assert isinstance(blk.inter, TimeGRU)
    # ...and the recurrence is unidirectional (a bidirectional GRU could not stream).
    assert not blk.inter.rnn.bidirectional


def test_hybrid_has_no_layernorm_module():
    """The 2-axis nn.LayerNorm blocker must be absent (batchnorm recipe)."""
    model = build_lisennet(HYBRID).eval()
    assert not any(isinstance(m, nn.LayerNorm) for m in model.modules())


def test_hybrid_is_causal():
    torch.manual_seed(0)
    model = build_lisennet(HYBRID).eval()
    mag, pha = _spec(1, 40, seed=1)
    off = _offline_est_mag(model, mag, pha)

    t0 = 20
    mag2, pha2 = mag.clone(), pha.clone()
    mag2[:, t0:] += 5.0
    pha2[:, t0:] = (torch.rand(1, mag.shape[1] - t0, mag.shape[2]) * 2 - 1) * torch.pi
    off2 = _offline_est_mag(model, mag2, pha2)

    delta = (off[:, :t0] - off2[:, :t0]).abs().max().item()
    assert delta == 0.0, f"future frame leaked into past outputs: max|delta|={delta:.2e}"


@pytest.mark.parametrize("b", [1, 3])
def test_hybrid_streaming_matches_offline(b):
    """Frame-by-frame (GRU hidden state + conv FIFOs) == offline whole-utterance."""
    torch.manual_seed(0)
    model = build_lisennet(HYBRID).eval()
    mag, pha = _spec(b, 30, seed=2)
    off = _offline_est_mag(model, mag, pha)
    s = stream_est_mag(model, mag, pha)
    err = (off - s).abs().max().item()
    assert err < 1e-4, f"hybrid streaming diverged from offline: {err:.3e}"


def test_toggling_streaming_leaves_offline_unchanged():
    torch.manual_seed(0)
    model = build_lisennet(HYBRID).eval()
    mag, pha = _spec(1, 20, seed=4)
    before = _offline_est_mag(model, mag, pha)
    enable_streaming(model)
    disable_streaming(model)
    after = _offline_est_mag(model, mag, pha)
    assert torch.allclose(before, after)


# --- deploy: the exported streaming graph ---


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_lisennet(HYBRID).eval()


@pytest.fixture
def onnx_path(model, tmp_path):
    return export_streaming_fp32(model, tmp_path / "lisennet_hybrid_stream.onnx")


def test_export_has_no_recurrent_op(model, onnx_path):
    """Recurrence must reach the NPU as convolutions, not as an ONNX ``GRU`` node.

    This is the claim that makes a recurrent bottleneck deployable on Neural-ART at
    all: the traced cell lowers to Conv/Add/Sigmoid/Tanh/Mul on 4-D tensors, with no
    GRU op and none of the reshapes the compiler rejects.
    """
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    ops = {n.op_type for n in m.graph.node}
    assert not ({"GRU", "LSTM", "RNN"} & ops), "a recurrent op survived export"
    assert "LayerNormalization" not in ops
    assert {"Conv", "Sigmoid", "Tanh"} <= ops, "the unrolled GRU cell is missing from the graph"


def test_export_state_slots_include_gru_hidden(model, onnx_path):
    """One (B, H, 1, F) hidden state per DPR block, alongside the conv FIFOs."""
    view = LiSenNetStreamingONNX(model)
    states = view.init_states(1)
    hidden = [tuple(s.shape) for s in states if tuple(s.shape) == (1, 18, 1, 32)]
    assert len(hidden) == HYBRID["n_blocks"], "expected one GRU hidden state per block"

    m = onnx.load(str(onnx_path))
    assert [i.name for i in m.graph.input] == ["feat"] + view.state_input_names
    assert [o.name for o in m.graph.output] == ["est_mag"] + view.state_output_names
    for inp in m.graph.input:                                  # static state shapes; batch dynamic
        if inp.name == "feat":
            continue
        dims = inp.type.tensor_type.shape.dim
        assert dims[0].dim_param
        assert all(d.dim_value > 0 for d in dims[1:])


def test_hybrid_carries_less_state_than_conv():
    """The trade this variant measures: recurrence compresses the past.

    The conv bottleneck holds a *receptive field* of activations (RF x C x F floats
    across many FIFOs); the GRU holds one H x F hidden state per block and its lookback
    is unbounded. At matched parameters the recurrent model must therefore stream with
    strictly less state and fewer state tensors — which is also what unloads the
    Slice/Concat state plumbing that dominates frame cost on the NPU.
    """
    def footprint(cfg):
        view = LiSenNetStreamingONNX(build_lisennet(cfg).eval())
        states = view.init_states(1)
        return len(states), sum(int(np.prod(s.shape)) for s in states)

    n_hyb, fl_hyb = footprint(HYBRID)
    n_conv, fl_conv = footprint(CONV)
    assert n_hyb < n_conv, f"hybrid should need fewer state tensors ({n_hyb} vs {n_conv})"
    assert fl_hyb < fl_conv / 2, f"hybrid state {fl_hyb:,} floats vs conv {fl_conv:,} — expected <half"


@pytest.mark.parametrize("B", [1, 2])
def test_ort_frame_loop_matches_offline(model, onnx_path, B):
    """The exported graph, driven frame-by-frame in ORT with threaded state, is the
    offline model — the end-to-end deploy contract."""
    sess = _session(onnx_path)

    T, F = 24, model.n_freqs
    mag, pha = _spec(B, T, seed=B + 7, f=F)
    feat_full = model.build_features(mag, pha)

    # The export fixture left the model in streaming mode; take the whole-utterance
    # reference from the offline path explicitly, then hand the model back to the
    # streaming view (which re-enables streaming and clears state).
    disable_streaming(model)
    off = _offline_est_mag(model, mag, pha).numpy()
    view = LiSenNetStreamingONNX(model)
    out_names = ["est_mag"] + view.state_output_names

    states = [s.numpy() for s in view.init_states(B)]
    outs = []
    for t in range(T):
        feed = {"feat": feat_full[:, :, t:t + 1, :].numpy()}
        feed.update(dict(zip(view.state_input_names, states)))
        res = sess.run(out_names, feed)
        outs.append(res[0])
        states = res[1:]
    err = float(np.abs(off - np.concatenate(outs, axis=1)).max())
    assert err < 1e-4, f"hybrid ORT frame loop diverged from offline: {err:.3e}"


def test_windowed_export_rejected_for_hybrid(model, tmp_path):
    """A GRU has no finite receptive field, so the stateless window is not defined.

    The conv variant's stateless windowed graph is bit-exact *because* its lookback is
    finite. Recurrence has to carry state — refusing the export is the honest answer,
    not silently shipping a truncated model.
    """
    with pytest.raises(ValueError, match="unbounded lookback"):
        export_windowed_fp32(model, tmp_path / "nope.onnx")
