"""Gates for N6Net-v2: causality, deploy-graph parity, and the compiler rules
the shape exists to follow (no time kernels, no Concat/Slice, ReLU only)."""

from __future__ import annotations

import io
import json

import pytest
import torch
from torch import nn

from n6net.model_v2 import N6NetV2, build_causal_model, cost_summary

F_BINS = 257


def _model(**over):
    h = dict(json.load(open("configs/n6net_v2.json")), **over)
    torch.manual_seed(0)
    return build_causal_model(h).eval()


def test_output_does_not_depend_on_future_frames():
    m = _model(channels=24)
    x = torch.randn(1, 1, F_BINS, 40, dtype=torch.complex64)
    with torch.no_grad():
        base = m(x)
        y = x.clone()
        y[..., 20] += 5.0
        out = m(y)
    assert (out[..., :20] - base[..., :20]).abs().max().item() == 0.0
    assert (out[..., 20] - base[..., 20]).abs().max().item() > 0


def test_receptive_field_is_31_frames():
    m = _model(channels=24)
    assert m.receptive_field_frames == 31
    x = torch.randn(1, 1, F_BINS, 50, dtype=torch.complex64)
    with torch.no_grad():
        base = m(x)[..., 45]
        inside, outside = x.clone(), x.clone()
        inside[..., 45 - 30] += 5.0
        outside[..., 45 - 31] += 5.0
        assert (m(inside)[..., 45] - base).abs().max() > 0
        assert (m(outside)[..., 45] - base).abs().max() == 0.0


def test_interleave_places_both_channels():
    m2 = torch.arange(8.0).view(1, 2, 4, 1)                  # ch0: 0..3, ch1: 4..7
    got = N6NetV2.interleave(m2)[0, 0, :, 0].tolist()
    assert got == [0, 4, 1, 5, 2, 6, 3, 7, 7]


def test_deploy_step_streams_to_the_offline_mask():
    """Three per-column k_f x 1 convs + a host ring == the dilated offline conv."""
    from n6net.export_npu import N6NetV2StreamStep

    m = _model(channels=24).double()
    step = N6NetV2StreamStep(m).double().eval()
    T = 40
    feat = torch.rand(1, 1, m.rows_in, T, dtype=torch.float64)
    with torch.no_grad():
        offline = step.offline_mask(feat)
        rings = [[c.double() for c in r] for r in step.init_states()]
        masks = []
        for t in range(T):
            ins = step.graph_inputs(rings)
            assert len(ins) == len(step.input_names) - 1
            outs = step(feat[..., t:t + 1], *ins)
            masks.append(outs[0])
            rings = step.advance(rings, outs[1:])
    streamed = torch.cat(masks, dim=-1)
    assert torch.allclose(offline, streamed, atol=1e-10), (offline - streamed).abs().max()


def test_deploy_graph_follows_the_compiler_rules():
    from n6net.export_npu import N6NetV2StreamStep

    onnx = pytest.importorskip("onnx")
    step = N6NetV2StreamStep(_model(channels=24)).eval()
    buf = io.BytesIO()
    ins = step.graph_inputs(step.init_states())
    torch.onnx.export(step, (torch.rand(1, 1, step.feat_rows, 1), *ins), buf,
                      input_names=step.input_names, output_names=step.output_names,
                      opset_version=17, dynamo=False)
    g = onnx.load_from_string(buf.getvalue()).graph
    ops = {n.op_type for n in g.node}
    assert not ops & {"Concat", "Slice", "Gather", "PRelu"}, ops
    for n in g.node:
        if n.op_type == "Conv":
            ks = next(a.ints for a in n.attribute if a.name == "kernel_shape")
            assert ks[1] == 1, f"kernel along time in the deploy graph: {list(ks)}"


def test_cost_accounting():
    """Per block: 3 taps x 128 x 5 x C^2 (temporal) + 128 x 5 x C^2 (spectral)."""
    C = 96
    c = cost_summary(_model(channels=C))
    stem = 128 * 4 * 1 * C
    head = 128 * 3 * C * 2
    assert c["macs_per_frame"] == 4 * (3 + 1) * 128 * 5 * C * C + stem + head
    assert c["fifo_bytes_int8"] == (2 + 4 + 8 + 16) * 128 * C


def test_no_prelu_no_norm_no_recurrence():
    for mod in _model(channels=24).modules():
        assert not isinstance(mod, (nn.PReLU, nn.BatchNorm2d, nn.LayerNorm, nn.GRU, nn.LSTM))


# --- frequency reach ----------------------------------------------------------

FULL_BAND = [
    dict(full_band="gap", full_band_blocks=[1]),
    dict(full_band="pool", full_band_blocks=[1], freq_pos_emb=True),
    dict(full_band="gap", freq_pos_emb=True, stem_stride=4),
]


def test_plain_v2_reach_is_local():
    from n6net.model_v2 import frequency_reach
    assert frequency_reach(_model(channels=8)) == 72


@pytest.mark.parametrize("over", FULL_BAND)
def test_full_band_branch_reaches_every_bin(over):
    """One full-band branch is enough for every input bin to move the centre bin."""
    from n6net.model_v2 import frequency_reach
    assert frequency_reach(_model(channels=8, **over)) == 256


def test_full_band_config_reaches_every_bin_for_under_one_percent_mac():
    base = cost_summary(_model(channels=96))
    fb = cost_summary(build_causal_model(json.load(open("configs/n6net_v2_fullband.json"))))
    assert fb["frequency_reach_bins"] == 256
    assert fb["macs_per_frame"] < 1.01 * base["macs_per_frame"]


@pytest.mark.parametrize("over", FULL_BAND)
def test_full_band_deploy_step_streams_to_the_offline_mask(over):
    from n6net.export_npu import N6NetV2StreamStep

    m = _model(channels=16, **over).double()
    step = N6NetV2StreamStep(m).double().eval()
    T = 36
    feat = torch.rand(1, 1, m.rows_in, T, dtype=torch.float64)
    with torch.no_grad():
        offline = step.offline_mask(feat)
        rings = [[c.double() for c in r] for r in step.init_states()]
        masks = []
        for t in range(T):
            outs = step(feat[..., t:t + 1], *step.graph_inputs(rings))
            masks.append(outs[0])
            rings = step.advance(rings, outs[1:])
    assert torch.allclose(offline, torch.cat(masks, dim=-1), atol=1e-10)
    assert offline.shape[1] == m.stem_stride
    assert N6NetV2.interleave(offline).shape[2] == F_BINS


def test_gap_pools_over_both_axes_in_the_deploy_graph():
    """ReduceMean over frequency alone compiles to a hybrid Transpose epoch;
    over (H, W) it is a HW GlobalAveragePool."""
    from n6net.export_npu import N6NetV2StreamStep

    onnx = pytest.importorskip("onnx")
    from onnx import numpy_helper

    step = N6NetV2StreamStep(_model(channels=8, full_band="gap")).eval()
    buf = io.BytesIO()
    torch.onnx.export(step, (torch.rand(1, 1, step.feat_rows, 1),
                             *step.graph_inputs(step.init_states())), buf,
                      input_names=step.input_names, output_names=step.output_names,
                      opset_version=17, dynamo=False)
    g = onnx.load_from_string(buf.getvalue()).graph
    consts = {i.name: numpy_helper.to_array(i).tolist() for i in g.initializer}
    consts.update({n.output[0]: numpy_helper.to_array(n.attribute[0].t).tolist()
                   for n in g.node if n.op_type == "Constant"})
    means = [n for n in g.node if n.op_type == "ReduceMean"]
    assert means
    for n in means:
        axes = next((list(a.ints) for a in n.attribute if a.name == "axes"), None)
        if axes is None:
            axes = consts[n.input[1]]
        assert sorted(axes) == [2, 3], axes
