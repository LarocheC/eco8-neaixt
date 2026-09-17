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


# --- MP-SENet ingredients -----------------------------------------------------


def test_phase_features_shape_range_and_causality():
    m = _model(channels=8, input_features="mag_gd_ifd")
    T = 20
    x = torch.randn(1, 1, F_BINS, T, dtype=torch.complex64)
    f = m.features(x)
    assert f.shape == (1, 3, 256, T)
    assert f[:, 1:].abs().max() <= 1.0 + 1e-6                   # GD and IFD are / pi
    with torch.no_grad():                                        # IFD reads only the past
        base = m(x)
        y = x.clone()
        y[..., 10] += 5.0
        out = m(y)
    assert (out[..., :10] - base[..., :10]).abs().max().item() == 0.0
    assert (out[..., 10] - base[..., 10]).abs().max().item() > 0


def test_learnable_sigmoid_has_a_slope_per_bin_and_can_exceed_one():
    m = _model(channels=8, mask_act="lsigmoid", mask_beta=2.0)
    assert m.mask_slope.shape == (1, 2, 128, 1)
    with torch.no_grad():
        m.head.bias.fill_(10.0)                                  # saturate
        mask = m.mask_half(torch.rand(1, 1, 256, 4))
    assert 1.5 < mask.max() <= 2.0


def test_phase_head_starts_near_identity_and_rotates_by_unit_modulus():
    h = dict(json.load(open("configs/n6net_v2_fullband_phase.json")), channels=8)
    torch.manual_seed(0)
    m = build_causal_model(h).eval()
    x = torch.randn(1, 1, F_BINS, 6, dtype=torch.complex64)
    with torch.no_grad():
        y, pha = m.spectrum(x)
        rot = m.rotation(pha)
        mask = m.interleave(m.heads(m.features(x))[0])
    assert pha.shape == (1, 4, 128, 6)
    assert torch.allclose(rot.abs(), torch.ones_like(rot.abs()), atol=1e-5)
    assert rot.angle().abs().mean() < 0.2                        # ~ the noisy phase at init
    assert torch.allclose(y, x * mask * rot, atol=1e-5)


@pytest.mark.parametrize("over", [
    dict(mask_act="lsigmoid"),
    dict(input_features="mag_gd_ifd"),
    dict(objective="mpsenet", phase_head=True),
    dict(mask_act="lsigmoid", input_features="mag_gd_ifd", objective="mpsenet", phase_head=True),
])
def test_mpsenet_ingredients_stream_to_the_offline_heads(over):
    from n6net.export_npu import N6NetV2StreamStep

    m = _model(channels=16, **over).double()
    step = N6NetV2StreamStep(m).double().eval()
    T = 30
    feat = torch.rand(1, m.in_channels, m.rows_in, T, dtype=torch.float64) * 2 - 1
    with torch.no_grad():
        offline = step.offline_heads(feat)
        assert len(offline) == step.n_head
        assert len(step.output_names) == step.n_head + len(m.blocks)
        rings = [[c.double() for c in r] for r in step.init_states()]
        got = [[] for _ in offline]
        for t in range(T):
            outs = step(feat[..., t:t + 1], *step.graph_inputs(rings))
            for g, o in zip(got, outs[:step.n_head]):
                g.append(o)
            rings = step.advance(rings, outs[step.n_head:])
    for g, r in zip(got, offline):
        assert torch.allclose(torch.cat(g, dim=-1), r, atol=1e-10)


def test_all_ingredients_deploy_graph_follows_the_compiler_rules():
    from n6net.export_npu import N6NetV2StreamStep

    onnx = pytest.importorskip("onnx")
    step = N6NetV2StreamStep(_model(channels=8, mask_act="lsigmoid", input_features="mag_gd_ifd",
                                    objective="mpsenet", phase_head=True)).eval()
    buf = io.BytesIO()
    torch.onnx.export(step, (torch.rand(1, 3, step.feat_rows, 1),
                             *step.graph_inputs(step.init_states())), buf,
                      input_names=step.input_names, output_names=step.output_names,
                      opset_version=17, dynamo=False)
    g = onnx.load_from_string(buf.getvalue()).graph
    ops = {n.op_type for n in g.node}
    assert not ops & {"Concat", "Slice", "Gather", "PRelu", "Atan", "Div"}, ops
    assert [o.name for o in g.output][:2] == ["mask", "pha"]


def test_mpsenet_objective_trains_every_head():
    h = dict(json.load(open("configs/n6net_v2_fullband_phase.json")), channels=8)
    torch.manual_seed(0)
    m = build_causal_model(h).train()
    x_noisy, x_clean = torch.randn(2, 1, 8000), torch.randn(2, 1, 8000)
    ld = m.train_step(x_noisy, x_clean)
    for k in ("loss/magnitude", "loss/complex", "loss/consistency", "loss/time",
              "loss/phase", "loss/phase_unit"):
        assert k in ld and torch.isfinite(ld[k]), k
    assert ld["loss/consistency"] > 0                            # live, not identically zero
    assert "loss/metric" not in ld                               # the trainer adds it
    ld["loss"].backward()
    assert m.pha_head.weight.grad.abs().sum() > 0
    assert m.head.weight.grad.abs().sum() > 0
    m.eval()
    x_pred, x_clean_p, _, ld_v = m.valid_step(x_noisy, x_clean)
    assert x_pred.shape == x_clean_p.shape and torch.isfinite(ld_v["loss"])
    # mask-only model on the same objective: no phase terms
    ld2 = build_causal_model(dict(h, phase_head=False)).train_step(x_noisy, x_clean)
    assert "loss/phase" not in ld2 and "loss/phase_unit" not in ld2


def test_phase_head_requires_the_mpsenet_objective():
    with pytest.raises(ValueError):
        _model(channels=8, phase_head=True)


def test_trainer_gan_step_runs_on_the_mpsenet_objective():
    """The shared trainer's own metric-GAN step, on the phase-head config."""
    from common.discriminator import MetricDiscriminator
    from common.env import AttrDict
    from convfsenet.train import _gan_step

    h = AttrDict(dict(json.load(open("configs/n6net_v2_fullband_phase.json")), channels=8))
    torch.manual_seed(0)
    model = build_causal_model(h).train()
    disc = MetricDiscriminator()
    optim = torch.optim.AdamW(model.parameters(), 1e-4)
    optim_d = torch.optim.AdamW(disc.parameters(), 1e-4)
    noisy, clean = torch.randn(2, 1, 16000), torch.randn(2, 1, 16000)
    before = model.pha_head.weight.detach().clone()
    metrics = _gan_step(model, disc, optim, optim_d, noisy, clean, h,
                        metric_lambda=0.05, disc_compress=0.3, device="cpu")
    for k in ("loss", "base_loss", "loss_metric", "loss_disc"):
        assert k in metrics and metrics[k] == metrics[k]              # finite, not NaN
    assert not torch.equal(before, model.pha_head.weight)              # the step trained it


def test_pool_branch_keeps_its_checkpoint_keys():
    """Trained full-band checkpoints load by these names; renaming breaks them."""
    keys = set(_model(channels=8, full_band="pool", full_band_blocks=[1]).state_dict())
    for k in ("d1", "d2", "fh"):
        assert f"blocks.1.full_band.{k}.weight" in keys


def test_pool_native_reaches_every_bin_without_kernels_the_compiler_rewrites():
    from n6net.model_v2 import frequency_reach
    m = _model(channels=8, full_band="pool_native", full_band_blocks=[1], freq_pos_emb=True)
    assert frequency_reach(m) == 256
    fb = m.blocks[1].full_band
    assert [c.kernel_size[0] for c in (fb.d1, fb.d2, fb.d3, fb.fh)] == [4, 4, 4, 2]


def test_native_rewrite_of_the_full_height_conv_is_exact():
    """The export-time rewrite must not change a trained 'pool' model."""
    from n6net.model_v2 import native_full_height

    torch.manual_seed(0)
    fh = nn.Conv2d(48, 96, (8, 1)).double()
    y = torch.randn(3, 48, 8, 5, dtype=torch.float64)
    native = native_full_height(fh)
    assert [c.kernel_size for c in native] == [(4, 1), (2, 1)]
    assert torch.allclose(fh(y), native(y), atol=1e-12)

    m = _model(channels=16, full_band="pool", full_band_blocks=[1], freq_pos_emb=True).double()
    feat = torch.rand(1, 1, 256, 12, dtype=torch.float64)
    with torch.no_grad():
        before = m.mask_half(feat)
        assert m.blocks[1].full_band.use_native_rewrite()
        assert not m.blocks[1].full_band.use_native_rewrite()      # idempotent
        after = m.mask_half(feat)
    assert torch.allclose(before, after, atol=1e-12)


def test_rewritten_pool_graph_has_no_kernel_taller_than_four():
    from n6net.export_npu import N6NetV2StreamStep

    onnx = pytest.importorskip("onnx")
    m = _model(channels=8, full_band="pool", full_band_blocks=[1])
    m.blocks[1].full_band.use_native_rewrite()
    step = N6NetV2StreamStep(m).eval()
    buf = io.BytesIO()
    torch.onnx.export(step, (torch.rand(1, 1, step.feat_rows, 1),
                             *step.graph_inputs(step.init_states())), buf,
                      input_names=step.input_names, output_names=step.output_names,
                      opset_version=17, dynamo=False)
    for n in onnx.load_from_string(buf.getvalue()).graph.node:
        if n.op_type == "Conv":
            ks = next(a.ints for a in n.attribute if a.name == "kernel_shape")
            assert ks[0] <= 5 and ks[1] == 1, list(ks)
