"""Gates for the structured (Monarch / block-diagonal) pointwise convs.

These run before any 12-hour training arm starts. What they defend:

  - The structured layers ARE the torch_structured layers NSNet2 swept —
    bit-identical init from a common seed, and forward equality — so the
    ConvFSENet block-count sweep can be read next to the NSNet2 one.
  - The dense path is untouched: no config block ⇒ same parameter count, same
    state_dict keys, same numbers.
  - The arm arithmetic (params and MACs/frame) the sweep matches on.
  - BN folding into a Monarch layer, including the negative control that the
    fold is silently wrong without the running_mean term.
  - Streaming / windowed parity, and an ONNX graph free of the ops the deploy
    path asserts against.
"""

from __future__ import annotations

import collections
import json
import math

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

from common.env import AttrDict
from common.quant_audit import float_weight_operands
from convfsenet.export_onnx import export_streaming_fp32, export_windowed_fp32
from convfsenet.layers import (
    BlockdiagPointwise,
    MonarchPointwise,
    clone_pointwise,
    fold_bn_into_pointwise,
    make_pointwise,
)
from convfsenet.model import build_causal_model
from convfsenet.streaming import (
    ConvFSENetStreaming,
    ConvFSENetStreamingFast,
    ConvFSENetStreamingONNX,
    ConvFSENetWindowedONNX,
)

BASE_CFG = json.load(open("configs/convfsenet.json"))

# (nblocks, params, MACs/frame) for the Monarch arms, and (width, params, MACs)
# for their MAC-matched dense controls. Every number here was produced by
# building the model; they are the numbers RESULTS_CONVFSENET.md reports and
# the whole sweep is a comparison between the two columns.
MONARCH_ARMS = {4: (873_281, 855_552), 8: (500_033, 482_304),
                16: (313_409, 295_680), 32: (220_097, 202_368)}
DENSE_ARMS = {146: (863_847, 850_304), 108: (491_333, 481_248),
              83: (302_958, 295_148), 67: (206_014, 199_660)}


def _macs_per_frame(model) -> int:
    """Multiply-accumulates per output frame. For a 1x1 conv that is in*out;
    for a structured one it is exactly the factor parameter count; the
    depthwise conv costs C*K. Matches the closed forms in the sweep tables."""
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            total += (m.in_channels // m.groups) * m.out_channels * m.kernel_size[0]
        elif isinstance(m, MonarchPointwise):
            total += m.w1.numel() + m.w2.numel()
        elif isinstance(m, BlockdiagPointwise):
            total += m.weight.numel()
    return total


def _build(pointwise=None, **overrides):
    h = dict(BASE_CFG, **overrides)
    if pointwise is not None:
        h["pointwise"] = pointwise
    return build_causal_model(h)


def _randomize_bn(m: nn.Module) -> None:
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm1d):
                nn.init.normal_(mod.weight, 1.0, 0.1)
                nn.init.normal_(mod.bias, 0.0, 0.1)
                mod.running_mean.normal_(0.0, 0.1)
                mod.running_var.uniform_(0.5, 1.5)
                mod.num_batches_tracked.fill_(1)


def _small(pointwise=None):
    """A narrow model — same topology, 48/96 channels — for the parity gates."""
    torch.manual_seed(0)
    m = _build(pointwise, n_channels_res=48, n_channels_conv=96)
    _randomize_bn(m)
    return m.eval()


# --- V0: the layer is the torch_structured layer ----------------------------


@pytest.mark.parametrize("ci,co,nblocks", [(192, 384, 4), (384, 192, 8),
                                           (192, 384, 32), (384, 192, 16)])
def test_monarch_pointwise_matches_monarch_linear(ci, co, nblocks):
    """Same factors and same map as torch_structured's MonarchLinear.

    Init equality is bit-exact, not statistical: reset_parameters draws from
    the RNG in the same order MonarchLinear does, so from a common seed the
    two layers hold identical weights. That is what licenses reading this
    sweep next to the NSNet2 Monarch sweep.
    """
    MonarchLinear = pytest.importorskip(
        "torch_structured.monarch.monarch_linear").MonarchLinear

    torch.manual_seed(7)
    ref = MonarchLinear(ci, co, bias=True, nblocks=nblocks).double()
    torch.manual_seed(7)
    mine = MonarchPointwise(ci, co, nblocks, bias=True).double()

    assert torch.equal(ref.w1, mine.w1)
    assert torch.equal(ref.w2, mine.w2)
    assert torch.equal(ref.bias, mine.bias)

    x = torch.randn(3, ci, 9, dtype=torch.float64)
    expected = ref(x.transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(mine(x), expected, atol=1e-12)

    # The static-shape path used for tracing must be the same function.
    mine.t_size = 9
    assert torch.allclose(mine(x), expected, atol=1e-12)


@pytest.mark.parametrize("ci,co,nblocks", [(192, 384, 4), (384, 192, 8)])
def test_blockdiag_pointwise_matches_blockdiag_linear(ci, co, nblocks):
    BlockdiagLinear = pytest.importorskip(
        "torch_structured.monarch.blockdiag_linear").BlockdiagLinear
    torch.manual_seed(3)
    ref = BlockdiagLinear(ci, co, bias=True, nblocks=nblocks).double()
    torch.manual_seed(3)
    mine = BlockdiagPointwise(ci, co, nblocks, bias=True).double()
    assert torch.equal(ref.weight, mine.weight)
    x = torch.randn(2, ci, 5, dtype=torch.float64)
    assert torch.allclose(mine(x), ref(x.transpose(1, 2)).transpose(1, 2), atol=1e-12)


def _touched_outputs(layer, in_channels):
    """How many output channels one input channel reaches."""
    x = torch.zeros(1, in_channels, 1, dtype=torch.float64)
    base = layer(x)
    probe = x.clone()
    probe[0, 0, 0] = 1.0
    return int(((layer(probe) - base).abs().squeeze() > 1e-12).sum())


@pytest.mark.parametrize("ci,co,nblocks", [
    (192, 384, 4), (192, 384, 8), (192, 384, 16), (192, 384, 32),
    (384, 192, 4), (384, 192, 8), (384, 192, 16), (384, 192, 32),
])
def test_monarch_support_matches_the_documented_rule(ci, co, nblocks):
    """The permutation is the point, but its reach is NOT unconditional.

    A two-factor Monarch mixes fully only while ``in_blksz >= nblocks`` (i.e.
    ``nblocks <= sqrt(in_channels)``); past that each output channel sees
    ``min(in_blksz, nblocks)`` of the ``nblocks`` input blocks. At the sweep's
    own shapes that is 100% at nblocks 4 and 8, 75% at 16 (192->384) and
    18.8% / 37.5% at 32 — a covariate the results table has to carry, not a
    constant. This is Monarch's own property (MonarchLinear measures
    identically), not an artifact of the grouped-conv lowering.

    Parametrized over every shape the sweep trains precisely because the
    earlier single-shape version of this test only covered 64/nblocks=8 — the
    one boundary case where the reach is total — so it would have passed while
    the documentation was wrong.
    """
    torch.manual_seed(0)
    layer = MonarchPointwise(ci, co, nblocks, bias=False).double()
    expected = min(ci // nblocks, nblocks) * (co // nblocks)
    assert _touched_outputs(layer, ci) == expected


@pytest.mark.parametrize("ci,co,nblocks", [(64, 64, 8), (192, 384, 16), (384, 192, 32)])
def test_monarch_reaches_further_than_blockdiag(ci, co, nblocks):
    """Whatever the exact reach, Monarch must beat one block — otherwise a bug
    that dropped the shuffle would leave a silently block-diagonal arm."""
    torch.manual_seed(0)
    mon = _touched_outputs(MonarchPointwise(ci, co, nblocks, bias=False).double(), ci)
    bd = _touched_outputs(BlockdiagPointwise(ci, co, nblocks, bias=False).double(), ci)
    assert bd == co // nblocks, "block-diagonal must touch exactly its own block"
    assert mon >= 2 * bd, f"Monarch reach {mon} is not beyond block-diagonal's {bd}"


def test_monarch_init_scale_matches_dense_conv():
    """Output scale must match a dense Conv1d's, or a PESQ delta could just be
    a broken init. (Naively Kaiming-initing both factors undershoots by orders
    of magnitude — reset_parameters variance-matches the composed product.)"""
    torch.manual_seed(0)
    x = torch.randn(64, 192, 50)
    dense_std = nn.Conv1d(192, 384, 1)(x).std().item()
    for nblocks in (4, 8, 16, 32):
        std = MonarchPointwise(192, 384, nblocks)(x).std().item()
        assert abs(std - dense_std) / dense_std < 0.05, (
            f"nblocks={nblocks}: output std {std:.4f} vs dense {dense_std:.4f}"
        )


def test_rejects_indivisible_nblocks_and_bad_config():
    with pytest.raises(ValueError, match="must divide"):
        MonarchPointwise(192, 384, 5)
    with pytest.raises(ValueError, match="unknown pointwise kind"):
        make_pointwise(192, 384, cfg={"kind": "butterfly"})
    with pytest.raises(ValueError, match="unknown pointwise scope"):
        make_pointwise(192, 384, cfg={"kind": "monarch", "scope": "ends"})
    with pytest.raises(ValueError, match="must divide"):
        _build({"kind": "monarch", "nblocks": 5})


# --- V1/V2: arm arithmetic, and the dense path is a no-op -------------------


def test_dense_path_unchanged():
    """No pointwise block ⇒ the model before this feature existed."""
    plain = _build()
    assert sum(p.numel() for p in plain.parameters()) == 1_453_889
    assert _macs_per_frame(plain) == 1_436_160
    assert all(isinstance(b.conv1x1, nn.Conv1d) for b in plain.tcm)
    # An explicit kind="conv" is the same model, key for key.
    assert set(plain.state_dict()) == set(_build({"kind": "conv"}).state_dict())


@pytest.mark.parametrize("nblocks,params,macs", [(k, *v) for k, v in MONARCH_ARMS.items()])
def test_monarch_arm_arithmetic(nblocks, params, macs):
    m = _build({"kind": "monarch", "nblocks": nblocks, "scope": "tcm"})
    assert sum(p.numel() for p in m.parameters()) == params
    assert _macs_per_frame(m) == macs
    # Frontend/backend stay dense: 257 is prime, and the deploy path's
    # prologue walk and Nyquist slicing both assume a dense Conv there.
    assert isinstance(m.frontend[0], nn.Conv1d)
    assert isinstance(m.backend[0], nn.Conv1d)
    assert all(isinstance(b.conv1x1, MonarchPointwise) for b in m.tcm)
    assert all(isinstance(b.conv1x1_out, MonarchPointwise) for b in m.tcm)


@pytest.mark.parametrize("width,params,macs", [(k, *v) for k, v in DENSE_ARMS.items()])
def test_dense_control_arm_arithmetic(width, params, macs):
    m = _build(None, n_channels_res=width, n_channels_conv=2 * width)
    assert sum(p.numel() for p in m.parameters()) == params
    assert _macs_per_frame(m) == macs


def test_arms_are_mac_matched_within_one_and_a_half_percent():
    """The sweep's whole premise. Each Monarch arm is paired with the dense
    width whose MACs/frame match it; report the residual so no reader has to
    take 'matched' on trust."""
    for (nb, (_, m_macs)), (w, (_, d_macs)) in zip(MONARCH_ARMS.items(), DENSE_ARMS.items()):
        rel = abs(d_macs - m_macs) / m_macs
        assert rel < 0.015, f"nblocks={nb} vs width={w}: MAC mismatch {rel:.2%}"


def test_nblocks_two_is_larger_than_dense_only_when_rectangular():
    """The rectangular geometry's asymmetry, and its absence when square.

    w1 is square in the INPUT block size, so an expanding 192->384 layer
    compresses 2*nblocks/3 but the contracting 384->192 layer only nblocks/3 —
    at nblocks=2 the 'compressed' rectangular model is 1.12x BIGGER than dense.
    Square C->C makes both factors (nblocks, C/nblocks, C/nblocks), so
    compression is exactly nblocks/2 everywhere and nblocks=2 is exact parity.
    That parity point is the zero-compression control the square sweep uses.
    """
    rect = _macs_per_frame(_build({"kind": "monarch", "nblocks": 2}))
    assert rect > _macs_per_frame(_build())
    assert abs(rect / _macs_per_frame(_build()) - 1.1155) < 0.001

    sq = dict(n_features=256, n_channels_res=256, n_channels_conv=256)
    sq_dense = _macs_per_frame(_build(None, **sq))
    sq_mon2 = _macs_per_frame(_build({"kind": "monarch", "nblocks": 2, "scope": "all"}, **sq))
    assert sq_mon2 == sq_dense, "square nblocks=2 must be exact MAC parity with dense"


SQUARE_ARMS = {2: (1_329_664, 1_317_632), 4: (674_304, 662_272),
               8: (346_624, 334_592), 16: (182_784, 170_752)}


@pytest.mark.parametrize("nblocks,params,macs", [(k, *v) for k, v in SQUARE_ARMS.items()])
def test_square_scope_all_arm_arithmetic(nblocks, params, macs):
    """The square 256-bin arms, every matrix structured — no un-structured floor
    except the nine depthwise convs (6,912 MACs/frame, 0.5-4.0% of an arm)."""
    m = _build({"kind": "monarch", "nblocks": nblocks, "scope": "all"},
               n_features=256, n_channels_res=256, n_channels_conv=256)
    assert sum(p.numel() for p in m.parameters()) == params
    assert _macs_per_frame(m) == macs
    assert isinstance(m.frontend[0], MonarchPointwise)
    assert isinstance(m.backend[0], MonarchPointwise)
    assert all(isinstance(b.conv1x1, MonarchPointwise) for b in m.tcm)
    # compression is exactly nblocks/2, per layer, in both directions
    assert m.frontend[0].saving == pytest.approx(2.0 / nblocks)
    assert m.tcm[0].conv1x1.saving == pytest.approx(2.0 / nblocks)
    assert m.tcm[0].conv1x1_out.saving == pytest.approx(2.0 / nblocks)


@pytest.mark.parametrize("nblocks", [2, 4, 8, 16])
def test_square_arms_have_full_reach(nblocks):
    """Every square arm in the sweep mixes fully: nblocks <= sqrt(256) = 16.

    This is the property the rectangular sweep could not hold fixed — there,
    reaching 7.1x compression forced nblocks=32 and reach down to 18.8%, so
    compression and connectivity moved together.
    """
    layer = MonarchPointwise(256, 256, nblocks, bias=False).double()
    assert _touched_outputs(layer, 256) == 256


def test_native_sub_nyquist_is_identity_at_full_width():
    """The trim/pad must be a no-op for a 257-bin model, or every existing
    number in RESULTS_CONVFSENET.md moves."""
    torch.manual_seed(0)
    m = _small()
    stft = torch.randn(2, 257, 30, dtype=torch.complex64)
    with torch.no_grad():
        out = m(stft.unsqueeze(1))
    assert out.shape[-2] == 257
    # and a sub-Nyquist model zero-fills exactly the bins it dropped
    torch.manual_seed(0)
    m256 = _build(None, n_features=256, n_channels_res=48, n_channels_conv=96).eval()
    with torch.no_grad():
        out256 = m256(stft.unsqueeze(1))
    assert out256.shape[-2] == 257
    assert torch.equal(out256[:, :, 256, :], torch.zeros_like(out256[:, :, 256, :]))


def test_drop_nyquist_refuses_on_a_natively_narrow_model():
    """deploy/stm32n6/scripts/run_windowed_eval.sh passes --drop_nyquist
    unconditionally; on a natively-256 model that would silently build a
    255-wide graph."""
    m = _build(None, n_features=256, n_channels_res=48, n_channels_conv=96).eval()
    with pytest.raises(ValueError, match="already trained at"):
        ConvFSENetWindowedONNX(m, T=1, drop_nyquist=True)


def test_drop_nyquist_refuses_on_a_structured_end_model_too():
    """A scope='all' model is necessarily natively narrow (257 is prime, so no
    nblocks divides it), so the width guard is what fires. The structured-end
    guard behind it is defence-in-depth for any future padded-end variant."""
    m = _build({"kind": "monarch", "nblocks": 8, "scope": "all"},
               n_features=256, n_channels_res=64, n_channels_conv=64).eval()
    with pytest.raises(ValueError, match="already trained at|structured frontend"):
        ConvFSENetWindowedONNX(m, T=1, drop_nyquist=True)


# --- V5: BatchNorm folding --------------------------------------------------


@pytest.mark.parametrize("cls", [MonarchPointwise, BlockdiagPointwise])
@pytest.mark.parametrize("ci,co,bias", [(192, 384, True), (384, 192, False)])
def test_bn_fold_exact(cls, ci, co, bias):
    torch.manual_seed(0)
    layer = cls(ci, co, 8, bias=bias).double()
    bn = nn.BatchNorm1d(co).double()
    with torch.no_grad():
        bn.running_mean.normal_(0.0, 0.5)
        bn.running_var.uniform_(0.5, 2.0)
        bn.weight.uniform_(0.5, 1.5)
        bn.bias.normal_()
    bn.eval()
    x = torch.randn(4, ci, 7, dtype=torch.float64)
    folded = fold_bn_into_pointwise(layer, bn)
    assert torch.allclose(folded(x), bn(layer(x)), atol=1e-12)
    assert folded.bias is not None, "the fold must allocate a bias even if the source had none"


def test_bn_fold_without_running_mean_is_wrong():
    """Negative control. The `- running_mean` term is easy to drop and its
    omission is silent: the model still trains, exports and quantizes. It is
    simply a different, wrong model."""
    torch.manual_seed(0)
    layer = MonarchPointwise(192, 384, 8, bias=True).double()
    bn = nn.BatchNorm1d(384).double()
    with torch.no_grad():
        bn.running_mean.normal_(0.0, 0.5)
        bn.running_var.uniform_(0.5, 2.0)
        bn.weight.uniform_(0.5, 1.5)
        bn.bias.normal_()
    bn.eval()
    x = torch.randn(4, 192, 7, dtype=torch.float64)
    bad = fold_bn_into_pointwise(layer, bn)
    with torch.no_grad():
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bad.bias.copy_(layer.bias * scale + bn.bias)          # running_mean dropped
    assert (bad(x) - bn(layer(x))).abs().max() > 0.1


def test_clone_preserves_everything():
    torch.manual_seed(0)
    src = MonarchPointwise(192, 384, 8, bias=True)
    dst = clone_pointwise(src)
    assert torch.equal(src.w1, dst.w1) and torch.equal(src.w2, dst.w2)
    dst.w1.data.add_(1.0)
    assert not torch.equal(src.w1, dst.w1), "clone must not alias the source"


def test_dense_fold_helper_rejects_structured():
    """A structured layer must never reach the dense fold path: the layers
    expose no .weight precisely so this fails loudly instead of fabricating
    a wrong dense conv."""
    from convfsenet.streaming import _fold_bn_into_conv
    bn = nn.BatchNorm1d(384).eval()
    with pytest.raises(Exception):
        _fold_bn_into_conv(MonarchPointwise(192, 384, 8), bn)


# --- V6/V7: streaming and windowed parity -----------------------------------


@pytest.mark.parametrize("pointwise", [
    {"kind": "monarch", "nblocks": 8},
    {"kind": "blockdiag", "nblocks": 4},
])
def test_streaming_parity(pointwise):
    m = _small(pointwise)
    stft = torch.randn(2, 257, 40, dtype=torch.complex64)
    with torch.no_grad():
        offline = m(stft.unsqueeze(1)).squeeze(1)
        naive = ConvFSENetStreaming(m).forward_full(stft)
        fast = ConvFSENetStreamingFast(m).forward_full(stft)
    assert (offline - naive).abs().max() < 1e-5
    assert (offline - fast).abs().max() < 1e-5


def test_windowed_parity():
    m = _small({"kind": "monarch", "nblocks": 8})
    w = ConvFSENetWindowedONNX(m, T=8)
    assert w.L == 42, "receptive field depends on K and D only, never on structure"
    mag = torch.randn(2, 257, 40).abs()
    padded = torch.nn.functional.pad(mag, (w.L, 0))
    with torch.no_grad():
        windowed = w.forward_mask_window(padded[..., : w.L + 8])
        fast = ConvFSENetStreamingFast(m)
        states = fast.init_states(2, mag.device)
        masks = []
        for t in range(w.L + 8):
            mk, states = fast.forward_mask_step(padded[..., t], states)
            masks.append(mk)
    assert (windowed - torch.stack(masks[-8:], dim=-1)).abs().max() < 1e-5


# --- V8: the exported graph -------------------------------------------------


def _op_hist(path):
    return collections.Counter(n.op_type for n in onnx.load(str(path)).graph.node)


def test_streaming_export_graph_and_ort_parity(tmp_path):
    """The per-frame graph must carry no Shape/If/Pad/BatchNorm/Einsum, add no
    Gather beyond the dense graph's FIFO taps, and be correct at a batch size
    other than the traced one. (The dynamic unflatten form is numerically fine
    too — what it cannot do is survive quant_pre_process, which is why the
    layers pin static reshape targets for export.)"""
    m = _small({"kind": "monarch", "nblocks": 8})
    out = tmp_path / "mon_stream.onnx"
    export_streaming_fp32(m, out)
    hist = _op_hist(out)
    for banned in ("Shape", "If", "Pad", "BatchNormalization", "Einsum"):
        assert hist.get(banned, 0) == 0, f"{banned} in streaming graph: {dict(hist)}"
    # The per-frame graph legitimately gathers the dilated FIFO taps, one per
    # block. Structuring the pointwise convs must add none of its own.
    dense_out = tmp_path / "dense_stream.onnx"
    export_streaming_fp32(_small(), dense_out)
    assert hist.get("Gather", 0) == _op_hist(dense_out).get("Gather", 0)
    assert hist.get("Conv", 0) > 0

    fast = ConvFSENetStreamingFast(m).eval()
    view = ConvFSENetStreamingONNX(fast).eval()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for B in (1, 3):
        mag = torch.randn(B, view.n_features).abs()
        states = view.init_states(B, "cpu", dtype=torch.float32)
        feeds = {"noisy_mag": mag.numpy()}
        for name, s in zip(view.state_input_names, states):
            feeds[name] = s.numpy()
        got = sess.run(None, feeds)[0]
        with torch.no_grad():
            expected = view(mag, *states)[0]
        assert np.abs(got - expected.numpy()).max() < 1e-4, f"batch {B} mismatch"


def test_windowed_export_graph_and_ort_parity(tmp_path):
    m = _small({"kind": "monarch", "nblocks": 8})
    out = tmp_path / "mon_win.onnx"
    export_windowed_fp32(m, out, T=4)
    hist = _op_hist(out)
    # Gather == 0 and Pad == 0 are the deploy asserts quant_windowed.py makes.
    for banned in ("Shape", "Gather", "If", "Pad", "BatchNormalization", "Einsum"):
        assert hist.get(banned, 0) == 0, f"{banned} in windowed graph: {dict(hist)}"

    view = ConvFSENetWindowedONNX(m, T=4).eval()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for B in (1, 3):
        win = torch.randn(B, view.n_features, view.window_length).abs()
        got = sess.run(None, {"noisy_mag_window": win.numpy()})[0]
        with torch.no_grad():
            expected = view.forward_mask_window(win)
        assert np.abs(got - expected.numpy()).max() < 1e-4, f"batch {B} mismatch"


def test_export_leaves_t_size_restored(tmp_path):
    """A trace must not leave the model pinned to the traced width — that would
    make training after an export silently wrong for other batch/time shapes."""
    m = _small({"kind": "monarch", "nblocks": 8})
    export_streaming_fp32(m, tmp_path / "x.onnx")
    assert all(sub.t_size is None for sub in m.modules()
               if isinstance(sub, MonarchPointwise))


def test_no_float_weight_operands_after_quantization_is_checked(tmp_path):
    """The audit itself must be able to see an unquantized weight — otherwise
    it would pass on a hybrid-precision graph the way the NSNet2 Einsum bug did."""
    m = _small({"kind": "monarch", "nblocks": 8})
    out = tmp_path / "fp32.onnx"
    export_streaming_fp32(m, out)
    # An FP32 graph is by definition all-float-weights: the audit must flag it.
    offenders = float_weight_operands(onnx.load(str(out)))
    assert len(offenders) > 0
    assert all(op == "Conv" for op, _, _ in offenders)


# --- V10/V11: training-time hazards ----------------------------------------


def test_train_step_stays_float32():
    """torch_structured's fast Monarch multiply is decorated
    custom_fwd(cast_inputs=torch.bfloat16), so under any autocast wrapper it
    would silently drop to bf16 — even under fp16 AMP. This layer uses plain
    grouped convs and so has no such trapdoor; assert the dtype anyway, since
    an autocast added later is exactly the kind of change nobody re-checks."""
    m = _small({"kind": "monarch", "nblocks": 8})
    m.train()
    x = torch.randn(2, 1, 8000)
    out = m.train_step(x, x)["loss"]
    assert out.dtype == torch.float32
    out.backward()
    for name, p in m.named_parameters():
        if "w1" in name or "w2" in name:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_checkpoint_roundtrip_through_every_loader(tmp_path):
    """All three copies of _load_offline_from_checkpoint build the model from
    an explicit kwarg list rather than build_causal_model. Miss the pointwise
    key in one and it builds a dense model, then dies on strict=True — twelve
    hours after the training run that produced the checkpoint."""
    import convfsenet.export_onnx as ex
    import convfsenet.quant as qt
    import convfsenet.quant_windowed as qw

    cfg = dict(BASE_CFG, n_channels_res=48, n_channels_conv=96,
               pointwise={"kind": "monarch", "nblocks": 8, "scope": "tcm"})
    m = build_causal_model(cfg).eval()
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    torch.save({"generator": m.state_dict()}, tmp_path / "g_best")

    loaded = [ex._load_offline_from_checkpoint(tmp_path / "g_best"),
              qt._load_offline_from_checkpoint(tmp_path / "g_best")[0],
              qw._load_offline_from_checkpoint(tmp_path / "g_best")[0]]
    for got in loaded:
        assert isinstance(got.tcm[0].conv1x1, MonarchPointwise)
        assert torch.equal(got.tcm[0].conv1x1.w1, m.tcm[0].conv1x1.w1)


def test_eager_fake_quant_covers_the_structured_layers():
    """common/quant_fake.py skips leaves it does not recognise SILENTLY.

    Before the structured branches existed, `eval_ptq --configs w8a8` on a
    Monarch arm would quantize only the frontend, backend and depthwise convs
    and still print a "w8a8" PESQ — 92% of the MACs left in FP32, a
    hybrid-precision number of exactly the kind that forced the int8 retraction
    in RESULTS_NSNET2.md. Assert coverage in weights, not in trust.
    """
    from common.quant_fake import _quantizable_weight_attrs

    for pointwise in ({"kind": "monarch", "nblocks": 8},
                      {"kind": "blockdiag", "nblocks": 4}):
        m = _build(pointwise, n_channels_res=48, n_channels_conv=96)
        covered = {
            id(getattr(mod, attr))
            for mod in m.modules()
            for attr, _ in _quantizable_weight_attrs(mod)
            if getattr(mod, attr, None) is not None
        }
        for name, mod in m.named_modules():
            if isinstance(mod, (MonarchPointwise, BlockdiagPointwise)):
                factors = [p for n, p in mod.named_parameters() if n != "bias"]
                assert factors, name
                for p in factors:
                    assert id(p) in covered, (
                        f"{name}: a weight factor is invisible to the eager "
                        f"fake-quant walker, so PTQ/QAT would leave it FP32"
                    )


def test_activation_hooks_reach_the_structured_layers():
    from common.quant_fake import install_static_activation_fake_quant

    dense_points = install_static_activation_fake_quant(
        _build(None, n_channels_res=48, n_channels_conv=96), bits=8)
    mon_points = install_static_activation_fake_quant(
        _build({"kind": "monarch", "nblocks": 8}, n_channels_res=48,
               n_channels_conv=96), bits=8)
    # 2 (frontend/backend) + 9 blocks x 3 quantizable leaves, structured or not.
    assert dense_points == mon_points == 29
