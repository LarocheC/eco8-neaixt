"""Gates for N6Net — the Neural-ART-native architecture.

The two that matter before any training run:

  * **Causality.** The whole model is a streaming enhancer; a temporal kernel
    that peeks one frame into the future would train fine, score fine offline,
    and be undeployable. Tested by perturbation, not by reading the padding.
  * **FIFO parity.** The deployed form feeds ``k_t`` columns (state + current)
    per call. That must be the same arithmetic as the offline left-padded
    convolution, or the on-target model is not the model that was trained.

Plus the cost accounting, because the architecture's entire claim is that it
trades weight-bound MACs for reused ones.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from n6net.model import (
    N6Block,
    build_causal_model,
    cost_summary,
    macs_per_frame,
)

F_BINS = 257


def _model(**over):
    h = dict(json.load(open("configs/n6net_b3.json")), **over)
    torch.manual_seed(0)
    return build_causal_model(h).eval()


# --- causality --------------------------------------------------------------


def test_output_does_not_depend_on_future_frames():
    """Perturb frame t+1; every output at or before t must be bit-identical.

    This is the gate the left-only pad exists for. A symmetric pad would leak
    one frame of lookahead per block — silently, and only detectable here.
    """
    m = _model()
    T = 24
    x = torch.randn(1, 1, F_BINS, T, dtype=torch.complex64)
    with torch.no_grad():
        base = m(x)
        for t_edit in (12, 18):
            y = x.clone()
            y[..., t_edit] += 5.0
            out = m(y)
            past = out[..., :t_edit] - base[..., :t_edit]
            assert past.abs().max().item() == 0.0, (
                f"editing frame {t_edit} changed an output at an earlier frame — "
                "the temporal convolution is not causal"
            )
            # and it must actually influence its own frame, or the test is vacuous
            assert (out[..., t_edit] - base[..., t_edit]).abs().max().item() > 0


def test_receptive_field_is_exactly_as_designed():
    """1 + n_blocks*(k_t-1): three blocks of k_t=6 give 16 frames, ~256 ms."""
    for nb, rf in ((1, 6), (3, 16), (4, 21)):
        assert _model(n_blocks=nb).receptive_field_frames == rf

    # and the reach is real: a frame RF-1 back still moves the output, one
    # further back does not.
    m = _model()
    rf = m.receptive_field_frames
    T, t_out = 40, 30
    x = torch.randn(1, 1, F_BINS, T, dtype=torch.complex64)
    with torch.no_grad():
        base = m(x)[..., t_out]
        inside, outside = x.clone(), x.clone()
        inside[..., t_out - (rf - 1)] += 5.0
        outside[..., t_out - rf] += 5.0
        assert (m(inside)[..., t_out] - base).abs().max() > 0
        assert (m(outside)[..., t_out] - base).abs().max() == 0.0


# --- FIFO / streaming parity ------------------------------------------------


def test_block_fifo_matches_offline():
    """forward_window() on a FIFO must equal forward()'s last column."""
    torch.manual_seed(0)
    blk = N6Block(96, k_t=6, k_f=3).double().eval()
    x = torch.randn(2, 96, F_BINS, 9, dtype=torch.float64)
    with torch.no_grad():
        offline = blk(x)[..., -1:]
        win = x[..., -blk.k_t:]                  # 5 FIFO frames + current
        streamed = blk.forward_window(win, x[..., -1:])
    assert torch.allclose(offline, streamed, atol=1e-12)


def test_whole_model_streams_frame_by_frame():
    """Drive the full stack through per-frame FIFOs and match the offline run.

    Only the steady state is compared: before RF frames have arrived the FIFOs
    hold zeros where the offline model had zero-padding, which agrees, but the
    interesting assertion is that they stay in lockstep afterwards.
    """
    m = _model().double()
    T = 30
    stft = torch.randn(1, 1, F_BINS, T, dtype=torch.complex128)
    with torch.no_grad():
        offline = m(stft)

        feats = m.features(stft.abs())
        xs = m.stem(feats)                                    # (1, C, F, T)
        fifos = [torch.zeros(1, m.channels, F_BINS, b.fifo_frames,
                             dtype=torch.float64) for b in m.blocks]
        masks = []
        for t in range(T):
            cur = xs[..., t:t + 1]
            for i, blk in enumerate(m.blocks):
                win = torch.cat([fifos[i], cur], dim=-1)
                out = blk.forward_window(win, cur)
                fifos[i] = win[..., 1:]                        # slide the FIFO
                cur = out
            masks.append(torch.sigmoid(m.head(cur)))
        streamed = stft * torch.cat(masks, dim=-1)

    assert torch.allclose(offline, streamed, atol=1e-10), (
        (offline - streamed).abs().max().item()
    )


# --- cost accounting --------------------------------------------------------


def test_cost_matches_the_design():
    """The architecture is a claim about cost shape; hold it to the numbers.

    Per block at C=96: temporal 257*6*96*96 = 14,211,072 and spectral
    257*3*96*96 = 7,105,536, so 21,316,608. Stem 257*3*1*96 = 74,016 and head
    257*96 = 24,672 bring three blocks to 64,048,512.
    """
    c = cost_summary(_model())
    assert c["macs_per_frame"] == 3 * (257 * 6 * 96 * 96 + 257 * 3 * 96 * 96) \
        + 257 * 3 * 1 * 96 + 257 * 96
    assert c["macs_per_frame"] == 64_048_512
    assert c["params"] == 250_465
    # The thesis in one number: every weight is reused across all frequency rows.
    assert c["arithmetic_intensity"] > 200
    assert c["fifo_bytes_int8"] == 3 * 5 * 257 * 96


def test_one_block_is_a_third_of_three():
    """The 1-block arm exists as the controlled midpoint of the PoC."""
    c1, c3 = cost_summary(_model(n_blocks=1)), cost_summary(_model(n_blocks=3))
    per_block = 257 * 6 * 96 * 96 + 257 * 3 * 96 * 96
    assert c3["macs_per_frame"] - c1["macs_per_frame"] == 2 * per_block


@pytest.mark.parametrize("channels", [72, 96, 120, 128])
def test_width_sweep_builds(channels):
    """72/96/120/128 are the widths worth putting past the compiler; 96 = 4*24
    is the designed four-way CONV_ACC tiling."""
    m = _model(channels=channels)
    assert macs_per_frame(m) == 3 * (257 * 6 * channels * channels
                                     + 257 * 3 * channels * channels) \
        + 257 * 3 * 1 * channels + 257 * channels


def test_no_normalization_no_recurrence_no_depthwise():
    """The design rules out all three on purpose; a later edit must not sneak
    one back in. Depthwise in particular would cut MACs and defeat the point."""
    m = _model()
    for mod in m.modules():
        assert not isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm,
                                    nn.GroupNorm, nn.GRU, nn.LSTM, nn.RNN))
        if isinstance(mod, nn.Conv2d):
            assert mod.groups == 1, "depthwise/grouped conv defeats the reuse thesis"
            assert mod.dilation == (1, 1), "no dilation by design"


# --- deploy graph layouts ---------------------------------------------------


@pytest.mark.parametrize("layout", ["time_w", "time_c", "time_split"])
def test_deploy_layouts_are_the_same_arithmetic(layout):
    """Every NPU export layout must stream to the offline mask.

    ``time_c`` folds the 1 x k_t kernel into a 1 x 1 conv over k_t*C channels
    and ``time_split`` into k_t summed 1 x 1 convs with a host-side ring buffer;
    both are rewrites for the compiler, so neither may change the numbers.
    """
    from n6net.export_npu import N6NetStreamStep

    m = _model().double()
    step = N6NetStreamStep(m, layout).double().eval()
    T = 24
    feat = torch.rand(1, 1, F_BINS, T, dtype=torch.float64)
    with torch.no_grad():
        x = m.stem(feat)
        for b in m.blocks:
            x = b(x)
        offline = torch.sigmoid(m.head(x))

        states = [s.double() for s in step.init_states()]
        assert len(states) == len(step.input_names) - 1
        masks = []
        for t in range(T):
            outs = step(feat[..., t:t + 1], *states)
            assert len(outs) == len(step.output_names)
            masks.append(outs[0])
            states = step.advance(states, outs[1:])
        streamed = torch.cat(masks, dim=-1)
    assert torch.allclose(offline, streamed, atol=1e-10), (
        (offline - streamed).abs().max().item()
    )


def test_split_layout_has_no_concat_or_slice():
    """The point of ``time_split``: Neural-ART runs Concat on the M55, so the
    deploy graph must not contain one (nor the Slice the shift would need)."""
    import io

    from n6net.export_npu import N6NetStreamStep

    onnx = pytest.importorskip("onnx")
    step = N6NetStreamStep(_model(n_blocks=1), "time_split").eval()
    buf = io.BytesIO()
    torch.onnx.export(step, (torch.rand(1, 1, F_BINS, 1), *step.init_states()), buf,
                      input_names=step.input_names, output_names=step.output_names,
                      opset_version=17, dynamo=False)
    ops = {n.op_type for n in onnx.load_from_string(buf.getvalue()).graph.node}
    assert not ops & {"Concat", "Slice", "Gather"}, ops
