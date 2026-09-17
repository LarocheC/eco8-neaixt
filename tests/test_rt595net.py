"""Gates for RT595Net, the HiFi4-cost-law architecture.

The three that must hold before any training run:

  * **Causality** — tested by perturbation, not by reading the padding. The
    full-band branch is the live hazard: averaging over time as well as
    frequency would look identical at deploy (W = 1) and leak the future here.
  * **Frequency reach** — N6Net v1 trained fine and scored 2.657 for 64 M MAC
    because it reached 9 of 257 bins. A stride-4 stem with k_f = 3 reaches +-8
    without the full-band branch, so this is the same trap.
  * **Deploy-shape parity** — ``forward_window`` on a host-held FIFO must equal
    the offline convolution, or the trained model is not the deployed one.

Plus the op hygiene the whole design rests on: no depthwise, no grouped conv, no
PReLU.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from rt595net.model import (
    HIFI4_BUDGET_CYCLES,
    RT595Block,
    build_causal_model,
    cost_summary,
    frequency_reach,
    hifi4_cycles,
)

F_BINS = 257


def _model(**over):
    h = dict(json.load(open("configs/rt595b_fb.json")), **over)
    torch.manual_seed(0)
    return build_causal_model(h).eval()


# --- causality --------------------------------------------------------------


def test_output_does_not_depend_on_future_frames():
    m = _model()
    T = 24
    x = torch.randn(1, 1, F_BINS, T, dtype=torch.complex64)
    with torch.no_grad():
        base = m(x)
        for t_edit in (12, 18):
            y = x.clone()
            y[..., t_edit] += 5.0
            out = m(y)
            assert (out[..., :t_edit] - base[..., :t_edit]).abs().max().item() == 0.0, (
                f"editing frame {t_edit} changed an earlier output — not causal")
            assert (out[..., t_edit] - base[..., t_edit]).abs().max().item() > 0


def test_full_band_branch_is_causal():
    """The branch averages over frequency only; over (freq, time) it would leak.

    At deploy W == 1 so both spellings agree, which is exactly what makes this
    worth a test of its own.
    """
    m = _model(full_band_blocks=[0, 1, 2])
    assert sum(1 for b in m.blocks if b.ctx is not None) == 3
    T = 20
    x = torch.randn(1, 1, F_BINS, T, dtype=torch.complex64)
    with torch.no_grad():
        base = m(x)
        y = x.clone()
        y[..., 15] += 5.0
        assert (m(y)[..., :15] - base[..., :15]).abs().max().item() == 0.0


def test_receptive_field_is_as_designed():
    for nb, rf in ((1, 6), (3, 16), (4, 21)):
        assert _model(n_blocks=nb).receptive_field_frames == rf


# --- frequency reach --------------------------------------------------------


def test_full_band_branch_reaches_every_bin():
    assert frequency_reach(_model()) == 256


def test_without_the_branch_reach_collapses():
    """Guard the lesson: this is N6Net v1's failure, reproduced on purpose.

    40 of 256 bins -- wider than the +-8 the kernels alone suggest, because the
    stride-4 stem's kernel is 8 rows and overlaps, but still a minority of the
    spectrum and on the wrong side of what cost v1 its 0.16 PESQ."""
    assert frequency_reach(_model(full_band_blocks=[])) == 40


# --- deploy parity ----------------------------------------------------------


def test_block_window_matches_offline():
    torch.manual_seed(0)
    blk = RT595Block(48, k_t=6, k_f=3, full_band=True).double().eval()
    x = torch.randn(2, 48, 64, 9, dtype=torch.float64)
    with torch.no_grad():
        offline = blk(x)[..., -1:]
        streamed = blk.forward_window(x[..., -blk.k_t:])
    assert torch.allclose(offline, streamed, atol=1e-12)


def test_whole_model_streams_frame_by_frame():
    m = _model().double()
    T = 30
    stft = torch.randn(1, 1, F_BINS, T, dtype=torch.complex128)
    with torch.no_grad():
        offline = m(stft)
        x = torch.relu(m.stem(m.features(stft.abs())))
        fifos = [torch.zeros(1, m.channels, m.rows, b.history, dtype=torch.float64)
                 for b in m.blocks]
        masks = []
        for t in range(T):
            cur = x[..., t:t + 1]
            for i, blk in enumerate(m.blocks):
                win = torch.cat([fifos[i], cur], dim=-1)
                fifos[i] = win[..., 1:]
                cur = blk.forward_window(win)
            masks.append(torch.sigmoid(m.head(cur)))
        streamed = stft * m.unfold(torch.cat(masks, dim=-1), m.stem_stride)
    assert torch.allclose(offline, streamed, atol=1e-10), \
        (offline - streamed).abs().max().item()


# --- the cost law the architecture exists to satisfy ------------------------


def test_cost_matches_the_measured_tflite_twin():
    """deploy/rt595/host/gen_arch_candidates.py rt595b_fb_s4_C48_b3, ISS-measured
    at 2,201,242 cyc/frame with 65,860 parameters."""
    c = cost_summary(_model())
    assert c["params"] == 65_860
    assert abs(c["hifi4_cycles"] - 2_201_242) / 2_201_242 < 0.15
    assert c["hifi4_budget_fraction"] < 1.0
    assert c["frequency_reach_bins"] == 256


def test_it_fits_the_dsp_data_segment():
    """1.24 MB dram0_0_seg killed dense NSNet2 (2.89 MB) and ConvFSENet (1.71 MB)."""
    c = cost_summary(_model())
    assert c["weight_bytes_int8"] + c["fifo_bytes_int8"] < 1_240_000


@pytest.mark.parametrize("channels,blocks", [(32, 2), (32, 3), (48, 2), (48, 3), (64, 2)])
def test_width_sweep_stays_in_budget(channels, blocks):
    assert hifi4_cycles(_model(channels=channels, n_blocks=blocks)) < HIFI4_BUDGET_CYCLES


@pytest.mark.parametrize("channels,blocks", [(64, 3), (80, 2)])
def test_the_budget_ceiling_is_where_it_is(channels, blocks):
    """C64/b3 is 1.04x and C80/b2 1.05x. Recording the ceiling matters as much as
    recording what fits: it is the first thing a later widening attempt hits."""
    assert hifi4_cycles(_model(channels=channels, n_blocks=blocks)) > HIFI4_BUDGET_CYCLES


def test_no_depthwise_no_grouped_no_prelu():
    """The cost law says an op is output-bound below k*Cin = 67. Depthwise sits at
    3 and a 4-group conv at 24; PReLU is a whole extra op per output element."""
    m = _model()
    for mod in m.modules():
        assert not isinstance(mod, (nn.PReLU, nn.BatchNorm2d, nn.LayerNorm, nn.GRU))
        if isinstance(mod, nn.Conv2d):
            assert mod.groups == 1, "grouped/depthwise conv is output-bound here"
            if mod.in_channels == 1:
                continue        # the stem reads a 1-channel spectrogram; k*Cin = 8
            k = mod.kernel_size[0] * mod.kernel_size[1]
            assert k * mod.in_channels >= 48, (mod, k * mod.in_channels)
