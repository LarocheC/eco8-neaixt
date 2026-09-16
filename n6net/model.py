"""N6Net — an NSNet designed around Neural-ART silicon rather than made compatible with it.

The thesis, and the reason this model looks backwards under the usual TinyML
metric: NSNet2 at batch 1 runs one MAC per weight byte. Every weight is fetched
from memory, used once, and discarded — arithmetic intensity 1, one to two
orders of magnitude below any accelerator's compute roof. Measured on the
STM32N6 that shows up directly: memory-bound at ~27% core utilization streaming
weights from external flash, and 1.62x recovered purely by moving them on-chip.

So this architecture deliberately spends MANY cheap, highly-reused MACs instead
of few expensive memory-bound ones:

    NSNet2-ish          ~2.8 M params, ~2.8 M MAC/frame, reuse ~1x,  GEMV
    N6Net (3 blocks)    ~249 k params, ~64 M MAC/frame,  reuse 257x, dense conv

10x fewer parameters, 20x more MACs. Every weight in a temporal or spectral
layer is reused at all 257 frequency bins, which is what turns the workload into
something the four CONV_ACC units can actually fill.

Layout
------
``[N, C, H=frequency, W=time]``. Frequency is the HEIGHT axis on purpose: ST
constrains ``feature_width * C_in <= 2048``, so time-on-width gives ``6*96=576``
(comfortable) where frequency-on-width would give ``257*96=24672`` and force the
compiler to split into columns.

Shape of the network
--------------------
    Stem   Conv 3x1, 1->C
    3x   [ Conv 1x6, C->C   (temporal, causal: current frame + 5 FIFO frames)
           PReLU
           Conv 3x1, C->C   (spectral, symmetric padding over frequency)
           PReLU
           + skip ]
    Head   Conv 1x1, C->1  -> sigmoid -> 257-bin mask

No normalization, no GRU, no depthwise convolution, no dilation.

Why C = 96
----------
ST's mapping rules want ``OCH`` in multiples of 24, preferably ``M*24``, and
document that 128->96 costs the same as 32->96 because the four CONV_ACC units
run in parallel. 96 = 4*24 is the smallest clean four-way output tiling. 72 is
3*24 and plausibly leaves one path idle; 120 needs a fifth iteration. The width
is a config knob so 72/96/120/128 can be swept against the compiler's own
heuristics, but 96 is the designed point.

Kernel sizes are likewise chosen from the mapping rules, not from habit: ST
recommends kernel widths 3 or 6 at stride 1 and calls out kernels with one
dimension equal to 1 as efficient special cases. Both convolutions here are of
that form.

Temporal context comes from the FIFO rather than a recurrence: three blocks of
``k_t = 6`` give a receptive field of ``1 + 3*(6-1) = 16`` frames, about 256 ms
at a 16 ms hop — enough that the proof-of-concept needs no GRU, and adding one
would drag the architecture back toward the GEMV regime this exists to leave.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from convfsenet.model import (
    BaseModel,
    DynCompMSE,
    PreProcLoss,
    TrainValidTest_TimeDomain,
    _TorchInverseSpectrogram,
    _TorchSpectrogram,
    compress_magnitude,
)


def _activation(kind: str, channels: int) -> nn.Module:
    """PReLU per the design; ReLU available for the int8-friendliness ablation."""
    if kind == "prelu":
        return nn.PReLU(num_parameters=channels)
    if kind == "relu":
        return nn.ReLU()
    raise ValueError(f"unknown activation {kind!r}; expected 'prelu' or 'relu'")


class N6Block(nn.Module):
    """Temporal mix (causal, k_t frames) then spectral mix (k_f bins), plus skip.

    The temporal convolution is where the reuse lives: one ``k_t * C * C`` weight
    set is applied at every one of the 257 frequency rows. At C=96, k_t=6 that is
    55,296 weights doing 14.2 M MACs — 257x reuse, and an arithmetic intensity
    around 62 MAC/byte against NSNet2's 1.

    Causality is structural: the kernel spans ``[t-k_t+1 .. t]`` and nothing to
    the right of ``t``, enforced offline by a left-only pad and at inference by
    a FIFO holding exactly ``k_t - 1`` past frames.
    """

    def __init__(self, channels: int, k_t: int = 6, k_f: int = 3,
                 activation: str = "prelu"):
        super().__init__()
        self.channels = int(channels)
        self.k_t = int(k_t)
        self.k_f = int(k_f)
        if self.k_f % 2 == 0:
            raise ValueError(f"k_f must be odd for symmetric padding; got {k_f}")
        # Temporal: kernel (H=1, W=k_t). No padding here — forward() pads left
        # only, so the layer itself never sees a future frame.
        self.conv_t = nn.Conv2d(channels, channels, kernel_size=(1, self.k_t))
        self.act_t = _activation(activation, channels)
        # Spectral: kernel (H=k_f, W=1), symmetric padding over frequency.
        # Frequency is not a causal axis, so centring the kernel is correct.
        self.conv_f = nn.Conv2d(channels, channels, kernel_size=(self.k_f, 1),
                                padding=(self.k_f // 2, 0))
        self.act_f = _activation(activation, channels)

    @property
    def fifo_frames(self) -> int:
        """Past frames this block must retain between calls."""
        return self.k_t - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, F, T) -> (N, C, F, T). Offline/training path."""
        # Left-only pad over the time (W) axis: causal by construction.
        y = F.pad(x, (self.fifo_frames, 0, 0, 0))
        y = self.act_t(self.conv_t(y))
        y = self.act_f(self.conv_f(y))
        return y + x

    def forward_window(self, x_win: torch.Tensor, x_cur: torch.Tensor) -> torch.Tensor:
        """Streaming step. x_win: (N, C, F, k_t) FIFO+current; x_cur: (N, C, F, 1).

        Identical arithmetic to forward() on its last output column — the FIFO
        supplies exactly the frames the offline left-pad would have supplied.
        """
        y = self.act_t(self.conv_t(x_win))          # (N, C, F, 1)
        y = self.act_f(self.conv_f(y))
        return y + x_cur


class N6Net(BaseModel):
    """Causal STFT mask predictor built from N6Block stacks."""

    def __init__(self, n_fft, win_length, n_features, channels, n_blocks,
                 k_t, k_f, activation, extractor_type, compress_factor,
                 loss, preproc, postproc):
        super().__init__(loss, preproc, postproc)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.n_features = int(n_features)
        self.channels = int(channels)
        self.n_blocks = int(n_blocks)
        self.k_t = int(k_t)
        self.k_f = int(k_f)
        self.extractor_type = extractor_type
        self.compress_factor = compress_factor
        self.causal = True                       # structural, not a config choice

        # Stem: 1 -> C over frequency. The magnitude spectrogram arrives already
        # shaped (N, 1, F, T), so the single input channel is free.
        self.stem = nn.Conv2d(1, self.channels, kernel_size=(self.k_f, 1),
                              padding=(self.k_f // 2, 0))
        self.blocks = nn.ModuleList([
            N6Block(self.channels, self.k_t, self.k_f, activation)
            for _ in range(self.n_blocks)
        ])
        # Head: 1x1 back down to a single mask channel.
        self.head = nn.Conv2d(self.channels, 1, kernel_size=1)

    # -- feature / mask plumbing -------------------------------------------

    def features(self, mag: torch.Tensor) -> torch.Tensor:
        """(N, 1, F, T) magnitude -> compressed features, same shape.

        Power-law compression squashes the 60+ dB STFT range into an
        int8-friendly domain; ConvFSENet's int8 results show it is what makes
        static PTQ near loss-free on this kind of input, and it must stay FP32
        at deployment.
        """
        if self.extractor_type == "mag_compressed":
            return compress_magnitude(mag, float(self.compress_factor or 0.3))
        if self.extractor_type == "mag":
            return mag
        raise ValueError(f"unsupported extractor_type {self.extractor_type!r}")

    @property
    def receptive_field_frames(self) -> int:
        return 1 + self.n_blocks * (self.k_t - 1)

    def forward(self, stft_noisy: torch.Tensor) -> torch.Tensor:
        """stft_noisy: (N, 1, F, T) complex -> (N, 1, F, T) complex."""
        feats = self.features(stft_noisy.abs())          # (N, 1, F, T)
        x = self.stem(feats)                             # (N, C, F, T)
        for blk in self.blocks:
            x = blk(x)
        mask = torch.sigmoid(self.head(x))               # (N, 1, F, T)
        return stft_noisy * mask


class N6Net_TD(TrainValidTest_TimeDomain, N6Net):
    """N6Net trained end-to-end on the time-domain objective."""
    pass


# ---------------------------------------------------------------------------
# Cost accounting — the point of the architecture, so it is measured, not quoted
# ---------------------------------------------------------------------------


def macs_per_frame(model: N6Net) -> int:
    """Multiply-accumulates for ONE new frame of output.

    A Conv2d whose kernel spans the whole time window emits one column per call,
    so its per-frame cost is ``H_out * k_f * k_t * C_in * C_out``. Frequency
    rows are all computed every frame; time is the streaming axis.
    """
    F_ = model.n_features
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            kh, kw = m.kernel_size
            # Height is preserved by symmetric padding on every conv here.
            total += F_ * kh * kw * (m.in_channels // m.groups) * m.out_channels
    return total


def weight_bytes(model: N6Net) -> int:
    """int8 weight footprint — the quantity NSNet2 is actually bound by."""
    return sum(p.numel() for n, p in model.named_parameters() if "bias" not in n)


def cost_summary(model: N6Net) -> dict:
    macs = macs_per_frame(model)
    wb = weight_bytes(model)
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "weight_bytes_int8": wb,
        "macs_per_frame": macs,
        "arithmetic_intensity": macs / wb,
        "receptive_field_frames": model.receptive_field_frames,
        "fifo_bytes_int8": sum(
            b.fifo_frames * model.n_features * model.channels for b in model.blocks
        ),
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_causal_model(h=None) -> N6Net_TD:
    """Build N6Net from a config dict. Causal by construction — no knob for it."""
    h = h or {}
    n_fft = int(h.get("n_fft", 512))
    win_length = int(h.get("win_length", n_fft))
    hop_length = int(h.get("hop_size", n_fft // 2))
    n_features = int(h.get("n_features", n_fft // 2 + 1))

    preproc = _TorchSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    postproc = _TorchInverseSpectrogram(n_fft=n_fft, win_length=win_length,
                                        hop_length=hop_length)
    loss = PreProcLoss(
        loss=DynCompMSE(normalize=True, normalize_mode="threshold",
                        normalize_framelen=n_fft, normalize_threshold=0.025),
        preproc=_TorchSpectrogram(n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 2),
    )
    channels = int(h.get("channels", 96))
    if channels % 24:
        # Not fatal — the sweep deliberately includes 72/120/128 — but the
        # four-way CONV_ACC tiling wants multiples of 24.
        print(f"N6Net: channels={channels} is not a multiple of 24; "
              f"expect incomplete CONV_ACC output tiling on Neural-ART.")
    return N6Net_TD(
        n_fft=n_fft, win_length=win_length, n_features=n_features,
        channels=channels,
        n_blocks=int(h.get("n_blocks", 3)),
        k_t=int(h.get("k_t", 6)),
        k_f=int(h.get("k_f", 3)),
        activation=h.get("activation", "prelu"),
        extractor_type=h.get("extractor_type", "mag_compressed"),
        compress_factor=h.get("compress_factor", 0.3),
        loss=loss, preproc=preproc, postproc=postproc,
    )


if __name__ == "__main__":
    for nb in (1, 3):
        m = build_causal_model({"n_blocks": nb})
        c = cost_summary(m)
        print(f"n_blocks={nb}: params={c['params']:,}  MAC/frame={c['macs_per_frame']:,}  "
              f"AI={c['arithmetic_intensity']:.0f} MAC/B  "
              f"RF={c['receptive_field_frames']} frames  "
              f"FIFO={c['fifo_bytes_int8']/1024:.0f} KiB")
