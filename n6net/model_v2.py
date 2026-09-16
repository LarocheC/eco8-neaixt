"""N6Net-v2 — N6Net reshaped around what the Neural-ART compiler actually prices well.

The v1 compile study (deploy/stm32n6/N6NET_COMPILE.md) and single-conv probes
through stedgeai 4.0.1 gave four rules the v1 shape violates:

  * Kernels along FREQUENCY (k x 1, k = 3..7) run at 108-129 MAC per compute
    cycle. A 1 x 1 is capped at 96 MAC/cycle whatever its width, and any
    kernel along TIME (1 x 3, 3 x 3, v1's 1 x 6) drops to 48.
  * Elementwise-only epochs (PReLU expansion, adds) took 40 % of v1's NPU
    cycles for 0.5 % of its ops. ReLU fuses into the conv; PReLU does not
    (v1: 1.44 -> 1.12 ms from that swap alone).
  * Fewer frequency rows means fewer memory stalls per MAC.
  * The FIFO must never be a Concat (it runs on the M55): each past column is
    its own graph input, and the host owns the ring buffer.

Shape
-----
    |X|^0.3 on 256 bins (Nyquist dropped on the host)
    Stem   Conv 4x1 stride 2, 1->C                      -> 128 rows x C
    B x  [ Conv (k_f x 3, dilation d_b) over time       (d_b = 1, 2, 4, 8)
           ReLU
           Conv k_f x 1                                 (spectral)
           ReLU
           + skip ]
    Head   Conv 3x1, C->2 -> sigmoid -> interleave to 256 rows, Nyquist copied

Offline, the temporal layer is one dilated conv with a left-only pad. In the
deploy graph it is the SAME weights as three k_f x 1 convs, one per past
column {t-2d, t-d, t}, summed. Every heavy op is then a frequency kernel, and
dilation is only a matter of which ring-buffer slots the host passes in.
Receptive field: 1 + 2 * sum(d_b) = 31 frames (~0.5 s at a 16 ms hop).
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


class N6BlockV2(nn.Module):
    """Dilated causal time x frequency conv, spectral conv, both ReLU, plus skip."""

    TAPS = 3

    def __init__(self, channels: int, dilation: int, k_f: int = 5):
        super().__init__()
        if k_f % 2 == 0:
            raise ValueError(f"k_f must be odd; got {k_f}")
        self.dilation = int(dilation)
        self.k_f = int(k_f)
        self.conv_t = nn.Conv2d(channels, channels, (k_f, self.TAPS),
                                dilation=(1, self.dilation), padding=(k_f // 2, 0))
        self.conv_f = nn.Conv2d(channels, channels, (k_f, 1), padding=(k_f // 2, 0))

    @property
    def offsets(self):
        """Past-frame offsets the temporal kernel reads, oldest first (0 = now)."""
        return [self.dilation * (self.TAPS - 1 - m) for m in range(self.TAPS)]

    @property
    def history(self):
        """Past columns the host must keep for this block."""
        return self.dilation * (self.TAPS - 1)

    def forward(self, x):
        y = F.pad(x, (self.history, 0, 0, 0))            # causal: left pad only
        y = F.relu(self.conv_t(y))
        y = F.relu(self.conv_f(y))
        return x + y


class N6NetV2(BaseModel):
    def __init__(self, n_fft, win_length, n_features, channels, dilations, k_f,
                 extractor_type, compress_factor, loss, preproc, postproc):
        super().__init__(loss, preproc, postproc)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.n_features = int(n_features)                 # 257 STFT bins
        self.rows_in = self.n_features - 1                # 256: Nyquist dropped
        if self.rows_in % 2:
            raise ValueError("N6NetV2 needs an even number of sub-Nyquist bins")
        self.rows = self.rows_in // 2                     # 128 after the stem
        self.channels = int(channels)
        self.dilations = [int(d) for d in dilations]
        self.k_f = int(k_f)
        self.extractor_type = extractor_type
        self.compress_factor = compress_factor
        self.causal = True

        self.stem = nn.Conv2d(1, self.channels, (4, 1), stride=(2, 1), padding=(1, 0))
        self.blocks = nn.ModuleList([N6BlockV2(self.channels, d, self.k_f)
                                     for d in self.dilations])
        self.head = nn.Conv2d(self.channels, 2, (3, 1), padding=(1, 0))

    def features(self, mag):
        """(N, 1, 257, T) magnitude -> (N, 1, 256, T) compressed features."""
        mag = mag[:, :, :self.rows_in]
        if self.extractor_type == "mag_compressed":
            return compress_magnitude(mag, float(self.compress_factor or 0.3))
        if self.extractor_type == "mag":
            return mag
        raise ValueError(f"unsupported extractor_type {self.extractor_type!r}")

    @staticmethod
    def interleave(mask2):
        """(N, 2, R, T) half-resolution mask -> (N, 1, 2R + 1, T), Nyquist copied.

        Done on the host at deployment — a reshape there is free, where an
        upsampling op on the NPU is not.
        """
        n, _, r, t = mask2.shape
        m = mask2.permute(0, 2, 1, 3).reshape(n, 1, 2 * r, t)
        return torch.cat([m, m[:, :, -1:]], dim=2)

    @property
    def receptive_field_frames(self):
        return 1 + sum(b.history for b in self.blocks)

    def mask_half(self, feats):
        """(N, 1, 256, T) features -> (N, 2, 128, T) sigmoid mask."""
        x = self.stem(feats)
        for blk in self.blocks:
            x = blk(x)
        return torch.sigmoid(self.head(x))

    def forward(self, stft_noisy):
        mask = self.interleave(self.mask_half(self.features(stft_noisy.abs())))
        return stft_noisy * mask


class N6NetV2_TD(TrainValidTest_TimeDomain, N6NetV2):
    pass


def macs_per_frame(model: N6NetV2) -> int:
    """MACs for one new frame: every tap of every kernel, at its output height."""
    heights = {}
    hooks = [m.register_forward_hook(lambda mod, i, o: heights.__setitem__(mod, o.shape[2]))
             for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model.mask_half(torch.zeros(1, 1, model.rows_in, model.receptive_field_frames))
    for h in hooks:
        h.remove()
    return sum(heights[m] * m.kernel_size[0] * m.kernel_size[1]
               * (m.in_channels // m.groups) * m.out_channels for m in heights)


def cost_summary(model: N6NetV2) -> dict:
    macs = macs_per_frame(model)
    wb = sum(p.numel() for n, p in model.named_parameters() if "bias" not in n)
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "weight_bytes_int8": wb,
        "macs_per_frame": macs,
        "arithmetic_intensity": macs / wb,
        "receptive_field_frames": model.receptive_field_frames,
        "fifo_bytes_int8": sum(b.history for b in model.blocks) * model.rows * model.channels,
    }


def build_causal_model(h=None) -> N6NetV2_TD:
    h = h or {}
    n_fft = int(h.get("n_fft", 512))
    win_length = int(h.get("win_length", n_fft))
    hop_length = int(h.get("hop_size", n_fft // 2))
    preproc = _TorchSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    postproc = _TorchInverseSpectrogram(n_fft=n_fft, win_length=win_length,
                                        hop_length=hop_length)
    loss = PreProcLoss(
        loss=DynCompMSE(normalize=True, normalize_mode="threshold",
                        normalize_framelen=n_fft, normalize_threshold=0.025),
        preproc=_TorchSpectrogram(n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 2),
    )
    return N6NetV2_TD(
        n_fft=n_fft, win_length=win_length,
        n_features=int(h.get("n_features", n_fft // 2 + 1)),
        channels=int(h.get("channels", 128)),
        dilations=h.get("dilations", [1, 2, 4, 8]),
        k_f=int(h.get("k_f", 5)),
        extractor_type=h.get("extractor_type", "mag_compressed"),
        compress_factor=h.get("compress_factor", 0.3),
        loss=loss, preproc=preproc, postproc=postproc,
    )


if __name__ == "__main__":
    for c in (96, 128, 192):
        s = cost_summary(build_causal_model({"channels": c}))
        print(f"C={c}: params={s['params']:,}  MAC/frame={s['macs_per_frame'] / 1e6:.1f} M  "
              f"RF={s['receptive_field_frames']}  FIFO={s['fifo_bytes_int8'] / 1024:.0f} KiB")
