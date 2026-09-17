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

Frequency reach
---------------
The plain shape lets the centre bin see 72 of 256 input bins (+-1.1 kHz).
``full_band`` adds a ``FullBand`` branch (on ``full_band_blocks``) that reduces
the frequency axis to one row and broadcast-adds it back, so every bin sees the
whole spectrum; ``freq_pos_emb`` adds a learned per-row offset so the network
can tell bands apart. ``configs/n6net_v2_fullband.json`` (one position-aware
``pool`` branch + embedding) reaches 256/256 bins for +0.7 % MAC and ~+10 %
estimated NPU time. ``stem_stride=4`` (64 rows, 4 mask channels) doubles the
local reach instead, but costs weights and NPU time.
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


class FullBand(nn.Module):
    """Full-band context for every frequency row, at almost no MAC cost.

    Dilated convs would be the textbook way to widen the frequency reach, but
    Neural-ART runs them in software, and upsampling ops (Resize,
    ConvTranspose, H-only pixel shuffle) are partly software too. What does map
    entirely to HW is reducing the frequency axis to one row and
    broadcast-adding the result back:

    * ``"gap"``  — mean over frequency, 1x1 conv, ReLU. Position-agnostic
      summary; C^2 MAC per frame.
    * ``"pool"`` — two 4x1 stride-4 convs (128 -> 32 -> 8 rows at width
      ``c_g``), then a full-height conv to one row, ReLU. Position-aware:
      each input band has its own weights.

    Either way, every output row sees every input row.
    """

    def __init__(self, kind: str, channels: int, rows: int, c_g: int):
        super().__init__()
        self.kind = kind
        if kind == "gap":
            self.proj = nn.Conv2d(channels, channels, 1)
        elif kind == "pool":
            if rows % 16:
                raise ValueError(f"'pool' full-band branch needs rows % 16 == 0; got {rows}")
            self.d1 = nn.Conv2d(channels, c_g, (4, 1), stride=(4, 1))
            self.d2 = nn.Conv2d(c_g, c_g, (4, 1), stride=(4, 1))
            self.fh = nn.Conv2d(c_g, channels, (rows // 16, 1))
        else:
            raise ValueError(f"unknown full-band branch {kind!r}")

    def forward(self, y):
        if self.kind == "gap":
            # With one time column (the deploy graph) average over both axes:
            # ReduceMean over H alone compiles to a hybrid Transpose epoch,
            # over (H, W) it becomes a HW GlobalAveragePool. Same numbers.
            dims = (2, 3) if y.shape[3] == 1 else 2
            return F.relu(self.proj(y.mean(dim=dims, keepdim=True)))
        return F.relu(self.fh(F.relu(self.d2(F.relu(self.d1(y))))))


class N6BlockV2(nn.Module):
    """Dilated causal time x frequency conv, spectral conv, both ReLU, plus skip,
    and optionally a full-band branch (``FullBand``) on the block output."""

    TAPS = 3

    def __init__(self, channels: int, dilation: int, k_f: int = 5,
                 full_band: str | None = None, rows: int = 128, c_g: int = 48):
        super().__init__()
        if k_f % 2 == 0:
            raise ValueError(f"k_f must be odd; got {k_f}")
        self.dilation = int(dilation)
        self.k_f = int(k_f)
        self.conv_t = nn.Conv2d(channels, channels, (k_f, self.TAPS),
                                dilation=(1, self.dilation), padding=(k_f // 2, 0))
        self.conv_f = nn.Conv2d(channels, channels, (k_f, 1), padding=(k_f // 2, 0))
        self.full_band = FullBand(full_band, channels, rows, c_g) if full_band else None

    def mix(self, x, y):
        """Everything after the temporal conv + ReLU; shared with the deploy graph."""
        y = F.relu(self.conv_f(y))
        if self.full_band is not None:
            y = y + self.full_band(y)
        return x + y

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
        return self.mix(x, F.relu(self.conv_t(y)))


class N6NetV2(BaseModel):
    def __init__(self, n_fft, win_length, n_features, channels, dilations, k_f,
                 extractor_type, compress_factor, loss, preproc, postproc,
                 stem_stride=2, full_band=None, full_band_width=48, freq_pos_emb=False,
                 full_band_blocks=None):
        super().__init__(loss, preproc, postproc)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.n_features = int(n_features)                 # 257 STFT bins
        self.rows_in = self.n_features - 1                # 256: Nyquist dropped
        self.stem_stride = int(stem_stride)
        if self.rows_in % self.stem_stride:
            raise ValueError(f"{self.rows_in} bins not divisible by stem_stride {stem_stride}")
        self.rows = self.rows_in // self.stem_stride      # 128 (stride 2) or 64 (stride 4)
        self.channels = int(channels)
        self.dilations = [int(d) for d in dilations]
        self.k_f = int(k_f)
        self.extractor_type = extractor_type
        self.compress_factor = compress_factor
        self.causal = True

        s = self.stem_stride
        self.stem = nn.Conv2d(1, self.channels, (2 * s, 1), stride=(s, 1), padding=(s // 2, 0))
        # Learned per-row offset: lets convs and the broadcast full-band
        # context know WHERE in the spectrum they are. Small random init, not
        # zeros: an all-zero Add is folded away by the toolchain, which would
        # make an untrained compile unrepresentative.
        self.pos = (nn.Parameter(0.02 * torch.randn(1, self.channels, self.rows, 1))
                    if freq_pos_emb else None)
        # Which blocks get the full-band branch (default: all). One is enough
        # for full reach; each costs two tensor-wide passes (pool + broadcast add).
        fb_at = set(range(len(self.dilations)) if full_band_blocks is None else full_band_blocks)
        self.blocks = nn.ModuleList([
            N6BlockV2(self.channels, d, self.k_f, full_band if i in fb_at else None,
                      self.rows, int(full_band_width))
            for i, d in enumerate(self.dilations)])
        # One mask channel per input bin folded into each row; the host un-folds.
        self.head = nn.Conv2d(self.channels, s, (3, 1), padding=(1, 0))

    def features(self, mag):
        """(N, 1, 257, T) magnitude -> (N, 1, 256, T) compressed features."""
        mag = mag[:, :, :self.rows_in]
        if self.extractor_type == "mag_compressed":
            return compress_magnitude(mag, float(self.compress_factor or 0.3))
        if self.extractor_type == "mag":
            return mag
        raise ValueError(f"unsupported extractor_type {self.extractor_type!r}")

    @staticmethod
    def interleave(mask_s):
        """(N, s, R, T) folded mask -> (N, 1, s*R + 1, T), Nyquist copied.

        Done on the host at deployment — a reshape there is free, where an
        upsampling op on the NPU is not (Resize/ConvTranspose map partly to SW).
        """
        n, s, r, t = mask_s.shape
        m = mask_s.permute(0, 2, 1, 3).reshape(n, 1, s * r, t)
        return torch.cat([m, m[:, :, -1:]], dim=2)

    @property
    def receptive_field_frames(self):
        return 1 + sum(b.history for b in self.blocks)

    def embed(self, x):
        return x if self.pos is None else x + self.pos

    def mask_half(self, feats):
        """(N, 1, 256, T) features -> (N, stem_stride, rows, T) sigmoid mask."""
        x = self.embed(self.stem(feats))
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


def frequency_reach(model: N6NetV2) -> int:
    """Input bins that can move the centre output bin (perturbation, fp64)."""
    rows_in = model.rows_in
    m = model.double().eval()
    torch.manual_seed(0)
    x = torch.rand(1, 1, rows_in, 1, dtype=torch.float64)
    centre = (rows_in // 2) // model.stem_stride           # the row holding bin rows_in//2
    reach = 0
    with torch.no_grad():
        base = m.mask_half(x)[:, :, centre]
        for b in range(rows_in):
            y = x.clone()
            y[:, :, b] += 1.0
            reach += bool((m.mask_half(y)[:, :, centre] - base).abs().max() > 0)
    model.float()
    return reach


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
        "frequency_reach_bins": frequency_reach(model),
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
        stem_stride=int(h.get("stem_stride", 2)),
        full_band=h.get("full_band"),
        full_band_width=int(h.get("full_band_width", 48)),
        freq_pos_emb=bool(h.get("freq_pos_emb", False)),
        full_band_blocks=h.get("full_band_blocks"),
    )


if __name__ == "__main__":
    for c in (96, 128, 192):
        s = cost_summary(build_causal_model({"channels": c}))
        print(f"C={c}: params={s['params']:,}  MAC/frame={s['macs_per_frame'] / 1e6:.1f} M  "
              f"RF={s['receptive_field_frames']}  FIFO={s['fifo_bytes_int8'] / 1024:.0f} KiB")
