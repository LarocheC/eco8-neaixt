"""RT595Net — a streaming enhancer designed against the measured HiFi4 cost law.

The RT595's HiFi4 charges differently from the STM32N6's Neural-ART, and the
difference is not a detail. Measured on the Cadence ISS
(``deploy/rt595/iss/kernel_bench.cpp``, single int8 ops through the same TFLM
interpreter the firmware runs):

    cycles ~ MAC/3.2 + 20.9 * output_elements + 1300 per op

Three consequences shape every choice below.

**An output element costs about 67 MACs.** So an op is bound by its output count
rather than its arithmetic unless ``k * Cin > 67``. Depthwise convolution sits at
``k*Cin = 3`` and a 4-group conv at 24 — which is the whole reason they measure
0.17 and 0.87 MAC/cycle against a dense conv's 2.45. They are not slow per MAC;
they hardly do any MACs per output they still have to requantize. So: no
depthwise, no grouped convs, and every conv here has ``k*Cin >= 144``.

**Convolution over frequency is free.** 2.45 MAC/cycle against a dense matmul's
2.48. Reusing one kernel across all 256 bins costs nothing per MAC, so N6Net's
central thesis transfers to this chip intact — unlike its Neural-ART tiling.

**Op count is the real budget.** Applied to ConvFSENet 192-384 the law predicts
764 k cyc/frame against a measured 2,768 k: its convolutions are 28 % of its
time and the other 72 % is the ~160 non-conv ops in its 190-op graph. Hence
ReLU rather than PReLU (ReLU fuses into the conv; PReLU is a separate op over
every output element), a host-side FIFO so the temporal conv is one ``Conv2D``
with no ``Concat`` or ``Slice``, and no reshape or transpose anywhere.

This is the PyTorch twin of ``deploy/rt595/host/gen_arch_candidates.py``'s
``rt595b_fb_s4_C48_b3``, which measures **2,201,242 cyc/frame (0.69x the
3,168,000 real-time budget) in 16 ops with 64 KiB of weights** — 2.8x
ConvFSENet's arithmetic in 20 % fewer cycles.

The ``full_band`` branch is not optional in spirit. A stride-4 stem with
``k_f=3`` over three blocks reaches +-8 of 256 bins, which is N6Net v1's failure
exactly (9 of 257, and v1 scored 2.657 for 64 M MAC). It costs 2.6 % of cycles
here.
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

# The measured HiFi4 cost law, so the model can price itself.
HIFI4_MAC_PER_CYCLE = 3.20
HIFI4_CYCLES_PER_OUTPUT = 20.9
HIFI4_CYCLES_PER_OP = 1296
HIFI4_BUDGET_CYCLES = 3_168_000          # 16 ms hop at 198 MHz


class RT595Block(nn.Module):
    """Causal time conv -> frequency conv -> optional full-band context -> skip.

    Both convs are dense and ReLU-activated so the activation fuses into the
    conv on TFLite. ``conv_t`` is ``1 x k_t`` and left-padded only: at deploy the
    host hands the k_t-wide window in as a tensor and this is one valid ``Conv2D``.
    """

    def __init__(self, channels: int, k_t: int, k_f: int, full_band: bool = False):
        super().__init__()
        if k_f % 2 == 0:
            raise ValueError(f"k_f must be odd; got {k_f}")
        self.k_t, self.k_f = int(k_t), int(k_f)
        self.conv_t = nn.Conv2d(channels, channels, (1, k_t))
        self.conv_f = nn.Conv2d(channels, channels, (k_f, 1), padding=(k_f // 2, 0))
        self.ctx = nn.Conv2d(channels, channels, 1) if full_band else None

    @property
    def history(self):
        """Past columns the host must keep for this block."""
        return self.k_t - 1

    def mix(self, t):
        """Everything after the temporal conv; shared with the deploy graph."""
        f = F.relu(self.conv_f(t))
        if self.ctx is not None:
            # Mean over FREQUENCY ONLY. Averaging over time as well would read
            # future frames during offline training and silently break causality;
            # in the deploy graph W == 1 so the two agree.
            f = f + F.relu(self.ctx(f.mean(dim=2, keepdim=True)))
        return t + f

    def forward(self, x):
        return self.mix(F.relu(self.conv_t(F.pad(x, (self.history, 0, 0, 0)))))

    def forward_window(self, win):
        """Deploy form: (N, C, rows, k_t) window -> (N, C, rows, 1)."""
        return self.mix(F.relu(self.conv_t(win)))


class RT595Net(BaseModel):
    def __init__(self, n_fft, win_length, n_features, channels, n_blocks, k_t, k_f,
                 extractor_type, compress_factor, loss, preproc, postproc,
                 stem_stride=4, full_band_blocks=(0,)):
        super().__init__(loss, preproc, postproc)
        self.n_fft, self.win_length = int(n_fft), int(win_length)
        self.n_features = int(n_features)                    # 257 STFT bins
        self.rows_in = self.n_features - 1                   # 256; Nyquist is copied
        self.stem_stride = s = int(stem_stride)
        if self.rows_in % s:
            raise ValueError(f"stem_stride {s} must divide {self.rows_in}")
        self.rows = self.rows_in // s
        self.channels = int(channels)
        self.k_t, self.k_f = int(k_t), int(k_f)
        self.extractor_type = extractor_type
        self.compress_factor = compress_factor
        self.causal = True

        self.stem = nn.Conv2d(1, self.channels, (2 * s, 1), stride=(s, 1),
                              padding=(s // 2, 0))
        fb = set(full_band_blocks or ())
        self.blocks = nn.ModuleList([
            RT595Block(self.channels, self.k_t, self.k_f, full_band=(i in fb))
            for i in range(int(n_blocks))])
        self.head = nn.Conv2d(self.channels, s, (3, 1), padding=(1, 0))

    # --- feature / mask plumbing ------------------------------------------

    def features(self, mag):
        """(N, 1, 257, T) magnitude -> (N, 1, 256, T) compressed features."""
        mag = mag[:, :, :self.rows_in]
        if self.extractor_type == "mag_compressed":
            return compress_magnitude(mag, float(self.compress_factor or 0.3))
        if self.extractor_type == "mag":
            return mag
        raise ValueError(f"unsupported extractor_type {self.extractor_type!r}")

    @staticmethod
    def unfold(x, s):
        """(N, s, R, T) row-folded mask -> (N, 1, s*R + 1, T), Nyquist copied.

        A reshape on the host, where it is free; an upsampling op on the target,
        where it is not.
        """
        n, ks, r, t = x.shape
        y = x.view(n, ks // s, s, r, t).permute(0, 1, 3, 2, 4).reshape(n, ks // s, s * r, t)
        return torch.cat([y, y[:, :, -1:]], dim=2)

    @property
    def receptive_field_frames(self):
        return 1 + sum(b.history for b in self.blocks)

    def mask_half(self, feats):
        x = F.relu(self.stem(feats))
        for blk in self.blocks:
            x = blk(x)
        return torch.sigmoid(self.head(x))

    def forward(self, stft_noisy):
        mask = self.unfold(self.mask_half(self.features(stft_noisy.abs())), self.stem_stride)
        return stft_noisy * mask


class RT595Net_TD(TrainValidTest_TimeDomain, RT595Net):
    """RT595Net on the repo's time-domain objective."""
    pass


# --- cost accounting --------------------------------------------------------


def _conv_cost(model):
    """(MACs, output elements, conv op count) for one frame."""
    mac = out = ops = 0
    hooks, seen = [], {}

    def hook(mod, i, o):
        seen[mod] = (o.shape[2], o.shape[3])

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model.mask_half(torch.zeros(1, 1, model.rows_in, model.receptive_field_frames))
    for h in hooks:
        h.remove()
    for m, (h, w) in seen.items():
        k = m.kernel_size[0] * m.kernel_size[1]
        mac += h * k * m.in_channels * m.out_channels
        out += h * m.out_channels
        ops += 1
    return mac, out, ops


def hifi4_cycles(model):
    """Predicted HiFi4 cycles/frame from the measured cost law.

    Counts the convolutions plus one residual Add per block and the sigmoid; the
    ReLUs fuse into their convs and cost nothing. Validated against the ISS to
    within 10 % on this architecture (and to -72 % on ConvFSENet, whose 160
    shuffle ops this does not model — which is the point of the design).
    """
    mac, out, ops = _conv_cost(model)
    n_add = len(model.blocks) + sum(1 for b in model.blocks if b.ctx is not None)
    out += (n_add + 1) * model.rows * model.channels          # Adds + sigmoid
    ops += n_add + 1
    return (mac / HIFI4_MAC_PER_CYCLE + HIFI4_CYCLES_PER_OUTPUT * out
            + HIFI4_CYCLES_PER_OP * ops)


def macs_per_frame(model):
    return _conv_cost(model)[0]


def frequency_reach(model):
    """Output bins moved by perturbing ONE centre input bin.

    This is N6Net v1's measurement verbatim — perturb bin 128, count how many of
    the 257 output bins change — because that is the number that diagnosed v1
    (9 of 257) after it had already trained to a 2.657 ceiling. Note the
    direction: counting *input* bins that affect the output *somewhere* returns
    256 for any model with a strided stem and says nothing.
    """
    m = model.double().eval()
    torch.manual_seed(0)
    t = model.receptive_field_frames
    x = torch.rand(1, 1, model.rows_in, t, dtype=torch.float64)
    with torch.no_grad():
        base = m.mask_half(x)[..., -1]
        y = x.clone()
        y[:, :, model.rows_in // 2] += 1.0
        delta = (m.mask_half(y)[..., -1] - base).abs()
    m.float()
    # (1, stem_stride, rows) folded -> one entry per output bin
    return int((delta.reshape(-1) > 1e-12).sum().item())


def cost_summary(model):
    mac, _, _ = _conv_cost(model)
    params = sum(p.numel() for p in model.parameters())
    cyc = hifi4_cycles(model)
    return {
        "params": params,
        "macs_per_frame": mac,
        "weight_bytes_int8": params,
        "hifi4_cycles": cyc,
        "hifi4_budget_fraction": cyc / HIFI4_BUDGET_CYCLES,
        "receptive_field_frames": model.receptive_field_frames,
        "fifo_bytes_int8": sum(b.history for b in model.blocks) * model.rows * model.channels,
        "frequency_reach_bins": frequency_reach(model),
    }


def build_causal_model(h=None) -> RT595Net_TD:
    h = h or {}
    n_fft = int(h.get("n_fft", 512))
    win_length = int(h.get("win_length", n_fft))
    cf = float(h.get("compress_factor", 0.3))
    return RT595Net_TD(
        n_fft=n_fft, win_length=win_length,
        n_features=int(h.get("n_features", n_fft // 2 + 1)),
        channels=int(h.get("channels", 48)),
        n_blocks=int(h.get("n_blocks", 3)),
        k_t=int(h.get("k_t", 6)),
        k_f=int(h.get("k_f", 3)),
        extractor_type=h.get("extractor_type", "mag_compressed"),
        compress_factor=cf,
        stem_stride=int(h.get("stem_stride", 4)),
        full_band_blocks=h.get("full_band_blocks", [0]),
        loss=PreProcLoss(
            loss=DynCompMSE(normalize=True, normalize_framelen=n_fft,
                            normalize_threshold=0.025, exponent=cf, alpha=0.3),
            preproc=_TorchSpectrogram(n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 2),
        ),
        preproc=_TorchSpectrogram(n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 2),
        postproc=_TorchInverseSpectrogram(n_fft=n_fft, win_length=n_fft,
                                          hop_length=n_fft // 2),
    )
