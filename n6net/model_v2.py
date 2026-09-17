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

MP-SENet ingredients (all config switches, all verified to map to NPU HW)
--------------------------------------------------------------------------
``input_features="mag_gd_ifd"``  feed (|X|^0.3, group delay, IF-difference)
    instead of magnitude alone — LiSenNet's shift-invariant phase features,
    computed on the host. Stem 1 -> 3 input channels, +0.1 % MAC.
``mask_act="lsigmoid"``  MP-SENet's learnable sigmoid: ``beta * sigmoid(a_f x)``
    with one learned slope per bin and ``beta`` = 2, so the mask can exceed 1.
``phase_head=True``  a second head predicts (cos, sin) of a phase *correction*
    per bin; the host normalises it and rotates the noisy STFT, so there is no
    atan2 anywhere in the deployed path. Trained with MP-SENet's anti-wrapping
    IP/GD/IAF phase loss, so it needs ``objective="mpsenet"``.
``objective="mpsenet"``  train on the shared MP-SENet loss (common/losses.py):
    magnitude + complex + STFT-consistency + time-L1 (+ phase), with the
    MetricGAN term still added by the trainer from ``gan.metric_loss_lambda``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from common.losses import (
    generator_loss, generator_terms, loss_weights, mag_pha_from_complex,
)
from convfsenet.model import (
    BaseModel,
    DynCompMSE,
    PreProcLoss,
    TrainValidTest_TimeDomain,
    _TorchInverseSpectrogram,
    _TorchSpectrogram,
    compress_magnitude,
)


def native_full_height(fh: nn.Conv2d, stride: int = 4) -> nn.Sequential:
    """Exact rewrite of a full-height (k x 1, valid) conv into native shapes.

    The compiler splits kernels like the 8x1 into masked 1x1 sub-convs, which
    hangs the NPU. The same function as two native convs: a (stride x 1)
    stride-``stride`` conv whose output holds one channel block per kernel
    segment (segment j applied to every input segment), then an (m x 1) conv
    with fixed 0/1 weights that keeps block j at row j and adds the bias.
    Linear in between, so it matches the original exactly (up to float
    rounding); only the export uses it, so trained weights carry over.
    """
    c_out, c_in, k, kw = fh.weight.shape
    if kw != 1 or k % stride:
        raise ValueError(f"need a (k x 1) kernel with k divisible by {stride}; got {k}x{kw}")
    m = k // stride
    kw_ = {"device": fh.weight.device, "dtype": fh.weight.dtype}
    a = nn.Conv2d(c_in, m * c_out, (stride, 1), stride=(stride, 1), bias=False, **kw_)
    b = nn.Conv2d(m * c_out, c_out, (m, 1), **kw_)
    with torch.no_grad():
        a.weight.copy_(torch.cat([fh.weight[:, :, j * stride:(j + 1) * stride]
                                  for j in range(m)], dim=0))
        b.weight.zero_()
        for j in range(m):
            b.weight[torch.arange(c_out), j * c_out + torch.arange(c_out), j, 0] = 1.0
        b.bias.copy_(fh.bias if fh.bias is not None else torch.zeros(c_out))
    return nn.Sequential(a, b)


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
      each input band has its own weights. **Hangs the NPU on target**: the
      compiler rewrites the 8x1 full-height kernel into eight masked 1x1
      sub-convs (see deploy/stm32n6/host/fullband_probes.py).
    * ``"pool_native"`` — the same idea with no kernel the compiler rewrites:
      three 4x1 stride-4 convs (128 -> 32 -> 8 -> 2 rows) and a 2x1 conv to
      one row. Still position-aware: the strided convs do not overlap, so
      every input row reaches the output through its own weights.

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
            # attribute names are checkpoint keys: keep d1 / d2 / fh
            self.d1 = nn.Conv2d(channels, c_g, (4, 1), stride=(4, 1))
            self.d2 = nn.Conv2d(c_g, c_g, (4, 1), stride=(4, 1))
            self.fh = nn.Conv2d(c_g, channels, (rows // 16, 1))
        elif kind == "pool_native":
            if rows % 64:
                raise ValueError(f"'pool_native' needs rows % 64 == 0; got {rows}")
            self.d1 = nn.Conv2d(channels, c_g, (4, 1), stride=(4, 1))
            self.d2 = nn.Conv2d(c_g, c_g, (4, 1), stride=(4, 1))
            self.d3 = nn.Conv2d(c_g, c_g, (4, 1), stride=(4, 1))
            self.fh = nn.Conv2d(c_g, channels, (rows // 64, 1))
        else:
            raise ValueError(f"unknown full-band branch {kind!r}")

    def use_native_rewrite(self):
        """Swap the 8x1 full-height conv of a ``pool`` branch for its exact
        native equivalent (``native_full_height``). Export-time only."""
        if self.kind == "pool" and not isinstance(self.fh, nn.Sequential):
            self.fh = native_full_height(self.fh)
            return True
        return False

    def forward(self, y):
        if self.kind == "gap":
            # With one time column (the deploy graph) average over both axes:
            # ReduceMean over H alone compiles to a hybrid Transpose epoch,
            # over (H, W) it becomes a HW GlobalAveragePool. Same numbers.
            dims = (2, 3) if y.shape[3] == 1 else 2
            return F.relu(self.proj(y.mean(dim=dims, keepdim=True)))
        for name in ("d1", "d2", "d3", "fh"):
            if hasattr(self, name):
                y = F.relu(getattr(self, name)(y))
        return y


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
                 full_band_blocks=None, input_features="mag", mask_act="sigmoid",
                 mask_beta=2.0, phase_head=False, hop_length=None):
        super().__init__(loss, preproc, postproc)
        if input_features not in ("mag", "mag_gd_ifd"):
            raise ValueError(f"unknown input_features {input_features!r}")
        if mask_act not in ("sigmoid", "lsigmoid"):
            raise ValueError(f"unknown mask_act {mask_act!r}")
        self.input_features = input_features
        self.in_channels = 3 if input_features == "mag_gd_ifd" else 1
        self.mask_act = mask_act
        self.mask_beta = float(mask_beta)
        self.phase_head = bool(phase_head)
        self.hop_length = int(hop_length or n_fft // 2)   # IFD hop correction
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
        self.stem = nn.Conv2d(self.in_channels, self.channels, (2 * s, 1), stride=(s, 1),
                              padding=(s // 2, 0))
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
        # Learnable sigmoid (MP-SENet): one slope per output bin, beta * sigmoid(a * x).
        self.mask_slope = (nn.Parameter(torch.ones(1, s, self.rows, 1))
                           if mask_act == "lsigmoid" else None)
        # Phase head: (cos, sin) of a per-bin phase correction, channels
        # [0:s] = cos, [s:2s] = sin. Starts near the identity rotation (bias
        # (1, 0), small weights) so training begins from the noisy phase.
        if self.phase_head:
            self.pha_head = nn.Conv2d(self.channels, 2 * s, (3, 1), padding=(1, 0))
            with torch.no_grad():
                self.pha_head.weight.mul_(0.1)
                self.pha_head.bias.zero_()
                self.pha_head.bias[:s] = 1.0
        else:
            self.pha_head = None

    def features(self, stft):
        """(N, 1, 257, T) complex STFT -> (N, in_channels, 256, T) network input.

        Channel 0 is the compressed magnitude. With ``input_features="mag_gd_ifd"``
        channels 1-2 are LiSenNet's group delay and instantaneous-frequency
        difference (anti-wrapped, /pi), both causal: GD reads the current frame,
        IFD the current and previous one. At deployment the host computes them
        from the STFT it already has; the previous frame's phase is host state.
        """
        stft = stft[:, :, :self.rows_in]
        mag = stft.abs()
        if self.extractor_type == "mag_compressed":
            mag = compress_magnitude(mag, float(self.compress_factor or 0.3))
        elif self.extractor_type != "mag":
            raise ValueError(f"unsupported extractor_type {self.extractor_type!r}")
        if self.input_features == "mag":
            return mag
        pha = stft.angle()
        gd = torch.diff(pha, dim=2, prepend=torch.zeros_like(pha[:, :, :1]))
        ifd = torch.diff(pha, dim=3, prepend=torch.zeros_like(pha[..., :1]))
        k = torch.arange(pha.shape[2], device=pha.device, dtype=pha.dtype)
        ifd = ifd - 2 * torch.pi * (self.hop_length / self.n_fft) * k[None, None, :, None]
        wrap = lambda x: torch.atan2(x.sin(), x.cos())
        return torch.cat([mag, wrap(gd) / torch.pi, wrap(ifd) / torch.pi], dim=1)

    @staticmethod
    def unfold(x, s):
        """(N, k*s, R, T) row-folded output -> (N, k, s*R + 1, T), Nyquist copied.

        Channel ``i*s + j`` of row ``r`` is bin ``s*r + j`` of quantity ``i``.
        Done on the host at deployment — a reshape there is free, where an
        upsampling op on the NPU is not (Resize/ConvTranspose map partly to SW).
        """
        n, ks, r, t = x.shape
        y = x.view(n, ks // s, s, r, t).permute(0, 1, 3, 2, 4).reshape(n, ks // s, s * r, t)
        return torch.cat([y, y[:, :, -1:]], dim=2)

    @staticmethod
    def interleave(mask_s):
        """(N, s, R, T) folded mask -> (N, 1, 257, T); one mask channel per sub-bin."""
        return N6NetV2.unfold(mask_s, mask_s.shape[1])

    def rotation(self, pha_s):
        """(N, 2s, R, T) raw (cos, sin) pairs -> unit-modulus complex (N, 1, 257, T)."""
        cs = self.unfold(pha_s, self.stem_stride)               # (N, 2, 257, T)
        c, s = cs[:, :1], cs[:, 1:]
        norm = torch.sqrt(c * c + s * s + 1e-8)
        return torch.complex(c / norm, s / norm)

    @property
    def receptive_field_frames(self):
        return 1 + sum(b.history for b in self.blocks)

    def embed(self, x):
        return x if self.pos is None else x + self.pos

    def apply_heads(self, x):
        """Trunk output (N, C, rows, T) -> (folded mask, folded (cos, sin) or None)."""
        m = self.head(x)
        if self.mask_slope is not None:
            m = self.mask_beta * torch.sigmoid(self.mask_slope * m)
        else:
            m = torch.sigmoid(m)
        return m, (self.pha_head(x) if self.pha_head is not None else None)

    def heads(self, feats):
        """(N, in_channels, 256, T) features -> (mask (N, s, rows, T), pha or None)."""
        x = self.embed(self.stem(feats))
        for blk in self.blocks:
            x = blk(x)
        return self.apply_heads(x)

    def mask_half(self, feats):
        return self.heads(feats)[0]

    def spectrum(self, stft_noisy):
        """Noisy complex STFT -> (enhanced STFT, raw folded (cos, sin) or None)."""
        mask, pha = self.heads(self.features(stft_noisy))
        y = stft_noisy * self.interleave(mask)
        if pha is not None:
            y = y * self.rotation(pha)
        return y, pha

    def forward(self, stft_noisy):
        return self.spectrum(stft_noisy)[0]


class N6NetV2_TD(TrainValidTest_TimeDomain, N6NetV2):
    """N6Net-v2 on the repo's original objective: DynCompMSE of the output audio."""
    pass


class TrainValidTest_MPSENet:
    """MP-SENet's objective (common/losses.py) for a model that emits a spectrum.

    Same train/valid interface as ``TrainValidTest_TimeDomain``, but the
    magnitude / complex terms see the network's *direct* spectrum, the
    consistency term compares it with the iSTFT -> STFT round trip of the
    output audio, and a time-domain L1 is added. The MetricGAN term is not
    summed here: the trainer adds it with ``gan.metric_loss_lambda``.

    ``phase_unit_weight`` penalises ``(|cos, sin| - 1)^2`` on the phase head so
    its int8 output range stays well used — BASENet's atan2 phase path is what
    collapsed under static int8.
    """

    mpsenet_weights: dict = None
    phase_unit_weight: float = 0.0

    def mpsenet_terms(self, y, pha_raw, x_pred, x_clean):
        cf = float(self.compress_factor or 0.3)
        r = self.preproc(x_clean)
        y_hat = self.preproc(x_pred)                                # round trip
        n_t = min(y.shape[-1], r.shape[-1], y_hat.shape[-1])       # cropping trims the tail
        mag_g, pha_g, com_g = mag_pha_from_complex(y[..., :n_t], cf)
        mag_r, pha_r, com_r = mag_pha_from_complex(r[..., :n_t], cf)
        _, _, com_g_hat = mag_pha_from_complex(y_hat[..., :n_t], cf)
        terms = generator_terms(mag_g, com_g, mag_r, com_r, com_g_hat, x_pred, x_clean,
                                pha_g=pha_g if pha_raw is not None else None, pha_r=pha_r,
                                metric_g=None, freq_dim=2, time_dim=3)
        if pha_raw is not None and self.phase_unit_weight:
            s = self.stem_stride
            mod = torch.sqrt(pha_raw[:, :s] ** 2 + pha_raw[:, s:] ** 2 + 1e-8)
            terms["phase_unit"] = ((mod - 1.0) ** 2).mean()
        return terms

    def process_data(self, x_noisy, x_clean, crop_signals=False):
        y, pha_raw = self.spectrum(self.preproc(x_noisy))
        x_pred = self.postproc(y)
        if crop_signals:
            x_pred, x_clean = self._crop_signals_if_needed(x_pred, x_clean)
        self._check_signals_shape(x_pred, x_clean)
        terms = self.mpsenet_terms(y, pha_raw, x_pred, x_clean)
        weights = dict(self.mpsenet_weights, phase_unit=self.phase_unit_weight)
        weights.pop("metric", None)                                 # the trainer adds it
        loss_dict = {f"loss/{k}": v.detach() for k, v in terms.items()}
        loss_dict["loss"] = generator_loss(terms, weights)
        return x_pred, loss_dict

    def train_step(self, x_noisy, x_clean):
        return self.process_data(x_noisy, x_clean, crop_signals=True)[1]

    def valid_step(self, x_noisy, x_clean):
        x_noisy, x_clean = self._pad_signals_if_needed(x_noisy, x_clean)
        x_pred, loss_dict = self.process_data(x_noisy, x_clean)
        return x_pred, x_clean, x_noisy, loss_dict

    def test_step(self, x_noisy, x_clean):
        return self.valid_step(x_noisy, x_clean)


class N6NetV2_MP(TrainValidTest_MPSENet, N6NetV2):
    """N6Net-v2 on the MP-SENet objective."""
    pass


def macs_per_frame(model: N6NetV2) -> int:
    """MACs for one new frame: every tap of every kernel, at its output height."""
    heights = {}
    hooks = [m.register_forward_hook(lambda mod, i, o: heights.__setitem__(mod, o.shape[2]))
             for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model.heads(torch.zeros(1, model.in_channels, model.rows_in,
                                model.receptive_field_frames))
    for h in hooks:
        h.remove()
    return sum(heights[m] * m.kernel_size[0] * m.kernel_size[1]
               * (m.in_channels // m.groups) * m.out_channels for m in heights)


def frequency_reach(model: N6NetV2) -> int:
    """Input bins that can move the centre output bin (perturbation, fp64)."""
    rows_in = model.rows_in
    m = model.double().eval()
    torch.manual_seed(0)
    x = torch.rand(1, model.in_channels, rows_in, 1, dtype=torch.float64)
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
        "input_channels": model.in_channels,
        "phase_head": model.phase_head,
    }


def build_causal_model(h=None) -> N6NetV2:
    h = h or {}
    objective = h.get("objective", "dyncompmse")
    if objective not in ("dyncompmse", "mpsenet"):
        raise ValueError(f"unknown objective {objective!r}; expected 'dyncompmse' or 'mpsenet'")
    if h.get("phase_head") and objective != "mpsenet":
        raise ValueError("phase_head needs objective='mpsenet' (the anti-wrapping phase loss)")
    if objective == "mpsenet":
        weights = loss_weights(h)
        gan_lambda = float((h.get("gan") or {}).get("metric_loss_lambda", weights["metric"]))
        if "metric" in (h.get("loss") or {}) and weights["metric"] != gan_lambda:
            raise ValueError("loss.metric and gan.metric_loss_lambda disagree; the trainer "
                             "applies gan.metric_loss_lambda")
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
    cls = N6NetV2_MP if objective == "mpsenet" else N6NetV2_TD
    model = cls(
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
        input_features=h.get("input_features", "mag"),
        mask_act=h.get("mask_act", "sigmoid"),
        mask_beta=float(h.get("mask_beta", 2.0)),
        phase_head=bool(h.get("phase_head", False)),
        hop_length=hop_length,
    )
    if objective == "mpsenet":
        model.mpsenet_weights = weights
        model.phase_unit_weight = float(h.get("phase_unit_weight", 0.1))
    return model


if __name__ == "__main__":
    for c in (96, 128, 192):
        s = cost_summary(build_causal_model({"channels": c}))
        print(f"C={c}: params={s['params']:,}  MAC/frame={s['macs_per_frame'] / 1e6:.1f} M  "
              f"RF={s['receptive_field_frames']}  FIFO={s['fifo_bytes_int8'] / 1024:.0f} KiB")
