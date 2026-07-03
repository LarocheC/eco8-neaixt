"""LiSenNet — Lightweight Sub-band and Dual-Path Speech Enhancement.

Faithful port of the generator from

    H. Yan et al., "LiSenNet: Lightweight Sub-band and Dual-Path Modeling for
    Real-Time Speech Enhancement", arXiv:2409.13285.
    Original (MIT licensed): https://github.com/hyyan2k/LiSenNet

Ported verbatim into this framework so its quality can be measured on the same
VoiceBank-DEMAND pipeline as the other models here. The architecture is
unchanged; only two unused bits from the upstream file are dropped:
  * the ``mel_scale`` helper (its sole dependency, torchaudio, isn't installed,
    and it is not used by ``forward`` or the training losses), and
  * the standalone ``NoiseDetector`` (not referenced by ``LiSenNet.forward``).
The upstream MetricGAN discriminator is the CMGAN one already present here as
``common.discriminator.MetricDiscriminator``, so it is reused rather than copied.

Design notes (why it's lightweight):
  * Input is 3 channels — power-compressed magnitude + group delay (GD) +
    instantaneous-frequency difference (IFD) of the noisy STFT.
  * A U-Net with sub-band-split convs (DSConv/USConv process low/high bands with
    different kernels/strides) and Dual-Path-Recurrent (DPR) bottleneck blocks.
  * Only a *magnitude* mask is predicted; the phase is recovered by a 2-iteration
    Griffin-Lim seeded from the noisy phase (no phase decoder) — the key cost
    saving vs. mask-and-phase models.

forward(src, tgt=None) takes waveforms (B, samples); returns a results dict with
the enhanced waveform ``est`` plus spectra/magnitudes for the CMGAN losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Frame-by-frame streaming helper
# =============================================================================
#
# LiSenNet is causal in time by construction: every time convolution left-pads
# the time axis (``(kt-1, 0)``) so frame n never reads a future frame, and the
# only time-recurrent module — the DPR "inter" GRU — is already unidirectional.
# (The DPR "intra" GRU is bidirectional only over *frequency*, which is fully
# available at each frame, so it streams fine.) Real-time inference therefore
# replaces each causal conv's start-of-sequence zero padding with a ``(kt-1)``-
# frame ring buffer and carries the inter-GRU hidden state across frames — the
# same pattern as ``basenet/streaming.py``. Streaming is gated by a ``streaming``
# flag (default off, so training/export/whole-utterance forward are unchanged)
# and each stateful module preserves its ``state_dict`` keys, so a checkpoint
# trained offline streams without retraining.


def _causal_conv_stream(pad_mod, conv, buf, x):
    """One streaming step of a causal-in-time ``(ConstantPad2d, Conv2d)`` pair.

    Tensors are ``(B, C, T, Fr)`` (time = dim 2, freq = dim 3). ``x`` is a single
    time frame ``(B, C, 1, Fr)``; ``buf`` holds the previous ``(kt-1)*dilation``
    input frames (or ``None`` at stream start). The number of past frames is read
    straight from the offline ``ConstantPad2d``'s top padding, so this works for
    dilated causal time-convs too (top = ``(kt-1)*dilation``). The frequency
    padding is read from the same ``pad_mod`` so freq handling is bit-identical to
    offline; the time padding is supplied by the ring buffer instead of zeros,
    making the step numerically identical to the offline conv on a steady stream.
    """
    f_left, f_right, t_top, _t_bot = pad_mod.padding
    k = t_top                                            # past frames = (kt-1)*dilation
    if buf is None:
        buf = x.new_zeros(x.shape[0], x.shape[1], k, x.shape[3])
    win = torch.cat([buf, x], dim=2)                     # (B, C, k+1, Fr)
    new_buf = win[:, :, 1:, :] if k > 0 else buf         # keep the last k input frames
    win = F.pad(win, (f_left, f_right, 0, 0))            # freq pad only (time supplied by buf)
    return conv(win), new_buf


# =============================================================================
# Dual-Path-Recurrent building blocks (upstream model/generator/dpr_layer.py)
# =============================================================================


class CustomLayerNorm(nn.Module):
    def __init__(self, input_dims, stat_dims=(1,), num_dims=4, eps=1e-5):
        super().__init__()
        assert isinstance(input_dims, tuple) and isinstance(stat_dims, tuple)
        assert len(input_dims) == len(stat_dims)
        param_size = [1] * num_dims
        for input_dim, stat_dim in zip(input_dims, stat_dims):
            param_size[stat_dim] = input_dim
        self.gamma = nn.Parameter(torch.ones(*param_size, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(*param_size, dtype=torch.float32))
        self.eps = eps
        self.stat_dims = stat_dims
        self.num_dims = num_dims

    def forward(self, x):
        assert x.ndim == self.num_dims, \
            f"Expect x to have {self.num_dims} dimensions, but got {x.ndim}"
        mu_ = x.mean(dim=self.stat_dims, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.stat_dims, unbiased=False, keepdim=True) + self.eps)
        return ((x - mu_) / std_) * self.gamma + self.beta


def _make_norm(kind, bn_channels, ln_input_dims):
    """Normalization factory for the deploy-hardened path.

    ``"layernorm"`` (default) keeps the original 2-axis :class:`CustomLayerNorm`
    (per-frame stats over channel+freq) — faithful but NOT NPU-mappable: it lowers
    to ReduceMean/Sqrt/Div and crashes the ST Neural-ART compiler. ``"batchnorm"``
    returns a per-channel :class:`nn.BatchNorm2d` — fixed stats at inference, folds
    into the adjacent conv, and maps to the NPU (mirrors ConvFSENet's quant-friendly
    recipe). Swapping the norm changes behaviour, so ``"batchnorm"`` needs a retrain.
    """
    if kind == "batchnorm":
        return nn.BatchNorm2d(bn_channels)
    if kind == "layernorm":
        return CustomLayerNorm(ln_input_dims, stat_dims=(1, 3))
    raise ValueError(f"unknown norm {kind!r} (expected 'layernorm' or 'batchnorm')")


def _make_act(kind, channels):
    """Activation factory. ``"prelu"`` (default) = per-channel :class:`nn.PReLU`
    (a per-channel float slope that blocks int8 / the NPU); ``"relu"`` = parameter-free
    :class:`nn.ReLU` (NPU-native); ``"relu6"`` = :class:`nn.ReLU6` (NPU-native Clip),
    which bounds the activation dynamic range so per-tensor int8 can resolve
    long-receptive-field variants (the unbounded-ReLU deep/dil16 models lose ~0.08
    PESQ to PTQ — see RESULTS_LISENNET.md round 2). ``channels`` is ignored for the
    parameter-free kinds."""
    if kind == "relu":
        return nn.ReLU()
    if kind == "relu6":
        return nn.ReLU6()
    if kind == "prelu":
        return nn.PReLU(channels)
    raise ValueError(f"unknown act {kind!r} (expected 'prelu', 'relu' or 'relu6')")


class RNN(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout_p=0.1, bidirectional=False):
        super().__init__()
        self.rnn = nn.GRU(emb_dim, hidden_dim, 1, batch_first=True, bidirectional=bidirectional)
        self.dense = nn.Linear(hidden_dim * (2 if bidirectional else 1), emb_dim)

    def forward(self, x):                                    # (b, t, d)
        x, _ = self.rnn(x)
        return self.dense(x)


class DualPathRNN(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_freqs=32, dropout_p=0.1):
        super().__init__()
        self.intra_norm = nn.LayerNorm((n_freqs, emb_dim))
        self.intra_rnn_attn = RNN(emb_dim, hidden_dim // 2, dropout_p, bidirectional=True)
        self.inter_norm = nn.LayerNorm((n_freqs, emb_dim))
        self.inter_rnn_attn = RNN(emb_dim, hidden_dim, dropout_p, bidirectional=False)
        # Streaming: the inter-time GRU hidden state, carried across frames. The
        # intra GRU runs over frequency (fully available per frame) and is
        # stateless across time. Default off.
        self.streaming = False
        self._h = None

    def stream_reset(self):
        self._h = None

    def forward(self, x):                                    # (b, d, t, f)
        b, d, t, f = x.size()
        x = x.permute(0, 2, 3, 1)                            # (b, t, f, d)

        x_res = x
        x = self.intra_norm(x).reshape(b * t, f, d)          # intra-frequency (within a frame)
        x = self.intra_rnn_attn(x).reshape(b, t, f, d)
        x = x + x_res

        x_res = x
        x = self.inter_norm(x).permute(0, 2, 1, 3).reshape(b * f, t, d)   # inter-time
        if self.streaming:                                   # carry GRU hidden across frames
            g, self._h = self.inter_rnn_attn.rnn(x, self._h)
            x = self.inter_rnn_attn.dense(g)
        else:
            x = self.inter_rnn_attn(x)
        x = x.reshape(b, f, t, d).permute(0, 2, 1, 3)
        x = x + x_res

        return x.permute(0, 3, 1, 2)                         # (b, d, t, f)


class _CausalTimeConv(nn.Module):
    """Causal dilated depthwise time-conv + pointwise mix, streamable.

    Operates on ``(b, d, t, f)``; mixes only along time (kernel ``(kernel, 1)``,
    dilation ``(dilation, 1)``), left-padded ``(kernel-1)*dilation`` so frame n
    never reads a future frame. A depthwise conv over time followed by a 1x1
    pointwise conv — both map to the NPU — replaces one step of the inter-time
    GRU's recurrence with a finite (dilated) receptive field. Streams via the
    shared ``_causal_conv_stream`` ring buffer (the pad's top value = the FIFO
    length, so dilation is handled).
    """

    def __init__(self, emb_dim, kernel=3, dilation=1, act="prelu"):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.pad = nn.ConstantPad2d((0, 0, pad, 0), value=0.0)   # causal time pad (freq untouched)
        self.dw = nn.Conv2d(emb_dim, emb_dim, (kernel, 1), dilation=(dilation, 1), groups=emb_dim)
        self.pw = nn.Conv2d(emb_dim, emb_dim, 1)
        self.act = _make_act(act, emb_dim)
        self.streaming = False
        self._buf = None

    def stream_reset(self):
        self._buf = None

    def forward(self, x):                                    # (b, d, t, f)
        if self.streaming:
            y, self._buf = _causal_conv_stream(self.pad, self.dw, self._buf, x)
        else:
            y = self.dw(self.pad(x))
        return self.act(self.pw(y))


class DualPathConv(nn.Module):
    """NPU-mappable drop-in for :class:`DualPathRNN` (no GRU, no ``nn.LayerNorm``).

    Keeps LiSenNet's two-path structure but replaces both GRUs with convolutions
    that map to the Neural-ART NPU and stream with bounded FIFO state:

      * **intra-frequency** mixer (was the bidirectional-over-freq GRU): a
        depthwise conv over the frequency axis with *symmetric* padding — the
        whole freq axis is available per frame, so bidirectional context needs no
        state — followed by a 1x1 pointwise mix.
      * **inter-time** mixer (was the unidirectional-over-time GRU): a stack of
        causal dilated :class:`_CausalTimeConv` layers whose dilations grow the
        temporal receptive field the recurrence used to provide.

    Norms are :class:`CustomLayerNorm` (already lowered to primitive ops), so the
    exported graph carries no 2-axis ``LayerNormalization`` primitive — clearing
    the Neural-ART LayerNorm blocker at the source. ``(b, d, t, f)`` in and out,
    same interface as :class:`DualPathRNN`.
    """

    def __init__(self, emb_dim, n_freqs=32, intra_kernel=7, inter_kernel=3,
                 inter_dilations=(1, 2, 4), norm="layernorm", act="prelu"):
        super().__init__()
        self.intra_norm = _make_norm(norm, emb_dim, (emb_dim, n_freqs))
        self.intra_dw = nn.Conv2d(emb_dim, emb_dim, (1, intra_kernel),
                                  padding=(0, intra_kernel // 2), groups=emb_dim)   # symmetric over freq
        self.intra_pw = nn.Conv2d(emb_dim, emb_dim, 1)
        self.intra_act = _make_act(act, emb_dim)

        self.inter_norm = _make_norm(norm, emb_dim, (emb_dim, n_freqs))
        self.inter = nn.ModuleList(
            _CausalTimeConv(emb_dim, inter_kernel, d, act=act) for d in inter_dilations)

    def forward(self, x):                                    # (b, d, t, f)
        y = self.intra_act(self.intra_pw(self.intra_dw(self.intra_norm(x))))
        x = x + y                                            # intra-frequency residual
        y = self.inter_norm(x)
        for layer in self.inter:
            y = layer(y)
        return x + y                                         # inter-time residual


class ConvolutionalGLU(nn.Module):
    def __init__(self, emb_dim, n_freqs=32, expansion_factor=2, dropout_p=0.1,
                 norm="layernorm", act="prelu"):
        super().__init__()
        hidden_dim = int(emb_dim * expansion_factor)
        self.norm = _make_norm(norm, emb_dim, (emb_dim, n_freqs))
        self.fc1 = nn.Conv2d(emb_dim, hidden_dim * 2, 1)
        self.dwconv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 2, 0), value=0.0),       # causal-in-time pad (2,0), freq (1,1)
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, groups=hidden_dim),
        )
        self.act = _make_act(act, hidden_dim) if act in ("relu", "relu6") else nn.Mish()
        self.fc2 = nn.Conv2d(hidden_dim, emb_dim, 1)
        self.dropout = nn.Dropout(dropout_p)
        # Streaming: the causal depthwise conv keeps a (kt-1)-frame ring buffer.
        self.streaming = False
        self._buf = None

    def stream_reset(self):
        self._buf = None

    def forward(self, x):                                    # (b, d, t, f)
        res = x
        x = self.norm(x)
        x, v = self.fc1(x).chunk(2, dim=1)
        if self.streaming:
            x, self._buf = _causal_conv_stream(self.dwconv[0], self.dwconv[1], self._buf, x)
        else:
            x = self.dwconv(x)
        x = self.act(x) * v
        x = self.dropout(x)
        return self.fc2(x) + res


class DPR(nn.Module):
    """Dual-path block: a two-path mixer (``bottleneck``) then a ConvolutionalGLU.

    ``bottleneck="rnn"`` (default) is the faithful LiSenNet :class:`DualPathRNN`;
    ``bottleneck="conv"`` swaps in the NPU-mappable :class:`DualPathConv`. Both
    keep the attribute name ``dp_rnn_attn`` so the surrounding code and the RNN
    checkpoints are unaffected.
    """

    def __init__(self, emb_dim=16, hidden_dim=24, n_freqs=32, dropout_p=0.1,
                 bottleneck="rnn", intra_kernel=7, inter_kernel=3, inter_dilations=(1, 2, 4),
                 norm="layernorm", act="prelu"):
        super().__init__()
        if bottleneck == "conv":
            self.dp_rnn_attn = DualPathConv(emb_dim, n_freqs, intra_kernel, inter_kernel,
                                            inter_dilations, norm=norm, act=act)
        elif bottleneck == "rnn":
            self.dp_rnn_attn = DualPathRNN(emb_dim, hidden_dim, n_freqs, dropout_p)
        else:
            raise ValueError(f"unknown bottleneck {bottleneck!r} (expected 'rnn' or 'conv')")
        self.conv_glu = ConvolutionalGLU(emb_dim, n_freqs=n_freqs, expansion_factor=2,
                                         dropout_p=dropout_p, norm=norm, act=act)

    def forward(self, x):
        return self.conv_glu(self.dp_rnn_attn(x))


# =============================================================================
# Sub-band U-Net (upstream model/generator/generator.py)
# =============================================================================


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1, 1))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class SPConvTranspose2d(nn.Module):
    """Sub-pixel transposed conv: upsample the frequency axis by factor r."""

    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super().__init__()
        self.pad = nn.ConstantPad2d((kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[0] - 1, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad(x)
        out = self.conv(x)
        b, c, hh, ww = out.shape
        out = out.view((b, self.r, c // self.r, hh, ww))
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        return out.view((b, c // self.r, hh, -1))


class DSConv(nn.Module):
    """Down-sampling sub-band conv: low band kept dense, high band strided (÷3)."""

    def __init__(self, in_channels, out_channels, n_freqs, norm="layernorm", act="prelu"):
        super().__init__()
        self.low_freqs = n_freqs // 4
        self.low_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3)),
        )
        self.high_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 5), stride=(1, 3)),
        )
        self.norm = _make_norm(norm, out_channels, (1, n_freqs // 2))
        self.act = _make_act(act, out_channels)
        # Streaming: low and high sub-band convs each keep their own ring buffer.
        self.streaming = False
        self._buf_low = None
        self._buf_high = None

    def stream_reset(self):
        self._buf_low = None
        self._buf_high = None

    def forward(self, x):                                    # (b, d, t, f)
        if self.streaming:
            x_low, self._buf_low = _causal_conv_stream(
                self.low_conv[0], self.low_conv[1], self._buf_low, x[..., :self.low_freqs])
            x_high, self._buf_high = _causal_conv_stream(
                self.high_conv[0], self.high_conv[1], self._buf_high, x[..., self.low_freqs:])
        else:
            x_low = self.low_conv(x[..., :self.low_freqs])
            x_high = self.high_conv(x[..., self.low_freqs:])
        return self.act(self.norm(torch.cat([x_low, x_high], dim=-1)))


class USConv(nn.Module):
    """Up-sampling sub-band conv (mirror of DSConv)."""

    def __init__(self, in_channels, out_channels, n_freqs, upsample="subpixel"):
        super().__init__()
        self.low_freqs = n_freqs // 2
        self.low_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3)),
        )
        # High-band frequency upsampling (x3). "subpixel" = the original
        # SPConvTranspose2d (a view->permute->view that emits 5-D tensors the
        # Neural-ART NPU cannot map — it segfaults atonn). "convtranspose" = the
        # NPU-native equivalent: a plain transposed conv, stride 3 over freq gives
        # the same x3 upsample (out_freq = 3*in_freq) with 4-D tensors only, and
        # ~3x fewer MACC than the sub-pixel conv-to-out*r.
        if upsample == "convtranspose":
            self.high_conv = nn.ConvTranspose2d(in_channels, out_channels,
                                                kernel_size=(1, 3), stride=(1, 3))
        else:
            self.high_conv = SPConvTranspose2d(in_channels, out_channels, kernel_size=(1, 3), r=3)

    def forward(self, x):                                    # (b, d, t, f)
        x_low = self.low_conv(x[..., :self.low_freqs])
        x_high = self.high_conv(x[..., self.low_freqs:])
        return torch.cat([x_low, x_high], dim=-1)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_channels=16, norm="layernorm", act="prelu"):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels // 4, (1, 1), (1, 1)),
            _make_norm(norm, num_channels // 4, (1, 257)),
            _make_act(act, num_channels // 4),
        )
        self.conv_2 = DSConv(num_channels // 4, num_channels // 2, n_freqs=257, norm=norm, act=act)
        self.conv_3 = DSConv(num_channels // 2, num_channels // 4 * 3, n_freqs=128, norm=norm, act=act)
        self.conv_4 = DSConv(num_channels // 4 * 3, num_channels, n_freqs=64, norm=norm, act=act)

    def forward(self, x):
        out_list = []
        x = self.conv_1(x)
        x = self.conv_2(x); out_list.append(x)               # 128
        x = self.conv_3(x); out_list.append(x)               # 64
        x = self.conv_4(x); out_list.append(x)               # 32
        return out_list


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channels=64, out_channel=2, beta=1,
                 norm="layernorm", act="prelu", upsample="subpixel"):
        super().__init__()
        self.up1 = USConv(num_channels * 2, num_channels // 4 * 3, n_freqs=32, upsample=upsample)
        self.up2 = USConv(num_channels // 4 * 3 * 2, num_channels // 2, n_freqs=64, upsample=upsample)
        self.up3 = USConv(num_channels // 2 * 2, num_channels // 4, n_freqs=128, upsample=upsample)
        self.mask_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(num_channels // 4, out_channel, (2, 2)),
            _make_norm(norm, out_channel, (1, 257)),
            _make_act(act, out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1)),
        )
        self.lsigmoid = nn.Sigmoid() if act in ("relu", "relu6") else LearnableSigmoid2d(num_features, beta=beta)
        # Streaming: the first mask conv (kernel (2,2)) is causal in time and keeps
        # a 1-frame ring buffer; the rest of mask_conv is per-frame.
        self.streaming = False
        self._buf = None

    def stream_reset(self):
        self._buf = None

    def forward(self, x, encoder_out_list):
        x = self.up1(torch.cat([x, encoder_out_list.pop()], dim=1))
        x = self.up2(torch.cat([x, encoder_out_list.pop()], dim=1))
        x = self.up3(torch.cat([x, encoder_out_list.pop()], dim=1))
        if self.streaming:
            x, self._buf = _causal_conv_stream(self.mask_conv[0], self.mask_conv[1], self._buf, x)
            x = self.mask_conv[2](x)                          # CustomLayerNorm (per-frame)
            x = self.mask_conv[3](x)                          # PReLU
            x = self.mask_conv[4](x)                          # 1x1 conv
        else:
            x = self.mask_conv(x)                            # (B, out_channel, T, F)
        x = x.permute(0, 3, 2, 1)                            # (B, F, T, out_channel)
        return self.lsigmoid(x).permute(0, 3, 2, 1)


class LiSenNet(nn.Module):
    def __init__(self, num_channels=16, n_blocks=2, n_fft=512, hop_length=256, compress_factor=0.3,
                 bottleneck="rnn", intra_kernel=7, inter_kernel=3, inter_dilations=(1, 2, 4),
                 norm="layernorm", act="prelu", upsample="subpixel"):
        super().__init__()
        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1
        self.hop_length = hop_length
        self.compress_factor = compress_factor
        self.bottleneck = bottleneck

        self.encoder = Encoder(in_channels=3, num_channels=num_channels, norm=norm, act=act)
        self.blocks = nn.Sequential(*[
            DPR(emb_dim=num_channels, hidden_dim=num_channels // 2 * 3,
                n_freqs=self.n_freqs // (2 ** 3), dropout_p=0.1,
                bottleneck=bottleneck, intra_kernel=intra_kernel,
                inter_kernel=inter_kernel, inter_dilations=inter_dilations,
                norm=norm, act=act)
            for _ in range(n_blocks)
        ])
        self.decoder = MaskDecoder(self.n_freqs, num_channels=num_channels, out_channel=2, beta=1,
                                   norm=norm, act=act, upsample=upsample)

    # ----- spectral helpers -------------------------------------------------
    def apply_stft(self, x, return_complex=True):
        spec = torch.stft(x, self.n_fft, self.hop_length,
                          window=torch.hann_window(self.n_fft).to(x.device),
                          onesided=True, return_complex=return_complex)
        return spec.transpose(1, 2)                          # (B, T, F)

    def apply_istft(self, x, length=None):
        x = x.transpose(1, 2)                                # (B, F, T)
        return torch.istft(x, self.n_fft, self.hop_length,
                           window=torch.hann_window(self.n_fft).to(x.device),
                           onesided=True, length=length, return_complex=False)

    def power_compress(self, x):
        mag = torch.abs(x) ** self.compress_factor
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    def power_uncompress(self, x):
        mag = torch.abs(x) ** (1.0 / self.compress_factor)
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    @staticmethod
    def cal_gd(x):
        """Group delay: anti-wrapped difference along frequency."""
        b, t, f = x.size()
        x_gd = torch.diff(x, dim=2, prepend=torch.zeros(b, t, 1, device=x.device))
        return torch.atan2(x_gd.sin(), x_gd.cos())

    def cal_ifd(self, x):
        """Instantaneous-frequency difference: anti-wrapped, hop-corrected, over time."""
        b, t, f = x.size()
        x_if = torch.diff(x, dim=1, prepend=torch.zeros(b, 1, f, device=x.device))
        x_ifd = x_if - 2 * torch.pi * (self.hop_length / self.n_fft) * torch.arange(f, device=x.device)[None, None, :]
        return torch.atan2(x_ifd.sin(), x_ifd.cos())

    def griffinlim(self, mag, pha=None, length=None, n_iter=2, momentum=0.99):
        mag = mag.detach()
        mag = mag ** (1.0 / self.compress_factor)            # uncompress
        assert 0 <= momentum < 1
        momentum = momentum / (1 + momentum)
        if pha is None:
            pha = torch.rand(mag.size(), dtype=mag.dtype, device=mag.device)
        tprev = torch.tensor(0.0, dtype=mag.dtype, device=mag.device)
        for _ in range(n_iter):
            inverse = self.apply_istft(torch.complex(mag * pha.cos(), mag * pha.sin()), length=length)
            rebuilt = self.apply_stft(inverse)
            pha = rebuilt
            pha = pha - tprev.mul_(momentum)
            pha = pha.angle()
            tprev = rebuilt
        return pha

    # ----- mask sub-network (the deployable / streamable core) --------------
    def build_features(self, src_mag, src_pha):
        """Stack the 3-channel network input: [compressed mag, GD/pi, IFD/pi]."""
        src_gd = self.cal_gd(src_pha)
        src_ifd = self.cal_ifd(src_pha)
        return torch.stack([src_mag, src_gd / torch.pi, src_ifd / torch.pi], dim=1)  # (B, 3, T, F)

    def predict_mask(self, feat):
        """Encoder -> DPR blocks -> decoder. feat (B, 3, T, F) -> mask (B, 2, T, F).

        Pure tensor ops (no STFT / Griffin-Lim), causal in time — this is the
        sub-network that is exported to ONNX and run frame-by-frame for real-time
        inference. STFT, feature extraction, the magnitude combination and the
        phase recovery live outside it.
        """
        encoder_out_list = self.encoder(feat)
        x = self.blocks(encoder_out_list[-1])
        return self.decoder(x, encoder_out_list)              # (B, 2, T, F)

    @staticmethod
    def apply_mask(mask, src_mag):
        """Combine the 2-channel mask with the noisy magnitude (upstream Eq.)."""
        return (mask[:, 0] + 1e-8) * src_mag + (mask[:, 1] + 1e-8) * src_mag

    # ----- forward ----------------------------------------------------------
    def forward(self, src, tgt=None):
        if tgt is None:
            tgt = src
        src_spec = self.power_compress(self.apply_stft(src))  # (B, T, F)
        src_mag = src_spec.abs()
        src_pha = src_spec.angle()

        tgt_spec = self.power_compress(self.apply_stft(tgt))
        tgt_mag = tgt_spec.abs()

        x = self.build_features(src_mag, src_pha)             # (B, 3, T, F)
        x = self.predict_mask(x)                              # (B, 2, T, F)

        est_mag = self.apply_mask(x, src_mag)

        est_pha = self.griffinlim(est_mag.detach(), src_pha, tgt.size(-1))
        est_spec = torch.complex(est_mag * est_pha.cos(), est_mag * est_pha.sin())
        est = self.apply_istft(self.power_uncompress(est_spec), length=tgt.size(-1))

        return {
            "tgt": tgt, "tgt_spec": tgt_spec, "tgt_mag": tgt_mag,
            "est": est, "est_spec": est_spec, "est_mag": est_mag,
        }


def build_lisennet(h=None):
    """Instantiate LiSenNet from a dict-like config (defaults = upstream config.yaml)."""
    h = h or {}

    def g(key, default):
        return h.get(key, default) if hasattr(h, "get") else getattr(h, key, default)

    return LiSenNet(
        num_channels=int(g("num_channels", 16)),
        n_blocks=int(g("n_blocks", 2)),
        n_fft=int(g("n_fft", 512)),
        hop_length=int(g("hop_length", 256)),
        compress_factor=float(g("compress_factor", 0.3)),
        bottleneck=str(g("bottleneck", "rnn")),
        intra_kernel=int(g("intra_kernel", 7)),
        inter_kernel=int(g("inter_kernel", 3)),
        inter_dilations=tuple(g("inter_dilations", (1, 2, 4))),
        norm=str(g("norm", "layernorm")),
        act=str(g("act", "prelu")),
        upsample=str(g("upsample", "subpixel")),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_lisennet().eval()
    print(f"LiSenNet params: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(2, 32000)
    with torch.no_grad():
        r = model(x, x)
    print(f"forward ok: est={tuple(r['est'].shape)}, est_mag={tuple(r['est_mag'].shape)}, "
          f"est_spec={tuple(r['est_spec'].shape)}")
