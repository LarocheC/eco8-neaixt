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

    def forward(self, x):                                    # (b, d, t, f)
        b, d, t, f = x.size()
        x = x.permute(0, 2, 3, 1)                            # (b, t, f, d)

        x_res = x
        x = self.intra_norm(x).reshape(b * t, f, d)          # intra-frequency (within a frame)
        x = self.intra_rnn_attn(x).reshape(b, t, f, d)
        x = x + x_res

        x_res = x
        x = self.inter_norm(x).permute(0, 2, 1, 3).reshape(b * f, t, d)   # inter-time
        x = self.inter_rnn_attn(x).reshape(b, f, t, d).permute(0, 2, 1, 3)
        x = x + x_res

        return x.permute(0, 3, 1, 2)                         # (b, d, t, f)


class ConvolutionalGLU(nn.Module):
    def __init__(self, emb_dim, n_freqs=32, expansion_factor=2, dropout_p=0.1):
        super().__init__()
        hidden_dim = int(emb_dim * expansion_factor)
        self.norm = CustomLayerNorm((emb_dim, n_freqs), stat_dims=(1, 3))
        self.fc1 = nn.Conv2d(emb_dim, hidden_dim * 2, 1)
        self.dwconv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 2, 0), value=0.0),       # causal-in-time pad (2,0), freq (1,1)
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, groups=hidden_dim),
        )
        self.act = nn.Mish()
        self.fc2 = nn.Conv2d(hidden_dim, emb_dim, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):                                    # (b, d, t, f)
        res = x
        x = self.norm(x)
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.dropout(x)
        return self.fc2(x) + res


class DPR(nn.Module):
    def __init__(self, emb_dim=16, hidden_dim=24, n_freqs=32, dropout_p=0.1):
        super().__init__()
        self.dp_rnn_attn = DualPathRNN(emb_dim, hidden_dim, n_freqs, dropout_p)
        self.conv_glu = ConvolutionalGLU(emb_dim, n_freqs=n_freqs, expansion_factor=2, dropout_p=dropout_p)

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

    def __init__(self, in_channels, out_channels, n_freqs):
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
        self.norm = CustomLayerNorm((1, n_freqs // 2), stat_dims=(1, 3))
        self.act = nn.PReLU(out_channels)

    def forward(self, x):                                    # (b, d, t, f)
        x_low = self.low_conv(x[..., :self.low_freqs])
        x_high = self.high_conv(x[..., self.low_freqs:])
        return self.act(self.norm(torch.cat([x_low, x_high], dim=-1)))


class USConv(nn.Module):
    """Up-sampling sub-band conv (mirror of DSConv)."""

    def __init__(self, in_channels, out_channels, n_freqs):
        super().__init__()
        self.low_freqs = n_freqs // 2
        self.low_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3)),
        )
        self.high_conv = SPConvTranspose2d(in_channels, out_channels, kernel_size=(1, 3), r=3)

    def forward(self, x):                                    # (b, d, t, f)
        x_low = self.low_conv(x[..., :self.low_freqs])
        x_high = self.high_conv(x[..., self.low_freqs:])
        return torch.cat([x_low, x_high], dim=-1)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_channels=16):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels // 4, (1, 1), (1, 1)),
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(num_channels // 4),
        )
        self.conv_2 = DSConv(num_channels // 4, num_channels // 2, n_freqs=257)
        self.conv_3 = DSConv(num_channels // 2, num_channels // 4 * 3, n_freqs=128)
        self.conv_4 = DSConv(num_channels // 4 * 3, num_channels, n_freqs=64)

    def forward(self, x):
        out_list = []
        x = self.conv_1(x)
        x = self.conv_2(x); out_list.append(x)               # 128
        x = self.conv_3(x); out_list.append(x)               # 64
        x = self.conv_4(x); out_list.append(x)               # 32
        return out_list


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channels=64, out_channel=2, beta=1):
        super().__init__()
        self.up1 = USConv(num_channels * 2, num_channels // 4 * 3, n_freqs=32)
        self.up2 = USConv(num_channels // 4 * 3 * 2, num_channels // 2, n_freqs=64)
        self.up3 = USConv(num_channels // 2 * 2, num_channels // 4, n_freqs=128)
        self.mask_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(num_channels // 4, out_channel, (2, 2)),
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1)),
        )
        self.lsigmoid = LearnableSigmoid2d(num_features, beta=beta)

    def forward(self, x, encoder_out_list):
        x = self.up1(torch.cat([x, encoder_out_list.pop()], dim=1))
        x = self.up2(torch.cat([x, encoder_out_list.pop()], dim=1))
        x = self.up3(torch.cat([x, encoder_out_list.pop()], dim=1))
        x = self.mask_conv(x)                                # (B, out_channel, T, F)
        x = x.permute(0, 3, 2, 1)                            # (B, F, T, out_channel)
        return self.lsigmoid(x).permute(0, 3, 2, 1)


class LiSenNet(nn.Module):
    def __init__(self, num_channels=16, n_blocks=2, n_fft=512, hop_length=256, compress_factor=0.3):
        super().__init__()
        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1
        self.hop_length = hop_length
        self.compress_factor = compress_factor

        self.encoder = Encoder(in_channels=3, num_channels=num_channels)
        self.blocks = nn.Sequential(*[
            DPR(emb_dim=num_channels, hidden_dim=num_channels // 2 * 3,
                n_freqs=self.n_freqs // (2 ** 3), dropout_p=0.1)
            for _ in range(n_blocks)
        ])
        self.decoder = MaskDecoder(self.n_freqs, num_channels=num_channels, out_channel=2, beta=1)

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

    # ----- forward ----------------------------------------------------------
    def forward(self, src, tgt=None):
        if tgt is None:
            tgt = src
        src_spec = self.power_compress(self.apply_stft(src))  # (B, T, F)
        src_mag = src_spec.abs()
        src_pha = src_spec.angle()
        src_gd = self.cal_gd(src_pha)
        src_ifd = self.cal_ifd(src_pha)

        tgt_spec = self.power_compress(self.apply_stft(tgt))
        tgt_mag = tgt_spec.abs()

        x = torch.stack([src_mag, src_gd / torch.pi, src_ifd / torch.pi], dim=1)  # (B, 3, T, F)

        encoder_out_list = self.encoder(x)
        x = self.blocks(encoder_out_list[-1])
        x = self.decoder(x, encoder_out_list)                 # (B, 2, T, F)

        est_mag = (x[:, 0] + 1e-8) * src_mag + (x[:, 1] + 1e-8) * src_mag

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
