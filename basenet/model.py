"""BASENet — Band-Adapted Speech Enhancement Network with Cross-Band Attention.

Faithful PyTorch implementation of

    D. Martins Gomes, F. Capman, "BASENet: Band-Adapted Speech Enhancement
    Network with Cross-Band Attention", arXiv:2606.12662 (Thales SIX GTS, 2026).

This module focuses on the **causal** variant (the paper's "BASENet-3 (Causal)",
PESQ 3.44 / STOI 96%, 0.81 M params), but a non-causal path is kept too so the
single `causal` flag matches the paper's two reported configurations.

Pipeline (Fig. 1 of the paper)
------------------------------
    [Y_m^c, Y_p]                       compressed magnitude + phase, (B, 2, F, N)
      -> 1x1 projection to C channels
      -> split into B Bark-scale frequency bands
      -> per-band Efficient-DenseNet encoder, depth L_b scaled by band density
      -> Cross-Band Attention (per-frame inter-band exchange, Eqs. 6-12)
      -> Fusion & Projection: full-res skip (S) + freq-pooled CRN input
      -> CRN temporal model (GRU + GLU conv block)
      -> decoders fuse the CRN output with the full-res skip S, then:
         -> Efficient magnitude-mask decoder -> M_hat  (LearnableSigmoid)
         -> Efficient phase decoder          -> X_hat_p (atan2, anti-wrapping)
    X_hat_m^c = Y_m^c (x) M_hat            (Eq. 1)

Frequency skip connection
-------------------------
The CRN pools 201 frequency bins down to `crn_freq` for the recurrent
bottleneck. Magnitude tolerates that coarse representation (spectral envelopes
are smooth), but *phase* decorrelates rapidly across frequency and cannot be
reconstructed from it — in a no-skip version the phase loss stays pinned at its
random-init level and validation PESQ never rises above the noisy input. So
each decoder also receives the full-resolution fusion output as a skip
connection (the U-Net structure of the CRN the paper cites, ref. [20]),
carrying per-bin spectral detail around the bottleneck. This is causal: the
skip is built from causal ops and merged by a 1x1 conv.

The input representation is exactly what ``common.dataset.mag_pha_stft`` already
produces (power-law compressed magnitude with ``compress_factor=0.3`` and phase),
and the output ``(mag, pha)`` feed straight back into ``mag_pha_istft``.

Causality — what the paper glosses over
---------------------------------------
The paper states that causal streaming needs only "replacing the bidirectional
GRU with a unidirectional variant". That is necessary but **not sufficient**: a
naive port leaks future context through three more components, each of which
pools or normalises across the time axis. For the causal path we therefore also:

  * **convolutions** — left-pad the time axis (`(kt-1, 0)`) instead of centring,
    so frame n never reads frames > n (`TFConv`). Dilation stays on the
    *frequency* axis as the paper specifies.
  * **normalisation** — replace InstanceNorm (which averages over (freq, time))
    with `CausalFreqNorm`, normalising over frequency *within each frame*.
  * **squeeze-excitation** — pool the SE descriptor over frequency only, giving a
    per-frame channel gate instead of a clip-global one.

Cross-band attention is already causal: every frame is attended independently
(Eq. 11), which is exactly why the paper highlights its O(N F_b B) cost.

The accompanying test (`tests/test_basenet_model.py`) verifies genuine causality
by perturbing a future frame and asserting that all earlier outputs are bit-for-
bit unchanged.

Hyperparameters default to the paper's experimental setup (Sec. 3.1):
n_fft=400, hop=100, 16 kHz, B=3 bands at [0, 1, 4, 8] kHz with depths [4, 3, 2]
(L_max=4). Widths (channels, CRN size) are not fully specified in the paper and
are exposed as constructor arguments; the defaults give a small, streamable
model in the spirit of the reported 0.81 M parameters.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


# =============================================================================
# Bark-scale band allocation (Sec. 2.2, Eqs. 2-4)
# =============================================================================


def bark_scale(f_hz):
    """Traunmüller's analytic Bark warping z(f) (Eq. 2).

    z(f) = 13 * arctan(0.00076 f) + 3.5 * arctan((f / 7500)^2)   [Bark].
    """
    f = torch.as_tensor(f_hz, dtype=torch.float64)
    return 13.0 * torch.atan(0.00076 * f) + 3.5 * torch.atan((f / 7500.0) ** 2)


def critical_band_densities(edges_hz):
    """Per-band critical-band density rho_b (Eq. 3): Bark units spanned per Hz.

    `edges_hz` is the list of B+1 band boundaries (ascending, Hz). Returns a
    list of B densities. A higher rho_b means finer auditory resolution and,
    per the paper, deserves a deeper encoder branch.
    """
    z = bark_scale(edges_hz)
    return [
        float((z[b] - z[b - 1]) / (edges_hz[b] - edges_hz[b - 1]))
        for b in range(1, len(edges_hz))
    ]


def derive_band_depths(edges_hz, l_max):
    """Density-driven depth allocation L_b = round(L_max * rho_b / max_b rho_b) (Eq. 4).

    Provided for completeness / documentation of the paper's "single-parameter
    design rule". NOTE: with the paper's own band edges this formula yields
    depths [4, 1, 0] rather than the [4, 3, 2] it reports using — the preprint
    is internally inconsistent here — so `BASENet` takes the explicitly reported
    `band_depths=(4, 3, 2)` as the canonical default and treats this helper as a
    reference implementation of the stated rule, not the source of truth.
    """
    rho = critical_band_densities(edges_hz)
    rho_max = max(rho)
    return [max(1, round(l_max * r / rho_max)) for r in rho]


def hz_edges_to_bins(edges_hz, n_fft, sampling_rate, n_features):
    """Map band boundaries in Hz to STFT bin indices, covering all `n_features` bins.

    Returns B+1 ascending bin indices with the first == 0 and the last ==
    n_features, so the bands tile the whole spectrum with no gaps/overlap.
    """
    bins = [int(round(f * n_fft / sampling_rate)) for f in edges_hz]
    bins[0] = 0
    bins[-1] = n_features  # extend the top band to include the Nyquist bin
    for a, b in zip(bins, bins[1:]):
        assert b > a, f"band edges must be strictly increasing in bins, got {bins}"
    return bins


# =============================================================================
# Causal-aware primitives
# =============================================================================


class TFConv(nn.Module):
    """Conv2d over (frequency, time) with 'same' frequency padding.

    Tensors are ``(B, C, F, N)`` with F=frequency (height) and N=time (width).
    Frequency uses symmetric 'same' padding with the requested dilation. Time
    uses left-only (causal) padding when ``causal`` so output frame n depends
    only on input frames <= n; otherwise symmetric 'same' padding.

    Dilation is applied to the **frequency** axis only (the paper dilates the
    dense blocks along frequency); the time kernel is never dilated.
    """

    def __init__(self, in_ch, out_ch, kf=3, kt=3, freq_dilation=1, groups=1,
                 bias=True, causal=True):
        super().__init__()
        self.causal = causal
        self.pad_freq = (kf - 1) * freq_dilation // 2
        self.pad_time = (kt - 1, 0) if causal else ((kt - 1) // 2, (kt - 1) - (kt - 1) // 2)
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=(kf, kt), padding=0,
            dilation=(freq_dilation, 1), groups=groups, bias=bias,
        )
        # Frame-by-frame streaming state (see basenet/streaming.py). When
        # `streaming` is on, forward expects a single time frame (N=1) and keeps
        # the last (kt-1) input frames in `_buf` to reconstruct the causal
        # receptive field — numerically identical to the offline conv. Default
        # off, so training and ONNX export are unaffected.
        self.streaming = False
        self._buf = None

    def stream_reset(self):
        self._buf = None

    def forward(self, x):
        if self.streaming:
            return self._forward_stream(x)
        # F.pad order for (B, C, F, N): (time_left, time_right, freq_left, freq_right)
        x = F.pad(x, (self.pad_time[0], self.pad_time[1], self.pad_freq, self.pad_freq))
        return self.conv(x)

    def _forward_stream(self, x):                       # x: (B, C, F, 1)
        assert self.causal, "streaming requires causal convolutions"
        k = self.pad_time[0]                            # kt-1 past frames
        if self._buf is None:
            self._buf = x.new_zeros(x.shape[0], x.shape[1], x.shape[2], k)
        win = torch.cat([self._buf, x], dim=-1)         # (B, C, F, kt)
        self._buf = win[..., 1:]                        # keep the last kt-1 input frames
        win = F.pad(win, (0, 0, self.pad_freq, self.pad_freq))
        return self.conv(win)                           # (B, C, F, 1)


class CausalFreqNorm(nn.Module):
    """InstanceNorm-style affine norm that normalises over **frequency only**.

    Statistics for frame n are computed from that frame alone (mean/var across
    the F axis, per channel), so no future information leaks in — the streamable
    replacement for InstanceNorm2d, whose stats span the whole time axis.
    """

    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):  # (B, C, F, N)
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def make_norm(num_channels, causal):
    """Frequency-only norm when causal, standard InstanceNorm2d otherwise."""
    if causal:
        return CausalFreqNorm(num_channels)
    return nn.InstanceNorm2d(num_channels, affine=True)


def _adaptive_avg_pool_matrix(in_size, out_size):
    """(out_size, in_size) matrix replicating AdaptiveAvgPool1d, derived by
    pooling the identity so it matches PyTorch's variable-window averaging
    exactly."""
    ident = torch.eye(in_size).unsqueeze(0)                  # (1, in, in)
    pooled = F.adaptive_avg_pool1d(ident, out_size)[0]       # (in, out)
    return pooled.t().contiguous()                           # (out, in)


def _nearest_upsample_matrix(in_size, out_size):
    """(out_size, in_size) 0/1 matrix replicating nearest-neighbour upsampling,
    derived from F.interpolate itself so the indexing matches exactly."""
    ident = torch.eye(in_size).unsqueeze(0)                  # (1, in, in)
    up = F.interpolate(ident, size=out_size, mode="nearest")[0]   # (in, out)
    return up.t().contiguous()                               # (out, in)


class FreqResample(nn.Module):
    """Resample the frequency axis by a fixed matrix, leaving time untouched.

    Replaces AdaptiveAvgPool2d (down) and nearest interpolation (up) over the
    frequency axis with an equivalent constant matmul. Two wins: it exports to
    ONNX cleanly (ONNX rejects adaptive pooling / Resize with a non-constant
    output size), and it stays causal (per-frame — the time axis is never
    mixed). The matrix is a non-persistent buffer (no parameters), so a
    checkpoint trained against the original pooling/interp ops loads with
    strict=True and behaves identically.
    """

    def __init__(self, in_size, out_size, mode):
        super().__init__()
        if mode == "avg":
            w = _adaptive_avg_pool_matrix(in_size, out_size)
        elif mode == "nearest":
            w = _nearest_upsample_matrix(in_size, out_size)
        else:
            raise ValueError(f"unsupported FreqResample mode {mode!r}")
        self.register_buffer("w", w, persistent=False)       # (out_size, in_size)

    def forward(self, x):                                    # (B, C, F_in, N)
        return torch.einsum("of,bcfn->bcon", self.w, x)


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel gate (Hu et al., used inside the IR block).

    Causal variant squeezes over frequency only (a per-frame descriptor); the
    non-causal variant squeezes over the whole (freq, time) plane as usual.
    """

    def __init__(self, channels, reduction=4, causal=True):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.causal = causal
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):  # (B, C, F, N)
        if self.causal:
            s = x.mean(dim=2, keepdim=True)            # (B, C, 1, N) per-frame
        else:
            s = x.mean(dim=(2, 3), keepdim=True)       # (B, C, 1, 1) clip-global
        s = self.fc2(self.act(self.fc1(s)))
        return x * self.gate(s)


# =============================================================================
# Efficient DenseNet building blocks (Sec. 2.4)
# =============================================================================


class IRBlock(nn.Module):
    """Inverted-residual block (Eq. 13): x + Proj(SE(DWConv(Expand(x)))).

    Expand (1x1, C -> rC) -> depthwise 3x3 dilated conv -> SE -> Project (1x1,
    rC -> C), Hardswish activations, residual add. Input and output both have C
    channels so the residual is well-defined.
    """

    def __init__(self, channels, expand_ratio=4, freq_dilation=1, causal=True,
                 se_reduction=4):
        super().__init__()
        mid = channels * expand_ratio
        self.expand = nn.Conv2d(channels, mid, 1)
        self.expand_norm = make_norm(mid, causal)
        self.act1 = nn.Hardswish()
        self.dwconv = TFConv(mid, mid, kf=3, kt=3, freq_dilation=freq_dilation,
                             groups=mid, bias=False, causal=causal)
        self.dw_norm = make_norm(mid, causal)
        self.act2 = nn.Hardswish()
        self.se = SEBlock(mid, se_reduction, causal)
        self.project = nn.Conv2d(mid, channels, 1)
        self.proj_norm = make_norm(channels, causal)

    def forward(self, x):
        y = self.act1(self.expand_norm(self.expand(x)))
        y = self.act2(self.dw_norm(self.dwconv(y)))
        y = self.se(y)
        y = self.proj_norm(self.project(y))
        return x + y


class DenseLayer(nn.Module):
    """One dense layer: IR-style transform producing `growth` new channels.

    Same Expand/DWConv/SE/Project structure as `IRBlock` but with no internal
    residual (dense concatenation provides the feature reuse) and a fixed
    expansion width so parameters stay bounded as the concatenated input grows.
    """

    def __init__(self, in_ch, growth, expand_width, freq_dilation, causal,
                 se_reduction=4):
        super().__init__()
        self.expand = nn.Conv2d(in_ch, expand_width, 1)
        self.expand_norm = make_norm(expand_width, causal)
        self.act1 = nn.Hardswish()
        self.dwconv = TFConv(expand_width, expand_width, kf=3, kt=3,
                             freq_dilation=freq_dilation, groups=expand_width,
                             bias=False, causal=causal)
        self.dw_norm = make_norm(expand_width, causal)
        self.act2 = nn.Hardswish()
        self.se = SEBlock(expand_width, se_reduction, causal)
        self.project = nn.Conv2d(expand_width, growth, 1)
        self.proj_norm = make_norm(growth, causal)

    def forward(self, x):
        y = self.act1(self.expand_norm(self.expand(x)))
        y = self.act2(self.dw_norm(self.dwconv(y)))
        y = self.se(y)
        y = self.proj_norm(self.project(y))
        return y


class DenseBlock(nn.Module):
    """DenseNet-style stack of `n_layers` dense layers (Sec. 2.4).

    z_i = DenseLayer_i([z_0, ..., z_{i-1}]) with frequency dilation d = 2^i, then
    a 1x1 bottleneck reduces the concatenated features back to `channels`.
    `growth` is the per-layer feature increment (defaults to `channels`).
    """

    def __init__(self, channels, n_layers, expand_ratio=4, causal=True, growth=None):
        super().__init__()
        growth = growth or channels
        expand_width = channels * expand_ratio
        self.layers = nn.ModuleList()
        in_ch = channels
        for i in range(n_layers):
            self.layers.append(DenseLayer(
                in_ch, growth, expand_width, freq_dilation=2 ** i, causal=causal,
            ))
            in_ch += growth
        self.bottleneck = nn.Conv2d(in_ch, channels, 1)
        self.bn_norm = make_norm(channels, causal)
        self.bn_act = nn.Hardswish()

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            feats.append(layer(torch.cat(feats, dim=1)))
        out = self.bottleneck(torch.cat(feats, dim=1))
        return self.bn_act(self.bn_norm(out))


class EfficientDenseNet(nn.Module):
    """A DenseBlock of depth `depth` followed by a standalone IR block (Eq. 5).

    This is the encoder branch E_b (and the body of each decoder). Deeper
    branches (larger `depth`) reach exponentially further along the frequency
    axis via the dense block's d = 2^i dilations.

    The paper's IR block spec ("Expand projects C channels to rC, r in {2,4}")
    doesn't say which blocks get which ratio, so the dense layers' expand
    ratio (`expand_ratio`) and the trailing standalone IR block's
    (`ir_expand_ratio`) are independent knobs. `ir_expand_ratio=None` (the
    default) reuses `expand_ratio`, reproducing the old single-ratio behaviour.
    """

    def __init__(self, channels, depth, expand_ratio=4, causal=True, growth=None,
                 ir_expand_ratio=None):
        super().__init__()
        self.dense = DenseBlock(channels, depth, expand_ratio, causal, growth=growth)
        self.ir = IRBlock(channels, ir_expand_ratio or expand_ratio,
                          freq_dilation=1, causal=causal)

    def forward(self, x):
        return self.ir(self.dense(x))


# =============================================================================
# Cross-Band Attention (Sec. 2.3, Eqs. 6-12)
# =============================================================================


class CrossBandAttention(nn.Module):
    """Per-frame attention exchanging information across the B frequency bands.

    For each band b: project to per-band Q/K/V (Eqs. 6-8), average K and V over
    frequency to form compact per-band summaries, concatenate the summaries
    across bands (Eqs. 9-10), then let every (freq-bin, frame) query attend over
    the B band summaries (Eq. 11) and add the result back through W_O (Eq. 12).

    Every frame is processed independently, so this is causal as-is — no masking
    or future context is involved.
    """

    def __init__(self, channels, n_bands, reduction=4, causal=True):
        super().__init__()
        self.channels = channels
        self.n_bands = n_bands
        self.d_k = max(1, channels // reduction)
        self.scale = 1.0 / math.sqrt(self.d_k)
        # Per-band projections (the superscript-b weights in Eqs. 6-8, 12).
        self.to_q = nn.ModuleList(nn.Conv2d(channels, self.d_k, 1) for _ in range(n_bands))
        self.to_k = nn.ModuleList(nn.Conv2d(channels, self.d_k, 1) for _ in range(n_bands))
        self.to_v = nn.ModuleList(nn.Conv2d(channels, channels, 1) for _ in range(n_bands))
        self.to_out = nn.ModuleList(nn.Conv2d(channels, channels, 1) for _ in range(n_bands))

    def forward(self, band_feats):
        # band_feats: list of B tensors, each (B, C, F_b, N).
        q = [self.to_q[b](h) for b, h in enumerate(band_feats)]          # (B, d_k, F_b, N)
        k = [self.to_k[b](h) for b, h in enumerate(band_feats)]          # (B, d_k, F_b, N)
        v = [self.to_v[b](h) for b, h in enumerate(band_feats)]          # (B, C,   F_b, N)
        # Compact per-band context: average K, V over frequency, stack across bands.
        k_bar = torch.stack([ki.mean(dim=2) for ki in k], dim=2)         # (B, d_k, B, N)
        v_bar = torch.stack([vi.mean(dim=2) for vi in v], dim=2)         # (B, C,   B, N)
        out = []
        for b, h in enumerate(band_feats):
            # logits over the B band summaries, independently per (freq bin, frame).
            logits = torch.einsum("bdfn,bdmn->bfmn", q[b], k_bar) * self.scale  # (B, F_b, B, N)
            attn = torch.softmax(logits, dim=2)
            ctx = torch.einsum("bfmn,bcmn->bcfn", attn, v_bar)            # (B, C, F_b, N)
            out.append(h + self.to_out[b](ctx))                          # Eq. 12 residual
        return out


# =============================================================================
# Fusion, CRN temporal model, decoders (Sec. 2.5)
# =============================================================================


class FusionProjection(nn.Module):
    """Fuse the attended bands and pool frequency for the recurrent bottleneck.

    Concatenated band features (B, C, F, N) -> Conv -> norm -> Hardswish -> IR
    block -> adaptive average pool over frequency to `crn_freq` (the 'AvgPool2d'
    in Fig. 1). Pooling is over frequency only, so it stays causal.
    """

    def __init__(self, channels, n_features, crn_freq, expand_ratio=4, causal=True,
                 use_ir=True):
        super().__init__()
        self.conv = TFConv(channels, channels, kf=3, kt=3, causal=causal)
        self.norm = make_norm(channels, causal)
        self.act = nn.Hardswish()
        # Fig. 1's Fusion & Proj lists Conv2d + InstanceNorm + Hardswish +
        # AvgPool2d only; the extra IR block predates the fidelity audit and is
        # kept behind `use_ir` for checkpoint compatibility.
        self.ir = IRBlock(channels, expand_ratio, freq_dilation=1, causal=causal) if use_ir else None
        self.pool = FreqResample(n_features, crn_freq, "avg")

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        if self.ir is not None:
            x = self.ir(x)                          # (B, C, F, N) full frequency resolution
        return x, self.pool(x)                      # (skip, pooled CRN input)


class CRN(nn.Module):
    """Convolutional Recurrent Network for temporal modelling (Sec. 2.5, ref. [20]).

    Two GRU wirings are supported (`mode`):

    * ``"flat"`` — the GRU input is the flattened (C * crn_freq) bottleneck.
      This was the original guess; its parameter count is dominated by the
      GRU input matrix, which is inconsistent with the paper's tiny reported
      bi->uni delta (0.83M vs 0.81M, i.e. ~0.02M).
    * ``"per_bin"`` — one GRU over time with input size C, **shared across
      frequency bins** (BSRNN-style sequence modelling). This matches the
      paper's numbers: bi->uni changes params by ~0.02M while MACs change by
      ~0.2G because the same small weights are re-applied at every bin.
      Residual wiring follows Fig. 1: x + LN(Linear(GRU(x))), then the GLU
      conv block with its own residual.

    Unidirectional when causal. O(N) and frame-by-frame either way.
    """

    def __init__(self, channels, crn_freq, hidden, causal=True, mode="flat"):
        super().__init__()
        assert mode in ("flat", "per_bin"), mode
        self.channels = channels
        self.crn_freq = crn_freq
        self.mode = mode
        in_feat = channels * crn_freq if mode == "flat" else channels
        self.gru = nn.GRU(in_feat, hidden, batch_first=True, bidirectional=not causal)
        gru_out = hidden * (1 if causal else 2)
        self.linear = nn.Linear(gru_out, in_feat)
        self.ln = nn.LayerNorm(in_feat)
        self.conv1 = TFConv(channels, channels * 2, kf=3, kt=3, causal=causal)
        self.norm1 = make_norm(channels * 2, causal)
        self.conv2 = TFConv(channels, channels, kf=3, kt=3, causal=causal)
        self.norm2 = make_norm(channels, causal)
        self.act = nn.PReLU()
        # Streaming state: the GRU hidden state carried across frames. The conv
        # layers stream via their own TFConv buffers. Default off.
        self.streaming = False
        self._h = None

    def stream_reset(self):
        self._h = None

    def _gru_pass(self, g):
        if self.streaming:
            g, self._h = self.gru(g, self._h)             # carry hidden state across frames
        else:
            g, _ = self.gru(g)
        return g

    def forward(self, x):  # (B, C, F, N)
        b, c, fr, n = x.shape
        if self.mode == "flat":
            g = x.permute(0, 3, 1, 2).reshape(b, n, c * fr)   # (B, N, C*F)
            g = self.ln(self.linear(self._gru_pass(g)))       # (B, N, C*F)
            g = g.reshape(b, n, c, fr).permute(0, 2, 3, 1)    # (B, C, F, N)
        else:
            g = x.permute(0, 2, 3, 1).reshape(b * fr, n, c)   # (B*F, N, C) shared weights
            g = self.ln(self.linear(self._gru_pass(g)))       # (B*F, N, C)
            g = g.reshape(b, fr, n, c).permute(0, 3, 1, 2)    # (B, C, F, N)
            g = x + g                                         # Fig. 1: residual after LN
        y = self.norm1(self.conv1(g))                     # (B, 2C, F, N)
        y = F.glu(y, dim=1)                               # (B, C, F, N)
        y = self.act(self.norm2(self.conv2(y)))
        return g + y


class EfficientDecoder(nn.Module):
    """Decoder body (Fig. 1): EfficientDenseNet -> upscale frequency -> IR -> norm.

    With ``upsample_first`` the dense body runs at full frequency resolution
    (upscale -> dense -> IR -> norm) instead of at the CRN bottleneck
    resolution. Parameter count is identical; MACs scale with the resolution
    the dense body runs at.
    """

    def __init__(self, channels, depth, in_freq, full_freq, expand_ratio=4, causal=True,
                 growth=None, upsample_first=False, ir_expand_ratio=None):
        super().__init__()
        self.upsample_first = upsample_first
        self.dense = EfficientDenseNet(channels, depth, expand_ratio, causal, growth=growth,
                                       ir_expand_ratio=ir_expand_ratio)
        self.upsample = FreqResample(in_freq, full_freq, "nearest")
        self.ir = IRBlock(channels, ir_expand_ratio or expand_ratio,
                          freq_dilation=1, causal=causal)
        self.norm = make_norm(channels, causal)

    def forward(self, x):
        if self.upsample_first:
            x = self.dense(self.upsample(x))
        else:
            x = self.upsample(self.dense(x))
        return self.norm(self.ir(x))


class LearnableSigmoid2d(nn.Module):
    """Bounded learnable sigmoid mask (MP-SENet): beta * sigmoid(slope_f * x).

    `beta` caps the gain (default 2.0, so the magnitude mask can exceed 1 to
    restore over-suppressed bins); `slope` is a per-frequency learnable scale.
    """

    def __init__(self, n_freq, beta=2.0):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(n_freq))

    def forward(self, x):  # (B, F, N)
        return self.beta * torch.sigmoid(self.slope[None, :, None] * x)


class MagMaskDecoder(nn.Module):
    """Predict a bounded magnitude mask and apply it to the (compressed) input.

    Fuses the decoder body's output with the full-resolution `skip` features
    (the fusion output before the CRN's frequency pooling) so the head sees
    per-bin spectral detail, not just the coarse recurrent bottleneck.
    """

    def __init__(self, channels, depth, in_freq, n_freq, expand_ratio=4, causal=True,
                 growth=None, upsample_first=False, use_skip=True, ir_expand_ratio=None):
        super().__init__()
        self.use_skip = use_skip
        self.body = EfficientDecoder(channels, depth, in_freq, n_freq, expand_ratio, causal,
                                     growth=growth, upsample_first=upsample_first,
                                     ir_expand_ratio=ir_expand_ratio)
        if use_skip:
            self.merge = nn.Conv2d(channels * 2, channels, 1)
            self.merge_norm = make_norm(channels, causal)
        self.act = nn.Hardswish()
        self.out = nn.Conv2d(channels, 1, 1)
        self.lsig = LearnableSigmoid2d(n_freq)

    def forward(self, x, skip, mag_in):            # x: (B,C,crn_freq,N), skip: (B,C,F,N)
        y = self.body(x)                           # (B, C, F, N)
        if self.use_skip:
            y = self.merge_norm(self.merge(torch.cat([y, skip], dim=1)))
        y = self.act(y)
        mask = self.lsig(self.out(y).squeeze(1))   # (B, F, N)
        return mag_in * mask, mask


class PhaseDecoder(nn.Module):
    """Predict enhanced phase via a two-head atan2 projection (anti-wrapping).

    Like the magnitude decoder, it fuses in the full-resolution `skip` features.
    This matters most here: phase decorrelates rapidly across frequency, so the
    head needs per-bin detail the coarse CRN bottleneck cannot provide.
    """

    def __init__(self, channels, depth, in_freq, n_freq, expand_ratio=4, causal=True,
                 growth=None, upsample_first=False, use_skip=True, ir_expand_ratio=None):
        super().__init__()
        self.use_skip = use_skip
        self.body = EfficientDecoder(channels, depth, in_freq, n_freq, expand_ratio, causal,
                                     growth=growth, upsample_first=upsample_first,
                                     ir_expand_ratio=ir_expand_ratio)
        if use_skip:
            self.merge = nn.Conv2d(channels * 2, channels, 1)
            self.merge_norm = make_norm(channels, causal)
        self.act = nn.Hardswish()
        self.conv_r = nn.Conv2d(channels, 1, 1)
        self.conv_i = nn.Conv2d(channels, 1, 1)

    def forward(self, x, skip):
        y = self.body(x)
        if self.use_skip:
            y = self.merge_norm(self.merge(torch.cat([y, skip], dim=1)))
        y = self.act(y)
        real = self.conv_r(y).squeeze(1)           # (B, F, N)
        imag = self.conv_i(y).squeeze(1)
        return torch.atan2(imag, real)


# =============================================================================
# BASENet
# =============================================================================


class BASENet(nn.Module):
    """BASENet speech-enhancement model operating on compressed magnitude + phase.

    forward(mag, pha) -> (mag_g, pha_g, com_g) where `mag`/`pha` are the
    power-law-compressed magnitude and phase of the noisy STFT, shape (B, F, N),
    and `com_g` is the enhanced complex spectrogram stacked as (B, F, N, 2)
    (real, imag) — matching ``common.dataset.mag_pha_stft``'s `com` layout.
    """

    def __init__(self, n_fft=400, hop_size=100, sampling_rate=16000,
                 compress_factor=0.3,
                 base_channels=32, expand_ratio=4, attn_reduction=4,
                 band_edges_hz=(0, 1000, 4000, 8000), band_depths=(4, 3, 2),
                 crn_freq=16, crn_hidden=128, dec_depth=2, causal=True,
                 growth=None, crn_mode="flat", dec_upsample_first=False,
                 dec_skip=True, band_stems=False, fusion_ir=True,
                 enc_dense_expand_ratio=None, enc_ir_expand_ratio=None,
                 dec_dense_expand_ratio=None, dec_ir_expand_ratio=None):
        super().__init__()
        # The paper's IR block spec ("r in {2,4}") doesn't say which modules
        # get which ratio; each defaults to the single `expand_ratio` so
        # existing configs are unaffected unless they set these explicitly.
        enc_dense_expand_ratio = enc_dense_expand_ratio or expand_ratio
        enc_ir_expand_ratio = enc_ir_expand_ratio or expand_ratio
        dec_dense_expand_ratio = dec_dense_expand_ratio or expand_ratio
        dec_ir_expand_ratio = dec_ir_expand_ratio or expand_ratio
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.sampling_rate = sampling_rate
        self.compress_factor = compress_factor
        self.causal = causal
        self.n_features = n_fft // 2 + 1

        n_bands = len(band_edges_hz) - 1
        assert len(band_depths) == n_bands, \
            f"band_depths {band_depths} must have one entry per band ({n_bands})"
        self.band_bins = hz_edges_to_bins(band_edges_hz, n_fft, sampling_rate, self.n_features)
        self.band_slices = list(zip(self.band_bins, self.band_bins[1:]))

        C = base_channels
        if band_stems:
            # Fig. 1: each H^(b) block starts with its own Conv2d + norm +
            # Hardswish stem; the 1x1 projection to C channels is per-band.
            self.input_proj = None
            self.band_stem = nn.ModuleList(
                nn.Sequential(nn.Conv2d(2, C, 1), make_norm(C, causal), nn.Hardswish())
                for _ in band_depths
            )
        else:
            self.input_proj = nn.Conv2d(2, C, 1)   # single shared 1x1 projection
            self.band_stem = None
        self.encoders = nn.ModuleList(
            EfficientDenseNet(C, depth, enc_dense_expand_ratio, causal, growth=growth,
                              ir_expand_ratio=enc_ir_expand_ratio)
            for depth in band_depths
        )
        self.cross_band = CrossBandAttention(C, n_bands, attn_reduction, causal)
        self.fusion = FusionProjection(C, self.n_features, crn_freq, expand_ratio, causal,
                                       use_ir=fusion_ir)
        self.crn = CRN(C, crn_freq, crn_hidden, causal, mode=crn_mode)
        self.mag_decoder = MagMaskDecoder(
            C, dec_depth, crn_freq, self.n_features, dec_dense_expand_ratio, causal,
            growth=growth, upsample_first=dec_upsample_first, use_skip=dec_skip,
            ir_expand_ratio=dec_ir_expand_ratio)
        self.phase_decoder = PhaseDecoder(
            C, dec_depth, crn_freq, self.n_features, dec_dense_expand_ratio, causal,
            growth=growth, upsample_first=dec_upsample_first, use_skip=dec_skip,
            ir_expand_ratio=dec_ir_expand_ratio)

    def forward(self, mag, pha):                   # (B, F, N) each
        x = torch.stack([mag, pha], dim=1)         # (B, 2, F, N)
        if self.band_stem is not None:
            bands = [stem(x[:, :, lo:hi, :])
                     for stem, (lo, hi) in zip(self.band_stem, self.band_slices)]
        else:
            x = self.input_proj(x)                 # (B, C, F, N)
            bands = [x[:, :, lo:hi, :] for lo, hi in self.band_slices]
        feats = [enc(band) for enc, band in zip(self.encoders, bands)]
        feats = self.cross_band(feats)
        fused = torch.cat(feats, dim=2)            # (B, C, F, N)
        skip, z = self.fusion(fused)               # full-res skip + pooled CRN input
        z = self.crn(z)                            # (B, C, crn_freq, N)
        mag_g, _ = self.mag_decoder(z, skip, mag)
        pha_g = self.phase_decoder(z, skip)
        com_g = torch.stack((mag_g * torch.cos(pha_g), mag_g * torch.sin(pha_g)), dim=-1)
        return mag_g, pha_g, com_g


# =============================================================================
# Factory
# =============================================================================


def build_basenet(h=None, causal=True):
    """Instantiate BASENet, optionally overriding defaults from a dict-like `h`.

    Mirrors ``convfsenet.model.build_causal_model``: pass an AttrDict / dict
    (e.g. loaded from ``configs/*.json``) to override hyperparameters. Defaults
    follow the paper's experimental setup (n_fft=400, hop=100, 16 kHz, B=3 Bark
    bands at [0, 1, 4, 8] kHz, depths [4, 3, 2]). `causal=True` selects the
    streamable variant.
    """
    h = h or {}

    def g(key, default):
        return h.get(key, default) if hasattr(h, "get") else getattr(h, key, default)

    return BASENet(
        n_fft=int(g("n_fft", 400)),
        hop_size=int(g("hop_size", 100)),
        sampling_rate=int(g("sampling_rate", 16000)),
        compress_factor=float(g("compress_factor", 0.3)),
        base_channels=int(g("base_channels", 32)),
        expand_ratio=int(g("expand_ratio", 4)),
        attn_reduction=int(g("attn_reduction", 4)),
        band_edges_hz=tuple(g("band_edges_hz", (0, 1000, 4000, 8000))),
        band_depths=tuple(g("band_depths", (4, 3, 2))),
        crn_freq=int(g("crn_freq", 16)),
        crn_hidden=int(g("crn_hidden", 128)),
        dec_depth=int(g("dec_depth", 2)),
        causal=bool(g("causal", causal)),
        growth=g("growth", None),
        crn_mode=str(g("crn_mode", "flat")),
        dec_upsample_first=bool(g("dec_upsample_first", False)),
        dec_skip=bool(g("dec_skip", True)),
        band_stems=bool(g("band_stems", False)),
        fusion_ir=bool(g("fusion_ir", True)),
        enc_dense_expand_ratio=g("enc_dense_expand_ratio", None) and int(g("enc_dense_expand_ratio", None)),
        enc_ir_expand_ratio=g("enc_ir_expand_ratio", None) and int(g("enc_ir_expand_ratio", None)),
        dec_dense_expand_ratio=g("dec_dense_expand_ratio", None) and int(g("dec_dense_expand_ratio", None)),
        dec_ir_expand_ratio=g("dec_ir_expand_ratio", None) and int(g("dec_ir_expand_ratio", None)),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_basenet(causal=True).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {model.__class__.__name__} (causal), params: {n_params:,}")
    print(f"bands (bins): {model.band_slices}")

    # smoke test: 40 STFT frames of a 201-bin spectrogram, batch 2
    b, fbins, n = 2, model.n_features, 40
    mag = torch.rand(b, fbins, n)
    pha = (torch.rand(b, fbins, n) * 2 - 1) * math.pi
    with torch.no_grad():
        mag_g, pha_g, com_g = model(mag, pha)
    print(f"forward ok: mag_g={tuple(mag_g.shape)}, pha_g={tuple(pha_g.shape)}, "
          f"com_g={tuple(com_g.shape)}")

    # quick causality probe: perturbing a future frame must not change past outputs
    t_f = n // 2
    mag2, pha2 = mag.clone(), pha.clone()
    mag2[:, :, t_f:] += 5.0
    pha2[:, :, t_f:] = (torch.rand(b, fbins, n - t_f) * 2 - 1) * math.pi
    with torch.no_grad():
        mag_g2, pha_g2, _ = model(mag2, pha2)
    max_delta = (mag_g[:, :, :t_f] - mag_g2[:, :, :t_f]).abs().max().item()
    print(f"causality: max |delta| on past mag outputs = {max_delta:.2e} (expect ~0)")
