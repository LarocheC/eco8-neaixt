"""FSPEN — Full-band and Sub-band Path Enhancement Network.

Re-implementation of

    L. Yang, W. Liu, R. Meng, G. Lee, S. Baek and H.-G. Moon, "FSPEN: An
    Ultra-Lightweight Network for Real Time Speech Enhancement",
    ICASSP 2024. https://ieeexplore.ieee.org/document/10446016

Written from the paper's architecture description for this framework, so its
quality can be measured on the same VoiceBank-DEMAND pipeline as the other
models here. The exact layer hyperparameters (encoder kernel/stride ladders,
the 5-group sub-band split of the 257-bin spectrum, the dual-path sizes) are
pinned to the unofficial reference implementation at
https://github.com/gitwukeyi/FSPEN (no license file, hence a re-implementation
rather than a port; hyperparameter values are facts, not expression). Verified
against the reference: with weights copied across, ``enhance_spectrum`` is
bit-exact to the reference forward — modulo two deliberate deviations:

  * the STFT uses the Hann window shared by every model in this repo (the
    reference profiles with a Hamming window) so results stay comparable
    across families;
  * the sub-band decoders read *consecutive* band groups of the dual-path
    feature ([0:8], [8:14], ... [26:32]), as the paper intends. The reference
    never advances its ``start_idx``, so all five of its decoder groups read
    bands [0:bands_g] and bands 8..31 never reach the sub-band decoders — an
    upstream indexing bug, not architecture.

Architecture (per 16 ms hop, 257-bin frames):
  * **full-band path** — the complex spectrum ``(re, im)`` runs through a
    3-level Conv1d encoder over frequency (2→4→16→32 channels, /2 stride each
    level) capturing global spectral structure, and a mirrored
    ConvTranspose1d U-Net decoder predicts a 2-channel complex mask.
  * **sub-band path** — the magnitude spectrum is split into 5 groups of
    increasing bandwidth (2/3/6/11/20 bins per band, ERB-like); one Conv1d per
    group summarizes each band into a 32-dim feature. A per-group Linear
    decoder predicts magnitude gains, upsampling each band back to its bins.
  * **dual-path extension (DPE)** — the merged full+sub features
    ``(B, bands=32, T, N=16)`` pass through ``dpe_modules`` dual-path blocks:
    a bidirectional *intra* GRU over the 32 bands of one frame, then a
    unidirectional grouped *inter* GRU along time (8 groups of width 2). Only
    the inter GRUs carry state across frames, so the whole model streams
    frame-by-frame with ``dpe_modules`` hidden-state tensors and nothing else.
  * output = complex mask × complex spectrum + magnitude gains × magnitude
    (the sub-band term is broadcast onto both real and imaginary channels,
    matching the reference implementation).

forward(src, tgt=None) takes waveforms (B, samples) and returns the results
dict consumed by the CMGAN trainer (``est`` waveform + compressed spectra /
magnitudes for the losses), like ``lisennet/model.py``. The deployable core is
``enhance_spectrum`` (pure tensor ops, explicit hidden state, exported to ONNX
by ``fspen/export_onnx.py`` and streamed by ``fspen/streaming.py``).
"""

import torch
import torch.nn as nn


# =============================================================================
# Reference hyperparameters (gitwukeyi/FSPEN config for 16 kHz, n_fft 512)
# =============================================================================

# Full-band encoder ladder: 257 -> 128 -> 64 -> 32 frequency, 2 -> 32 channels.
DEFAULT_FULL_BAND_ENCODER = (
    {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2},
    {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3},
    {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2},
)

# Mirrored decoder ladder: 32 -> 64 -> 128 -> 256 frequency (DC bin re-added by
# the zero mask pad). in_channels counts the skip concat (2x the running width).
DEFAULT_FULL_BAND_DECODER = (
    {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2},
    {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3},
    {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2},
)

# Sub-band split of the 257 bins: 5 groups of increasing bandwidth. Each conv
# summarizes one group into (bins - k + 2p) / s + 1 bands (8/6/6/6/6 = 32).
DEFAULT_SUB_BAND_ENCODER = (
    {"start": 0, "end": 16, "kernel_size": 4, "stride": 2, "padding": 1},
    {"start": 16, "end": 34, "kernel_size": 7, "stride": 3, "padding": 2},
    {"start": 34, "end": 70, "kernel_size": 11, "stride": 5, "padding": 2},
    {"start": 70, "end": 136, "kernel_size": 20, "stride": 10, "padding": 4},
    {"start": 136, "end": 257, "kernel_size": 30, "stride": 20, "padding": 5},
)


def _sub_band_geometry(specs):
    """Per-group (n_bands, band_width) implied by the sub-band conv ladders."""
    bands, widths = [], []
    for s in specs:
        group_width = s["end"] - s["start"]
        n = (group_width - s["kernel_size"] + 2 * s["padding"]) // s["stride"] + 1
        bands.append(n)
        widths.append(group_width // n)
    return tuple(bands), tuple(widths)


def _conv_out_len(length, kernel_size, stride, padding):
    return (length + 2 * padding - kernel_size) // stride + 1


# =============================================================================
# Building blocks
# =============================================================================


class FullBandEncoderBlock(nn.Module):
    """Conv1d over frequency + BatchNorm + ELU on ``(B*T, C, F)`` frames."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU()

    def forward(self, x):                                    # (B*T, C, F)
        return self.act(self.norm(self.conv(x)))


class FullBandDecoderBlock(nn.Module):
    """Skip-concat -> 1x1 channel halving -> ConvTranspose1d upsample (x2 freq)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels // 2, kernel_size=1)
        self.conv_t = nn.ConvTranspose1d(in_channels // 2, out_channels,
                                         kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU()

    def forward(self, x, skip):                              # both (B*T, C, F)
        x = torch.cat([x, skip], dim=1)
        return self.act(self.norm(self.conv_t(self.conv(x))))


class SubBandEncoderBlock(nn.Module):
    """One frequency group -> per-band features: slice + Conv1d + ReLU."""

    def __init__(self, start, end, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.start = start
        self.end = end
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.ReLU()

    def forward(self, amp):                                  # (B*T, 1, F)
        return self.act(self.conv(amp[:, :, self.start:self.end]))


class SubBandDecoderBlock(nn.Module):
    """Per-group gain decoder: band features -> ``band_width`` gains per band.

    Concatenates the DPE feature slice for this group's bands with the group's
    encoder output along channels, then a per-band Linear maps the 2x-channel
    feature to the group's ``band_width`` bins (ReLU keeps gains non-negative).
    """

    def __init__(self, in_features, band_width, start_band, end_band):
        super().__init__()
        self.start_band = start_band
        self.end_band = end_band
        self.fc = nn.Linear(in_features, band_width)
        self.act = nn.ReLU()

    def forward(self, feature, sub_encode):                  # (B*T, C, bands_total/bands_g)
        x = torch.cat([feature[:, :, self.start_band:self.end_band], sub_encode], dim=1)
        x = self.fc(x.transpose(1, 2))                       # (B*T, bands_g, band_width)
        x = self.act(x)
        return x.reshape(x.shape[0], -1)                     # (B*T, bands_g * band_width)


class GroupedGRU(nn.Module):
    """Grouped unidirectional GRU: ``groups`` independent GRUs over channel slices.

    The "path extension" trick that keeps the inter-frame RNN tiny: the N-dim
    feature is split into ``groups`` slices of width N/groups, each with its own
    GRU of hidden size ``hidden_size/groups``; outputs are re-concatenated.
    Hidden state is one stacked tensor ``(groups, B', hidden_size/groups)``
    (``None`` = zeros), so callers carry a single tensor per module.
    """

    def __init__(self, input_size, hidden_size, groups):
        super().__init__()
        assert input_size % groups == 0 and hidden_size % groups == 0, \
            f"input/hidden ({input_size}/{hidden_size}) must divide groups ({groups})"
        self.groups = groups
        self.rnns = nn.ModuleList(
            nn.GRU(input_size // groups, hidden_size // groups, batch_first=True)
            for _ in range(groups)
        )

    def forward(self, x, h=None):                            # (B', T, N)
        width = x.shape[2] // self.groups
        outs, states = [], []
        for g, rnn in enumerate(self.rnns):
            out, state = rnn(x[:, :, g * width:(g + 1) * width],
                             None if h is None else h[g:g + 1].contiguous())
            outs.append(out)                                 # (B', T, hidden/groups)
            states.append(state)                             # (1, B', hidden/groups)
        return torch.cat(outs, dim=2), torch.cat(states, dim=0)


class DualPathExtensionBlock(nn.Module):
    """One DPE block: bidirectional intra-band GRU + grouped inter-frame GRU.

    Operates on ``(B, bands, T, N)``. The intra path mixes the ``bands`` of one
    frame (bidirectional over frequency — fully available per frame, so it
    streams); the inter path runs along time per band (unidirectional, grouped)
    and is the only stateful piece of the whole model. Both paths are residual.
    """

    def __init__(self, input_size, intra_hidden, inter_hidden, groups):
        super().__init__()
        self.intra_rnn = nn.GRU(input_size, intra_hidden, batch_first=True,
                                bidirectional=True)
        self.intra_fc = nn.Linear(2 * intra_hidden, input_size)
        self.intra_norm = nn.LayerNorm(input_size)
        self.inter_rnn = GroupedGRU(input_size, inter_hidden, groups)
        self.inter_fc = nn.Linear(inter_hidden, input_size)

    def forward(self, x, h=None):                            # (B, F, T, N)
        b, f, t, n = x.shape

        intra = x.transpose(1, 2).reshape(b * t, f, n)       # (B*T, F, N)
        intra, _ = self.intra_rnn(intra)
        intra = self.intra_fc(intra)
        intra = intra.reshape(b, t, f, n).transpose(1, 2)    # (B, F, T, N)
        intra = x + self.intra_norm(intra)                   # residual

        inter = intra.reshape(b * f, t, n)                   # (B*F, T, N)
        inter, h_out = self.inter_rnn(inter, h)
        inter = self.inter_fc(inter.reshape(b, f, t, -1))    # (B, F, T, N)

        return intra + inter, h_out                          # residual


# =============================================================================
# FSPEN
# =============================================================================


class FSPEN(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, compress_factor=0.3,
                 sub_band_channels=32, compress_rate=2,
                 dpe_modules=3, dpe_groups=8, dpe_intra_hidden=16, dpe_inter_hidden=16,
                 full_band_encoder=DEFAULT_FULL_BAND_ENCODER,
                 full_band_decoder=DEFAULT_FULL_BAND_DECODER,
                 sub_band_encoder=DEFAULT_SUB_BAND_ENCODER):
        super().__init__()
        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1
        self.hop_length = hop_length
        self.compress_factor = compress_factor              # loss domain only

        assert sub_band_encoder[-1]["end"] == self.n_freqs, \
            "sub-band split must cover the full spectrum"

        # ----- full-band path ------------------------------------------------
        self.full_band_encoders = nn.ModuleList(
            FullBandEncoderBlock(**spec) for spec in full_band_encoder)
        fb_channels = full_band_encoder[-1]["out_channels"]
        self.global_feature = nn.Conv1d(fb_channels, fb_channels, kernel_size=1)
        self.full_band_decoders = nn.ModuleList(
            FullBandDecoderBlock(**spec) for spec in full_band_decoder)

        fb_freq = self.n_freqs
        for spec in full_band_encoder:
            fb_freq = _conv_out_len(fb_freq, spec["kernel_size"],
                                    spec["stride"], spec["padding"])

        # ----- sub-band path -------------------------------------------------
        assert fb_channels == sub_band_channels, \
            "full-band and sub-band encoder widths must match for the merge"
        self.sub_band_encoders = nn.ModuleList(
            SubBandEncoderBlock(spec["start"], spec["end"], 1, sub_band_channels,
                                spec["kernel_size"], spec["stride"], spec["padding"])
            for spec in sub_band_encoder)
        self.bands_per_group, self.band_widths = _sub_band_geometry(sub_band_encoder)
        n_bands = sum(self.bands_per_group)

        starts = [sum(self.bands_per_group[:i]) for i in range(len(self.bands_per_group))]
        self.sub_band_decoders = nn.ModuleList(
            SubBandDecoderBlock(2 * sub_band_channels, width, start, start + bands)
            for width, bands, start in zip(self.band_widths, self.bands_per_group, starts))

        # ----- merge -> DPE -> split ------------------------------------------
        # After the freq-axis concat the feature is (B*T, C=32, fb_freq + n_bands);
        # a Linear halves the frequency axis and a 1x1 Conv1d halves the channels,
        # giving the (B, bands, T, N) dual-path feature (reference: 32 bands, N=16).
        merge_freq = fb_freq + n_bands
        self.dpe_bands = merge_freq // compress_rate
        self.dpe_input = fb_channels // compress_rate
        # The split doubles the band axis back to 2*dpe_bands and hands one
        # half to each decoder, so both decoders require exactly dpe_bands
        # bands — the full-band skip-concat needs fb_freq, the sub-band
        # slices need n_bands. Fail here with a clear message instead of an
        # opaque cat() error deep in forward.
        assert self.dpe_bands == fb_freq == n_bands, (
            f"decoder shape contract requires (fb_freq + n_bands) // compress_rate"
            f" == fb_freq == n_bands; got fb_freq={fb_freq}, n_bands={n_bands}, "
            f"compress_rate={compress_rate}")
        self.feature_merge = nn.Sequential(
            nn.Linear(merge_freq, self.dpe_bands),
            nn.ELU(),
            nn.Conv1d(fb_channels, self.dpe_input, kernel_size=1),
        )

        self.dpe_groups = dpe_groups
        self.dpe_state_width = dpe_inter_hidden // dpe_groups
        self.dpe_blocks = nn.ModuleList(
            DualPathExtensionBlock(self.dpe_input, dpe_intra_hidden,
                                   dpe_inter_hidden, dpe_groups)
            for _ in range(dpe_modules))

        # Split mirrors the merge and doubles the frequency axis; the trailing
        # x2 is reshaped into (full-band, sub-band) decoder features.
        self.feature_split = nn.Sequential(
            nn.Conv1d(self.dpe_input, fb_channels, kernel_size=1),
            nn.Linear(self.dpe_bands, 2 * self.dpe_bands),
            nn.ELU(),
        )

        # Masks are predicted for bins 1..256; the DC bin is zero-padded away
        # (removes any DC component), restoring the 257-bin spectrum.
        self.mask_pad = nn.ConstantPad2d((1, 0, 0, 0), value=0.0)

    # ----- spectral helpers ---------------------------------------------------
    def apply_stft(self, x):
        """Waveform (B, samples) -> complex spectrum (B, T, F)."""
        spec = torch.stft(x, self.n_fft, self.hop_length,
                          window=torch.hann_window(self.n_fft).to(x.device),
                          onesided=True, return_complex=True)
        return spec.transpose(1, 2)                          # (B, T, F)

    def apply_istft(self, x, length=None):
        """Complex spectrum (B, T, F) -> waveform (B, samples)."""
        return torch.istft(x.transpose(1, 2), self.n_fft, self.hop_length,
                           window=torch.hann_window(self.n_fft).to(x.device),
                           onesided=True, length=length)

    def power_compress(self, x):
        """Power-compress a complex spectrum (loss domain, CMGAN convention).

        Returns ``(compressed complex, compressed magnitude)``, computed as
        ``x * (|x|^2 + eps)^((c-1)/2)`` and ``(|x|^2 + eps)^(c/2)`` — no
        abs/angle, so both stay differentiable at exactly-zero bins. That
        matters here and not in the other families: the *estimated* spectrum
        is compressed for the loss, so this is on the gradient path, and the
        dataset zero-pads utterances shorter than segment_size
        (common/dataset.py), making all-zero frames routine — abs/angle have
        NaN gradients at 0+0j and would NaN every parameter on the first such
        batch. The ``+1e-9`` guard mirrors ``mag_pha_stft``'s.
        """
        mag2 = x.real ** 2 + x.imag ** 2 + 1e-9
        return x * mag2 ** ((self.compress_factor - 1) / 2), mag2 ** (self.compress_factor / 2)

    @staticmethod
    def spec_features(spec):
        """Complex spectrum (B, T, F) -> network inputs (B, T, 2, F) + (B, T, 1, F)."""
        stacked = torch.stack([spec.real, spec.imag], dim=2)  # (B, T, 2, F)
        amp = spec.abs().unsqueeze(2)                         # (B, T, 1, F)
        return stacked, amp

    # ----- streaming state ------------------------------------------------------
    def init_hidden(self, batch_size, device="cpu", dtype=torch.float32):
        """Zero inter-GRU states: one (groups, B*bands, width) tensor per DPE block."""
        return [torch.zeros(self.dpe_groups, batch_size * self.dpe_bands,
                            self.dpe_state_width, device=device, dtype=dtype)
                for _ in self.dpe_blocks]

    # ----- the deployable core --------------------------------------------------
    def enhance_spectrum(self, spec, amp, hidden=None):
        """The streamable enhancer: (spec, amp) frames -> enhanced spectrum.

        :param spec: (B, T, 2, F) noisy complex spectrum as stacked re/im.
        :param amp: (B, T, 1, F) noisy magnitude spectrum.
        :param hidden: list of ``dpe_modules`` inter-GRU states
            (``init_hidden`` layout), or ``None`` for zeros (start of stream /
            whole utterance).
        :return: (est (B, T, 2, F), hidden_out list).
        """
        b, t, _, f = spec.shape
        x = spec.reshape(b * t, 2, f)                        # frames as batch
        a = amp.reshape(b * t, 1, f)

        # Encoders.
        skips = []
        for encoder in self.full_band_encoders:
            x = encoder(x)
            skips.append(x)
        global_feature = self.global_feature(x)              # (B*T, C, fb_freq)

        sub_encodes = [encoder(a) for encoder in self.sub_band_encoders]
        local_feature = torch.cat(sub_encodes, dim=2)        # (B*T, C, n_bands)

        # Merge -> dual-path over (bands, time) -> split.
        feature = torch.cat([global_feature, local_feature], dim=2)
        feature = self.feature_merge(feature)                # (B*T, N, bands)
        feature = feature.reshape(b, t, self.dpe_input, self.dpe_bands)
        feature = feature.permute(0, 3, 1, 2).contiguous()   # (B, bands, T, N)

        hidden_out = []
        for i, block in enumerate(self.dpe_blocks):
            feature, h = block(feature, None if hidden is None else hidden[i])
            hidden_out.append(h)

        feature = feature.permute(0, 2, 3, 1).contiguous()   # (B, T, N, bands)
        feature = feature.reshape(b * t, self.dpe_input, self.dpe_bands)
        feature = self.feature_split(feature)                # (B*T, C, 2*bands)
        feature = feature.reshape(b * t, -1, self.dpe_bands, 2)

        # Full-band decoder: complex mask from the U-Net path.
        full = feature[..., 0]
        for decoder, skip in zip(self.full_band_decoders, reversed(skips)):
            full = decoder(full, skip)                       # (B*T, 2, F-1)

        # Sub-band decoder: magnitude gains from the per-group Linear path.
        sub = feature[..., 1]
        gains = torch.cat([decoder(sub, enc) for decoder, enc
                           in zip(self.sub_band_decoders, sub_encodes)], dim=1)

        full_mask = self.mask_pad(full.reshape(b, t, 2, f - 1))    # (B, T, 2, F)
        sub_gains = self.mask_pad(gains.reshape(b, t, 1, f - 1))   # (B, T, 1, F)

        # Sub-band term broadcasts onto both re and im channels (reference
        # behaviour: the magnitude correction is added to each component).
        return spec * full_mask + amp * sub_gains, hidden_out

    # ----- forward ----------------------------------------------------------
    def forward(self, src, tgt=None):
        if tgt is None:
            tgt = src
        src_spec = self.apply_stft(src)                      # (B, T, F) complex
        spec, amp = self.spec_features(src_spec)

        est, _ = self.enhance_spectrum(spec, amp)            # (B, T, 2, F)
        est_spec = torch.complex(est[:, :, 0], est[:, :, 1])  # (B, T, F)
        est_wav = self.apply_istft(est_spec, length=tgt.size(-1))

        # Losses live in the compressed domain (CMGAN convention shared with
        # the other trainers here); the network itself runs uncompressed.
        est_spec_c, est_mag = self.power_compress(est_spec)
        tgt_spec_c, tgt_mag = self.power_compress(self.apply_stft(tgt))

        return {
            "tgt": tgt, "tgt_spec": tgt_spec_c, "tgt_mag": tgt_mag,
            "est": est_wav, "est_spec": est_spec_c, "est_mag": est_mag,
        }


def build_fspen(h=None):
    """Instantiate FSPEN from a dict-like config (defaults = the reference config)."""
    h = h or {}

    def g(key, default):
        return h.get(key, default) if hasattr(h, "get") else getattr(h, key, default)

    return FSPEN(
        n_fft=int(g("n_fft", 512)),
        hop_length=int(g("hop_size", 256)),
        compress_factor=float(g("compress_factor", 0.3)),
        sub_band_channels=int(g("sub_band_channels", 32)),
        compress_rate=int(g("compress_rate", 2)),
        dpe_modules=int(g("dpe_modules", 3)),
        dpe_groups=int(g("dpe_groups", 8)),
        dpe_intra_hidden=int(g("dpe_intra_hidden", 16)),
        dpe_inter_hidden=int(g("dpe_inter_hidden", 16)),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_fspen().eval()
    print(f"FSPEN params: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(2, 32000)
    with torch.no_grad():
        r = model(x, x)
    print(f"forward ok: est={tuple(r['est'].shape)}, est_mag={tuple(r['est_mag'].shape)}, "
          f"est_spec={tuple(r['est_spec'].shape)}")
