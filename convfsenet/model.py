"""
Standalone version of the model described by
`config/model/convfsenet/quant_td.yaml` from the `convolve_dyn_experiments` repo.

It bundles every class the model needs (BaseModel, TCMBlock, TCM, ConvFSENet and
the quant-friendly variants) into a single file. Build with `build_model()`.

The model carries no loss: training uses the MP-SENet objective from
common/losses.py, driven by convfsenet/train.py::forward_spectra. (It previously
owned a PreProcLoss(DynCompMSE); that is gone.)

Composed config (resolved from quant_td -> quant -> shared):

    _target_: dyn_experiments.models.ConvFSENet_QuantFriendly_TD
    n_fft: 512
    win_length: 512
    n_features: 257             # nfft_to_bins(512) = 512//2 + 1
    n_channels_res: 128
    n_channels_conv: 256
    kernel_size: 3
    n_blocks: 3
    n_stacks: 3
    extractor_type: "mag"
    compress_factor: null
    causal: false
    preproc:  Spectrogram(n_fft=512, win_length=512, hop_length=256, power=None)
    postproc: InverseSpectrogram(n_fft=512, win_length=512, hop_length=256)
"""

import torch
from torch import nn
from torch.nn import functional as F


# =============================================================================
# Building blocks
# =============================================================================


class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., : -self.chomp_size].contiguous()


def _get_normalization(norm_type):
    # quant variant pins norm_type="batch"; keep the whole dispatch for clarity.
    if norm_type == "batch":
        return nn.BatchNorm1d
    if norm_type in (None, "none"):
        return nn.Identity
    raise ValueError(f"unsupported norm_type for the standalone: {norm_type}")


# Magnitude-compression epsilon. (|stft| + eps) ** c keeps the d/dx gradient
# finite at |stft|=0 (raw x**c has an infinite slope there for c<1).
_MAG_EPS = 1e-9


def compress_magnitude(mag, compress_factor):
    """Power-law magnitude compression: (mag + eps) ** compress_factor.

    Shared single source of truth for the offline feature extractor and the
    streaming wrappers, so the +eps and exponent never drift between them.
    """
    return (mag + _MAG_EPS).pow(compress_factor)


def _get_feature_extractor(extractor_type, compress_factor=None):
    """Return a callable complex-STFT -> real-feature extractor.

    'mag'            : plain magnitude |stft|.
    'mag_compressed' : power-compressed magnitude (|stft|+eps)**compress_factor.
                       Compressing the input squashes the 60+ dB dynamic range
                       of speech spectra into an int8-friendly range — the
                       same trick NSNet2 uses (compress_factor 0.3) that makes
                       its static int8 near loss-free.
    """
    if extractor_type == "mag":
        return lambda x_stft: x_stft.abs()
    if extractor_type == "mag_compressed":
        c = float(compress_factor) if compress_factor else 0.3
        return lambda x_stft: compress_magnitude(x_stft.abs(), c)
    raise ValueError(f"unsupported extractor_type {extractor_type!r}")


def _get_masker(extractor_type):
    # The mask is a real gain applied to the complex STFT — independent of how
    # input features were extracted, so 'mag' and 'mag_compressed' share it.
    if extractor_type in ("mag", "mag_compressed"):
        return lambda x_stft, mask: x_stft * mask
    raise ValueError(f"unsupported extractor_type {extractor_type!r}")


# =============================================================================
# TCM (Temporal Conv Module) blocks
# =============================================================================


class TCMBlock(nn.Module):
    """Original TCM block (op order: conv -> act -> dropout -> norm)."""

    def __init__(self, n_channels_res, n_channels_conv, kernel_size, dilation, dropout, norm_type, causal):
        super().__init__()
        self.n_channels_res = n_channels_res
        self.n_channels_conv = n_channels_conv
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        self.norm_type = norm_type
        self.causal = causal
        self.padding = (kernel_size - 1) * dilation if causal else dilation
        NormClass = _get_normalization(norm_type)
        self.conv1x1 = nn.Conv1d(n_channels_res, n_channels_conv, 1)
        self.act1 = nn.PReLU(num_parameters=1)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = NormClass(n_channels_conv)
        self.dconv = nn.Conv1d(
            n_channels_conv, n_channels_conv, kernel_size, 1, self.padding, dilation,
            groups=n_channels_conv, bias=False,
        )
        self.chomp = Chomp(self.padding) if causal else nn.Identity()
        self.act2 = nn.PReLU(num_parameters=1)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = NormClass(n_channels_conv)
        self.conv1x1_out = nn.Conv1d(n_channels_conv, n_channels_res, 1, bias=False)

    def forward_no_residual(self, x):
        x = self.conv1x1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.norm1(x)
        x = self.dconv(x)
        x = self.chomp(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.norm2(x)
        x = self.conv1x1_out(x)
        return x

    def forward(self, x):
        return self.forward_no_residual(x) + x


class TCMBlock_QuantFriendly(TCMBlock):
    """Quant-friendly TCM block: op order is conv -> bn -> relu, no dropout, ReLU instead of PReLU."""

    def __init__(self, n_channels_res, n_channels_conv, kernel_size, dilation, causal):
        super().__init__(n_channels_res, n_channels_conv, kernel_size, dilation, 0.0, "batch", causal)
        del self.dropout1
        del self.dropout2
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward_no_residual(self, x):
        x = self.conv1x1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dconv(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.chomp(x)
        x = self.conv1x1_out(x)
        return x

    def forward(self, x):
        return self.forward_no_residual(x) + x


class TCM(nn.Sequential):
    def __init__(self, n_channels_res, n_channels_conv, kernel_size,
                 n_blocks, n_stacks, dropout, norm_type, causal):
        blocks = []
        for _ in range(n_stacks):
            for n in range(n_blocks):
                blocks.append(TCMBlock(
                    n_channels_res, n_channels_conv, kernel_size,
                    dilation=2 ** n, dropout=dropout, norm_type=norm_type, causal=causal,
                ))
        super().__init__(*blocks)


# =============================================================================
# Base model + mixins
# =============================================================================


def _pad_to_valid_length(x, hop_len):
    x_len = x.shape[-1]
    x_len_valid = (x_len // hop_len + 1) * hop_len
    padding = x_len_valid - x_len
    return F.pad(x, (0, padding), mode="constant", value=0)


def _pad_signals_to_valid_length(*signals, hop_len):
    return tuple(_pad_to_valid_length(s, hop_len) for s in signals)


class BaseModel(nn.Module):
    """Spectrogram in -> masked spectrogram out, plus the STFT pair around it.

    The model owns no loss: training goes through common/losses.py (the MP-SENet
    objective), which needs the *pre-iSTFT* spectrum and so drives preproc /
    forward / postproc itself. See convfsenet/train.py::forward_spectra.
    """

    def __init__(self, preproc, postproc):
        super().__init__()
        self.preproc = preproc
        self.postproc = postproc
        self.eps = 1e-9

    def _pad_signals_if_needed(self, *signals):
        if not hasattr(self.preproc, "hop_length"):
            return signals
        return _pad_signals_to_valid_length(*signals, hop_len=self.preproc.hop_length)

    def _crop_signals_if_needed(self, *signals):
        min_length = min(s.shape[-1] for s in signals)
        return tuple(s[..., :min_length] for s in signals)

    def _check_signals_shape(self, sig_pred, sig_clean):
        assert sig_pred.shape == sig_clean.shape, \
            f"prediction and target shapes do not match: pred={sig_pred.shape} vs clean={sig_clean.shape}"


# =============================================================================
# ConvFSENet (and its quant-friendly subclass)
# =============================================================================


class ConvFSENet(BaseModel):
    """ConvTasNet-derived speech enhancement model operating on STFTs.

    Frontend (1x1 conv + ReLU) -> stacked TCM blocks -> backend (1x1 conv + sigmoid mask)
    -> apply mask to noisy STFT.
    """

    def __init__(self, n_fft, win_length, n_features,
                 n_channels_res, n_channels_conv, kernel_size,
                 n_blocks, n_stacks, dropout, norm_type,
                 extractor_type, compress_factor, causal,
                 preproc, postproc):
        super().__init__(preproc, postproc)
        self.n_fft = n_fft
        self.win_length = win_length
        self.n_features = n_features
        self.n_channels_res = n_channels_res
        self.n_channels_conv = n_channels_conv
        self.kernel_size = kernel_size
        self.n_blocks = n_blocks
        self.n_stacks = n_stacks
        self.dropout = dropout
        self.norm_type = norm_type
        self.extractor_type = extractor_type
        self.compress_factor = compress_factor
        self.causal = causal
        # layers
        self.features_extractor = _get_feature_extractor(extractor_type, compress_factor)
        self.frontend = nn.Sequential(nn.Conv1d(n_features, n_channels_res, 1), nn.ReLU())
        self.backend = nn.Sequential(nn.Conv1d(n_channels_res, n_features, 1), nn.Sigmoid())
        self.tcm = TCM(n_channels_res, n_channels_conv, kernel_size, n_blocks, n_stacks, dropout, norm_type, causal)
        self.masker = _get_masker(extractor_type)

    def forward(self, stft_noisy):
        stft_noisy = stft_noisy.squeeze(1)
        feats = self.features_extractor(stft_noisy)
        x = self.frontend(feats)
        x = self.tcm(x)
        mask = self.backend(x)
        stft_pred = self.masker(stft_noisy, mask)
        return stft_pred.unsqueeze(1)


class ConvFSENet_QuantFriendly(ConvFSENet):
    """ConvFSENet whose TCM blocks use the quant-friendly op order (conv -> bn -> relu, no dropout)."""

    def __init__(self, n_fft, win_length, n_features,
                 n_channels_res, n_channels_conv, kernel_size,
                 n_blocks, n_stacks,
                 extractor_type, compress_factor, causal,
                 preproc, postproc):
        super().__init__(
            n_fft, win_length, n_features,
            n_channels_res, n_channels_conv, kernel_size,
            n_blocks, n_stacks, 0.0, "batch",
            extractor_type, compress_factor, causal,
            preproc, postproc,
        )
        for i, m in enumerate(self.tcm):
            self.tcm[i] = TCMBlock_QuantFriendly(
                n_channels_res, n_channels_conv, kernel_size, m.dilation, causal,
            )


class ConvFSENet_QuantFriendly_TD(ConvFSENet_QuantFriendly):
    """Quant-friendly ConvFSENet, trained end-to-end in the time domain."""
    pass


# =============================================================================
# Factory
# =============================================================================


def build_model():
    """Instantiate the model with the exact hyperparameters from quant_td.yaml."""
    import torchaudio  # local import: only the factory needs it (Spectrogram/InverseSpectrogram).
    n_fft = 512
    win_length = 512
    hop_length = 256  # = int(win_length * 0.5)
    n_features = n_fft // 2 + 1  # 257

    preproc = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None,
    )
    postproc = torchaudio.transforms.InverseSpectrogram(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length,
    )
    return ConvFSENet_QuantFriendly_TD(
        n_fft=n_fft,
        win_length=win_length,
        n_features=n_features,
        n_channels_res=128,
        n_channels_conv=256,
        kernel_size=3,
        n_blocks=3,
        n_stacks=3,
        extractor_type="mag",
        compress_factor=None,
        causal=False,
        preproc=preproc,
        postproc=postproc,
    )


# -----------------------------------------------------------------------------
# Torch-only Spectrogram / InverseSpectrogram (avoid the torchaudio dep).
#
# torch.stft + torch.istft give the same numerical result as
# torchaudio.transforms.Spectrogram(power=None) / InverseSpectrogram when the
# window matches. We add a tiny channel-dim passthrough so the (B, 1, samples)
# input from dataset.py round-trips to (B, 1, F, T) complex (and back), which is
# what TrainValidTest_TimeDomain.process_data expects.
# -----------------------------------------------------------------------------


class _TorchSpectrogram(nn.Module):
    def __init__(self, n_fft, win_length, hop_length):
        super().__init__()
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    def forward(self, x):
        # Handle the channel dim (B, 1, samples) <-> (B, samples).
        squeeze_back = x.dim() == 3 and x.shape[1] == 1
        if squeeze_back:
            x = x.squeeze(1)
        X = torch.stft(
            x, self.n_fft,
            hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=True, normalized=False, return_complex=True,
        )                                                            # (B, F, T) complex
        if squeeze_back:
            X = X.unsqueeze(1)                                       # (B, 1, F, T)
        return X


class _TorchInverseSpectrogram(nn.Module):
    def __init__(self, n_fft, win_length, hop_length):
        super().__init__()
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    def forward(self, X):
        squeeze_back = X.dim() == 4 and X.shape[1] == 1
        if squeeze_back:
            X = X.squeeze(1)
        wav = torch.istft(
            X, self.n_fft,
            hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=True, normalized=False, return_complex=False,
        )                                                            # (B, samples)
        if squeeze_back:
            wav = wav.unsqueeze(1)                                   # (B, 1, samples)
        return wav


def build_causal_model(h=None):
    """Instantiate ConvFSENet_QuantFriendly_TD with causal=True and no torchaudio dep.

    Uses torch.stft / torch.istft internally so the training pipeline doesn't
    pull a new dependency. Hyperparams default to the quant_td.yaml values;
    pass an AttrDict / dict-like `h` to override (e.g., from configs/*.json).

    Defaults flipped from build_model():
        causal     = True       (required for the streaming wrapper)
        bias       = matches the parent class definition

    All other hyperparams match build_model() / the YAML by default.
    """
    h = h or {}
    n_fft = int(h.get("n_fft", 512))
    win_length = int(h.get("win_length", n_fft))
    hop_length = int(h.get("hop_size", n_fft // 2))
    n_features = int(h.get("n_features", n_fft // 2 + 1))
    n_channels_res = int(h.get("n_channels_res", 128))
    n_channels_conv = int(h.get("n_channels_conv", 256))
    kernel_size = int(h.get("kernel_size", 3))
    n_blocks = int(h.get("n_blocks", 3))
    n_stacks = int(h.get("n_stacks", 3))
    causal = bool(h.get("causal", True))                                  # default True for the trainer
    extractor_type = h.get("extractor_type", "mag")
    compress_factor = h.get("compress_factor", None)

    preproc = _TorchSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    postproc = _TorchInverseSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    return ConvFSENet_QuantFriendly_TD(
        n_fft=n_fft, win_length=win_length, n_features=n_features,
        n_channels_res=n_channels_res, n_channels_conv=n_channels_conv,
        kernel_size=kernel_size, n_blocks=n_blocks, n_stacks=n_stacks,
        extractor_type=extractor_type, compress_factor=compress_factor,
        causal=causal,
        preproc=preproc, postproc=postproc,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {model.__class__.__name__}, params: {n_params:,}")

    # smoke test: 1 s of mono audio at 16 kHz, shape [batch, channels, samples]
    sr = 16000
    x_noisy = torch.randn(2, 1, sr)
    x_clean = torch.randn(2, 1, sr)

    model.eval()
    with torch.no_grad():
        x_pred = model.postproc(model(model.preproc(x_noisy)))
    print(f"forward ok: x_pred shape={tuple(x_pred.shape)}")

    model.train()
    x_pred = model.postproc(model(model.preproc(x_noisy)))
    x_pred, x_clean_c = model._crop_signals_if_needed(x_pred, x_clean)
    F.l1_loss(x_pred, x_clean_c).backward()
    print("backward ok")
