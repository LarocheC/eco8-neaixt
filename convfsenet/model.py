"""
Standalone version of the model described by
`config/model/convfsenet/quant_td.yaml` from the `convolve_dyn_experiments` repo.

It bundles every class the model needs (BaseModel, TCMBlock, TCM, ConvFSENet,
the quant-friendly variants, the time-domain training mixin, and the
DynCompMSE / PreProcLoss combo) into a single file. Build with `build_model()`.

Composed config (resolved from quant_td -> quant -> shared, with the
`dyncompmse_from_waveform` loss override):

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
    loss: PreProcLoss(
        preproc=Spectrogram(n_fft=512, power=None),
        loss=DynCompMSE(normalize=True, normalize_mode="threshold",
                        normalize_framelen=512, normalize_threshold=0.025),
    )
"""

import torch
from torch import nn
from torch.nn import functional as F


# =============================================================================
# Losses
# =============================================================================


class CompositeLoss(nn.Module):
    """Marker class — BaseModel.loss_function() checks `isinstance(loss, CompositeLoss)`."""
    pass


def _stft_to_rms(stft_clean, normalize_factor):
    x = torch.linalg.norm(stft_clean.squeeze(1), dim=1)
    x *= normalize_factor
    return x


def _get_active_speech_level_threshold(stft_clean, normalize_threshold, normalize_factor):
    energy = _stft_to_rms(stft_clean, normalize_factor)
    active_frames = energy > normalize_threshold
    active_energy = energy[active_frames]
    return active_energy.mean()


def _normalize_several(stft_clean, stft_pred_list, normalize_threshold, normalize_factor, eps):
    std_signal = _get_active_speech_level_threshold(stft_clean, normalize_threshold, normalize_factor)
    stft_pred_list = [stft_pred.clone() / (std_signal + eps) for stft_pred in stft_pred_list]
    stft_clean = stft_clean.clone() / (std_signal + eps)
    return stft_clean, stft_pred_list


class DynCompMSE(nn.Module):
    """Magnitude+complex compressed MSE with active-speech-level normalization.

    Only the `normalize_mode='threshold'` path is kept (matches `dyncompmse_norm.yaml`).
    """

    def __init__(self, normalize, normalize_framelen, normalize_threshold,
                 normalize_mode="threshold", exponent=0.3, alpha=0.3, eps=1e-9):
        super().__init__()
        assert normalize_mode == "threshold", "Standalone only supports normalize_mode='threshold'"
        self.normalize = normalize
        self.normalize_threshold = normalize_threshold
        self.normalize_factor = torch.sqrt(torch.tensor(2.0)) / normalize_framelen
        self.exponent = exponent
        self.alpha = alpha
        self.eps = eps

    def _r_pow(self, x, a):
        return torch.pow(x + self.eps, a)

    def forward(self, stft_pred, stft_clean, **kwargs):
        if self.normalize:
            stft_clean, stft_pred_list = _normalize_several(
                stft_clean, [stft_pred],
                self.normalize_threshold, self.normalize_factor, self.eps,
            )
            stft_pred = stft_pred_list[0]
        pred_magnitude = self._r_pow(stft_pred.abs(), self.exponent)
        clean_magnitude = self._r_pow(stft_clean.abs(), self.exponent)
        loss_magnitude = torch.pow(pred_magnitude - clean_magnitude, 2).mean()
        pred_complex = stft_pred * self._r_pow(stft_pred.abs(), self.exponent - 1.0)
        clean_complex = stft_clean * self._r_pow(stft_clean.abs(), self.exponent - 1.0)
        loss_complex = torch.pow((pred_complex - clean_complex).abs(), 2).mean()
        return (1 - self.alpha) * loss_magnitude + self.alpha * loss_complex


class PreProcLoss(nn.Module):
    """Apply a preprocessing transform (here: Spectrogram) before computing the inner loss."""

    def __init__(self, loss, preproc):
        super().__init__()
        self.loss_fnc = loss
        self.preproc = preproc

    def forward(self, x_pred, x_clean, **kwargs):
        return self.loss_fnc(self.preproc(x_pred), self.preproc(x_clean), **kwargs)


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
    def __init__(self, loss, preproc, postproc):
        super().__init__()
        self.loss_fnc = loss
        self.preproc = preproc
        self.postproc = postproc
        self.eps = 1e-9

    def loss_function(self, *args, **kwargs):
        if isinstance(self.loss_fnc, CompositeLoss):
            loss, loss_dict = self.loss_fnc(*args, **kwargs)
            loss_dict = {f"loss/{k}": v for k, v in loss_dict.items()}
            loss_dict["loss"] = loss
        else:
            loss = self.loss_fnc(*args, **kwargs)
            loss_dict = {"loss": loss}
        return loss_dict

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


class TrainValidTest_TimeDomain:
    """End-to-end loss on the time domain (preproc -> forward -> postproc -> loss(x_pred, x_clean))."""

    def process_data(self, x_noisy, x_clean, crop_signals=False):
        repr_noisy = self.preproc(x_noisy)
        repr_pred = self.forward(repr_noisy)
        x_pred = self.postproc(repr_pred)
        if crop_signals:
            x_pred, x_clean = self._crop_signals_if_needed(x_pred, x_clean)
        self._check_signals_shape(x_pred, x_clean)
        loss_dict = self.loss_function(x_pred, x_clean)
        return x_pred, loss_dict

    def train_step(self, x_noisy, x_clean):
        _, loss_dict = self.process_data(x_noisy, x_clean, crop_signals=True)
        return loss_dict

    def valid_step(self, x_noisy, x_clean):
        x_noisy, x_clean = self._pad_signals_if_needed(x_noisy, x_clean)
        x_pred, loss_dict = self.process_data(x_noisy, x_clean)
        return x_pred, x_clean, x_noisy, loss_dict

    def test_step(self, x_noisy, x_clean):
        return self.valid_step(x_noisy, x_clean)


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
                 loss, preproc, postproc):
        super().__init__(loss, preproc, postproc)
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
                 loss, preproc, postproc):
        super().__init__(
            n_fft, win_length, n_features,
            n_channels_res, n_channels_conv, kernel_size,
            n_blocks, n_stacks, 0.0, "batch",
            extractor_type, compress_factor, causal,
            loss, preproc, postproc,
        )
        for i, m in enumerate(self.tcm):
            self.tcm[i] = TCMBlock_QuantFriendly(
                n_channels_res, n_channels_conv, kernel_size, m.dilation, causal,
            )


class ConvFSENet_QuantFriendly_TD(TrainValidTest_TimeDomain, ConvFSENet_QuantFriendly):
    """Quant-friendly ConvFSENet trained with an end-to-end time-domain loss."""
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
    inner_loss = DynCompMSE(
        normalize=True, normalize_mode="threshold",
        normalize_framelen=n_fft, normalize_threshold=0.025,
    )
    loss = PreProcLoss(
        loss=inner_loss,
        preproc=torchaudio.transforms.Spectrogram(n_fft=n_fft, power=None),
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
        loss=loss,
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
    # Loss preproc matches the YAML's Spectrogram(n_fft=512, power=None) — torchaudio
    # defaults are hop=n_fft/2, win=n_fft. Reuse the same window/hop.
    loss_preproc = _TorchSpectrogram(n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 2)
    inner_loss = DynCompMSE(
        normalize=True, normalize_mode="threshold",
        normalize_framelen=n_fft, normalize_threshold=0.025,
    )
    loss = PreProcLoss(loss=inner_loss, preproc=loss_preproc)

    return ConvFSENet_QuantFriendly_TD(
        n_fft=n_fft, win_length=win_length, n_features=n_features,
        n_channels_res=n_channels_res, n_channels_conv=n_channels_conv,
        kernel_size=kernel_size, n_blocks=n_blocks, n_stacks=n_stacks,
        extractor_type=extractor_type, compress_factor=compress_factor,
        causal=causal,
        loss=loss, preproc=preproc, postproc=postproc,
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
        x_pred, x_clean_p, x_noisy_p, loss_dict = model.valid_step(x_noisy, x_clean)
    print(f"valid_step ok: x_pred shape={tuple(x_pred.shape)}, loss={loss_dict['loss'].item():.4f}")

    model.train()
    loss_dict = model.train_step(x_noisy, x_clean)
    loss_dict["loss"].backward()
    print(f"train_step ok: loss={loss_dict['loss'].item():.4f}")
