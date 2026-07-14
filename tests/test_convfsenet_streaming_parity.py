"""Parity gate: offline ConvFSENet_QuantFriendly (causal=True) vs.
its two streaming views — naive (step 1) and BN-folded fast (step 3).

Hard tolerance: max_abs_err < 1e-5 on a random complex-STFT input.

  - Naive path: shares parameters by reference, so the error floor is pure
    floating-point. Empirically < 1e-7 on CPU float32.
  - Fast path: holds folded copies (gamma*W/sigma, bias rewritten). The
    fold introduces one extra mul + add per channel in the offline-equivalent
    math, so the error floor sits one order higher (~1e-6) but is still
    safely under 1e-5.

Tests are parameterized over both streaming classes via the `streaming`
fixture. Bucket diagnostic in the parity-gate failure message mirrors
models/streaming.py / tests/test_streaming_parity.py style.

The offline model is built directly (not via build_model()) to skip the
torchaudio Spectrogram preproc/postproc — they're irrelevant to the
streaming math, which operates on complex STFT frames directly.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from convfsenet.model import ConvFSENet_QuantFriendly
from convfsenet.streaming import (
    ConvFSENetStreaming,
    ConvFSENetStreamingFast,
)


# ---- model fixtures ---------------------------------------------------------


def _fresh_offline(causal: bool = True, extractor_type: str = "mag") -> ConvFSENet_QuantFriendly:
    """Build the quant_td hyperparams with causal swapped on. preproc/postproc/
    loss are None — irrelevant for forward(). extractor_type selects plain vs
    compressed-magnitude features (compress_factor 0.3 for the latter)."""
    return ConvFSENet_QuantFriendly(
        n_fft=512, win_length=512, n_features=257,
        n_channels_res=128, n_channels_conv=256, kernel_size=3,
        n_blocks=3, n_stacks=3,
        extractor_type=extractor_type,
        compress_factor=0.3 if extractor_type == "mag_compressed" else None,
        causal=causal,
        preproc=None, postproc=None,
    )


def _randomize_bn_running_stats(model: nn.Module) -> None:
    """BN1d defaults: running_mean=0, running_var=1, weight=1, bias=0 — i.e., an
    identity affine. That hides any bug that mishandles the time axis or treats
    streaming BN differently from offline BN. Replace with non-trivial values so
    a regression surfaces clearly in the parity diff."""
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm1d):
                nn.init.normal_(mod.weight, mean=1.0, std=0.1)
                nn.init.normal_(mod.bias, mean=0.0, std=0.1)
                mod.running_mean.normal_(mean=0.0, std=0.1)
                mod.running_var.uniform_(0.5, 1.5)        # must stay > 0
                mod.num_batches_tracked.fill_(1)


# Parameterized over the feature extractor so every parity test below covers
# both plain and compressed-magnitude models.
@pytest.fixture(params=["mag", "mag_compressed"])
def offline_model(request) -> ConvFSENet_QuantFriendly:
    torch.manual_seed(0)
    m = _fresh_offline(causal=True, extractor_type=request.param)
    _randomize_bn_running_stats(m)
    m.eval()
    return m


# Parameterize the `streaming` fixture over both implementations so every test
# below runs against naive AND fast paths. ids= shows up in pytest -v output.
@pytest.fixture(
    params=[ConvFSENetStreaming, ConvFSENetStreamingFast],
    ids=["naive", "fast"],
)
def streaming_cls(request):
    return request.param


@pytest.fixture
def streaming(offline_model, streaming_cls):
    return streaming_cls(offline_model).eval()


# ---- shape / self-parity smoke tests ---------------------------------------


def test_forward_step_io_shapes(streaming):
    """forward_step IO shape contract: (stft_t [B, F] complex, list of state) ->
    (stft_pred_t [B, F] complex, list of state)."""
    B, F = 2, 257
    stft_t = torch.randn(B, F, dtype=torch.complex64)
    states_in = streaming.init_states(B, stft_t.device, dtype=torch.float32)
    out_t, states_out = streaming.forward_step(stft_t, states_in)
    assert out_t.shape == (B, F)
    assert out_t.dtype == torch.complex64
    assert len(states_out) == streaming.n_blocks_total
    for s_in, s_out in zip(states_in, states_out):
        assert s_out.shape == s_in.shape, (
            f"state shape changed across step: {tuple(s_in.shape)} -> "
            f"{tuple(s_out.shape)} — buffer width must be invariant."
        )


def test_forward_full_drives_forward_step(streaming):
    """forward_full(stft) is byte-identical to a manual T-step loop of
    forward_step. Mirrors test_streaming_parity.py:test_forward_full_drives_forward_step."""
    torch.manual_seed(0)
    B, F, T = 2, 257, 6
    stft = torch.randn(B, F, T, dtype=torch.complex64)
    with torch.no_grad():
        out_full = streaming.forward_full(stft)
        states = streaming.init_states(B, stft.device, dtype=torch.float32)
        outs = []
        for t in range(T):
            o, states = streaming.forward_step(stft[..., t], states)
            outs.append(o)
        out_loop = torch.stack(outs, dim=-1)
    err = (out_full - out_loop).abs().max().item()
    assert err == 0.0, (
        f"forward_full vs manual time-loop diverge (err = {err:.3e}). "
        f"Suggests forward_full does not thread states correctly between iterations."
    )


# ---- the parity gate -------------------------------------------------------


def test_parity_offline_vs_streaming(offline_model, streaming):
    """STR-01 (this module): max_abs_err < 1e-5 between offline ConvFSENet
    forward (whole-utterance, causal=True) and streaming forward_full (per-frame).

    Both paths share parameters by reference — error floor is FP noise from
    F.conv1d implementation differences between a long padded input and a
    sliding (K-1)*D+1 window. Empirically < 1e-6 on CPU float32.
    """
    torch.manual_seed(0)
    B, F, T = 2, 257, 20
    stft = torch.randn(B, F, T, dtype=torch.complex64)
    with torch.no_grad():
        # Offline forward expects [B, 1, F, T] complex and returns [B, 1, F, T].
        out_offline = offline_model(stft.unsqueeze(1)).squeeze(1)      # [B, F, T] complex
        out_streaming = streaming.forward_full(stft)                   # [B, F, T] complex
    err = (out_offline - out_streaming).abs().max().item()
    assert err < 1e-5, (
        f"Streaming parity FAILED: max_abs_err = {err:.3e}. Expected < 1e-5. "
        f"Bucket diagnostic: "
        f"1e-5 pass / "
        f"1e-4 suspect dconv buffer alignment (off-by-one in window indexing — "
        f"state at step t must hold the last (K-1)*D frames of the "
        f"conv1x1/norm1/act1 output stream, and init_state must be zeros to "
        f"match offline causal left-padding) / "
        f"1e-3 suspect BN eval-mode (one path in train mode, or running stats "
        f"not propagated to the streaming wrapper) / "
        f"1e-2 suspect residual placement (streaming residual must add x_t — "
        f"the block INPUT frame BEFORE conv1x1 — to the conv1x1_out output, "
        f"matching TCMBlock_QuantFriendly.forward: forward_no_residual(x) + x) / "
        f"any: suspect chomp regression — streaming must SKIP self.src.chomp "
        f"(it would return an empty [B, C, 0] tensor on T=1 input)."
    )


def test_first_frame_uses_zero_padded_state(offline_model, streaming):
    """Initial state must be all-zero, matching the implicit left-padding
    that the offline causal Conv1d applies. If init_state were random or
    learned, the streaming first frame would diverge from offline at t=0."""
    torch.manual_seed(0)
    B, F = 1, 257
    states = streaming.init_states(B, "cpu", dtype=torch.float32)
    for s in states:
        assert torch.all(s == 0.0), (
            "init_state must be zeros to match offline causal padding."
        )


def test_non_causal_base_rejected(streaming_cls):
    """Both streaming wrappers must hard-fail when wrapping a non-causal base —
    silent acceptance would produce wrong outputs without warning."""
    base = _fresh_offline(causal=False)
    base.eval()
    with pytest.raises(ValueError, match="causal=True"):
        streaming_cls(base)


def test_fast_rejects_train_mode_base(offline_model):
    """The fast wrapper folds BN at __init__ — running stats must be in eval mode."""
    offline_model.train()
    with pytest.raises(ValueError, match="eval"):
        ConvFSENetStreamingFast(offline_model)


def test_naive_and_fast_agree(offline_model):
    """Cross-check: naive and fast paths agree to 1e-5 on the same input.

    Both reduce to the same offline-equivalent math, but via different op
    sequences (naive: BN explicit; fast: folded into preceding conv; dilation
    in the kernel vs absorbed into a gather). They should still agree to
    within float32 noise after the fold.
    """
    torch.manual_seed(0)
    naive = ConvFSENetStreaming(offline_model).eval()
    fast = ConvFSENetStreamingFast(offline_model).eval()
    B, F, T = 2, 257, 30
    stft = torch.randn(B, F, T, dtype=torch.complex64)
    with torch.no_grad():
        out_naive = naive.forward_full(stft)
        out_fast = fast.forward_full(stft)
    err = (out_naive - out_fast).abs().max().item()
    assert err < 1e-5, (
        f"Naive vs Fast streaming diverge: max_abs_err = {err:.3e}. "
        f"Expected < 1e-5. Suspect: BN fold formula bug "
        f"(W'=W*gamma/sigma, b'=(b_w-mean)*gamma/sigma+beta) or tap_indices "
        f"layout drift (must be [i*D for i in 0..K-2] into the FIFO buffer "
        f"where buf[..., j] = y_{{t-(K-1)*D+j}})."
    )
