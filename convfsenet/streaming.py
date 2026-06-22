"""Streaming-friendly view of ConvFSENet_QuantFriendly — per-frame forward
for real-time inference and (later) ONNX export.

Wraps a ConvFSENet_QuantFriendly with causal=True and runs each TCM block
as a single-frame F.conv1d on a sliding window of length (K-1)*D + 1.
Maintains a per-block state buffer [B, n_channels_conv, (K-1)*D] holding
the past dconv-input frames.

Op order within a TCM block (matches TCMBlock_QuantFriendly.forward_no_residual):

    conv1x1 -> norm1 -> act1 -> dconv -> norm2 -> act2 -> conv1x1_out
    out = conv1x1_out(...) + x_t   (residual on the block's INPUT frame)

Streaming math vs offline (causal padding=(K-1)*D, chomp last (K-1)*D):

    offline:   out[t] = w0*y[t-2D] + w1*y[t-D] + w2*y[t]   (K=3)
               where y[t-i] for i>0 is zero-padding when t<i.
    streaming: window = state ++ [y_t]                     # length (K-1)*D+1
               F.conv1d(window, dilation=D, kernel=K, padding=0) → 1 frame.
               state at step t holds the last (K-1)*D frames of {y_0..y_{t-1}}
               (zeros at the start, matching the offline left-padding).

Two implementations land in this module:

  - ConvFSENetStreaming        — naive shift-and-cat buffer, composes the
                                 base model by reference (no clones). Used
                                 as the parity reference.

  - ConvFSENetStreamingFast    — step 3 efficient form. BatchNorm folded
                                 into the preceding Conv1d (norm1 -> conv1x1,
                                 norm2 -> dconv), so each block runs three
                                 convs instead of two-convs-plus-two-BNs.
                                 The dconv input becomes a K-frame window
                                 gathered from the FIFO buffer at indices
                                 [0, D, 2D, ...], and the conv is K-kernel
                                 dilation-1 (the dilation is absorbed into
                                 the gather pattern). Holds its own copies
                                 of all weights (folding mutates them).

Both classes share the same state shape — list of [B, C', (K-1)*D] FIFO
buffers per TCM block, init_states() returns zeros — so they're drop-in
parity-comparable.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from convfsenet.model import (
    ConvFSENet_QuantFriendly,
    TCMBlock_QuantFriendly,
    compress_magnitude,
)

_SUPPORTED_EXTRACTORS = ("mag", "mag_compressed")


class _StreamingTCMBlockQF(nn.Module):
    """Per-frame forward for one TCMBlock_QuantFriendly (causal=True).

    State is a single tensor [B, n_channels_conv, (K-1)*D] holding the
    past dconv-input frames (i.e., the conv1x1/norm1/act1 outputs for
    the last (K-1)*D time steps).
    """

    def __init__(self, src: TCMBlock_QuantFriendly):
        super().__init__()
        if not src.causal:
            raise ValueError(
                "Streaming requires causal=True; got causal=False on a TCM "
                "block. Train the model with causal=true in config."
            )
        self.src = src
        K = src.kernel_size
        D = src.dilation
        self.kernel_size = int(K)
        self.dilation = int(D)
        self.groups = int(src.dconv.groups)
        self.buffer_size = int((K - 1) * D)            # (K-1)*D past frames
        self.n_channels_res = int(src.n_channels_res)
        self.n_channels_conv = int(src.n_channels_conv)

    def init_state(self, batch_size: int, device, dtype) -> torch.Tensor:
        # Zeros match the implicit left-padding of the offline causal Conv1d.
        return torch.zeros(
            batch_size, self.n_channels_conv, self.buffer_size,
            device=device, dtype=dtype,
        )

    def forward_step(
        self,
        x_t: torch.Tensor,        # [B, n_channels_res, 1]
        state: torch.Tensor,      # [B, n_channels_conv, (K-1)*D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (out [B, n_channels_res, 1], new_state [B, n_channels_conv, (K-1)*D])."""
        y = self.src.conv1x1(x_t)                                       # K=1, time-pointwise
        y = self.src.norm1(y)                                           # BN1d eval = channel affine
        y = self.src.act1(y)                                            # ReLU
        # Build the dconv input window: past buffer + current frame.
        window = torch.cat([state, y], dim=-1)                          # [B, C', (K-1)*D + 1]
        # Explicit F.conv1d with padding=0 — bypass src.dconv's left-padding.
        dconv_out = F.conv1d(
            window,
            self.src.dconv.weight,
            bias=self.src.dconv.bias,
            stride=1,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )                                                                # [B, C', 1]
        y2 = self.src.norm2(dconv_out)
        y2 = self.src.act2(y2)
        # NB: offline applies self.src.chomp here, which trims (K-1)*D frames
        # from the right. Streaming intentionally skips it — the F.conv1d above
        # already produced exactly 1 output frame (no padding to trim). Calling
        # Chomp on a [B, C, 1] tensor would return an empty [B, C, 0] tensor.
        y2 = self.src.conv1x1_out(y2)                                    # K=1 → [B, n_channels_res, 1]
        out = y2 + x_t                                                   # residual on block INPUT
        # State for next step: drop oldest frame, keep last (K-1)*D frames.
        new_state = window[..., 1:]
        return out, new_state


class ConvFSENetStreaming(nn.Module):
    """Streaming view of ConvFSENet_QuantFriendly (causal=True).

    Per-frame forward over complex STFT frames. Composes the base model
    by reference (frontend, per-block params, backend); only adds the
    per-block state-buffer plumbing.

    forward_step(stft_t, states_in) -> (stft_pred_t, states_out)
    forward_full(stft_noisy)        -> stft_pred   (drives forward_step in a loop)
    """

    def __init__(self, base: ConvFSENet_QuantFriendly):
        super().__init__()
        if not base.causal:
            raise ValueError(
                "ConvFSENetStreaming requires causal=True on the base model "
                "(non-causal dilated conv has no streaming form). Re-train "
                "with causal=true."
            )
        if base.extractor_type not in _SUPPORTED_EXTRACTORS:
            raise NotImplementedError(
                f"Streaming supports extractor_type in {_SUPPORTED_EXTRACTORS} "
                f"(got {base.extractor_type!r})."
            )
        self.base = base
        self.tcm_blocks = nn.ModuleList(
            _StreamingTCMBlockQF(m) for m in base.tcm
        )
        self.n_blocks_total = len(self.tcm_blocks)

    # ---- state helpers ----------------------------------------------------

    def init_states(self, batch_size: int, device, dtype=torch.float32) -> List[torch.Tensor]:
        return [blk.init_state(batch_size, device, dtype) for blk in self.tcm_blocks]

    # ---- per-frame forward ------------------------------------------------

    def forward_step(
        self,
        stft_t: torch.Tensor,             # [B, F] complex
        states_in: List[torch.Tensor],    # one [B, C', (K-1)*D] per block
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Single complex-STFT frame in, single predicted complex frame out."""
        # 1. feature extraction — reuse the base model's extractor so the
        #    'mag' / 'mag_compressed' choice and its eps stay in lockstep.
        feats = self.base.features_extractor(stft_t)      # [B, F] real
        # 2. frontend: Conv1d(F -> n_channels_res, K=1) + ReLU, on shape [B, F, 1].
        x = feats.unsqueeze(-1)                           # [B, F, 1]
        x = self.base.frontend(x)                         # [B, n_channels_res, 1]
        # 3. TCM blocks, one per-frame call each, threading state.
        states_out: List[torch.Tensor] = []
        for blk, state in zip(self.tcm_blocks, states_in):
            x, new_state = blk.forward_step(x, state)
            states_out.append(new_state)
        # 4. backend: Conv1d(n_channels_res -> F, K=1) + Sigmoid → mask [B, F, 1].
        mask = self.base.backend(x).squeeze(-1)           # [B, F] real
        # 5. mask the noisy STFT (complex × real → complex).
        stft_pred_t = stft_t * mask                       # [B, F] complex
        return stft_pred_t, states_out

    # ---- python time-loop driver (parity reference) -----------------------

    def forward_full(self, stft_noisy: torch.Tensor) -> torch.Tensor:
        """Drive forward_step in a Python time-loop. Used by the parity gate.

        stft_noisy: [B, F, T] complex, OR [B, 1, F, T] (the base model's shape).
        returns:    [B, F, T] complex (no leading 1-channel axis).
        """
        if stft_noisy.dim() == 4 and stft_noisy.shape[1] == 1:
            stft_noisy = stft_noisy.squeeze(1)
        B, F_, T = stft_noisy.shape
        states = self.init_states(B, stft_noisy.device, dtype=torch.float32)
        outs = []
        for t in range(T):
            stft_t = stft_noisy[..., t]                   # [B, F]
            stft_pred_t, states = self.forward_step(stft_t, states)
            outs.append(stft_pred_t)
        return torch.stack(outs, dim=-1)                  # [B, F, T]

    # Alias so torch.onnx.export traces forward_step by default (step 5).
    forward = forward_step


# =============================================================================
# Step 3: efficient streaming — BN folded, dilation absorbed into the gather.
# =============================================================================


def _fold_bn_into_conv(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> nn.Conv1d:
    """Return a new Conv1d that absorbs BatchNorm1d into its weight + bias.

    For y = BN_eval(W·x + b_w) with BN_eval(z) = (z − μ)/σ · γ + β, where
    σ = sqrt(running_var + eps):

        W' = W · (γ/σ)              # scale per output channel
        b' = (b_w − μ) · (γ/σ) + β

    Caller must put BN in eval mode first — running_mean / running_var are
    only stable in eval. Asserts on this so silent garbage is impossible.

    The returned Conv1d has the same in/out_channels, kernel_size, dilation,
    groups as `conv`, but padding=0 (streaming feeds an exact-length window)
    and bias=True (post-fold there's always a bias).
    """
    if bn.training:
        raise ValueError(
            "BN must be in eval mode before folding; running_mean / running_var "
            "only stabilize after .eval(). Call model.eval() before constructing "
            "ConvFSENetStreamingFast."
        )
    if bn.num_features != conv.out_channels:
        raise ValueError(
            f"BN.num_features={bn.num_features} != Conv1d.out_channels="
            f"{conv.out_channels}; BN must follow conv on out_channels."
        )

    folded = nn.Conv1d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size[0],
        stride=conv.stride[0],
        padding=0,                                # streaming feeds an exact-length window
        dilation=conv.dilation[0],
        groups=conv.groups,
        bias=True,
    )
    with torch.no_grad():
        gamma = bn.weight                         # (C_out,)
        beta = bn.bias                            # (C_out,)
        mean = bn.running_mean                    # (C_out,)
        var = bn.running_var                      # (C_out,)
        sigma = torch.sqrt(var + bn.eps)          # (C_out,)
        scale = gamma / sigma                     # (C_out,)
        # conv.weight: (C_out, C_in/groups, K) — scale broadcasts along dim 0.
        folded.weight.copy_(conv.weight * scale.view(-1, 1, 1))
        b_w = conv.bias if conv.bias is not None else torch.zeros_like(beta)
        folded.bias.copy_((b_w - mean) * scale + beta)
    return folded


def _clone_conv1d(src: nn.Conv1d) -> nn.Conv1d:
    """Deep-copy a Conv1d (weight + bias)."""
    dst = nn.Conv1d(
        in_channels=src.in_channels,
        out_channels=src.out_channels,
        kernel_size=src.kernel_size[0],
        stride=src.stride[0],
        padding=src.padding[0] if isinstance(src.padding, tuple) else src.padding,
        dilation=src.dilation[0],
        groups=src.groups,
        bias=src.bias is not None,
    )
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if src.bias is not None:
            dst.bias.copy_(src.bias)
    return dst


class _StreamingTCMBlockQF_Fast(nn.Module):
    """Efficient per-frame TCM block.

    Differences from _StreamingTCMBlockQF:
      - conv1x1+norm1 -> a single folded Conv1d.
      - dconv+norm2 -> a single folded Conv1d (depthwise, K, but called with
        dilation=1 — the dilation is absorbed into a Gather over the FIFO
        buffer at indices [0, D, 2D, ..., (K−2)·D]).
      - conv1x1_out cloned (no BN follows it).

    State (same shape as the naive block): [B, n_channels_conv, (K−1)·D]
    FIFO buffer. Layout invariant: buf[..., j] = y_{t-(K-1)*D + j}, oldest
    on the left.
    """

    def __init__(self, src: TCMBlock_QuantFriendly):
        super().__init__()
        if not src.causal:
            raise ValueError(
                "Streaming requires causal=True; got causal=False on a TCM block."
            )
        K = int(src.kernel_size)
        D = int(src.dilation)
        self.kernel_size = K
        self.dilation = D
        self.groups = int(src.dconv.groups)
        self.buffer_size = (K - 1) * D
        self.n_channels_res = int(src.n_channels_res)
        self.n_channels_conv = int(src.n_channels_conv)

        # Folded modules — own copies, not by-reference.
        self.conv1x1_folded = _fold_bn_into_conv(src.conv1x1, src.norm1)
        self.dconv_folded = _fold_bn_into_conv(src.dconv, src.norm2)
        self.conv1x1_out = _clone_conv1d(src.conv1x1_out)

        # K−1 past tap indices into the FIFO buffer. For K=3, D=2: [0, 2].
        # General: [i·D for i in 0..K−2]. Empty when K=1.
        tap_indices = torch.tensor(
            [i * D for i in range(K - 1)], dtype=torch.long,
        )
        self.register_buffer("tap_indices", tap_indices, persistent=False)

    def init_state(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.n_channels_conv, self.buffer_size,
            device=device, dtype=dtype,
        )

    def forward_step(
        self,
        x_t: torch.Tensor,        # [B, n_channels_res, 1]
        buffer: torch.Tensor,     # [B, n_channels_conv, (K−1)·D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Folded conv1x1 (absorbs norm1) + ReLU.
        y = self.conv1x1_folded(x_t)             # [B, C', 1]
        y = torch.relu(y)
        # 2. Build the K-frame window: K−1 dilated taps from the FIFO buffer
        #    + the current frame y.
        if self.buffer_size > 0:
            past = buffer.index_select(dim=-1, index=self.tap_indices)   # [B, C', K-1]
            window = torch.cat([past, y], dim=-1)                        # [B, C', K]
        else:
            window = y                                                   # K=1 fallback
        # 3. K-kernel dilation=1 depthwise conv (folded with norm2).
        dconv_out = F.conv1d(
            window,
            self.dconv_folded.weight,
            bias=self.dconv_folded.bias,
            stride=1,
            padding=0,
            dilation=1,                          # dilation already absorbed into the gather
            groups=self.groups,
        )                                        # [B, C', 1]
        z = torch.relu(dconv_out)
        z = self.conv1x1_out(z)                  # [B, n_channels_res, 1]
        out = z + x_t                            # residual on block INPUT
        # 4. FIFO update: drop oldest frame, append current.
        if self.buffer_size > 0:
            new_buffer = torch.cat([buffer[..., 1:], y], dim=-1)
        else:
            new_buffer = buffer
        return out, new_buffer


class ConvFSENetStreamingFast(nn.Module):
    """Efficient streaming view of ConvFSENet_QuantFriendly.

    Built once from a trained / eval-mode base via the constructor. Holds
    its own copies of all weights with BN folded into the conv1x1 and dconv
    of every TCM block. Frontend / backend convs are cloned (no surrounding
    BN to fold).

    State shape and init_states() match ConvFSENetStreaming exactly, so the
    two classes are drop-in interchangeable in the parity tests and in
    benchmarks.
    """

    def __init__(self, base: ConvFSENet_QuantFriendly):
        super().__init__()
        if not base.causal:
            raise ValueError(
                "ConvFSENetStreamingFast requires causal=True on the base model."
            )
        if base.extractor_type not in _SUPPORTED_EXTRACTORS:
            raise NotImplementedError(
                f"Streaming supports extractor_type in {_SUPPORTED_EXTRACTORS} "
                f"(got {base.extractor_type!r})."
            )
        if base.training:
            raise ValueError(
                "Base model must be in .eval() mode before folding BatchNorm."
            )
        # Frontend / backend: no surrounding BN — clone the Conv1d, keep ReLU/Sigmoid inline.
        self.frontend_conv = _clone_conv1d(base.frontend[0])
        self.backend_conv = _clone_conv1d(base.backend[0])
        self.n_features = int(base.n_features)
        self.n_channels_res = int(base.n_channels_res)
        # Magnitude-compression: applied as the first op in forward_mask_step
        # (and so the first op in the exported ONNX graph). The ONNX `noisy_mag`
        # input stays plain |stft| — the compression Pow runs inside the graph.
        self.extractor_type = base.extractor_type
        self.compress_factor = (
            float(base.compress_factor)
            if base.extractor_type == "mag_compressed" and base.compress_factor
            else 0.3
        )

        self.tcm_blocks = nn.ModuleList(
            _StreamingTCMBlockQF_Fast(m) for m in base.tcm
        )
        self.n_blocks_total = len(self.tcm_blocks)

    # ---- state helpers (identical shape contract to ConvFSENetStreaming) ----

    def init_states(self, batch_size: int, device, dtype=torch.float32) -> List[torch.Tensor]:
        return [blk.init_state(batch_size, device, dtype) for blk in self.tcm_blocks]

    # ---- per-frame forward ------------------------------------------------

    def forward_mask_step(
        self,
        noisy_mag: torch.Tensor,          # [B, F] real
        states_in: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Core real-valued mask computation — the ONNX-exportable body.

        `noisy_mag` is plain |stft_t| (caller computes the .abs()). Magnitude
        compression, if enabled, is applied here so it becomes the first op of
        the exported ONNX graph (an FP32 Pow, ahead of every quantized conv —
        the quantized activations then see the compressed, int8-friendly range).
        Returned mask is in (0, 1) per the sigmoid backend.
        """
        if self.extractor_type == "mag_compressed":
            feats = compress_magnitude(noisy_mag, self.compress_factor)
        else:
            feats = noisy_mag
        x = feats.unsqueeze(-1)                           # [B, F, 1]
        x = self.frontend_conv(x)
        x = torch.relu(x)                                 # frontend ReLU
        states_out: List[torch.Tensor] = []
        for blk, state in zip(self.tcm_blocks, states_in):
            x, new_state = blk.forward_step(x, state)
            states_out.append(new_state)
        mask = torch.sigmoid(self.backend_conv(x)).squeeze(-1)   # [B, F]
        return mask, states_out

    def forward_step(
        self,
        stft_t: torch.Tensor,             # [B, F] complex
        states_in: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Complex-IO wrapper for parity tests / Python-side inference.
        Delegates to forward_mask_step for the real-valued core, then
        applies the mask back to the complex STFT (preserves phase)."""
        mask, states_out = self.forward_mask_step(stft_t.abs(), states_in)
        return stft_t * mask, states_out

    def forward_full(self, stft_noisy: torch.Tensor) -> torch.Tensor:
        if stft_noisy.dim() == 4 and stft_noisy.shape[1] == 1:
            stft_noisy = stft_noisy.squeeze(1)
        B, F_, T = stft_noisy.shape
        states = self.init_states(B, stft_noisy.device, dtype=torch.float32)
        outs = []
        for t in range(T):
            stft_t = stft_noisy[..., t]
            stft_pred_t, states = self.forward_step(stft_t, states)
            outs.append(stft_pred_t)
        return torch.stack(outs, dim=-1)

    forward = forward_step


# =============================================================================
# Step 5: ONNX-exportable view (real-valued IO, flat state args).
# =============================================================================


class ConvFSENetStreamingONNX(nn.Module):
    """ONNX-friendly view of ConvFSENetStreamingFast.

    Differences from ConvFSENetStreamingFast:
      - Input:  real magnitude `noisy_mag` [B, F] (caller computes |stft_t|
                before the call).
      - Output: real `mask` [B, F] (caller applies `stft_t * mask` after).
      - Per-block FIFO state passed as N positional args, returned as a flat
        tuple. Every input and output tensor in the exported graph is float32.

    Why a separate class: torch.onnx.export captures positional `forward`
    inputs and outputs into the ONNX graph. Flat positional state args
    produce a clean `state_i_in / state_i_out` IO naming scheme; nested
    list/tuple state would collapse into one opaque input.

    Mirrors NSNet2 streaming.py's split: ONNX produces only mask + new state;
    the complex multiply and STFT/iSTFT live in the Python wrapper.
    """

    def __init__(self, fast: ConvFSENetStreamingFast):
        super().__init__()
        self.fast = fast
        self.n_blocks = int(fast.n_blocks_total)
        self.buffer_sizes = [int(blk.buffer_size) for blk in fast.tcm_blocks]
        self.n_channels_conv = int(fast.tcm_blocks[0].n_channels_conv)
        self.n_features = int(fast.n_features)

    @property
    def state_input_names(self) -> List[str]:
        return [f"state_{i}_in" for i in range(self.n_blocks)]

    @property
    def state_output_names(self) -> List[str]:
        return [f"state_{i}_out" for i in range(self.n_blocks)]

    def init_states(self, batch_size: int, device, dtype=torch.float32) -> List[torch.Tensor]:
        return self.fast.init_states(batch_size, device, dtype)

    def forward(self, noisy_mag: torch.Tensor, *states_in: torch.Tensor):
        """
        noisy_mag: [B, F] float32
        states_in: n_blocks positional tensors, each [B, n_channels_conv, buffer_size_i]
        returns:   (mask, state_0_out, ..., state_{n_blocks-1}_out)
        """
        mask, states_out = self.fast.forward_mask_step(noisy_mag, list(states_in))
        return (mask, *states_out)


# =============================================================================
# Stateless-windowed view (no FIFO state — the STM32N6 efficiency rework).
#
# Removes the per-block FIFO state plumbing that is ConvFSENet's M55 floor on
# the Neural-ART (the Slice/Concat/Gather class, always Hybrid). Instead of
# threading per-block state across single frames, the host keeps a ring buffer
# of the last L = sum_blocks (K-1)*D magnitude columns and feeds a fixed window
#
#     noisy_mag_window  [B, n_freq, L + T]
#
# through the BN-folded offline-causal model run with VALID (padding-0) convs.
# Each TCM block's valid dilated dconv shrinks the time axis by (K-1)*D and
# crops the residual to match, so the window collapses L+T -> T with no padding
# and no state I/O. The emitted T mask columns are bit-exact to the offline
# causal model on those frames (full real left context — same receptive-field
# algebra the FIFO wrappers above parity-test). See deploy/stm32n6/
# EFFICIENCY_REWORK_PLAN.md (Track 1).
# =============================================================================


class _WindowedTCMBlockQF(nn.Module):
    """Valid-conv windowed forward for one TCMBlock_QuantFriendly (causal=True).

    Runs the BN-folded block over a [B, n_channels_res, W] window with VALID
    (padding-0) convs. The dilated dconv shrinks the time axis by trim=(K-1)*D;
    the residual is cropped by the same trim so it aligns with the surviving
    (rightmost) outputs. Equivalent to the offline causal block on its last
    (W-trim) output positions.

    Reuses the same fold/clone helpers as ConvFSENetStreamingFast — the only
    difference vs the FIFO block is that there is NO state buffer and NO gather:
    the dilation lives natively in the conv (padding=0), operating over the full
    context columns the host already supplied.
    """

    def __init__(self, src: TCMBlock_QuantFriendly):
        super().__init__()
        if not src.causal:
            raise ValueError(
                "Windowed wrapper requires causal=True; got causal=False on a "
                "TCM block. Train the model with causal=true in config."
            )
        K = int(src.kernel_size)
        D = int(src.dilation)
        self.kernel_size = K
        self.dilation = D
        self.groups = int(src.dconv.groups)
        self.trim = (K - 1) * D                       # time columns the valid dconv removes
        self.n_channels_res = int(src.n_channels_res)
        self.n_channels_conv = int(src.n_channels_conv)

        # Folded (BN absorbed) + padding=0 — _fold_bn_into_conv copies the
        # dilation, so dconv_folded is a *native* valid dilated conv (no gather).
        self.conv1x1_folded = _fold_bn_into_conv(src.conv1x1, src.norm1)   # k=1
        self.dconv_folded = _fold_bn_into_conv(src.dconv, src.norm2)       # k=K, dilation=D
        self.conv1x1_out = _clone_conv1d(src.conv1x1_out)                  # k=1, no BN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_channels_res, W] -> [B, n_channels_res, W - trim]."""
        y = self.conv1x1_folded(x)                    # [B, C', W]
        y = torch.relu(y)
        y = self.dconv_folded(y)                      # valid -> [B, C', W - trim]
        y = torch.relu(y)
        z = self.conv1x1_out(y)                       # [B, n_channels_res, W - trim]
        # Residual: valid output[j] aligns to causal position j+trim (its
        # right-most input), so add the trim-cropped block input.
        res = x[..., self.trim:] if self.trim > 0 else x
        return z + res


class ConvFSENetWindowedONNX(nn.Module):
    """Stateless windowed view of ConvFSENet_QuantFriendly (causal=True).

    ONNX-exportable. No FIFO state — the entire left context arrives as columns
    of a fixed window.

      Input:  noisy_mag_window [B, n_freq, L + T]  — plain |stft| context window.
      Output: mask            [B, n_freq, T]       — last T mask columns, in (0,1).

    L = sum_blocks (K-1)*D (the receptive field minus one). Magnitude
    compression (mag_compressed models) is the first op of the graph — an FP32
    Pow over the window, ahead of every quantized conv, exactly like
    ConvFSENetStreamingFast.forward_mask_step.

    256-bin alignment (`drop_nyquist=True`): the trained model carries 257 freq
    bins; the Neural-ART maps a power-of-two channel count more cleanly. With
    drop_nyquist the frontend Conv's Nyquist *input* column and the backend
    Conv's Nyquist *output* row+bias are sliced off, so n_freq becomes 256. The
    host then drops the Nyquist magnitude bin before the graph and pads the mask
    back to 257 after it. This slices trained weights (an approximation, not
    bit-exact to the 257 model) — verify PESQ; fine-tune if it shifts results.
    """

    def __init__(self, base: ConvFSENet_QuantFriendly, T: int = 1, drop_nyquist: bool = False):
        super().__init__()
        if not base.causal:
            raise ValueError(
                "ConvFSENetWindowedONNX requires causal=True on the base model "
                "(non-causal dilated conv has no streaming/windowed form). "
                "Re-train with causal=true."
            )
        if base.extractor_type not in _SUPPORTED_EXTRACTORS:
            raise NotImplementedError(
                f"Windowed wrapper supports extractor_type in {_SUPPORTED_EXTRACTORS} "
                f"(got {base.extractor_type!r})."
            )
        if base.training:
            raise ValueError(
                "Base model must be in .eval() mode before folding BatchNorm "
                "(running_mean / running_var only stabilize after .eval())."
            )
        self.T = int(T)
        if self.T < 1:
            raise ValueError(f"T (emitted frames) must be >= 1; got {self.T}.")
        self.drop_nyquist = bool(drop_nyquist)
        self.n_features_full = int(base.n_features)
        self.n_features = self.n_features_full - (1 if self.drop_nyquist else 0)

        # Frontend / backend convs (k=1, no surrounding BN). Optionally slice
        # the Nyquist bin so the graph is 256-wide.
        front = _clone_conv1d(base.frontend[0])
        back = _clone_conv1d(base.backend[0])
        if self.drop_nyquist:
            front, back = self._slice_nyquist(front, back)
        self.frontend_conv = front
        self.backend_conv = back

        self.blocks = nn.ModuleList(_WindowedTCMBlockQF(m) for m in base.tcm)
        self.n_blocks_total = len(self.blocks)
        # L = total time shrink across all blocks = receptive field minus one.
        self.L = int(sum(blk.trim for blk in self.blocks))

        self.extractor_type = base.extractor_type
        self.compress_factor = (
            float(base.compress_factor)
            if base.extractor_type == "mag_compressed" and base.compress_factor
            else 0.3
        )

    @staticmethod
    def _slice_nyquist(front: nn.Conv1d, back: nn.Conv1d):
        """Drop the Nyquist bin: front loses its last *input* channel, back
        loses its last *output* channel (+bias). Both are k=1 convs."""
        new_front = nn.Conv1d(
            front.in_channels - 1, front.out_channels, 1,
            bias=front.bias is not None,
        )
        new_back = nn.Conv1d(
            back.in_channels, back.out_channels - 1, 1,
            bias=back.bias is not None,
        )
        with torch.no_grad():
            new_front.weight.copy_(front.weight[:, :-1, :])
            if front.bias is not None:
                new_front.bias.copy_(front.bias)
            new_back.weight.copy_(back.weight[:-1, :, :])
            if back.bias is not None:
                new_back.bias.copy_(back.bias[:-1])
        return new_front, new_back

    @property
    def window_length(self) -> int:
        """L + T — the time width of the input window the host must feed."""
        return self.L + self.T

    def forward_mask_window(self, noisy_mag_window: torch.Tensor) -> torch.Tensor:
        """noisy_mag_window [B, n_freq, L+T] -> mask [B, n_freq, T]."""
        if self.extractor_type == "mag_compressed":
            feats = compress_magnitude(noisy_mag_window, self.compress_factor)
        else:
            feats = noisy_mag_window
        x = self.frontend_conv(feats)                 # [B, C_res, L+T]
        x = torch.relu(x)
        for blk in self.blocks:
            x = blk(x)                                # shrinks by trim each -> [B, C_res, T]
        mask = torch.sigmoid(self.backend_conv(x))    # [B, n_freq, T]
        return mask

    def forward(self, noisy_mag_window: torch.Tensor) -> torch.Tensor:
        return self.forward_mask_window(noisy_mag_window)
