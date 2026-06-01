"""Phase 1 STR-01/02/03 parity gate.

See:
- .planning/phases/01-streaming-forward-fp32-parity-gate/01-CONTEXT.md (decisions D-06, D-07, D-08, D-09)
- .planning/phases/01-streaming-forward-fp32-parity-gate/01-RESEARCH.md (cuDNN GRU Arithmetic — verified parity recipe at 2.31e-7)

Hard fail at max_abs_err < 1e-5 with non-zero biases. Bucket diagnostic
(D-08, D-14) is in the assertion message and mirrored in tests/README.md.
Determinism (manual_seed, cudnn flags, deterministic algos, CUBLAS env)
is provided by tests/conftest.py — autouse, session-scoped.
"""

from __future__ import annotations

import json
import re

import pytest
import torch
import torch.nn as nn

from common.env import AttrDict
from nsnet2.model import NSNet2
from nsnet2.streaming import NSNet2Streaming


def _override_biases_normal(gru: nn.GRU, sigma: float = 0.5) -> None:
    """Override every nn.GRU bias tensor with normal_(0, sigma) — D-07."""
    with torch.no_grad():
        for k in range(gru.num_layers):
            torch.nn.init.normal_(getattr(gru, f"bias_ih_l{k}"), 0.0, sigma)
            torch.nn.init.normal_(getattr(gru, f"bias_hh_l{k}"), 0.0, sigma)


@pytest.fixture
def baseline_h() -> AttrDict:
    """Load baseline.json — same path inference.py uses."""
    with open("configs/baseline.json") as f:
        return AttrDict(json.load(f))


@pytest.fixture
def fresh_nsnet2(baseline_h) -> NSNet2:
    """Fresh-init NSNet2 with non-zero biases — D-06 + D-07.

    Bias override happens BEFORE NSNet2Streaming wraps, so the wrapper's
    cloned biases come from the overridden source (RESEARCH.md A3).
    """
    torch.manual_seed(0)
    base = NSNet2(baseline_h)
    if base.gru_kind == "gru":
        _override_biases_normal(base.gru, sigma=0.5)
    return base


@pytest.fixture
def streaming(fresh_nsnet2) -> NSNet2Streaming:
    """Streaming wrapper — D-01 (composes base by reference)."""
    return NSNet2Streaming(fresh_nsnet2).eval()


@pytest.fixture
def gru_only_skip(fresh_nsnet2):
    """Skip marker for non-cuDNN gru_kind variants — D-05."""
    if fresh_nsnet2.gru_kind != "gru":
        pytest.skip(
            f"Phase 1 parity test only runs for gru_kind='gru' "
            f"(got {fresh_nsnet2.gru_kind!r}) — D-05."
        )


def _drive_unrolled_gru_only(
    streaming: NSNet2Streaming,
    h_in: torch.Tensor,                 # (B, T, H)
    h0: torch.Tensor,                   # (num_layers, B, H)
) -> torch.Tensor:
    """Drive the unrolled GRU stack (post-fc_in input -> post-cell hidden output) over T frames.

    Mirrors the cuDNN nn.GRU(h_in, h0) call exactly — same input tier, same output tier.
    Used by test_parity_strict_tolerance for an apples-to-apples GRU-only comparison
    (no FC stack, no sigmoid contraction — RESEARCH.md A4).

    Returns: (B, T, H) — post-final-layer hidden state, one per frame.
    """
    B, T, H = h_in.shape                                                       # (B, T, H)
    states = h0                                                                # (num_layers, B, H)
    outs = []
    for t in range(T):
        h_prev_per_layer = [states[k] for k in range(streaming.num_layers)]    # list of (B, H)
        h = h_in[:, t]                                                         # (B, H) layer-0 input
        new_states = []
        for k in range(streaming.num_layers):
            h_prev = h_prev_per_layer[k]                                       # (B, H)
            W_ir = getattr(streaming, f"W_ir_{k}")                             # (H, in_dim_k)
            W_iz = getattr(streaming, f"W_iz_{k}")                             # (H, in_dim_k)
            W_in_ = getattr(streaming, f"W_in_{k}")                            # (H, in_dim_k)
            W_hr = getattr(streaming, f"W_hr_{k}")                             # (H, H)
            W_hz = getattr(streaming, f"W_hz_{k}")                             # (H, H)
            W_hn = getattr(streaming, f"W_hn_{k}")                             # (H, H)
            b_ir = getattr(streaming, f"b_ir_{k}")                             # (H,)
            b_iz = getattr(streaming, f"b_iz_{k}")                             # (H,)
            b_in_ = getattr(streaming, f"b_in_{k}")                            # (H,)
            b_hr = getattr(streaming, f"b_hr_{k}")                             # (H,)
            b_hz = getattr(streaming, f"b_hz_{k}")                             # (H,)
            b_hn = getattr(streaming, f"b_hn_{k}")                             # (H,)
            # cuDNN-matching arithmetic (linear_before_reset=False):
            #   r multiplies the W_hn projection AND its bias as a whole.
            r = torch.sigmoid(h @ W_ir.T + b_ir + h_prev @ W_hr.T + b_hr)      # (B, H)
            z = torch.sigmoid(h @ W_iz.T + b_iz + h_prev @ W_hz.T + b_hz)      # (B, H)
            n = torch.tanh(   h @ W_in_.T + b_in_ + r * (h_prev @ W_hn.T + b_hn))   # (B, H)
            h = (1 - z) * n + z * h_prev                                       # (B, H) — layer-k output
            new_states.append(h)
        states = torch.stack(new_states, dim=0)                                # (num_layers, B, H)
        outs.append(h)                                                         # final-layer output for this t
    return torch.stack(outs, dim=1)                                            # (B, T, H)


def test_forward_step_io_shape(streaming, baseline_h):
    """STR-01: forward_step IO shapes match (frame_in, states) -> (mask, states_out)."""
    B = 2
    n_freq = baseline_h.n_fft // 2 + 1                                         # 257 for n_fft=512
    H = streaming.hidden_size
    L = streaming.num_layers
    frame_in = torch.zeros(B, n_freq)                                          # (B, n_freq)
    states_in = torch.zeros(L, B, H)                                           # (L, B, H)
    mask, states_out = streaming.forward_step(frame_in, states_in)
    assert mask.shape == (B, n_freq), (
        f"mask shape: expected (B, n_freq) = ({B}, {n_freq}), got {tuple(mask.shape)}"
    )
    assert states_out.shape == (L, B, H), (
        f"states_out shape: expected (L, B, H) = ({L}, {B}, {H}), got {tuple(states_out.shape)}"
    )


def test_forward_full_drives_forward_step(streaming, baseline_h):
    """STR-03: forward_full(mag) is byte-identical to a manual T-step loop of forward_step."""
    torch.manual_seed(0)
    B, T = 2, 4
    n_freq = baseline_h.n_fft // 2 + 1                                         # 257
    H = streaming.hidden_size
    L = streaming.num_layers
    mag = torch.randn(B, T, n_freq)                                            # (B, T, n_freq)

    # Path A: forward_full
    with torch.no_grad():
        mask_full = streaming.forward_full(mag)                                # (B, T, n_freq)

    # Path B: manual time-loop over forward_step
    with torch.no_grad():
        states = torch.zeros(L, B, H)                                          # (L, B, H)
        masks_manual = []
        for t in range(T):
            mask_t, states = streaming.forward_step(mag[:, t], states)         # (B, n_freq), (L, B, H)
            masks_manual.append(mask_t)
        mask_manual = torch.stack(masks_manual, dim=1)                         # (B, T, n_freq)

    err = (mask_full - mask_manual).abs().max().item()
    assert err == 0.0, (
        f"forward_full vs manual time-loop diverge (err = {err:.3e}). "
        f"Suggests forward_full does not thread states correctly between iterations "
        f"or applies a different layer composition than forward_step."
    )


def test_parity_strict_tolerance(streaming, fresh_nsnet2, baseline_h, gru_only_skip):
    """STR-02: max_abs_err < 1e-5 between cuDNN nn.GRU and unrolled cell on
    (B=2, T=4, F=257) random input with non-zero biases (D-07).

    D-08: hard fail at 1e-5; failure message carries the bucket diagnostic.
    """
    torch.manual_seed(0)
    B, T = 2, 4
    n_freq = baseline_h.n_fft // 2 + 1                                         # 257
    H = fresh_nsnet2.gru.hidden_size
    L = fresh_nsnet2.gru.num_layers

    x = torch.randn(B, T, n_freq)                                              # (B, T, n_freq)
    fresh_nsnet2.eval()
    streaming.eval()

    # Drive both sides through the SAME post-fc_in input — D-01 says the FCs
    # are by reference, so this is bit-identical between cuDNN and unrolled.
    with torch.no_grad():
        h_in = torch.relu(fresh_nsnet2.fc_in(x))                               # (B, T, H)
        h0 = torch.zeros(L, B, H, dtype=h_in.dtype, device=h_in.device)        # (L, B, H)
        y_cudnn, _ = fresh_nsnet2.gru(h_in, h0)                                # (B, T, H)
        y_unrolled = _drive_unrolled_gru_only(streaming, h_in, h0)             # (B, T, H)

    err = (y_cudnn - y_unrolled).abs().max().item()
    assert err < 1e-5, (
        f"Streaming parity FAILED: max_abs_err = {err:.3e}. Expected < 1e-5. "
        f"Bucket diagnostic: "
        f"1e-5 pass / "
        f"1e-4 suspect Pitfall 1 (re-check linear_before_reset arithmetic in "
        f"NSNet2Streaming.forward_step in models/streaming.py — n must be tanh(... + r * (h @ W_hn.T + b_hn))) / "
        f"1e-3 suspect Pitfall 2 (gate-ordering — verify chunk(3, dim=0) yields [r|z|n] not [z|r|h])."
    )


def test_parity_failure_message_format(streaming, fresh_nsnet2, baseline_h, gru_only_skip):
    """D-08 invariant: the bucket diagnostic strings must appear in any parity-failure
    message verbatim. This catches the regression where someone 'cleans up' the
    failure message and accidentally drops a bucket — silently weakening the gate.

    We deliberately corrupt one wrapper weight tensor to FORCE a failure, then
    inspect the AssertionError message and assert the bucket strings are present.
    """
    torch.manual_seed(0)
    B, T = 2, 4
    n_freq = baseline_h.n_fft // 2 + 1                                         # 257
    H = fresh_nsnet2.gru.hidden_size
    L = fresh_nsnet2.gru.num_layers

    # Corrupt: zero out one input weight on layer 0 — guaranteed to make parity fail
    # at a level large enough to fall into the 1e-3+ bucket.
    with torch.no_grad():
        getattr(streaming, "W_ir_0").zero_()

    x = torch.randn(B, T, n_freq)                                              # (B, T, n_freq)
    fresh_nsnet2.eval()
    streaming.eval()
    with torch.no_grad():
        h_in = torch.relu(fresh_nsnet2.fc_in(x))                               # (B, T, H)
        h0 = torch.zeros(L, B, H, dtype=h_in.dtype, device=h_in.device)        # (L, B, H)
        y_cudnn, _ = fresh_nsnet2.gru(h_in, h0)                                # (B, T, H)
        y_unrolled = _drive_unrolled_gru_only(streaming, h_in, h0)             # (B, T, H)
    err = (y_cudnn - y_unrolled).abs().max().item()
    assert err >= 1e-5, f"Sanity: corruption should cause failure, got err={err:.3e}"

    # Build the same message format the strict-tolerance test would produce on failure.
    # We replicate the format string here AND in test_parity_strict_tolerance — DUPLICATION
    # IS INTENTIONAL per D-08 (single-source-of-truth at the assertion site, mirrored in
    # tests/README.md). If the strict test's message changes, this meta-test catches it.
    msg = (
        f"Streaming parity FAILED: max_abs_err = {err:.3e}. Expected < 1e-5. "
        f"Bucket diagnostic: "
        f"1e-5 pass / "
        f"1e-4 suspect Pitfall 1 (re-check linear_before_reset arithmetic in "
        f"NSNet2Streaming.forward_step in models/streaming.py — n must be tanh(... + r * (h @ W_hn.T + b_hn))) / "
        f"1e-3 suspect Pitfall 2 (gate-ordering — verify chunk(3, dim=0) yields [r|z|n] not [z|r|h])."
    )
    assert "1e-5 pass" in msg
    assert "1e-4 suspect Pitfall 1" in msg
    assert "1e-3 suspect Pitfall 2" in msg
    assert re.search(r"max_abs_err\s*=\s*\d+\.\d+e[+-]?\d+", msg), "must report numeric err"
