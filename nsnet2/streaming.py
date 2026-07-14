"""Streaming-friendly view of NSNet2 — unrolled per-step GRU for ONNX export.

Wraps a trained ``NSNet2`` and replaces its cuDNN ``nn.GRU`` with a
hand-rolled per-step cell. The cell holds twelve ``nn.Parameter`` tensors
per layer (sliced once at ``__init__`` from ``nn.GRU.weight_ih_l[k]`` /
``weight_hh_l[k]`` / ``bias_ih_l[k]`` / ``bias_hh_l[k]``) so the exported
ONNX graph contains only ``MatMul`` / ``Add`` / ``Sigmoid`` / ``Tanh``
primitives — no opaque ``GRU`` op (Pitfall 11), no ``aten::*gru*``
symbolic miss (Pitfall 10).

Per-step formula (matches cuDNN ``nn.GRU`` with ``linear_before_reset=False``):

    r_t = sigmoid(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)
    z_t = sigmoid(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)
    n_t = tanh   (W_in x_t + b_in + r_t * (W_hn h_{t-1} + b_hn))
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}

This is NOT the textbook formula used by ``models.layers.StructuredGRUCell``
(which puts the reset gate before the W_hn projection). See Pitfall 1.

NSNet2 itself stays untouched (training stays on cuDNN) — D-01.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from common.env import AttrDict
from nsnet2.model import NSNet2


class NSNet2Streaming(nn.Module):
    """Streaming view of NSNet2 with an unrolled GRU cell.

    Composes a trained ``NSNet2`` (FC layers reused by reference per D-01)
    and replaces ``base.gru`` with twelve per-gate ``nn.Parameter`` tensors
    per layer so ``forward_step`` exports cleanly via
    ``torch.onnx.export(dynamo=False, opset_version=17)``.

    For ``base.gru_kind != "gru"`` (structured variants — butterfly, blockdiag,
    monarch, and the triton_* kinds), the wrapper delegates ``forward_step`` to
    ``base.gru`` per-step (D-05); the Phase 1 parity test and ONNX export
    smoke test SKIP that branch.
    """

    def __init__(self, base: NSNet2):
        super().__init__()
        # D-01: hold base by reference; FC layers are reused without cloning.
        self.base = base
        self.gru_kind = base.gru_kind

        if self.gru_kind == "gru":
            # cuDNN path — slice the packed nn.GRU parameters once into
            # per-gate nn.Parameter tensors (D-02).
            if not isinstance(base.gru, nn.GRU):
                raise ValueError(
                    f"NSNet2Streaming expected base.gru to be nn.GRU when "
                    f"gru_kind='gru', got {type(base.gru).__name__!r}"
                )
            src: nn.GRU = base.gru
            self.num_layers = int(src.num_layers)
            self.hidden_size = int(src.hidden_size)
            for k in range(self.num_layers):
                # Packed shapes (PyTorch convention, gates [r | z | n]):
                W_ih = getattr(src, f"weight_ih_l{k}")  # (3*H, input_dim_k) packed [r | z | n]
                W_hh = getattr(src, f"weight_hh_l{k}")  # (3*H, H)            packed [r | z | n]
                b_ih = getattr(src, f"bias_ih_l{k}")    # (3*H,)              packed [r | z | n]
                b_hh = getattr(src, f"bias_hh_l{k}")    # (3*H,)              packed [r | z | n]
                # chunk(3, dim=0) preserves PyTorch [r | z | n] order — NOT ONNX [z | r | h].
                W_ir, W_iz, W_in = W_ih.chunk(3, dim=0)
                W_hr, W_hz, W_hn = W_hh.chunk(3, dim=0)
                b_ir, b_iz, b_in = b_ih.chunk(3, dim=0)
                b_hr, b_hz, b_hn = b_hh.chunk(3, dim=0)
                # Clone — wrapper owns its own copy; base.gru weights stay untouched
                # (the GRU is the bit being replaced — there is no by-reference contract
                # here; D-01 is about FC weights only).
                setattr(self, f"W_ir_{k}", nn.Parameter(W_ir.clone()))
                setattr(self, f"W_iz_{k}", nn.Parameter(W_iz.clone()))
                setattr(self, f"W_in_{k}", nn.Parameter(W_in.clone()))
                setattr(self, f"W_hr_{k}", nn.Parameter(W_hr.clone()))
                setattr(self, f"W_hz_{k}", nn.Parameter(W_hz.clone()))
                setattr(self, f"W_hn_{k}", nn.Parameter(W_hn.clone()))
                setattr(self, f"b_ir_{k}", nn.Parameter(b_ir.clone()))
                setattr(self, f"b_iz_{k}", nn.Parameter(b_iz.clone()))
                setattr(self, f"b_in_{k}", nn.Parameter(b_in.clone()))
                setattr(self, f"b_hr_{k}", nn.Parameter(b_hr.clone()))
                setattr(self, f"b_hz_{k}", nn.Parameter(b_hz.clone()))
                setattr(self, f"b_hn_{k}", nn.Parameter(b_hn.clone()))
        else:
            # D-05: structured variant — delegate per-step to base.gru.
            # Phase 1 parity + ONNX smoke tests SKIP this branch.
            # We still expose num_layers/hidden_size so callers can build h0.
            # StructuredGRU exposes these attributes; assume duck-typed access.
            self.num_layers = int(getattr(base.gru, "num_layers", 0))
            self.hidden_size = int(getattr(base.gru, "hidden_size", 0))

    def forward_step(
        self,
        frame_in: torch.Tensor,           # (B, n_freq)
        states_in: torch.Tensor,          # (num_layers, B, hidden)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Mask shape contract: (mask: (B, n_freq), states_out: (num_layers, B, hidden))
        if self.gru_kind != "gru":
            return self._forward_step_structured(frame_in, states_in)

        # 1. fc_in + ReLU — D-04: torch.relu inline (no nn.ReLU module ref).
        h = torch.relu(self.base.fc_in(frame_in))    # (B, H)
        # 2. unrolled GRU stack — per-layer per-step compute.
        new_states = []
        for k in range(self.num_layers):
            h_prev = states_in[k]                     # (B, H)
            W_ir = getattr(self, f"W_ir_{k}")
            W_iz = getattr(self, f"W_iz_{k}")
            W_in_ = getattr(self, f"W_in_{k}")
            W_hr = getattr(self, f"W_hr_{k}")
            W_hz = getattr(self, f"W_hz_{k}")
            W_hn = getattr(self, f"W_hn_{k}")
            b_ir = getattr(self, f"b_ir_{k}")
            b_iz = getattr(self, f"b_iz_{k}")
            b_in_ = getattr(self, f"b_in_{k}")
            b_hr = getattr(self, f"b_hr_{k}")
            b_hz = getattr(self, f"b_hz_{k}")
            b_hn = getattr(self, f"b_hn_{k}")
            # cuDNN GRU formula (linear_before_reset=False):
            #   r multiplies the W_hn projection AND its bias as a whole.
            r = torch.sigmoid(h @ W_ir.T + b_ir + h_prev @ W_hr.T + b_hr)            # (B, H)
            z = torch.sigmoid(h @ W_iz.T + b_iz + h_prev @ W_hz.T + b_hz)            # (B, H)
            n = torch.tanh(   h @ W_in_.T + b_in_ + r * (h_prev @ W_hn.T + b_hn))    # (B, H)
            h = (1 - z) * n + z * h_prev                                              # (B, H) — layer-k output
            new_states.append(h)
        states_out = torch.stack(new_states, dim=0)   # (num_layers, B, H)
        # 3. fc1 / fc2 / fc_out — D-04 inline activations.
        h = torch.relu(self.base.fc1(h))              # (B, fc_hidden)
        h = torch.relu(self.base.fc2(h))              # (B, fc_hidden)
        mask = torch.sigmoid(self.base.fc_out(h))     # (B, n_freq)
        return mask, states_out

    def _forward_step_structured(
        self,
        frame_in: torch.Tensor,           # (B, n_freq)
        states_in: torch.Tensor,          # (num_layers, B, hidden)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # D-05: structured variant fallback — delegate per-frame to base.gru.
        # base.gru is StructuredGRU (or similar) with forward(x, h0) signature
        # matching nn.GRU. We feed a (B, T=1, H) input.
        h = torch.relu(self.base.fc_in(frame_in))    # (B, H)
        h_in = h.unsqueeze(1)                         # (B, T=1, H)
        y, states_out = self.base.gru(h_in, states_in)  # y: (B, 1, H); states_out: (num_layers, B, H)
        h = y.squeeze(1)                              # (B, H)
        h = torch.relu(self.base.fc1(h))              # (B, fc_hidden)
        h = torch.relu(self.base.fc2(h))              # (B, fc_hidden)
        mask = torch.sigmoid(self.base.fc_out(h))     # (B, n_freq)
        return mask, states_out

    def forward_full(self, mag: torch.Tensor) -> torch.Tensor:
        # D-11: stops at mask. mag: (B, T, n_freq) -> mask: (B, T, n_freq).
        # STR-03: drives forward_step in a Python time-loop.
        B, T, _ = mag.shape
        states = torch.zeros(
            self.num_layers, B, self.hidden_size,
            dtype=mag.dtype, device=mag.device,
        )                                              # (num_layers, B, H)
        masks = []
        for t in range(T):
            frame_in = mag[:, t]                       # (B, n_freq)
            mask_t, states = self.forward_step(frame_in, states)
            masks.append(mask_t)
        return torch.stack(masks, dim=1)               # (B, T, n_freq)

    @classmethod
    def from_checkpoint(cls, path):
        """Load a trained NSNet2 from a checkpoint file and wrap it for streaming.

        path: checkpoint file path (e.g., ``cp_baseline/g_best``). Sibling
        ``config.json`` is read from ``Path(path).parent / 'config.json'`` per
        the inference.py / repo convention. DET-02: torch.load uses
        ``weights_only=True`` (Pitfall 13 closure for the new code site).

        Raises FileNotFoundError if path or sibling config.json is missing.
        Raises KeyError if the checkpoint is not a {'generator': state_dict} dict.
        """
        ckpt_path = Path(path)
        config_path = ckpt_path.parent / "config.json"

        with open(config_path) as f:
            h = AttrDict(json.load(f))

        base = NSNet2(h)
        ckpt = torch.load(str(ckpt_path), weights_only=True, map_location="cpu")  # DET-02 / D-12
        base.load_state_dict(ckpt["generator"], strict=True)
        base.eval()
        return cls(base)

    # Alias so torch.onnx.export traces forward_step by default — Plan 04 export
    # smoke test depends on this. Class-level attribute, NOT a method redefinition.
    forward = forward_step
