"""Smoke tests for ConvFSENet QAT scaffolding.

Verifies prepare_for_qat (extended for nn.Conv1d in quant_fake.py) targets
every conv in ConvFSENet, that forward/backward flows through the STE
fake-quant to the underlying weights, that the parametrized weight is
genuinely int8-representable, and that _save_qat_checkpoint produces a plain
checkpoint loadable into a fresh (non-QAT) model — the contract the export
pipeline depends on.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from convfsenet.model import ConvFSENet_QuantFriendly
from convfsenet.qat_train import _save_qat_checkpoint
from common.quant_fake import QSpec, is_qat_prepared, prepare_for_qat
from common.utils import load_checkpoint


def _fresh(extractor_type: str = "mag_compressed") -> ConvFSENet_QuantFriendly:
    return ConvFSENet_QuantFriendly(
        n_fft=512, win_length=512, n_features=257,
        n_channels_res=128, n_channels_conv=256, kernel_size=3,
        n_blocks=3, n_stacks=3,
        extractor_type=extractor_type,
        compress_factor=0.3 if extractor_type == "mag_compressed" else None,
        causal=True,
        loss=None, preproc=None, postproc=None,
    )


def _n_conv1d(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, nn.Conv1d))


def test_prepare_for_qat_targets_every_conv():
    """frontend + 9*(conv1x1, dconv, conv1x1_out) + backend = 29 Conv1d."""
    torch.manual_seed(0)
    m = _fresh()
    n_conv = _n_conv1d(m)
    assert n_conv == 29, f"expected 29 Conv1d, got {n_conv}"

    nw, na = prepare_for_qat(
        m, weight=QSpec(bits=8, axis=None), activation=QSpec(bits=8),
    )
    assert nw == 29, f"expected all 29 conv weights parametrized, got {nw}"
    assert na == 29, f"expected all 29 conv inputs hooked, got {na}"
    assert is_qat_prepared(m)


def test_qat_forward_backward_reaches_raw_weights():
    """Forward through the fake-quant + backward; STE must route grads to the
    underlying (raw) nn.Parameter behind each parametrization."""
    torch.manual_seed(0)
    m = _fresh()
    prepare_for_qat(m, weight=QSpec(bits=8, axis=None), activation=QSpec(bits=8))
    m.train()

    stft = torch.randn(2, 1, 257, 40, dtype=torch.complex64)
    out = m(stft)
    assert out.shape == (2, 1, 257, 40)
    out.abs().mean().backward()

    for mod in m.modules():
        if isinstance(mod, nn.Conv1d):
            raw = mod.parametrizations.weight.original
            assert raw.grad is not None, "STE did not route grad to raw weight"
            assert torch.isfinite(raw.grad).all()


def test_parametrized_weight_is_int8_representable():
    """Reading conv.weight through the parametrization yields a fake-quantized
    tensor: per output channel, at most 2^8-1 distinct dequantized values."""
    torch.manual_seed(0)
    m = _fresh()
    prepare_for_qat(m, weight=QSpec(bits=8, axis=None), activation=QSpec(bits=8))

    # frontend conv: weight (128, 257, 1) — 257 values per output channel,
    # enough to make the <=255-distinct bound a real check.
    front = m.frontend[0]
    assert isinstance(front, nn.Conv1d)
    w = front.weight                                  # parametrized -> fake-quant
    for c in range(min(6, w.shape[0])):
        distinct = torch.unique(w[c]).numel()
        assert distinct <= 256, f"channel {c}: {distinct} distinct values (> 256)"


def test_save_qat_checkpoint_loads_into_plain_model(tmp_path):
    """_save_qat_checkpoint must emit a plain {'generator': state_dict} that
    strict-loads into a fresh non-QAT ConvFSENet (export pipeline contract),
    write qat_recipe.json, and leave the live model still QAT-prepared."""
    torch.manual_seed(0)
    m = _fresh()
    prepare_for_qat(m, weight=QSpec(bits=8, axis=None), activation=QSpec(bits=8))

    ckpt = tmp_path / "g_best"
    _save_qat_checkpoint(m, str(ckpt), w_bits=8, a_bits=8, source="unit-test")
    assert ckpt.exists()

    recipe_path = tmp_path / "qat_recipe.json"
    assert recipe_path.exists()
    recipe = json.loads(recipe_path.read_text())
    assert recipe == {"w_bits": 8, "a_bits": 8, "source": "unit-test"}

    # Strict-load into a fresh plain model — no parametrization keys.
    fresh = _fresh()
    sd = load_checkpoint(str(ckpt), torch.device("cpu"))
    fresh.load_state_dict(sd["generator"], strict=True)

    # The live QAT model is re-armed for continued training.
    assert any(
        isinstance(mod, nn.Conv1d) and hasattr(mod, "parametrizations")
        and "weight" in mod.parametrizations
        for mod in m.modules()
    ), "parametrizations were not re-installed after save"


def test_mag_path_also_qat_preparable():
    """QAT works on the plain-magnitude extractor too, not just compressed."""
    torch.manual_seed(0)
    m = _fresh(extractor_type="mag")
    nw, na = prepare_for_qat(
        m, weight=QSpec(bits=8, axis=None), activation=QSpec(bits=8),
    )
    assert nw == 29 and na == 29
