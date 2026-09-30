"""Tests for the Cadence-NNE kernel-shape variant (configs/lisennet_hybrid_nc24_nne.json).

Probing the NNE scheduler (xnneSched) with single-conv models showed it hands five
of the hybrid nc24's 36 convs to the HiFi: the two ``1x11`` intra depthwise convs
(width >= 7 never offloads) and the three encoder high-band ``(2,5)`` stride-``(1,3)``
convs (stride 3 never offloads; width 5 on the 195-wide first-stage input doesn't
either). The variant swaps exactly those — ``intra_kernel=[5, 5]`` and
``downsample="slice"`` — so these tests pin that:

  * the exported streaming graph carries no conv the probes sent to the HiFi;
  * the encoder widths (and so the decoder) are unchanged;
  * it is still causal and still streams with parity;
  * configs that don't set the new keys build the exact same modules as before.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from lisennet.export_onnx import LiSenNetONNX, export_streaming_fp32
from lisennet.model import build_lisennet
from lisennet.streaming import LiSenNetStreamingONNX, disable_streaming

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _model(name):
    torch.manual_seed(0)
    model = build_lisennet(json.loads((CONFIGS / f"{name}.json").read_text())).eval()
    for mod in model.modules():                               # non-trivial BN so parity means something
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5)
            mod.running_var.uniform_(0.5, 2.0)
    return model


def _convs(path):
    g = onnx.load(str(path))
    for n in g.graph.node:
        if n.op_type == "Conv":
            a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
            yield list(a["kernel_shape"]), list(a.get("strides", [1, 1])), a.get("group", 1)


def test_streaming_graph_has_no_hifi_bound_conv(tmp_path):
    path = tmp_path / "nne.onnx"
    export_streaming_fp32(_model("lisennet_hybrid_nc24_nne"), path)
    convs = list(_convs(path))
    assert all(max(k) < 7 for k, _, _ in convs), "a conv of width >= 7 stays on the HiFi"
    assert all(3 not in s for _, s, _ in convs), "a stride-3 conv stays on the HiFi"
    # Width 5 is only confirmed on NNE as the bottleneck's 24-channel depthwise.
    wide = [(k, g) for k, _, g in convs if k[1] == 5]
    assert len(wide) == 4 and all(g == 24 for _, g in wide)


def test_original_graph_has_the_five_probed_convs(tmp_path):
    """Guards the audit above: it does catch the five layers in the original."""
    path = tmp_path / "orig.onnx"
    export_streaming_fp32(_model("lisennet_hybrid_nc24"), path)
    convs = list(_convs(path))
    assert len(convs) == 36
    assert sum(k == [1, 11] for k, _, _ in convs) == 2
    assert sum(k == [2, 5] and s == [1, 3] for k, s, _ in convs) == 3


def test_encoder_widths_unchanged():
    feat = torch.randn(1, 3, 8, 257)
    with torch.no_grad():
        a = [x.shape for x in _model("lisennet_hybrid_nc24").encoder(feat)]
        b = [x.shape for x in _model("lisennet_hybrid_nc24_nne").encoder(feat)]
    assert a == b and [s[-1] for s in b] == [128, 64, 32]


def test_causal_and_streams_with_parity(tmp_path):
    model = _model("lisennet_hybrid_nc24_nne")
    n = 60
    feat = torch.randn(1, 3, n, 257)
    alt = feat.clone()
    alt[:, :, n // 2:] = torch.randn_like(alt[:, :, n // 2:])
    with torch.no_grad():
        ref = LiSenNetONNX(model)(feat).numpy()
        ref_alt = LiSenNetONNX(model)(alt).numpy()
    assert np.abs(ref[:, :n // 2] - ref_alt[:, :n // 2]).max() == 0.0

    path = tmp_path / "s.onnx"
    export_streaming_fp32(model, path)
    disable_streaming(model)
    sess = ort.InferenceSession(str(path))
    names = [i.name for i in sess.get_inputs()]
    states = [s.numpy() for s in LiSenNetStreamingONNX(model).init_states(1)]
    out = []
    for t in range(n):
        r = sess.run(None, dict(zip(names, [feat[:, :, t:t + 1].numpy()] + states)))
        out.append(r[0])
        states = r[1:]
    np.testing.assert_allclose(np.concatenate(out, 1), ref, atol=1e-4)


def test_defaults_build_the_original_modules():
    """Existing configs/checkpoints: same keys and shapes as before the new options."""
    sd = _model("lisennet_hybrid_nc24").state_dict()
    assert sd["encoder.conv_2.high_conv.1.weight"].shape[-2:] == (2, 5)
    assert sd["blocks.0.dp_rnn_attn.intra_dw.weight"].shape[-1] == 11
