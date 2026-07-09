"""Export the end-to-end butterfly NSNet2 to a single waveform->waveform ONNX.

Unlike every other model in this repo, the STFT is *inside* the graph — because
it is a learned butterfly (structured MatMuls), not ``torch.stft``. The exported
graph is therefore fully self-contained: noisy waveform in, enhanced waveform
out, no external framing / FFT / masking wrapper.

The butterfly custom op does not trace directly, so we reuse
``_patch_structured_for_export`` from ``nsnet2/export_onnx.py`` — it swaps
``Butterfly.forward`` for a structure-preserving, ONNX-friendly equivalent
(the twiddle factors survive as structure, not a dense N x N matrix).

Batch is fixed to 1 (whole-utterance enhancement); the sample/time axis is
dynamic.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from common.env import AttrDict
from common.utils import load_checkpoint
from nsnet2.model_e2e import NSNet2E2E
from nsnet2.export_onnx import _patch_structured_for_export


class _E2EExport(nn.Module):
    """Waveform->waveform, no data-dependent padding/cropping. The caller feeds
    a frame-aligned length ``L = (T-1)*hop + win`` so analysis/synthesis round-
    trip exactly; output length equals input length."""

    def __init__(self, model: NSNet2E2E):
        super().__init__()
        self.m = model

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        w = self.m.transform.analyze(wav)        # (B, T, N)
        mask = self.m.core.predict_mask(w)       # (B, T, N)
        return self.m.transform.synthesize(w * mask)   # (B, L)


def _valid_len(win: int, hop: int, n_frames: int) -> int:
    return (n_frames - 1) * hop + win


def _load_from_checkpoint(checkpoint_file):
    cfg_path = Path(checkpoint_file).parent / "config.json"
    with open(cfg_path) as f:
        h = AttrDict(json.loads(f.read()))
    model = NSNet2E2E(h)
    state = load_checkpoint(checkpoint_file, torch.device("cpu"))
    model.load_state_dict(state["generator"])
    model.eval()
    return model, h


def export_e2e_fp32(model: NSNet2E2E, output_path, opset: int = 17, n_frames: int = 64):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    L = _valid_len(model.win, model.hop, n_frames)
    wav = torch.zeros(1, L, dtype=torch.float32)

    wrapper = _E2EExport(model).eval()
    with _patch_structured_for_export(wrapper):
        torch.onnx.export(
            wrapper,
            (wav,),
            str(output_path),
            input_names=["noisy_wav"],
            output_names=["enhanced_wav"],
            dynamic_axes={"noisy_wav": {1: "L"}, "enhanced_wav": {1: "L"}},
            opset_version=opset,
            dynamo=False,
        )

    model_onnx = onnx.load(str(output_path))
    onnx.checker.check_model(model_onnx)
    op_hist = Counter(n.op_type for n in model_onnx.graph.node)
    sorted_ops = dict(sorted(op_hist.items(), key=lambda kv: -kv[1]))
    size_mib = output_path.stat().st_size / (1024 * 1024)
    print("Exported e2e FP32 ONNX to {}".format(output_path))
    print("  size : {:.3f} MiB".format(size_mib))
    print("  nodes: {}".format(len(model_onnx.graph.node)))
    print("  ops  : {}".format(sorted_ops))
    # STFT lives in the loss only — it must NOT be in the graph.
    for forbidden in ("STFT", "DFT"):
        assert op_hist.get(forbidden, 0) == 0, f"{forbidden} leaked into the e2e graph"
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export e2e butterfly NSNet2 to FP32 ONNX.")
    parser.add_argument("--checkpoint_file", required=True,
                        help="Path to a g_best checkpoint; sibling config.json is auto-loaded.")
    parser.add_argument("--output", default=None,
                        help="Output .onnx path. Defaults to <ckpt_dir>/g_best_e2e_fp32.onnx.")
    a = parser.parse_args()
    model, _ = _load_from_checkpoint(a.checkpoint_file)
    out = a.output or (Path(a.checkpoint_file).parent / "g_best_e2e_fp32.onnx")
    export_e2e_fp32(model, out)


if __name__ == "__main__":
    main()
