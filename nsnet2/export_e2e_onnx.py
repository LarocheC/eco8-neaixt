"""Export the end-to-end butterfly NSNet2 to a single waveform->waveform ONNX.

Unlike every other model in this repo, the STFT is *inside* the graph — because
it is a learned FFT-initialised butterfly, not ``torch.stft``. The exported graph
is fully self-contained: noisy waveform in, enhanced waveform out.

The transform uses a *complex* butterfly (FFT/iFFT). ONNX has no complex dtype,
so ``_ComplexButterflyReal`` re-expresses each complex butterfly as real
arithmetic on separate (real, imag) tensors — numerically identical to the eager
``torch_structured`` complex butterfly (verified < 1e-6) and structure-preserving
(the twiddle factors survive as constants; the data path is Mul/Sub/Add/ReduceSum
per stage). Magnitude masking is a real gain on the complex coefficients (phase
preserved), so no atan2/cos/sin is needed. Batch is fixed to 1; time is dynamic.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.env import AttrDict
from common.utils import load_checkpoint
from nsnet2.model_e2e import NSNet2E2E


def _complex_butterfly_real(tw_r, tw_i, xr, xi, increasing_stride, out_size):
    """Real-arithmetic equivalent of a torch_structured complex Butterfly.forward.

    Mirrors ``butterfly_multiply_torch`` stage loop with complex mult expanded:
    ``(tr + i·ti)(pr + i·pi)``. tw_*: (nstacks, nblocks, log_n, n//2, 2, 2) (the
    real/imag parts of the twiddle, constant at export). xr, xi: (batch, in_size).
    Returns (out_r, out_i) of shape (batch, out_size); assumes nstacks == 1.
    """
    nstacks, nblocks, log_n = tw_r.shape[0], tw_r.shape[1], tw_r.shape[2]
    n = 1 << log_n
    insz = xr.shape[-1]
    if insz < n:
        xr = F.pad(xr, (0, n - insz)); xi = F.pad(xi, (0, n - insz))
    elif insz > n:
        xr = xr[:, :n]; xi = xi[:, :n]
    orr = xr.reshape(-1, nstacks, n)
    ori = xi.reshape(-1, nstacks, n)
    cur = increasing_stride
    for block in range(nblocks):
        for idx in range(log_n):
            log_stride = idx if cur else log_n - 1 - idx
            s = 1 << log_stride
            tr = tw_r[:, block, idx].view(nstacks, n // (2 * s), s, 2, 2).permute(0, 1, 3, 4, 2)
            ti = tw_i[:, block, idx].view(nstacks, n // (2 * s), s, 2, 2).permute(0, 1, 3, 4, 2)
            pr = orr.reshape(-1, nstacks, n // (2 * s), 1, 2, s)
            pi = ori.reshape(-1, nstacks, n // (2 * s), 1, 2, s)
            orr = (tr * pr - ti * pi).sum(dim=4).reshape(-1, nstacks, n)
            ori = (tr * pi + ti * pr).sum(dim=4).reshape(-1, nstacks, n)
        cur = not cur
    return orr[:, 0, :out_size], ori[:, 0, :out_size]


class _E2EExportReal(nn.Module):
    """Waveform->waveform, pure real ops. Feed a frame-aligned length
    ``L = (T-1)*hop + win`` (output length == input length)."""

    def __init__(self, model: NSNet2E2E):
        super().__init__()
        self.m = model
        t = model.transform
        self.win, self.hop = t.win, t.hop
        self.compress = t.compress_factor
        # constants (twiddle real/imag, windows, framing kernels)
        self.register_buffer("ana_tw_r", t.analysis.twiddle.detach().real.contiguous())
        self.register_buffer("ana_tw_i", t.analysis.twiddle.detach().imag.contiguous())
        self.register_buffer("syn_tw_r", t.synthesis.twiddle.detach().real.contiguous())
        self.register_buffer("syn_tw_i", t.synthesis.twiddle.detach().imag.contiguous())

    def forward(self, wav):
        t = self.m.transform
        frames = F.conv1d(wav.unsqueeze(1), t.frame_kernel, stride=self.hop)
        frames = frames.transpose(1, 2) * t.ana_window          # (B, T, win)
        B = frames.shape[0]
        f2d = frames.reshape(-1, self.win)
        Xr, Xi = _complex_butterfly_real(self.ana_tw_r, self.ana_tw_i,
                                         f2d, torch.zeros_like(f2d), True, self.win)
        Xr = Xr.reshape(B, -1, self.win); Xi = Xi.reshape(B, -1, self.win)
        mag = (Xr.pow(2) + Xi.pow(2) + 1e-9).pow(0.5 * self.compress)   # compressed magnitude
        mask = self.m.core.predict_mask(mag)                    # (B, T, win) in [0,1]
        gain = mask.pow(1.0 / self.compress)
        Xr = (Xr * gain).reshape(-1, self.win); Xi = (Xi * gain).reshape(-1, self.win)
        fr, _ = _complex_butterfly_real(self.syn_tw_r, self.syn_tw_i, Xr, Xi, False, self.win)
        fr = fr.reshape(B, -1, self.win) * t.syn_window
        wav_hat = F.conv_transpose1d(fr.transpose(1, 2), t.oa_kernel, stride=self.hop)
        return wav_hat.squeeze(1)


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
    wrapper = _E2EExportReal(model).eval()
    torch.onnx.export(
        wrapper, (wav,), str(output_path),
        input_names=["noisy_wav"], output_names=["enhanced_wav"],
        dynamic_axes={"noisy_wav": {1: "L"}, "enhanced_wav": {1: "L"}},
        opset_version=opset, dynamo=False,
    )
    model_onnx = onnx.load(str(output_path))
    onnx.checker.check_model(model_onnx)
    op_hist = Counter(n.op_type for n in model_onnx.graph.node)
    print("Exported e2e FP32 ONNX to {}".format(output_path))
    print("  size : {:.3f} MiB".format(output_path.stat().st_size / (1024 * 1024)))
    print("  nodes: {}".format(len(model_onnx.graph.node)))
    print("  ops  : {}".format(dict(sorted(op_hist.items(), key=lambda kv: -kv[1]))))
    for forbidden in ("STFT", "DFT"):
        assert op_hist.get(forbidden, 0) == 0, f"{forbidden} leaked into the e2e graph"
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export e2e butterfly NSNet2 to FP32 ONNX.")
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--output", default=None)
    a = parser.parse_args()
    model, _ = _load_from_checkpoint(a.checkpoint_file)
    out = a.output or (Path(a.checkpoint_file).parent / "g_best_e2e_fp32.onnx")
    export_e2e_fp32(model, out)


if __name__ == "__main__":
    main()
