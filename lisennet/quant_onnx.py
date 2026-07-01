"""Int8 quantization of the exported LiSenNet ONNX graph.

Mirrors ``basenet/quant_onnx.py`` — two PTQ paths with very different outcomes:

  * **dynamic weight-only int8** — weights int8, activations stay fp32 (ranges
    computed per inference). Robust and near-lossless; the recommended PTQ path.
    LiSenNet's graph is small (~0.26 MiB fp32), so the absolute size win is modest
    but the weights shrink ~4x and the math stays accurate.
  * **static full int8 (QDQ)** — both weights and activations int8 against fixed
    calibrated scales. Provided as the starting point for a future QAT effort; as
    with BASENet and ConvFSENet here, static PTQ on a spectral model with wide-
    dynamic-range compressed magnitudes tends to need quantization-aware training
    to stay lossless. Measured quality of both paths is reported by the deploy
    eval, not asserted here.

The graph input is the 3-channel feature map ``feat`` produced outside the model
(compressed magnitude + group delay + instantaneous-frequency difference); the
static calibrator therefore feeds real VoiceBank-DEMAND ``feat`` tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
    quantize_dynamic, quantize_static,
)


def quantize_dynamic_int8(fp32_path, out_path) -> Path:
    """Weight-only dynamic int8 — the robust, near-lossless path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(fp32_path), str(out_path), weight_type=QuantType.QInt8)
    return out_path


def quantize_static_int8(fp32_path, out_path, calib_reader, per_channel=True) -> Path:
    """Static full int8 (QDQ, percentile calibration).

    Calibrate on short crops (the 2 s training segment) to bound the histogram
    memory — full-length utterances can OOM the calibrator.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(fp32_path), str(out_path), calib_reader,
        quant_format=QuantFormat.QDQ, per_channel=per_channel,
        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.Percentile,
    )
    return out_path


class VBDCalibrationReader(CalibrationDataReader):
    """Feeds 3-channel ``feat`` tensors from VoiceBank-DEMAND 2 s crops.

    Reuses the model's own STFT + feature extraction so the calibration inputs
    match exactly what the exported graph consumes at inference.
    """

    def __init__(self, h, n=16, split="train"):
        # Local imports: only static calibration needs the data + model stacks.
        import torch
        from common.dataset import Dataset, load_voicebank_demand
        from lisennet.model import build_lisennet

        model = build_lisennet(h).eval()
        hf = load_voicebank_demand()
        ds = Dataset(hf[split], h.segment_size, h.sampling_rate,
                     split=True, shuffle=True, seed=0)
        self.items = []
        with torch.no_grad():
            for i in range(n):
                _, noisy = ds[i]
                spec = model.power_compress(model.apply_stft(noisy.unsqueeze(0)))
                feat = model.build_features(spec.abs(), spec.angle())   # (1, 3, T, F)
                self.items.append({"feat": feat.numpy()})
        self._it = iter(self.items)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self.items)


class VBDStreamingCalibrationReader(CalibrationDataReader):
    """Calibration for the *streaming* graph — feeds ``feat`` + all ``state_i_in``.

    The streaming graph (``export_streaming_fp32``) takes one frame plus the N
    FIFO state tensors, so the calibrator must provide realistic values for the
    states too. This runs the conv model frame-by-frame over a few VoiceBank
    crops via ``LiSenNetStreamingONNX`` and records, at each frame, the full input
    feed ``{feat, state_0_in, ...}`` with the *actual* propagated state — the
    activation ranges the quantizer sees then match real streaming inference.
    """

    def __init__(self, h, n_utts=4, max_frames=300, split="train"):
        import torch
        from common.dataset import Dataset, load_voicebank_demand
        from lisennet.model import build_lisennet
        from lisennet.streaming import LiSenNetStreamingONNX

        model = build_lisennet(h).eval()
        view = LiSenNetStreamingONNX(model)
        names = view.state_input_names
        hf = load_voicebank_demand()
        ds = Dataset(hf[split], h.segment_size, h.sampling_rate,
                     split=True, shuffle=True, seed=0)
        self.items = []
        with torch.no_grad():
            for i in range(n_utts):
                if len(self.items) >= max_frames:
                    break
                _, noisy = ds[i]
                spec = model.power_compress(model.apply_stft(noisy.unsqueeze(0)))
                feat_full = model.build_features(spec.abs(), spec.angle())   # (1, 3, T, F)
                states = view.init_states(1)
                for t in range(feat_full.shape[2]):
                    feat_t = feat_full[:, :, t:t + 1, :]
                    sample = {"feat": feat_t.numpy()}
                    sample.update({nm: s.numpy() for nm, s in zip(names, states)})
                    self.items.append(sample)
                    states = list(view(feat_t, *states)[1:])                # propagate real state
                    if len(self.items) >= max_frames:
                        break
        self._it = iter(self.items)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self.items)


def main():
    parser = argparse.ArgumentParser(description="Int8-quantize an exported LiSenNet ONNX.")
    parser.add_argument("--fp32", required=True, help="fp32 .onnx from export_onnx.py")
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", choices=["dynamic", "static"], default="dynamic",
                        help="dynamic = robust weight-only (recommended); "
                             "static = full int8 (QAT-grade calibration needed)")
    parser.add_argument("--config", default=None,
                        help="config.json next to the checkpoint (static mode only).")
    parser.add_argument("--calib_utts", type=int, default=16)
    a = parser.parse_args()

    out = Path(a.output) if a.output else Path(a.fp32).with_suffix(f".int8_{a.mode}.onnx")
    if a.mode == "dynamic":
        quantize_dynamic_int8(a.fp32, out)
    else:
        from common.env import AttrDict
        with open(a.config) as f:
            h = AttrDict(json.load(f))
        # per_channel=False: ORT's per-channel int32-bias scale adjustment trips
        # on this graph's biases (see module docstring / RESULTS_LISENNET.md).
        quantize_static_int8(a.fp32, out, VBDCalibrationReader(h, a.calib_utts),
                             per_channel=False)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB, mode={a.mode})")


if __name__ == "__main__":
    main()
