"""Int8 quantization of the exported BASENet ONNX graph.

Two PTQ paths, with very different outcomes on this model (measured on the
trained g_best over a VoiceBank-DEMAND test subset, wideband PESQ):

  * **dynamic weight-only int8** -> 3.24 -> 3.19  (-0.05 PESQ, ~34% smaller).
    Weights are int8; activations stay fp32 (ranges computed per inference).
    Robust and near-lossless — the recommended PTQ path here.
  * **static full int8 (QDQ)** -> 3.24 -> ~1.2  (collapse), with MinMax,
    percentile, or entropy calibration alike. The activations of this
    architecture (wide-dynamic-range compressed spectra + the atan2 phase path)
    are too sensitive to fixed int8 scales. Getting full int8 near-lossless
    needs quantization-aware training — the same conclusion the repo's
    ConvFSENet int8 work reached. The static helper is provided as the starting
    point for a future QAT effort, NOT as a deployable artifact as-is.

The eventual STM32 int8 conversion is done by stedgeai on the deploy box; these
helpers characterise quantizability and produce a quick int8 ONNX to sanity it.
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
    """Weight-only dynamic int8 — the robust, near-lossless path for BASENet."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(fp32_path), str(out_path), weight_type=QuantType.QInt8)
    return out_path


def quantize_static_int8(fp32_path, out_path, calib_reader, per_channel=True) -> Path:
    """Static full int8 (QDQ, percentile calibration).

    WARNING: collapses BASENet's PESQ without QAT (see module docstring).
    Calibrate on short crops (e.g. the 2 s training segment) to bound the
    histogram memory — full-length utterances can OOM the calibrator.
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
    """Feeds (mag, pha) from VoiceBank-DEMAND 2 s crops for static calibration."""

    def __init__(self, h, n=16, split="train"):
        # Local imports: only static calibration needs the data stack.
        from common.dataset import Dataset, load_voicebank_demand, mag_pha_stft
        hf = load_voicebank_demand()
        ds = Dataset(hf[split], h.segment_size, h.sampling_rate,
                     split=True, shuffle=True, seed=0)
        cf = float(h.compress_factor)
        self.items = []
        for i in range(n):
            _, noisy = ds[i]
            mag, pha, _ = mag_pha_stft(noisy.unsqueeze(0), h.n_fft, h.hop_size,
                                       h.win_size, cf)
            self.items.append({"mag": mag.numpy(), "pha": pha.numpy()})
        self._it = iter(self.items)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self.items)


def main():
    parser = argparse.ArgumentParser(description="Int8-quantize an exported BASENet ONNX.")
    parser.add_argument("--fp32", required=True, help="fp32 .onnx from export_onnx.py")
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", choices=["dynamic", "static"], default="dynamic",
                        help="dynamic = robust weight-only (recommended); "
                             "static = full int8 (needs QAT, collapses as PTQ)")
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
        quantize_static_int8(a.fp32, out, VBDCalibrationReader(h, a.calib_utts))
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB, mode={a.mode})")


if __name__ == "__main__":
    main()
