"""Int8 quantization of the exported FSPEN ONNX graph.

Mirrors ``lisennet/quant_onnx.py`` — two PTQ paths with different outcomes:

  * **dynamic weight-only int8** — weights int8, activations stay fp32 (ranges
    computed per inference). Robust and near-lossless; the recommended PTQ
    path. FSPEN's graph is tiny (~35k params), so the absolute size win is
    modest but the math stays accurate.
  * **static full int8 (QDQ)** — both weights and activations int8 against
    fixed calibrated scales. Provided as the starting point for a future QAT
    effort; as with the other spectral models here, static PTQ on
    wide-dynamic-range spectra tends to need quantization-aware training to
    stay lossless. Measured quality of both paths is reported by the deploy
    eval, not asserted here.

The graph input is the raw stacked-re/im noisy spectrum ``spec`` produced
outside the model, so the static calibrator feeds real VoiceBank-DEMAND
spectra built with the model's own STFT.
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


def quantize_static_int8(fp32_path, out_path, calib_reader, per_channel=False, signed=False) -> Path:
    """Static full int8 (QDQ, percentile calibration).

    Calibrate on short crops (the 2 s training segment) to bound the histogram
    memory — full-length utterances can OOM the calibrator.

    ``signed=True`` uses **QInt8** activations with ``ActivationSymmetric=False`` /
    ``WeightSymmetric=True`` — the stedgeai / Neural-ART deploy recipe used across
    this repo. ``signed=False`` keeps the QUInt8 activations of the host int8 study.
    ``per_channel`` defaults to False like LiSenNet's whole-utterance path (ORT's
    per-channel int32-bias scale adjustment trips on these small biased graphs).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    act_type = QuantType.QInt8 if signed else QuantType.QUInt8
    extra = {"ActivationSymmetric": False, "WeightSymmetric": True} if signed else {}
    quantize_static(
        str(fp32_path), str(out_path), calib_reader,
        quant_format=QuantFormat.QDQ, per_channel=per_channel,
        weight_type=QuantType.QInt8, activation_type=act_type,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options=extra,
    )
    return out_path


class VBDCalibrationReader(CalibrationDataReader):
    """Feeds stacked-re/im ``spec`` tensors from VoiceBank-DEMAND 2 s crops.

    Reuses the model's own STFT so the calibration inputs match exactly what
    the exported graph consumes at inference.
    """

    def __init__(self, h, n=16, split="train"):
        # Local imports: only static calibration needs the data + model stacks.
        import torch
        from common.dataset import Dataset, load_voicebank_demand
        from fspen.model import build_fspen

        model = build_fspen(h).eval()
        hf = load_voicebank_demand()
        ds = Dataset(hf[split], h.segment_size, h.sampling_rate,
                     split=True, shuffle=True, seed=0)
        self.items = []
        with torch.no_grad():
            for i in range(n):
                _, noisy = ds[i]
                spec, _ = model.spec_features(model.apply_stft(noisy.unsqueeze(0)))
                self.items.append({"spec": spec.numpy()})    # (1, T, 2, F)
        self._it = iter(self.items)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self.items)


class VBDStreamingCalibrationReader(CalibrationDataReader):
    """Calibration for the *streaming* graph — feeds ``spec`` + all ``state_i_in``.

    The streaming graph (``export_streaming_fp32``) takes one frame plus the
    per-DPE-block hidden-state tensors, so the calibrator must provide
    realistic values for the states too. This runs the model frame-by-frame
    over a few VoiceBank crops via ``FSPENStreamingONNX`` and records, at each
    frame, the full input feed ``{spec, state_0_in, ...}`` with the *actual*
    propagated state — the activation ranges the quantizer sees then match
    real streaming inference.
    """

    def __init__(self, h, checkpoint=None, n_utts=4, max_frames=300, split="train"):
        import torch
        from common.dataset import Dataset, load_voicebank_demand
        from fspen.model import build_fspen
        from fspen.streaming import FSPENStreamingONNX

        model = build_fspen(h).eval()
        if checkpoint is not None:
            # The reader propagates hidden state through the model, so the
            # trained weights are required for realistic state ranges.
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["generator"], strict=True)
        view = FSPENStreamingONNX(model)
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
                spec_full, _ = model.spec_features(model.apply_stft(noisy.unsqueeze(0)))
                states = view.init_states(1)
                for t in range(spec_full.shape[1]):
                    spec_t = spec_full[:, t:t + 1]
                    sample = {"spec": spec_t.numpy()}
                    sample.update({nm: s.numpy() for nm, s in zip(names, states)})
                    self.items.append(sample)
                    states = list(view(spec_t, *states)[1:])   # propagate real state
                    if len(self.items) >= max_frames:
                        break
        self._it = iter(self.items)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self.items)


def main():
    parser = argparse.ArgumentParser(description="Int8-quantize an exported FSPEN ONNX.")
    parser.add_argument("--fp32", required=True, help="fp32 .onnx from export_onnx.py")
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", choices=["dynamic", "static"], default="dynamic",
                        help="dynamic = robust weight-only (recommended); "
                             "static = full int8 (QAT-grade calibration needed)")
    parser.add_argument("--config", default=None,
                        help="config.json next to the checkpoint (static mode only).")
    parser.add_argument("--calib_utts", type=int, default=16)
    parser.add_argument("--streaming", action="store_true",
                        help="Calibrate the frame-by-frame streaming graph (spec + state_i_in "
                             "inputs) with real propagated state. Implies signed int8.")
    parser.add_argument("--checkpoint", default=None,
                        help="Trained g_best — streaming calibration propagates state with it.")
    parser.add_argument("--signed", action="store_true",
                        help="QInt8 (signed) activations — the stedgeai/Neural-ART deploy "
                             "recipe. Default is unsigned QUInt8 (the host int8 study).")
    a = parser.parse_args()

    out = Path(a.output) if a.output else Path(a.fp32).with_suffix(f".int8_{a.mode}.onnx")
    if a.mode == "dynamic":
        quantize_dynamic_int8(a.fp32, out)
    else:
        from common.env import AttrDict
        with open(a.config) as f:
            h = AttrDict(json.load(f))
        if a.streaming:
            # Frame-by-frame streaming graph -> signed int8, per-channel (the
            # deploy recipe shared with lisennet); calibration threads the real
            # hidden state so state_i ranges are realistic.
            reader = VBDStreamingCalibrationReader(h, a.checkpoint, n_utts=a.calib_utts)
            quantize_static_int8(a.fp32, out, reader, per_channel=True, signed=True)
        else:
            quantize_static_int8(a.fp32, out, VBDCalibrationReader(h, a.calib_utts),
                                 per_channel=False, signed=a.signed)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB, mode={a.mode})")


if __name__ == "__main__":
    main()
