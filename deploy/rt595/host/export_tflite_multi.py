#!/usr/bin/env python
"""Generalized streaming -> TFLite (fp32 + int8) exporter for the RT595 HiFi4 bench.

Companion to ``export_tflite.py`` (LiSenNet-only). This one dispatches on ``--model``
and reuses each model's *existing* checkpoint loader + streaming-ONNX view, so the
frame-by-frame graph that already exports to ONNX instead becomes a ``.tflite`` for the
TFLite-Micro / HiFi4 path.

Supported:
  convfsenet : ConvFSENetStreamingONNX  — (noisy_mag[B,F], *states) -> (mask, *states)
  nsnet2     : NSNet2Streaming          — (frame_in[B,F], states_in[L,B,H]) -> (mask, states_out)

Run in the export venv:
  ~/.venvs/rt595-export/bin/python deploy/rt595/host/export_tflite_multi.py \
      --model convfsenet --checkpoint_file cp_convfsenet_win/g_best \
      --output deploy/rt595/host_out/convfsenet_win_streaming --int8

NOTE ON CALIBRATION: for the cycle/latency measurement the int8 graph only needs valid
per-tensor scales with realistic *shapes* — the actual activation ranges don't change
op dispatch or cycle counts. So this exporter calibrates on random/zero frames by
default (a "flow-test" int8). That is fine for benchmarking; it is NOT a quality model.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]   # host -> rt595 -> deploy -> repo


def build_convfsenet(checkpoint_file):
    """(view, example_args, in_names) for ConvFSENet streaming."""
    import torch
    from convfsenet.export_onnx import _load_offline_from_checkpoint
    from convfsenet.streaming import ConvFSENetStreamingFast, ConvFSENetStreamingONNX

    base = _load_offline_from_checkpoint(Path(checkpoint_file))
    fast = ConvFSENetStreamingFast(base).eval()
    view = ConvFSENetStreamingONNX(fast).eval()
    F = view.n_features
    states = view.init_states(1, "cpu", dtype=torch.float32)
    args = (torch.zeros(1, F, dtype=torch.float32), *states)
    in_names = [f"args_{i}" for i in range(len(args))]
    import contextlib
    return view, args, in_names, contextlib.nullcontext


def build_nsnet2(checkpoint_file):
    """(view, example_args, in_names, patch_ctx) for NSNet2 streaming.

    Structured GRUs (blockdiag/butterfly) trace only inside
    ``_patch_structured_for_export`` — the same context manager the ONNX exporter
    uses to swap BlockdiagLinear/Butterfly for standard primitives. Returned as a
    zero-arg factory so main() can hold it open across the litert_torch trace.
    """
    import torch
    from nsnet2.streaming import NSNet2Streaming
    from nsnet2.export_onnx import _patch_structured_for_export

    view = NSNet2Streaming.from_checkpoint(str(checkpoint_file)).eval()
    n_freq = view.base.h.n_fft // 2 + 1
    frame_in = torch.zeros(1, n_freq, dtype=torch.float32)
    states_in = torch.zeros(view.num_layers, 1, view.hidden_size, dtype=torch.float32)
    args = (frame_in, states_in)
    in_names = ["args_0", "args_1"]
    return view, args, in_names, (lambda: _patch_structured_for_export(view))


def build_convfsenet_deploy(checkpoint_file):
    """ConvFSENet streaming view, TFLM-micro-deployable.

    Two changes vs build_convfsenet, both required to keep the int8 graph inside
    TFLM's op set: (1) extractor_type='mag' drops the FP32 Pow (no AddPow in TFLM);
    (2) the per-block FIFO tap is a *strided slice* instead of index_select, which
    litert_torch lowers to SLICE/GATHER_ND (both supported) rather than the
    index_select path that also emits REDUCE_ALL+SELECT_V2 (REDUCE_ALL has no TFLM
    kernel). Numerically identical to the gather (verified max|d|=0).
    """
    import torch
    import torch.nn.functional as F
    from convfsenet.export_onnx import _load_offline_from_checkpoint
    from convfsenet.streaming import (ConvFSENetStreamingFast, ConvFSENetStreamingONNX,
                                      _StreamingTCMBlockQF_Fast)
    import contextlib

    def forward_step_sliced(self, x_t, buffer):
        y = torch.relu(self.conv1x1_folded(x_t))
        if self.buffer_size > 0:
            step, n = self._tap_step, self._tap_n           # plain ints (set pre-trace)
            past = buffer[..., 0:(n - 1) * step + 1:step]
            window = torch.cat([past, y], dim=-1)
        else:
            window = y
        dconv_out = F.conv1d(window, self.dconv_folded.weight, bias=self.dconv_folded.bias,
                             stride=1, padding=0, dilation=1, groups=self.groups)
        z = self.conv1x1_out(torch.relu(dconv_out))
        out = z + x_t
        new_buffer = torch.cat([buffer[..., 1:], y], dim=-1) if self.buffer_size > 0 else buffer
        return out, new_buffer

    base = _load_offline_from_checkpoint(Path(checkpoint_file))
    base.extractor_type = "mag"
    fast = ConvFSENetStreamingFast(base).eval()
    for blk in fast.tcm_blocks:
        ti = blk.tap_indices
        blk._tap_step = int(ti[1] - ti[0]) if ti.numel() > 1 else 1
        blk._tap_n = int(ti.numel())
    _StreamingTCMBlockQF_Fast.forward_step = forward_step_sliced   # process-global, intentional
    view = ConvFSENetStreamingONNX(fast).eval()
    F_ = view.n_features
    states = view.init_states(1, "cpu", dtype=torch.float32)
    args = (torch.zeros(1, F_, dtype=torch.float32), *states)
    in_names = [f"args_{i}" for i in range(len(args))]
    return view, args, in_names, contextlib.nullcontext


BUILDERS = {"convfsenet": build_convfsenet,
            "convfsenet_sliced": build_convfsenet_deploy,
            "nsnet2": build_nsnet2}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--checkpoint_file", required=True)
    ap.add_argument("--output", required=True, help="output stem (no extension)")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib_frames", type=int, default=64)
    ap.add_argument("--repo", default=str(REPO_ROOT))
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    import torch
    import numpy as np
    import litert_torch

    view, args, in_names, patch_ctx = BUILDERS[a.model](a.checkpoint_file)

    out_stem = Path(a.output)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = out_stem.with_suffix(".fp32.tflite")
    int8_path = out_stem.with_suffix(".tflite")

    # Structured views must be traced inside their export patch (no-op for convfsenet).
    with patch_ctx():
        with torch.no_grad():
            out = view(*args)
        n_out = len(out) if isinstance(out, (tuple, list)) else 1
        print(f"[{a.model}] streaming view OK: {len(args)} inputs -> {n_out} outputs; "
              f"input0={tuple(args[0].shape)}")
        # FP32 conversion is the graph-conversion gate.
        litert_torch.convert(view, args).export(str(fp32_path))
    print(f"FP32 tflite: {fp32_path}  ({fp32_path.stat().st_size/1024:.1f} KiB)")

    if not a.int8:
        return

    from ai_edge_quantizer import quantizer, recipe

    # Random-frame representative data — valid scales, realistic shapes (see module docstring).
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(a.calib_frames):
        smp = {}
        for nm, t in zip(in_names, args):
            arr = t.detach().numpy().astype(np.float32)
            if nm == in_names[0]:                       # jitter the feature input only
                arr = (rng.standard_normal(arr.shape) * 0.5).astype(np.float32)
            smp[nm] = arr
        samples.append(smp)
    print(f"  calibrating on {len(samples)} random frames (flow-test int8)")

    q = quantizer.Quantizer(str(fp32_path), recipe.static_wi8_ai8())
    calib = q.calibrate({"serving_default": samples}) if q.need_calibration else None
    result = q.quantize(calib)
    result.export_model(str(int8_path))
    print(f"Wrote {int8_path}  ({int8_path.stat().st_size/1024:.1f} KiB, int8 w8/a8)")


if __name__ == "__main__":
    main()
