#!/usr/bin/env python
"""LiSenNet (streaming) -> TFLite via ai-edge-torch, for the RT595 M33 TFLM path.

This mirrors ``lisennet/export_onnx.py::export_streaming_fp32`` but targets TFLite
instead of ONNX, so the same frame-by-frame graph
``feat (B,3,1,F) + N state_i_in -> est_mag (B,1,F) + N state_i_out``
becomes a ``.tflite`` the eIQ TensorFlow-Lite-Micro runtime can run on the RT595's
Cortex-M33 (CMSIS-NN int8 kernels). The recurrence (hybrid TimeGRU) is already a
1x1-conv single-step cell in the streaming view, so there are no TFLite RNN ops.

Run in the dedicated env (isolated from the torch research .venv):
    ~/.venvs/rt595-export/bin/python deploy/rt595/host/export_tflite.py \
        --checkpoint_file cp_lisennet_hybrid_nc24/g_best --output host_out/lisennet_hybrid_nc24_streaming.tflite

Bring-up note: only a *dummy* hybrid-nc24 checkpoint exists today
(``cp_lisennet_hybrid_nc24_dummy/g_best``). FP32 conversion on it validates that
ai-edge-torch handles the 11-state streaming module + every op; int8/PESQ numbers
need the trained checkpoint (and VoiceBank-DEMAND for calibration) dropped in.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]   # host -> rt595 -> deploy -> repo


def build_streaming_view(checkpoint_file):
    """Load the trained model + wrap it as the frame-by-frame streaming module.

    Inlines ``lisennet.export_onnx._load_from_checkpoint`` so we don't import
    export_onnx (which pulls ``onnx``, absent in this TFLite-only venv).
    """
    import json
    import torch
    from common.env import AttrDict
    from lisennet.model import build_lisennet
    from lisennet.streaming import LiSenNetStreamingONNX

    checkpoint_file = Path(checkpoint_file)
    with open(checkpoint_file.parent / "config.json") as f:
        h = AttrDict(json.load(f))
    model = build_lisennet(h)
    ckpt = torch.load(str(checkpoint_file), weights_only=True, map_location="cpu")
    model.load_state_dict(ckpt["generator"], strict=True)
    model.eval()
    view = LiSenNetStreamingONNX(model).eval()
    return model, view


def sample_args(model, view):
    """The example inputs ai-edge-torch traces: (feat, *zero_states)."""
    import torch
    feat = torch.zeros(1, 3, 1, model.n_freqs, dtype=torch.float32)
    states = view.init_states(1)
    return (feat, *states)


def representative_dataset(view, model, checkpoint_file, n_utts, max_frames):
    """Reuse lisennet's streaming calibrator: real VBD feats + propagated state.

    Yields positional tuples (feat, state_0_in, ...) matching the module's forward
    signature, so PT2E calibration sees on-device-realistic activation ranges.
    Falls back to zero-state frames if VoiceBank-DEMAND isn't available (flow-test
    only — int8 quality would be meaningless without real calibration data).
    """
    import torch
    import json
    from common.env import AttrDict

    with open(Path(checkpoint_file).parent / "config.json") as f:
        h = AttrDict(json.load(f))
    names = view.state_input_names
    try:
        from lisennet.quant_onnx import VBDStreamingCalibrationReader
        reader = VBDStreamingCalibrationReader(
            h, checkpoint=str(checkpoint_file), n_utts=n_utts, max_frames=max_frames
        )
        for item in reader.items:
            feat = torch.from_numpy(item["feat"])
            states = [torch.from_numpy(item[nm]) for nm in names]
            yield (feat, *states)
    except Exception as e:  # dataset missing / reader change — degrade loudly
        print(f"  [warn] VBD calibration unavailable ({type(e).__name__}: {e}); "
              f"using zero-state frames — int8 is a FLOW TEST ONLY.")
        states = view.init_states(1)
        for _ in range(32):
            yield (torch.zeros(1, 3, 1, model.n_freqs), *states)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_file", required=True,
                    help="e.g. cp_lisennet_hybrid_nc24/g_best (sibling config.json auto-loaded)")
    ap.add_argument("--output", default=None, help="output .tflite path")
    ap.add_argument("--int8", action="store_true",
                    help="PT2E int8 PTQ with VBD-calibrated representative data (needs trained ckpt)")
    ap.add_argument("--calib_utts", type=int, default=4)
    ap.add_argument("--calib_frames", type=int, default=300)
    ap.add_argument("--repo", default=str(REPO_ROOT), help="repo root to import lisennet from")
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    import torch
    import litert_torch

    model, view = build_streaming_view(a.checkpoint_file)
    args = sample_args(model, view)

    # Sanity: the streaming module must run in eager mode before we trace it.
    with torch.no_grad():
        out = view(*args)
    print(f"streaming module OK: n_states={view.n_states}, "
          f"est_mag={tuple(out[0].shape)}, n_freqs={model.n_freqs}")

    out_path = Path(a.output) if a.output else (
        Path(a.checkpoint_file).parent /
        ("g_best_streaming_int8.tflite" if a.int8 else "g_best_streaming_fp32.tflite"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Always convert FP32 first (this is the graph-conversion gate). int8 is then a
    # TFLite-side PTQ pass with ai-edge-quantizer — avoids torch-side PT2E, which is
    # incompatible with the torch version litert_torch pins (needs >=2.11, has 2.9.1).
    fp32_path = out_path if not a.int8 else out_path.with_suffix(".fp32.tflite")
    litert_torch.convert(view, args).export(str(fp32_path))
    print(f"FP32 tflite: {fp32_path}  ({fp32_path.stat().st_size/1024:.1f} KiB)")

    if a.int8:
        import numpy as np
        from ai_edge_quantizer import quantizer, recipe

        # Calibration samples keyed by signature input name: args_0=feat, args_1..N=states
        # (positional order preserved by ai-edge-torch; verified via get_signature_list).
        in_names = ["args_0"] + [f"args_{i}" for i in range(1, view.n_states + 1)]
        samples = []
        for rep in representative_dataset(view, model, a.checkpoint_file,
                                          a.calib_utts, a.calib_frames):
            samples.append({nm: t.detach().numpy().astype(np.float32)
                            for nm, t in zip(in_names, rep)})
        print(f"  calibrating on {len(samples)} frames")

        q = quantizer.Quantizer(str(fp32_path), recipe.static_wi8_ai8())
        calib = q.calibrate({"serving_default": samples}) if q.need_calibration else None
        result = q.quantize(calib)
        result.export_model(str(out_path))
        print(f"Wrote {out_path}  ({out_path.stat().st_size/1024:.1f} KiB, int8 w8/a8)")
    else:
        print(f"Wrote {out_path}  ({out_path.stat().st_size/1024:.1f} KiB, fp32)")


if __name__ == "__main__":
    main()
