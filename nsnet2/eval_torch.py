"""eval_torch.py — PyTorch-eager streaming PESQ baseline.

Drives NSNet2Streaming.forward_step in a Python time-loop (eager mode, no
patches, no ONNX), computes PESQ on the VBD test split, and prints the mean.

Use this to A/B against FP32 ONNX PESQ (from inference_onnx.py): if the two
agree within run-to-run noise, the structure-preserving FP32 export is
faithful; if they diverge, the export itself is the source of degradation
before quant ever runs (avenue 2 in the butterfly investigation).

Mirrors inference_onnx.py's RMS norm + STFT + iSTFT + eval_pesq pipeline
verbatim so the only varying axis is the inference engine.
"""
import argparse
import json
import os

import numpy as np
import torch
from rich.progress import track

from common.dataset import load_voicebank_demand, mag_pha_stft, mag_pha_istft
from common.env import AttrDict
from common.metrics import eval_pesq
from nsnet2.streaming import NSNet2Streaming


@torch.no_grad()
def enhance_streaming_eager(noisy_wav, streaming, h):
    """RMS-norm, STFT, frame-by-frame streaming forward, mask*mag, iSTFT, unscale.

    Decorated with no_grad so it is self-contained: structured/triton
    checkpoints hold parameters that require grad, and the per-step forward
    would otherwise raise outside an enclosing no_grad context.
    """
    noisy_t = torch.from_numpy(np.asarray(noisy_wav, dtype=np.float32))
    norm_factor = torch.sqrt(len(noisy_t) / (torch.sum(noisy_t ** 2.0) + 1e-8))
    noisy_t = (noisy_t * norm_factor).unsqueeze(0)                       # (1, L)

    noisy_mag, noisy_pha, _ = mag_pha_stft(
        noisy_t, h.n_fft, h.hop_size, h.win_size, h.compress_factor,
    )                                                                     # (1, F, T)

    B, F, T = noisy_mag.shape
    states = torch.zeros(streaming.num_layers, B, streaming.hidden_size,
                         dtype=torch.float32)
    masks = []
    for t in range(T):
        frame_in = noisy_mag[:, :, t]                                     # (1, F)
        mask, states = streaming.forward_step(frame_in, states)
        masks.append(mask)
    mask_full = torch.stack(masks, dim=2)                                 # (1, F, T)

    enhanced_mag = mask_full * noisy_mag
    audio = mag_pha_istft(
        enhanced_mag, noisy_pha, h.n_fft, h.hop_size, h.win_size, h.compress_factor,
    )
    return (audio / norm_factor).squeeze().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_file", required=True,
                        help="Path to PyTorch g_best checkpoint (cp_<run>/g_best)")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--max_utterances", type=int, default=None)
    a = parser.parse_args()

    cp_dir = os.path.split(a.checkpoint_file)[0]
    with open(os.path.join(cp_dir, "config.json")) as f:
        h = AttrDict(json.loads(f.read()))
    torch.manual_seed(h.seed)

    streaming = NSNet2Streaming.from_checkpoint(a.checkpoint_file).eval()
    # If the checkpoint was trained with a triton_* GRU kind, flip Triton off so
    # the per-step Python path runs (matches eval_quant / sweep_hf_ptq / export).
    for sub in streaming.modules():
        if isinstance(getattr(sub, "use_triton", None), bool):
            sub.use_triton = False

    hf = load_voicebank_demand(cache_dir=a.hf_cache_dir)
    test_split = hf["test"]
    if a.max_utterances is not None:
        test_split = test_split.select(range(min(a.max_utterances, len(test_split))))

    pesq_scores = []
    with torch.no_grad():
        for item in track(test_split):
            noisy_wav = np.asarray(item["noisy"]["array"], dtype=np.float32)
            clean_wav = np.asarray(item["clean"]["array"], dtype=np.float32)
            enhanced = enhance_streaming_eager(noisy_wav, streaming, h)
            pesq_scores.append(eval_pesq(clean_wav, enhanced, h.sampling_rate))

    clean = [s for s in pesq_scores if s != -1]
    n_fail = len(pesq_scores) - len(clean)
    mean = float(np.mean(clean)) if clean else float("nan")
    print(f"Mean: PESQ Torch-streaming={mean:.3f}; failures={n_fail} / {len(pesq_scores)}")


if __name__ == "__main__":
    main()
