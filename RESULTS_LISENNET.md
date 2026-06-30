# LiSenNet — results

A lightweight, causal speech enhancer. LiSenNet is a ~37 K-parameter
sub-band U-Net with a dual-path-recurrent (DPR) bottleneck that predicts a
**magnitude-only** mask and recovers phase with a 2-iteration Griffin-Lim
seeded from the noisy phase (no phase decoder — the key cost saving). The
architecture is a faithful port of

> H. Yan, J. Zhou, K. Chen, J. Lu, *LiSenNet: Lightweight Sub-band and
> Dual-Path Modeling for Real-Time Speech Enhancement*,
> [arXiv:2409.13285](https://arxiv.org/abs/2409.13285),

from the authors' MIT-licensed reference implementation,
[hyyan2k/LiSenNet](https://github.com/hyyan2k/LiSenNet). The port keeps the
architecture unchanged; it only drops two unused upstream bits (the
`mel_scale` helper, whose sole dependency `torchaudio` is not installed, and
the standalone `NoiseDetector`) and reuses this repo's CMGAN
`MetricDiscriminator` (identical to LiSenNet's) rather than copying it.

It runs **frame-by-frame**: the mask sub-network is causal in time, so each
causal time-conv keeps a `(kt-1)`-frame ring buffer and the DPR inter-time GRU
carries its hidden state. The streaming reference, the FP32 ONNX export, and
the dynamic/static int8 quantization are all parity-checked against the offline
model (`tests/test_lisennet_*`).

See [RESULTS_CONVFSENET.md](RESULTS_CONVFSENET.md) and
[RESULTS_NSNET2.md](RESULTS_NSNET2.md) for the other model families, and the
[README](README.md) for setup and repo layout.

## Headline results

100-epoch CMGAN training on VoiceBank-DEMAND-16k (complex + magnitude +
PESQ-metric-GAN losses, `n_fft=512`/`hop=256`, AdamW). PESQ is wideband, on the
full 824-utterance VBD test split. The model is trained with the offline
Griffin-Lim phase; for real-time deployment the non-causal Griffin-Lim is
dropped in favour of the noisy phase (its own seed).

| metric                              |     value |
| ----------------------------------- | --------: |
| params                              | 36,783    |
| FP32 PESQ (torch, Griffin-Lim)      | **3.006** |
| FP32 PESQ (ONNX, Griffin-Lim)       | **3.006** |
| dynamic-int8 PESQ (Griffin-Lim)     |     2.995 |
| **real-time PESQ (int8 + noisy phase)** | **2.982** |
| int8 RTF (1 thread CPU)             |     0.15  |
| FP32 ONNX size                      | 0.27 MiB  |

The reproduction lands at **PESQ 3.006**, within ~0.06 of the paper's reported
~3.07 — a faithful from-scratch reproduction in this framework's pipeline.

## What is exported, and what stays outside the graph

Only the **mask sub-network** is exported / streamed — the pure tensor-op core
that maps the 3-channel feature map
`feat = [compressed_mag, group_delay/π, instantaneous_freq_diff/π]` to the
enhanced magnitude `est_mag`. As with the other models here, the STFT, the
feature extraction, and the phase recovery live *outside* the ONNX graph
(`lisennet/export_onnx.py`, dynamic batch/time axes; the DPR GRUs survive
export; ORT matches PyTorch to ~1e-6, including at an untraced time length).

LiSenNet is causal in time **by construction**, which the deploy work made
explicit rather than having to re-engineer:

* every time convolution left-pads the time axis `(kt-1, 0)`, so frame *n*
  never reads a future frame;
* the DPR **inter-time** GRU is already unidirectional;
* the DPR **intra** GRU is bidirectional only over *frequency*, which is fully
  available at each frame, so it streams fine.

The single non-causal component is the 2-iteration Griffin-Lim phase
refinement (its iSTFT/STFT span the whole utterance and need future enhanced
magnitudes). It is kept offline; the real-time path reuses the **noisy phase**
— the very seed Griffin-Lim starts from — for a causal iSTFT. The
frame-by-frame streamer reproduces the offline mask to **2e-7**
(`tests/test_lisennet_streaming_parity.py`).

## Deployment eval — isolating each cost

`lisennet/eval_deploy.py` measures PESQ across every combination of *mask
backend* × *phase method* on the full 824-utterance test split, so each
deployment step's cost is isolated. (Hold phase = Griffin-Lim and vary the
backend to read the export/quant cost; hold backend = torch and vary the phase
to read the cost of dropping Griffin-Lim.)

| variant                                          |  PESQ | Δ vs offline |
| ------------------------------------------------ | ----: | -----------: |
| torch + Griffin-Lim (offline recipe)             | 3.006 |            — |
| FP32 ONNX + Griffin-Lim                           | 3.006 |    **0.000** |
| dynamic-int8 ONNX + Griffin-Lim                   | 2.995 |       −0.012 |
| torch + noisy phase (real-time phase)             | 2.989 |       −0.017 |
| **dynamic-int8 ONNX + noisy phase (full real-time)** | 2.982 |   **−0.025** |
| static-int8 ONNX + Griffin-Lim                    | 2.897 |       −0.109 |

Takeaways:

* **The ONNX export is loss-free** (3.006 → 3.006): the exported mask
  sub-network is numerically the torch model.
* **Dynamic weight-only int8 is near-loss-free** (−0.012 PESQ) and is the
  recommended PTQ path.
* **Dropping the non-causal Griffin-Lim costs only ~0.017 PESQ**, so the
  causal real-time pipeline stays close to the offline model.
* The full real-time deployable config — **dynamic int8 mask + noisy phase** —
  lands at **2.982**, just 0.025 below the offline model.
* **Static full int8** degrades more (−0.109) and would need quantization-aware
  training, the same conclusion the BASENet and ConvFSENet int8 studies reached
  here — but it is a mild drop, not a collapse, because the mask is
  magnitude-only (no sensitive `atan2` phase path).

### Real-time factor

Frame-by-frame mask compute under a single onnxruntime / PyTorch CPU thread is
**~2.4 ms per 16 ms frame → RTF ≈ 0.15**, i.e. ~6.6× faster than real time,
with the bounded per-frame state described above.

### Note — int8 file size

Unlike the larger models here, the int8 ONNX **file is not smaller** than FP32
(~0.32 MiB vs 0.27 MiB). At ~37 K parameters the per-weight-tensor quant
metadata (DequantizeLinear nodes, scale / zero-point initializers, int32
biases) outweighs the byte saving on such small weights — the model is already
tiny in FP32. The win from int8 here is integer-arithmetic compute on edge
hardware, not graph footprint. (Static int8 additionally needs
`per_channel=False` to dodge an onnxruntime int32-bias-scale-adjustment bug on
this graph.)

## Reproducing

```bash
source .venv/bin/activate
# train (configs/lisennet.json — CMGAN, batch 16)
python -m lisennet.train --config configs/lisennet.json \
    --checkpoint_path cp_lisennet --training_epochs 100
# FP32 ONNX export of the mask sub-network (feat -> est_mag, dynamic B/T)
python -m lisennet.export_onnx --checkpoint_file cp_lisennet/g_best
# dynamic weight-only int8 (robust) — or --mode static (QAT-grade)
python -m lisennet.quant_onnx --fp32 cp_lisennet/g_best_fp32.onnx --mode dynamic
# full deploy eval: PESQ across backend × phase + streaming RTF
python -m lisennet.eval_deploy --checkpoint_file cp_lisennet/g_best --n_utts 824
```

Training is fast — the model is tiny, so each step is dominated by the CPU
PESQ + Griffin-Lim cost rather than GPU compute (~0.8 GB at batch 16,
~94 s/epoch, ~2.7 h for 100 epochs on an RTX 4090). The streaming reference is
`lisennet/streaming.py` (`LiSenNetStreamer`); streaming reuses the trained
`g_best` with no retraining (the `state_dict` keys are unchanged when the
streaming flag is off, and the offline forward stays bit-identical).

## Trained checkpoint

The best-PESQ generator (`g_best`), the FP32 ONNX (`g_best_fp32.onnx`), the
dynamic and static int8 ONNX (`g_best_int8_dyn.onnx`, `g_best_int8_static.onnx`),
and the training config are mirrored on HuggingFace at
[`claroche1/LiSenNet`](https://huggingface.co/claroche1/LiSenNet).

PyTorch:

```python
import json, torch
from huggingface_hub import hf_hub_download
from common.env import AttrDict
from lisennet.model import build_lisennet

REPO = "claroche1/LiSenNet"
cfg  = json.load(open(hf_hub_download(REPO, "config.json")))
ckpt = torch.load(hf_hub_download(REPO, "g_best"),
                  map_location="cpu", weights_only=True)
model = build_lisennet(AttrDict(cfg)).eval()
model.load_state_dict(ckpt["generator"])
# end-to-end waveform -> waveform: model(noisy_wav)["est"]
```

ONNX (the mask sub-network — `feat (B,3,T,F)` -> `est_mag (B,T,F)`):

```python
import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download

REPO = "claroche1/LiSenNet"
sess = ort.InferenceSession(
    hf_hub_download(REPO, "g_best_int8_dyn.onnx"),   # or _fp32 / _int8_static
    providers=["CPUExecutionProvider"],
)
est_mag = sess.run(["est_mag"], {"feat": np.zeros((1, 3, 100, 257), np.float32)})[0]
```

To (re-)publish from a fresh run:

```bash
python push_lisennet_hf.py            # cp_lisennet/ + deploy/lisennet/ -> claroche1/LiSenNet
```

(needs `huggingface-cli login` or `HF_TOKEN`; idempotent — re-running just makes
another HF commit.)
