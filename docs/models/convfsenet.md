# ConvFSENet — results

A fully-convolutional, causal speech enhancer. ConvFSENet is a
ConvTasNet-derived magnitude-mask predictor built from stacked Temporal
Conv Module (TCM) blocks (1×1 conv + depthwise dilated conv, BatchNorm,
ReLU, residual). The architecture here is the base enhancer from
Miccini, Laroche, Piechowiak & Pezzarossa, *Scalable Speech Enhancement
with Dynamic Channel Pruning*
([ICASSP 2025, arXiv:2412.17121](https://arxiv.org/abs/2412.17121)) —
this repo uses the static (un-pruned) variant of that model.
It runs **frame-by-frame**: each TCM block keeps a small FIFO state
buffer so the dilated convs stream with zero lookahead. The streaming
wrapper, FP32/int8 ONNX export and the ORT inference path are all
parity-checked against the offline model (`tests/test_convfsenet_*`).

See [../models/nsnet2.md](../models/nsnet2.md) for the recurrent model
family, and the [README](../../README.md) for setup and repo layout.

## Headline results

200-epoch training on VoiceBank-DEMAND-16k with an end-to-end
time-domain loss (`DynCompMSE`), a PESQ metric-GAN
(`MetricDiscriminator`), and a cosine LR schedule. PESQ on the full
824-utterance VBD test split; RTF is the int8 streaming session under
onnxruntime CPU (single thread), lower is faster.

| metric         | value     |
| -------------- | --------: |
| params         | 1.45 M    |
| FP32 PESQ      | **2.931** |
| int8 PESQ      | **2.911** |
| Δ (FP32→int8)  | +0.020    |
| int8 RTF       | 0.017     |
| int8 size      | 1.6 MiB   |

Static int8 PTQ is essentially loss-free (0.020 PESQ drop). The FP32
score of 2.931 beats NSNet2's dense baseline (2.845) and every
structured NSNet2 variant — at a fraction of the RTF (NSNet2 ranges
0.025–0.452 under the same onnxruntime-CPU conditions). The PESQ
metric-GAN is what pushes the FP32 score past 2.93; without it the same
architecture tops out around 2.77–2.79.

### Feature extractor — implementation note

ConvFSENet feeds `(|stft|+eps)^0.3` to the frontend Conv
(`extractor_type: mag_compressed` in the config). The compression
squashes the 60+ dB STFT magnitude range into an int8-friendly domain —
the same idea as NSNet2's log-magnitude input — and must stay in FP32 at
deployment.

`convfsenet/quant.py` automatically keeps the compression prologue
(`Add` / `Pow` / `Unsqueeze`) out of quantization: it walks the ONNX
graph from the magnitude input to the first Conv and adds those node
names to `quantize_static`'s `nodes_to_exclude`. Without this exclusion
`quantize_static` would treat the eps-`Add` as quantizable and drag the
raw `|stft|` onto a coarse int8 grid before the compression runs,
destroying exactly the low-energy detail the compression exists to
preserve.

## STM32N6 on-board deployment (preliminary)

First measurements of the int8 streaming model on real hardware: an
**STM32N6570-DK** (STM32N657 — Cortex-M55 @ 800 MHz + Neural-ART NPU @
1 GHz), compiled with ST Edge AI Core 4.0.1 and run via the bundled
`NPU_Validation` firmware. Single on-target `stedgeai validate` run
(10 random samples) of the per-frame graph, weights in external xSPI
flash, fully scripted — no STM32CubeIDE — through the `deploy/stm32n6/`
setup. **Preliminary** numbers, one model, one run.

| metric (on-target)                  |       value |
| ----------------------------------- | ----------: |
| inference latency / frame           | **7.21 ms** |
| frame period (hop 256 @ 16 kHz)     |       16 ms |
| hardware real-time factor           |  **≈ 0.45** |
| compute split (NPU / SW / SW-ctrl)  | 51.7% / 20.9% / 27.4% |
| mask cosine vs FP32 ONNX (on-target)|       0.990 |
| MACC / frame                        |      1.47 M |
| weights (external xSPI flash)       |    1.40 MiB |
| activations (on-chip SRAM)          |     ~34 KiB |

Real-time with ~2.2× headroom (7.2 ms inference against the 16 ms frame
budget, before the M55 STFT/iSTFT front-end). Note that **~48% of the
time runs on the Cortex-M55, not the NPU**: the convolutions map to the
Neural-ART accelerator, but the per-frame FIFO state handling
(Slice/Gather) and the int8 quant boundary fall back to software. That
software share — not the convs — is the lever for going faster. The
on-target int8 mask tracks the FP32 ONNX reference at cosine 0.990,
consistent with the loss-free PTQ above.

Caveats: only ConvFSENet currently compiles for the Neural-ART — NSNet2
(dense and structured) crashes the ST Edge AI compiler at this version;
the validation firmware is a volatile RAM image; and the headline
`int8 RTF 0.017` above is onnxruntime-CPU, not comparable to this 0.45
on-device factor. See `deploy/stm32n6/` for the generate → build → flash
→ gdb-load → validate procedure.

**Optimization — weight locality (npuRAM vs external flash).** The
`allmems` profile parks all 1.44 MB of weights in external octoFlash, and
the counters show the NPU is memory-bound: it re-reads the full weight set
from flash every frame (201 MB/s avg) at only ~27% core compute
utilization. The model is small enough to fit entirely on-chip, so
regenerating with the `n6-noextmem` profile packs weights into internal
npuRAM3/4/5/6. On-board latency then drops **7.14 → 4.40 ms/frame (1.62×;
RTF 0.45 → 0.275)**, NPU core time 3.73 → 1.26 ms, and core compute
utilization rises to **81%** — the NPU flips from memory-bound to
compute-bound. The remaining ~3.1 ms is the Cortex-M55 software share (the
per-frame FIFO state + int8 quant boundary). Caveat: in this layout the
weights load over the debugger (gdb) rather than being flashed; a
standalone power-on deploy needs an on-chip-resident boot layout.

## Stateless-windowed rework (Track 1 — removes the FIFO M55 floor)

That remaining ~3.1 ms M55 share is the per-block FIFO state plumbing
(`Slice`/`Concat`/`Gather`, always Hybrid). Track 1 of the efficiency rework
(`deploy/stm32n6/EFFICIENCY_REWORK_PLAN.md`) removes it entirely: instead of
threading per-block FIFO state across single frames, the host keeps a ring
buffer of the last `L = sum_blocks (K-1)*D = 42` magnitude columns and feeds a
fixed `[1, n_freq, L+T]` window through the BN-folded offline-causal model run as
**valid (padding-0) convs**. The dilated dconvs shrink the time axis by `(K-1)*D`
per block (residual cropped to match), so `L+T → T` with no state I/O and no Pad.
The exported int8 graph is **stateless** — `Conv`/`Add`/`Relu`/`Sigmoid` + 9
static residual-crop `Slice`s, **zero `Gather`/`Pad`/state/BatchNorm**
(`convfsenet/streaming.py:ConvFSENetWindowedONNX`, exported via
`export_onnx.py --windowed`, quantized via `quant_windowed.py`). FP32 is bit-exact
(<1e-6) to the offline causal model on full-context frames
(`tests/test_convfsenet_windowed_parity.py`, 34/34).

**Host PESQ (full 824-utt VBD test, no retrain — same v5 weights):**

| variant (int8) | FP32 PESQ | int8 PESQ | Δ |
| --- | ---: | ---: | ---: |
| streaming reference (deployed) | 2.931 | 2.911 | +0.020 |
| windowed-257, `coldstart=zero` | 2.858 | 2.836 | +0.022 |
| windowed-256, `coldstart=zero` | 2.865 | 2.843 | +0.022 |
| windowed-257, `coldstart=replicate` | 2.923 | 2.904 | +0.019 |
| **windowed-256, `coldstart=replicate` (deploy)** | **2.933** | **2.913** | +0.020 |

The 256-bin variant drops the Nyquist bin (frontend input + backend output) for
power-of-two HW alignment — it is **PESQ-neutral** (256 ≥ 257), so no 256-native
fine-tune was needed. The whole apparent gap was the **cold start**: the model
was trained with a zero-*activation* history before t=0, but a zero-*magnitude*
ring buffer feeds the frontend bias (`frontend(0) ≠ 0`) for the first `<L` frames
→ out-of-distribution, costing ~0.045 PESQ on short clips. Seeding the ring
buffer by **replicating the first frame** (`coldstart=replicate`, the default —
causal, no look-ahead, on-device deployable) recovers it: the windowed-256 int8
**2.913 matches/slightly beats the streaming 2.911**, at the documented int8 gate
≥2.85 (target 2.90–2.91, exceeded).

Net: same quality as the deployed streaming model, but the int8 deploy graph has
**no FIFO/state/Pad** class — the M55-Hybrid floor that capped ConvFSENet at 4.40
ms. On-board latency (does the per-frame epoch count drop, do the convs fill the
array at `h:43`?) is the Gate-0/Phase-4 verdict on the deploy box — see
[deploy/stm32n6/WINDOWED_DEPLOY_HANDOFF.md](../../deploy/stm32n6/WINDOWED_DEPLOY_HANDOFF.md).

### Track 2 (small-STFT block) — rejected on quality

Track 2 retrained the same 192/384 windowed backbone at a **smaller STFT**
(`n_fft 256 / hop 128`, 129 bins, 16 ms/8 ms framing, emit T=2) for 2× block
amortization under ~30 ms latency (`configs/convfsenet_win_smallstft.json`,
200-epoch GAN from scratch). The coarser frequency resolution costs too much
quality: windowed-128 int8 PESQ **2.725** (FP32 2.783) — **0.13 below the ≥2.85
gate**, ~0.19 below Track 1's 2.913. This is the plan's flagged Config-C risk
("accept only if PESQ holds ≥2.85") materializing. **Rejected** — Track 1
(512/256 windowed) stays the winner; the small-STFT latency win isn't worth the
PESQ loss.

## Low-bit weight PTQ study

`convfsenet/eval_ptq.py` sweeps `(w_bits, a_bits)` via the eager
fake-quant path in `common/quant_fake.py` (per-output-channel symmetric
weights, dynamic per-tensor symmetric activations) over the full
824-utterance test split. It is a study tool — the dynamic-symmetric
activation path is mildly optimistic versus the deployed
static-asymmetric int8 ONNX (so w8a8 reads slightly higher here than
the real ONNX number above).

| precision | PESQ  | note |
| --------- | ----: | --- |
| fp32      | 2.934 | reference |
| w8a8      | 2.928 | matches the deployed int8 ONNX within noise |
| **w4a8**  | 2.856 | 4-bit per-channel weights cost ~0.08 PESQ |

Even at 4-bit weights, ConvFSENet still beats every NSNet2 int8 variant.

## QAT (kept for the w4 study)

`convfsenet/qat_train.py` and the QAT machinery in `common/quant_fake.py`
(`StaticActFakeQuant`, `install_static_activation_fake_quant`) are kept
for the low-bit (w4) study and as a reference scaffold. They are **not
needed for int8 deployment** — static PTQ alone is loss-free thanks to
the compression-prologue exclusion above.

## Reproducing

```bash
source .venv/bin/activate
# train (configs/convfsenet.json — mag-compressed)
python -m convfsenet.train --config configs/convfsenet.json \
    --checkpoint_path cp_convfsenet --training_epochs 200
# streaming FP32 ONNX
python -m convfsenet.export_onnx --checkpoint_file cp_convfsenet/g_best
# static int8 ONNX (QDQ, per-channel, MinMax; compression prologue kept FP32)
python -m convfsenet.quant --checkpoint_dir cp_convfsenet --num_utterances 200
# dual FP32/int8 PESQ + RTF on the VBD test split
python -m convfsenet.inference_onnx --checkpoint_file cp_convfsenet/g_best.onnx
# low-bit weight PTQ study
python -m convfsenet.eval_ptq --checkpoint_file cp_convfsenet/g_best
```

FP32 training runs ~22 min on an RTX 4090; the metric-GAN runs take
6–7 h. The streaming wrappers live in `convfsenet/streaming.py`
(`ConvFSENetStreaming` naive per-frame, `ConvFSENetStreamingFast` with BN
folded into the convs, `ConvFSENetStreamingONNX` the real-valued export
wrapper); streaming requires `causal=True`.

## Trained checkpoint

The best-PESQ generator (`g_best`), the streaming FP32 ONNX
(`g_best_fp32.onnx`), the static int8 ONNX (`g_best.onnx`), and the
exact training config are mirrored on HuggingFace at
[`claroche1/convfsenet`](https://huggingface.co/claroche1/convfsenet).
The HF repo is a flat layout (one variant) — no per-run subdirs.

PyTorch:

```python
import json, torch
from huggingface_hub import hf_hub_download
from common.env import AttrDict
from convfsenet.model import build_causal_model

REPO = "claroche1/convfsenet"
cfg  = json.load(open(hf_hub_download(REPO, "config.json")))
ckpt = torch.load(hf_hub_download(REPO, "g_best"),
                  map_location="cuda", weights_only=False)
model = build_causal_model(AttrDict(cfg)).cuda().eval()
model.load_state_dict(ckpt["generator"])
```

ONNX (FP32 or int8):

```python
import onnxruntime as ort
from huggingface_hub import hf_hub_download

REPO = "claroche1/convfsenet"
sess = ort.InferenceSession(
    hf_hub_download(REPO, "g_best.onnx"),       # or g_best_fp32.onnx
    providers=["CPUExecutionProvider"],
)
# Streaming shape: feed one frame of magnitude STFT (B, n_freq) + the
# per-block FIFO state buffers per call. End-to-end RMS-norm + STFT +
# frame loop + iSTFT pipeline is in convfsenet/inference_onnx.py.
```

To (re-)publish from a fresh training run:

```bash
python push_convfsenet_hf.py --source cp_convfsenet
```

(needs `huggingface-cli login` or `HF_TOKEN` in the environment; the
script is idempotent — re-running just makes another HF commit.)
