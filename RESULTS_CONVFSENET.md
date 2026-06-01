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

See [RESULTS_NSNET2.md](RESULTS_NSNET2.md) for the recurrent model
family, and the [README](README.md) for setup and repo layout.

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
