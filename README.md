# eco8-neaixt

Efficient, quantized, streamable speech-enhancement models for edge-AI benchmarking in the context of the [NeAIxt](https://neaixt.eu/) project.

This repository contains experimental speech-enhancement models trained on [VoiceBank-DEMAND-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k). The goal is to benchmark compact neural audio models that can be exported, quantized, and evaluated for deployment on constrained edge-AI hardware.

## NeAIxt context

[NeAIxt — Next Generation of edge AI crossing technology fields](https://neaixt.eu/) is a European Chips Joint Undertaking / Horizon Europe project focused on next-generation secure, energy-efficient edge AI. The project combines advanced microelectronics, embedded AI, and non-volatile memory technologies to enable low-power AI directly on devices.

This repository contributes to the NeAIxt consumer use case on **ultra-low-power speech enhancement**. In that use case, the objective is to develop highly compressed neural speech-enhancement models capable of real-time noise reduction on ultra-low-power edge processors for next-generation hearing and audio devices.

The code here focuses on open benchmarking and model exploration: streamable architectures, ONNX export, post-training quantization, quantization-aware training experiments, and runtime/quality evaluation. It is not a hardware-specific SDK and does not include proprietary device integration code.

## Model families

The repo contains three model families. They share the dataset wrapper, training utilities, metrics, and quantization scaffolding, but each model family has its own architecture, configs, training entry point, ONNX export path, and quantization pipeline.

| family         | architecture                                                                                                                                                           | causal | streaming                               | role                                                         |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | --------------------------------------- | ------------------------------------------------------------ |
| **NSNet2**     | GRU recurrent enhancer based on Braun & Tashev, ICASSP 2021. FC and GRU layers can be swapped between dense, Butterfly, and Monarch structured factorizations.         | yes    | yes                                     | Structured-factorization, int8, and int4 quantization study. |
| **ConvFSENet** | Fully-convolutional ConvTasNet-derived magnitude-mask predictor using stacked Temporal Conv Module blocks. The architecture is based on Miccini *et al.*, ICASSP 2025. | yes    | yes, frame-by-frame with zero lookahead | Fast causal CNN enhancer and int8 quantization study.        |
| **LiSenNet**   | Lightweight (~37 K-param) sub-band U-Net with a dual-path-recurrent bottleneck and a magnitude-only mask (Griffin-Lim phase). A port of Yan *et al.*, arXiv:2409.13285.   | yes    | yes, frame-by-frame with bounded state  | Ultra-compact real-time enhancer and int8 quantization study. |

All three families export to streaming-shape ONNX and support int8 quantization, allowing comparison under the same evaluation conditions: PESQ, real-time factor, and model size under ONNX Runtime.

## Results

Detailed results are kept in separate files:

* [RESULTS_NSNET2.md](RESULTS_NSNET2.md): structured NSNet2 sweep, int8 quantization, and int4-weight PTQ/QAT experiments.
* [RESULTS_CONVFSENET.md](RESULTS_CONVFSENET.md): causal ConvFSENet models, FP32 vs int8 results, and the magnitude-compression fix required for robust int8 deployment.
* [RESULTS_LISENNET.md](RESULTS_LISENNET.md): ultra-compact LiSenNet, frame-by-frame streaming, FP32/static-int8 ONNX, and the real-time (noisy-phase) deployment eval.


## Setup

Requires:

* Python 3.11–3.12
* [`uv`](https://docs.astral.sh/uv/)
* CUDA-compatible PyTorch wheels if using GPU acceleration

```bash
git clone https://github.com/LarocheC/eco8-neaixt.git
cd eco8-neaixt
uv sync
```

The structured-matrix and GRU-QAT primitives come from the published PyPI packages:

* [`torch-structured`](https://pypi.org/project/torch-structured/)
* [`gru-qat`](https://pypi.org/project/gru-qat/)

No git submodules or local CUDA builds are required. The packages fall back to Triton or native PyTorch backends depending on availability.

## Repository layout

```text
common/                  Shared infrastructure
  dataset.py             VoiceBank-DEMAND-16k wrapper and STFT helpers
  env.py                 Config loader
  utils.py               Checkpoint and utility helpers
  metrics.py             PESQ helpers
  discriminator.py       PESQ-based metric discriminator
  quant_fake.py          Eager fake-quantization scaffold for PTQ/QAT

nsnet2/                  GRU recurrent enhancer
  model.py               NSNet2 model wiring
  layers.py              Dense / Butterfly / Monarch layer factories
  streaming.py           Streaming-shape wrapper for ONNX export
  train.py               Training loop
  export_onnx.py         FP32 ONNX export
  quant.py               Static int8 PTQ
  eval_quant.py          Int4/int8 fake-quant PTQ evaluation
  qat_train.py           Int4/a8 QAT fine-tuning

convfsenet/              Fully-convolutional causal enhancer
  model.py               ConvFSENet model and causal factory
  streaming.py           Streaming wrappers
  train.py               End-to-end training
  export_onnx.py         FP32 ONNX export
  quant.py               Static int8 PTQ
  eval_ptq.py            Low-bit PTQ study
  qat_train.py           QAT reference scaffold

lisennet/                Ultra-compact sub-band + dual-path enhancer
  model.py               LiSenNet model (sub-band U-Net + DPR bottleneck)
  streaming.py           Frame-by-frame streamer (bounded state)
  train.py               CMGAN training loop
  export_onnx.py         FP32 ONNX export of the mask sub-network
  quant_onnx.py          Static (+ dynamic) int8 quantization
  eval_deploy.py         PESQ (backend × phase) + streaming RTF eval

configs/                 Per-run configs for all model families
tests/                   Pytest suite
RESULTS_NSNET2.md        NSNet2 results
RESULTS_CONVFSENET.md    ConvFSENet results
RESULTS_LISENNET.md      LiSenNet results
```

Scripts are run as modules from the repository root, for example:

```bash
python -m nsnet2.train --config configs/baseline.json --checkpoint_path cp_baseline
python -m convfsenet.train --config configs/convfsenet.json --checkpoint_path cp_convfsenet
python -m lisennet.train --config configs/lisennet.json --checkpoint_path cp_lisennet
```

## NSNet2 experiments

The NSNet2 branch explores whether structured linear transforms can reduce model size and runtime while preserving speech-enhancement quality after quantization.

Supported layer types include:

```json
"linear": {"kind": "linear" | "butterfly" | "monarch"},
"gru": {"kind": "gru" | "butterfly" | "monarch"}
```

The main sweep scripts are:

```bash
./run_sweep.sh
./run_quantize_sweep.sh
./run_eval_sweep.sh
./run_qat_sweep.sh
```

The results show that Monarch-based variants are particularly friendly to int8 QDQ quantization, while Butterfly variants are more sensitive unless constrained by orthogonal initialization or recovered through QAT.

## ConvFSENet experiments

The ConvFSENet branch studies a fast causal CNN-based speech enhancer. It runs frame-by-frame using FIFO state buffers in each Temporal Conv Module block, enabling zero-lookahead streaming inference.

The deployed config is:

```text
configs/convfsenet.json
```

Typical workflow:

```bash
python -m convfsenet.train --config configs/convfsenet.json --checkpoint_path cp_convfsenet --training_epochs 200
python -m convfsenet.export_onnx --checkpoint_file cp_convfsenet/g_best
python -m convfsenet.quant --checkpoint_dir cp_convfsenet --num_utterances 200
python -m convfsenet.inference_onnx --checkpoint_file cp_convfsenet/g_best.onnx
```

ConvFSENet uses magnitude compression before the frontend convolution. The quantization script keeps this compression prologue in FP32 before applying int8 quantization, which is important for preserving low-energy spectral detail.

## LiSenNet experiments

LiSenNet is an ultra-compact (~37 K-param) sub-band U-Net with a dual-path-recurrent bottleneck, a magnitude-only mask, and Griffin-Lim phase. It is a faithful port of [Yan *et al.*, arXiv:2409.13285](https://arxiv.org/abs/2409.13285) (the authors' MIT reference at [hyyan2k/LiSenNet](https://github.com/hyyan2k/LiSenNet)).

The model is causal in time by construction, so the mask sub-network streams frame-by-frame with bounded state (per-conv ring buffers + the dual-path inter-time GRU hidden state). The only non-causal piece, the 2-iteration Griffin-Lim phase, is kept offline; the real-time path reuses the noisy phase (Griffin-Lim's own seed) for a causal iSTFT.

Typical workflow:

```bash
python -m lisennet.train --config configs/lisennet.json --checkpoint_path cp_lisennet --training_epochs 100
python -m lisennet.export_onnx --checkpoint_file cp_lisennet/g_best
python -m lisennet.quant_onnx --fp32 cp_lisennet/g_best_fp32.onnx --mode static --config cp_lisennet/config.json
python -m lisennet.eval_deploy --checkpoint_file cp_lisennet/g_best --n_utts 824
```

The reproduction reaches **PESQ 3.006** (full VBD test split, within ~0.06 of the paper's ~3.07). The ONNX export is loss-free, static int8 (the embedded-deployable quantization) costs ~0.086 PESQ, and the full real-time config (static int8 mask + noisy phase) lands at **2.930** at **RTF ≈ 0.13** on a single CPU thread. See [RESULTS_LISENNET.md](RESULTS_LISENNET.md) for the full backend × phase breakdown.

The GRU model above is the quality reference but does not compile to the STM32N6 Neural-ART NPU (GRU + 2-axis LayerNorm are compiler blockers). An **NPU-deployable variant** (`configs/lisennet_conv_wide.json`, `bottleneck: "conv"`) replaces the dual-path GRU with a dual-path conv bottleneck — exported graph has 0 GRU / 0 LayerNormalization nodes — and adds a frame-by-frame streaming graph with explicit FIFO state I/O (the stedgeai target). It reaches **PESQ 2.970** FP32 / **2.855** real-time int8.

## Trained checkpoints

Some trained checkpoints and exported ONNX models are mirrored on Hugging Face:

* [`claroche1/sparse-nsnet2-checkpoints`](https://huggingface.co/claroche1/sparse-nsnet2-checkpoints)
* [`claroche1/convfsenet`](https://huggingface.co/claroche1/convfsenet)
* [`claroche1/LiSenNet`](https://huggingface.co/claroche1/LiSenNet) — GRU quality reference: PyTorch `g_best`, FP32 ONNX, and static int8 ONNX
* [`claroche1/LiSenNet-npu`](https://huggingface.co/claroche1/LiSenNet-npu) — NPU-deployable conv variant: adds the frame-by-frame streaming ONNX (`g_best_streaming_fp32.onnx`, the stedgeai target)

See the result files for loading examples and evaluation commands.

## Acknowledgements

This work was carried out in the context of the NeAIxt project.

NeAIxt has received funding from the European Union’s Horizon Europe research and innovation programme under the HORIZON-JU-Chips-2024-1-IA grant agreement No. 101194172, with support from the participating Chips JU member states. The contents of this repository reflect the authors’ views only; the European Union and the granting authority are not responsible for them.

This repository also builds on:

* MP-SENet, for the training recipe and metric discriminator.
* NSNet2, for the recurrent speech-enhancement architecture.
* Miccini, Laroche, Piechowiak & Pezzarossa, *Scalable Speech Enhancement with Dynamic Channel Pruning*, ICASSP 2025, for the ConvFSENet base architecture.
* ConvTasNet, for the convolutional design principles used by ConvFSENet.
* Yan, Zhou, Chen & Lu, *LiSenNet: Lightweight Sub-band and Dual-Path Modeling for Real-Time Speech Enhancement*, [arXiv:2409.13285](https://arxiv.org/abs/2409.13285) ([hyyan2k/LiSenNet](https://github.com/hyyan2k/LiSenNet), MIT), for the LiSenNet architecture.
* Butterfly and Monarch structured matrix factorizations, as packaged in `torch-structured`.
* JacobLinCool’s resampled VoiceBank-DEMAND-16k dataset.

## License

MIT.

The repository inherits part of its training structure from MP-SENet. The `torch-structured` dependency is Apache-2.0; see its PyPI page and upstream repository for details.
