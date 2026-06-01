# sparse-nsnet2

Speech-enhancement models trained on
[VoiceBank-DEMAND-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k),
built for benchmarking efficient, quantized, streamable enhancers when
deployed on hardware.

The repo carries **two model families** side by side. They share the
dataset, training utilities and quantization scaffolding, but are
otherwise independent — each has its own model, configs, training entry
point and ONNX/quantization pipeline.

| family | architecture | causal | streams | role |
| --- | --- | --- | --- | --- |
| **NSNet2** | GRU recurrent enhancer (Braun & Tashev, ICASSP 2021), with FC/GRU layers swappable between dense, [Butterfly](https://arxiv.org/abs/1903.05895) and [Monarch](https://arxiv.org/abs/2204.00595) structured factorizations | yes | yes | structured-factorization + int8/int4 quantization study |
| **ConvFSENet** | fully-convolutional ConvTasNet-derived magnitude-mask predictor (stacked Temporal Conv Module blocks); architecture from [Miccini *et al.*, ICASSP 2025](https://arxiv.org/abs/2412.17121) | yes | yes (frame-by-frame, zero lookahead) | fast causal CNN enhancer + int8 quantization study |

Both export to streaming-shape ONNX and quantize to int8, so they can be
benchmarked on the same footing (PESQ, RTF, model size) under
onnxruntime.

## Results

Results live in dedicated files, one per model family:

* **[RESULTS_NSNET2.md](RESULTS_NSNET2.md)** — the 9-config structured
  sweep, int8 quantization findings, and the int4-weight PTQ + QAT study.
* **[RESULTS_CONVFSENET.md](RESULTS_CONVFSENET.md)** — the v3/v4/v5 causal
  models, FP32 vs int8, and the magnitude-compression fix that makes int8
  loss-free.

Headline: NSNet2 `wide_monarch` tops FP32 PESQ among the recurrent
variants; ConvFSENet `v5` beats every NSNet2 variant on both FP32 and
int8 PESQ while running causally at a fraction of the RTF.

## Setup

Requires `uv`, Python 3.11–3.12, and (for CUDA acceleration) a working
CUDA toolkit + matching PyTorch wheels.

```bash
git clone https://github.com/LarocheC/sparse-nsnet2.git
cd sparse-nsnet2
uv sync
```

The structured-matrix (Butterfly / Monarch factorizations) and
GRU-QAT primitives come from the published PyPI packages
[`torch-structured`](https://pypi.org/project/torch-structured/) and
[`gru-qat`](https://pypi.org/project/gru-qat/) — no git submodules or
local CUDA build required. Both ship pure-Python wheels that fall back to
the Triton / native-PyTorch backend. `torch` is pinned to the cu118 wheels
in `pyproject.toml` (covers Pascal sm\_61 onwards).

## Repo layout

Each model family lives in its own package, with a third package for the
infrastructure both share. Scripts are run as modules from the repo root,
e.g. `python -m nsnet2.train ...` or `python -m convfsenet.quant ...`.

### `common/` — shared infrastructure

```
common/dataset.py        HF VoiceBank-DEMAND-16k wrapper + STFT helpers
common/env.py            AttrDict config loader
common/utils.py          checkpoint / misc helpers
common/metrics.py        PESQ helpers (eval_pesq / pesq_score)
common/discriminator.py  PESQ-based metric discriminator (both train loops)
common/quant_fake.py     eager fake-quant (STE) — int4/int8 weight+activation
                         PTQ and QAT scaffold, shared by BOTH families
```

Other top-level files: `configs/` (per-run configs for both families,
incl. the `*_triton.json` variants), `pyproject.toml` / `uv.lock` (dependency lock),
`tests/` (pytest suite), and the `run_*.sh` NSNet2 sweep drivers. The
structured-matrix and GRU-QAT primitives now come from the published
`torch-structured` and `gru-qat` PyPI packages.

### `nsnet2/` — GRU recurrent enhancer

```
nsnet2/model.py          NSNet2 (~50 lines, pure wiring)
nsnet2/layers.py         make_linear / make_gru / StructuredGRU + cell
                         + butterfly_ortho_penalty (int8-friendly training)
nsnet2/streaming.py      streaming-shape view of NSNet2 for ONNX export
nsnet2/train.py          training loop (HF dataset, GAN, validation, ckpt)
nsnet2/inference.py      single-checkpoint inference (PyTorch)
nsnet2/inference_onnx.py streaming inference + dual-session PESQ (FP32 vs int8)
nsnet2/export_onnx.py    streaming FP32 ONNX export (variant-aware)
nsnet2/eval_torch.py     streaming PESQ eval (PyTorch, used by the int4 study)
nsnet2/quant.py          static int8 PTQ (QDQ, per-channel, MinMax)
nsnet2/quant_hook.py     in-training quant validation hook
nsnet2/calibration.py    VBDCalibrationReader for int8 calibration
nsnet2/eval_quant.py     int4/int8 fake-quant PTQ evaluation
nsnet2/sweep_hf_ptq.py   w4/a8 PTQ sweep over HF checkpoints
nsnet2/qat_train.py      int4/a8 QAT fine-tuning driver
nsnet2/bench_gru.py      GRU backend microbenchmark
nsnet2/analyze_sweep.ipynb        NSNet2 results notebook
nsnet2/inspect_butterfly_activations.py  per-stage activation magnitude trace
```

NSNet2 configs in `configs/`: baseline, butterfly_*, monarch_*, wide_monarch,
triton_* / tr_* (Triton GRU backend). Sweep drivers at the repo root:
`run_sweep.sh` (9-run training sweep), `run_quantize_sweep.sh` (per-cp FP32 +
int8 ONNX export), `run_eval_sweep.sh` (per-cp PESQ eval), `run_qat_sweep.sh`
(w4/a8 QAT over the HF checkpoints; `EPOCHS`-overridable). Each `run_*.sh`
overrides its run set and knobs via env vars — see the script header.

### `convfsenet/` — fully-convolutional enhancer

```
convfsenet/model.py           ConvFSENet model + build_causal_model factory
                              (TCM blocks; mag / mag_compressed extractors)
convfsenet/streaming.py       streaming wrappers (naive / BN-folded fast /
                              real-valued ONNX-export wrapper)
convfsenet/train.py           end-to-end time-domain training (+ metric-GAN)
convfsenet/export_onnx.py     streaming FP32 ONNX export
convfsenet/quant.py           static int8 PTQ (keeps compression prologue FP32)
convfsenet/calibration.py     int8 calibration reader
convfsenet/inference_onnx.py  dual FP32/int8 ORT eval — PESQ + RTF
convfsenet/eval_ptq.py        low-bit (w4/w8) fake-quant PTQ study
convfsenet/qat_train.py       QAT fine-tuning driver (legacy — see results doc)
```

ConvFSENet config in `configs/`: `convfsenet.json` (the deployed
model — mag-compressed feature extractor, 200-epoch GAN-trained).

## Acknowledgements

* MP-SENet (Lu et al., 2023) for the training recipe and discriminator.
* NSNet2 (Braun & Tashev, 2021) for the recurrent model architecture.
* Miccini, Laroche, Piechowiak & Pezzarossa, *Scalable Speech Enhancement with
  Dynamic Channel Pruning* ([ICASSP 2025, arXiv:2412.17121](https://arxiv.org/abs/2412.17121))
  — the ConvFSENet architecture used here is the base enhancer from this paper.
* ConvTasNet (Luo & Mesgarani, 2019) for the convolutional design ConvFSENet derives from.
* Butterfly factorization (Dao et al., 2019) and Monarch (Dao et al., 2022),
  as packaged in [torch-structured](https://github.com/LarocheC/torch-structured).
* JacobLinCool for the resampled VoiceBank-DEMAND-16k HF dataset.

## License

MIT (inherited from MP-SENet). The `torch-structured` dependency is
Apache-2.0 (see its PyPI page / upstream repository).
