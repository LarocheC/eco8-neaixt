# eco8-neaixt

Efficient, quantized, streamable speech-enhancement models, benchmarked for edge AI as part of the EU
NeAIxt project (its ultra-low-power speech-enhancement use case). There are three model families:
NSNet2 (GRU, with structured layers), ConvFSENet (causal CNN) and LiSenNet (a ~37 K-parameter sub-band
U-Net). All three are trained on VoiceBank-DEMAND-16k, exported to streaming ONNX, quantized to int8 and
scored with PESQ, DNSMOS, NISQA and SCOREQ. They are deployed on an NXP RT595 (HiFi4 DSP) and an
STM32N6570-DK (Neural-ART NPU). MIT licensed.

## Run it
```bash
uv sync
uv run pytest      # tests/; `slow` tests (model weights, the real VBD split) are deselected, run them with -m slow
```
Run scripts as modules from the repo root, inside the uv environment. LiSenNet as the example:
```bash
python -m lisennet.train --config configs/lisennet.json --checkpoint_path cp_lisennet --training_epochs 100
python -m lisennet.export_onnx --checkpoint_file cp_lisennet/g_best
python -m lisennet.quant_onnx --fp32 cp_lisennet/g_best_fp32.onnx --mode static --config cp_lisennet/config.json
python -m lisennet.eval_deploy --checkpoint_file cp_lisennet/g_best --n_utts 824
```
- ConvFSENet uses `convfsenet.{train,export_onnx,quant,inference_onnx}`.
- NSNet2 runs as sweeps: `./run_sweep.sh`, `./run_quantize_sweep.sh`, `./run_eval_sweep.sh`,
  `./run_qat_sweep.sh`.
- Scoring all published models: `python -m benchmarks.enhance`, then `python -m benchmarks.score`, then
  `python -m benchmarks.report`. The eval CLIs take an opt-in `--metrics`. DNSMOS alone takes about
  4 minutes on the full test split.
- STM32N6: `cd deploy/stm32n6 && make help`, then `make doctor`, `make bootstrap` and
  `make deploy MODEL=<onnx>`. There is also `make bench-cloud` and `make validate-target`.
  Machine-local paths go in `deploy/stm32n6/config.mk`. Training and the board live on different
  machines (`docs/targets/stm32n6.md`).

## Where things are
- `common/`: dataset, PESQ helpers, `quality.py` (DNSMOS, NISQA, SCOREQ) and the fake-quant scaffold.
- `nsnet2/`, `convfsenet/`, `lisennet/`: one package per family (model, streaming, train, export, quant).
- `configs/`: per-run configs. `configs/lisennet_conv_wide.json` is the NPU-deployable LiSenNet.
- `benchmarks/`: the metric harness. `summary.json` and `per_utterance.json.gz` are committed.
- `deploy/rt595/`, `deploy/stm32n6/`: the two hardware targets.
- `docs/`: the results index. Start at `docs/README.md`, then `models/`, `targets/`, `studies/` and
  `publishing/`.
- Checkpoints go to `cp_<run>/` (gitignored). Published checkpoints are on Hugging Face (see README).

## Status
As of 2026-10-02:
- `main` last changed on 2026-09-04 (`b72d38e`).
  [#5](https://github.com/LarocheC/eco8-neaixt/pull/5) made the RT595 a first-class deploy target, and
  `36aa725` split the results by model and by target.
- Earlier merged PRs:
  - [#1](https://github.com/LarocheC/eco8-neaixt/pull/1) LiSenNet (2026-07-03).
  - [#2](https://github.com/LarocheC/eco8-neaixt/pull/2) "monarch" renamed to "blockdiag", plus a
    genuine Monarch (2026-07-14).
  - [#3](https://github.com/LarocheC/eco8-neaixt/pull/3) DNSMOS, NISQA and SCOREQ (2026-07-14).
  - [#4](https://github.com/LarocheC/eco8-neaixt/pull/4) LiSenNet hybrid bottleneck (2026-07-14).
- Open: [#6](https://github.com/LarocheC/eco8-neaixt/pull/6), which drops the references to the old
  pre-public repo. Newer work sits on unmerged branches:
  - `feat/pesq-finetune` (2026-10-02): an optional differentiable PESQ loss (triton-pesq) and a
    fine-tune study runner.
  - `block-design` (2026-09-29): the sparse-kernel write-up and a ConvFSENet hand-off exporter.
  - `sparse-masks-rowfusion` (2026-09-27): the codebook, square and knee sparsity studies.
  - `N6Net` (2026-09-17): newer STM32N6 work.
  - `monarch-nblocks-sweep` (2026-09-04): holds the provisional `monarch_40` row in `docs/README.md`.
- Naming caveat from `docs/README.md`: the `deploy/stm32n6/` documents still use the pre-rename labels.
  Their `monarch_full` and `monarch_8` are the block-diagonal `blockdiag_full` and `blockdiag_8`.
  Genuine two-factor Monarch has never run on the STM32N6.

## Related repos
- [stm32n6-deployment-zoo](https://github.com/LarocheC/stm32n6-deployment-zoo): the STM32N6 hub and failure atlas, part of which was mined from this repo's N6 work.
- [stm32n6-stt](https://github.com/LarocheC/stm32n6-stt): Citrinet-256 speech-to-text on the same board and toolchain.
- [dnsmos_exported](https://github.com/LarocheC/dnsmos_exported): DNSMOS as an int8 metric and a trainable loss graph on the N6.
- [triton-pesq](https://github.com/LarocheC/triton-pesq): the GPU PESQ loss used by the `feat/pesq-finetune` branch.
- [torch-structured](https://github.com/LarocheC/torch-structured): the structured layers (Butterfly, block-diagonal, Monarch), a PyPI dependency.
- [gru-qat](https://github.com/LarocheC/gru-qat): GRU quantization-aware training, a PyPI dependency.

## Conventions
- Work on a branch and merge into `main` through a PR.
- Point at other repos with GitHub links or `repo:path`, never local paths.
- This repo is public. Name only public repos, and carry no unpublished numbers or paper plans. Toolchain
  locations come from the environment or `config.mk` overrides, never a committed home directory
  (`027edd4`, `f0f8c4c`).
- Every hardware number carries a provenance marker: **SILICON** (measured on a board, raw capture
  committed), **ISS** (instruction-set simulator) or **MODELLED** (datasheet constants). PESQ is always
  measured on the host, never on the device.
- Never commit training or benchmark outputs (`cp_*/`, `logs/`, `generated_files/`, `benchmarks/out/`).
  The durable results (`benchmarks/summary.json`, `benchmarks/per_utterance.json.gz`) are committed with
  their provenance block.
- Keep `uv run pytest` green.

## STM32N6 deployment
Before debugging anything on the STM32N6570-DK (ST Edge AI Core, the Neural-ART NPU, signing and
flashing, memory placement), read the hub:
[stm32n6-deployment-zoo/KNOWLEDGE.md](https://github.com/LarocheC/stm32n6-deployment-zoo/blob/main/KNOWLEDGE.md).
From a checkout of that repo, `uv run zoo atlas <words>` searches its failure atlas, and
`uv run zoo atlas --classify <log>` matches a build or flash log against it. Findings about the part,
the toolchain or the board belong there, as atlas entries with sources. Findings about this repo's
models stay here.
