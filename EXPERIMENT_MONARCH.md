# Experiment plan — genuine two-factor Monarch NSNet2 checkpoints

## Goal

Train a set of **genuine Monarch** NSNet2 checkpoints that parallel the existing
**block-diagonal** runs, so we can answer: *does the extra cross-channel mixing
of a true two-factor Monarch buy meaningful PESQ (FP32 and int8) over a single
block-diagonal factor at comparable parameter budgets?*

The current `*_monarch` HF checkpoints were mislabeled — they are single
block-diagonal factors and have been renamed `blockdiag_*` (see the HF rename in
`rename_hf_monarch_to_blockdiag.py`). This experiment produces the first
checkpoints that legitimately deserve the `monarch` name.

## Background: what changed

- `blockdiag` (`torch_structured.monarch.BlockdiagLinear`): one block-diagonal
  factor, `nblocks` blocks, **zero cross-block mixing**. `O(N²/nblocks)` params.
- `monarch` (`torch_structured.monarch.MonarchLinear`, torch-structured ≥1.3.0):
  two block-diagonal factors `w1`,`w2` with a permutation between them —
  block-diagonal × permutation × block-diagonal — giving **full cross-channel
  mixing**. Roughly `2×` the params of `blockdiag` at the same `nblocks`, but the
  composed map can reach full rank `min(in,out)` for any shape.

Both are wired in `nsnet2/layers.py` (`make_linear`/`make_gru`, kinds
`blockdiag`/`monarch` and `triton_blockdiag`/`triton_monarch`). The genuine
Monarch has a real fused Triton scan (`gru_qat` 0.4.0 `scan_monarch`) for the
GRU hidden recurrence; the block-diagonal path uses `scan_blockdiag`.

## Checkpoint set (launch-ready configs already committed)

Each `monarch_*` config mirrors its `blockdiag_*` sibling exactly (same dims,
LR, epochs, n_fft) — only `kind` differs. Param counts below are the full
NSNet2 (measured on CPU build).

| run             | linear   | gru      | nblocks | hidden | params (M) | blockdiag sibling params (M) |
| --------------- | -------- | -------- | ------: | -----: | ---------: | ---------------------------: |
| `monarch_8`     | monarch  | monarch  | 8       | 400    | 0.553      | 0.355 (`blockdiag_8`)        |
| `monarch_full`  | monarch  | monarch  | 4       | 400    | 1.099      | 0.700 (`blockdiag_full`)     |
| `monarch_fc`    | monarch  | gru      | 4       | 400    | 2.379      | 2.14  (`blockdiag_fc`)       |
| `wide_monarch`  | monarch  | monarch  | 4       | 768    | 3.635      | 2.36  (`wide_blockdiag`)     |

Config files: `configs/{monarch_8,monarch_full,monarch_fc,wide_monarch}.json`
(+ `_triton.json` variants for the gru-qat Triton GRU path, + `tr_monarch_8.json`
/ `tr_monarch_full.json` / `triton_monarch4.json` / `triton_monarch8.json` for
the bench/QAT-seed configs).

### Fairness note (two ways to compare)

A true Monarch at `nblocks=k` has ~2× the params of block-diagonal at the same
`k`. Two comparisons are worth running:

1. **Matched `nblocks`** (the configs above): monarch vs blockdiag at the same
   block count — monarch is bigger, so this tests "is the extra structure worth
   the extra params?".
2. **Matched params** (optional follow-up): bump the block-diagonal `nblocks`
   down (fewer, larger blocks → more params) or the monarch `nblocks` up until
   param counts line up, to isolate structure from capacity. E.g. compare
   `monarch_8` (0.55M) against a `blockdiag_4`-ish config near 0.55M.

## Initialization

`MonarchLinear` already ships a **variance-matched two-factor init**
(`set_weights_from_dense_init`, called from `reset_parameters`) that measures the
per-element variance a dense Kaiming init would produce and rescales `w1`,`w2`
so the *composed* output variance matches — important because naively
Kaiming-initing both factors compounds the variance twice and undershoots by
orders of magnitude. So the default random init is already correct; no action
needed for a from-scratch run.

Optional **warm-start from a trained dense/blockdiag checkpoint**: project a
trained dense weight into Monarch factors with
`torch_structured.monarch.blockdiag_butterfly_projection.blockdiag_butterfly_project(M)`
(returns `w1_bfly, w2_bfly` for square `M`). This is a good ablation if
from-scratch monarch trains slower than its blockdiag sibling.

## Commands

Train the genuine-Monarch sweep (writes `cp_monarch_*/`):

```bash
RUNS="$MONARCH_RUNS" ./run_sweep.sh                 # PyTorch/native path
TRITON=1 RUNS="$MONARCH_RUNS" ./run_sweep.sh        # gru-qat Triton GRU (CUDA); -> cp_monarch_*_triton/
```

(`$MONARCH_RUNS = "monarch_8 monarch_full monarch_fc wide_monarch"`, defined in
`run_sweep.sh`.) Single run, matching the block-diagonal training recipe:

```bash
python -m nsnet2.train --config configs/monarch_8.json \
    --checkpoint_path cp_monarch_8 --training_epochs 30 --validation_interval 200
```

FP32 eval, int8 PTQ, int8 eval (per-run, same recipe as the blockdiag sweep):

```bash
RUNS="$MONARCH_RUNS" ./run_eval_sweep.sh            # FP32 PESQ
RUNS="$MONARCH_RUNS" ./run_quantize_sweep.sh        # export FP32 ONNX + int8 PTQ
# int8 PESQ via nsnet2.eval_quant on cp_monarch_*/g_best.onnx
```

The int8 path already supports MonarchLinear end-to-end:
- ONNX export: `nsnet2/export_onnx.py` patches the two-factor
  `blockdiag_butterfly_multiply` with an export-friendly einsum
  (`_blockdiag_butterfly_multiply_export`, verified numerically identical to the
  fast op, ~1e-6).
- Fake-quant / PTQ / QAT: `common/quant_fake.py` detects `MonarchLinear` and
  quantizes both factors `w1`,`w2` per-output-row within each block (axis=1).

## Success metrics

Report, per run, alongside the `blockdiag_*` sibling:

- **FP32 PESQ** (full VBD test split, 824 utts) — primary quality signal.
- **int8 PESQ** and **Δ (FP32→int8)** — Monarch's two-factor structure is a
  richer graph; confirm it stays as int8-robust as block-diagonal (|Δ| ≤ ~0.02
  was the blockdiag result).
- **params** and **int8 RTF** (onnxruntime CPU) — Monarch costs ~2× params and
  an extra matmul stage per projection; quantify the quality-per-param and
  quality-per-latency trade vs block-diagonal.

Reference block-diagonal baselines (from the current HF table):

| sibling          | FP32 PESQ | int8 PESQ | int8 RTF |
| ---------------- | --------: | --------: | -------: |
| `wide_blockdiag` | 2.864     | 2.842     | 0.166    |
| `blockdiag_8`    | 2.832     | 2.826     | 0.025    |
| `blockdiag_full` | 2.827     | 2.848     | 0.039    |
| `blockdiag_fc`   | 2.805     | 2.789     | 0.448    |

**Headline question:** does `monarch_X` beat `blockdiag_X` on FP32 PESQ by more
than the extra params would buy a wider block-diagonal, and does it keep the
near-loss-free int8 behaviour?

## Publishing the results

Once trained/evaluated, upload with the existing pipeline (the monarch runs sit
next to the renamed blockdiag ones in the same HF repo):

```bash
python push_nsnet2_hf.py --runs monarch_8 monarch_full monarch_fc wide_monarch
```

Then add a genuine-Monarch results block to `RESULTS_NSNET2.md` and the HF model
card (`hf_readme_nsnet2.md`).

## Deployment note (STM32N6 NPU) — future work

`deploy/stm32n6/host/export_blockdiag_npu.py` is **block-diagonal-specific** (it
reads each projection's single `.weight` and emits per-block Slice+MatMul+Concat).
A genuine Monarch has no single `.weight` — it needs a **two-stage** NPU
decomposition: block-diagonal MatMul → channel permutation (Gather/Reshape) →
block-diagonal MatMul. That exporter does not exist yet; it's a prerequisite for
on-device deployment of any `monarch_*` checkpoint and should be scoped
separately (the permutation-between-factors is the new piece the Neural-ART
lowering has to accept).

## Risks / open questions

- **Trainability from scratch:** the composed two-factor map is deeper (two
  contractions, no nonlinearity between); if it trains slower than blockdiag,
  warm-start via `blockdiag_butterfly_project` from the trained dense baseline.
- **int8 sensitivity of the permutation stage:** the intermediate activation
  between `w1` and `w2` is a new quant point; watch its dynamic range during
  calibration (the fake-quant hook already covers the factor weights, but the
  inter-factor activation rides the module's output quant).
- **Triton scan_monarch** needs a CUDA device; CPU falls back to the per-step
  loop (fine for correctness, slow for training). Validate parity on GPU before
  a long run.
