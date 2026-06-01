# sparse-nsnet2 — Quantization & Streaming Export

## What This Is

sparse-nsnet2 is a research codebase exploring sparse / structured weight matrices in NSNet2 speech-enhancement models (dense, butterfly, monarch GRU/Linear variants, trained on VoiceBank-DEMAND-16k). This milestone adds an **int8 ONNX deployment path** so that any trunk variant can be exported as a streaming, frame-by-frame model with a stable `[frame_in, states] → [mask, states]` interface and evaluated under static quantization both at export time and periodically during training.

## Core Value

A trained NSNet2 variant can be exported to a static int8 ONNX model that runs frame-by-frame (one STFT frame per call, GRU state passed in/out) on ONNX Runtime CPU x86, with the post-quantization PESQ tracked alongside FP32 throughout training so quantization regressions are visible early — not discovered after weeks of training.

## Requirements

### Validated

<!-- Existing capabilities inferred from .planning/codebase/* (HEAD of main as of 2026-04-27) -->

- ✓ NSNet2 trunk model (FC → GRU stack → FC × 3 → sigmoid mask) — `models/model.py` — existing
- ✓ Config-driven structured-layer factory (`make_linear` / `make_gru` dispatch to dense / Butterfly / BlockdiagLinear) — `models/layers.py` — existing
- ✓ GAN training loop with PESQ-mimicking discriminator (`MetricDiscriminator`), GAN losses, rolling + best checkpointing — `train.py`, `models/discriminator.py` — existing
- ✓ VoiceBank-DEMAND-16k dataset loader, `mag_pha_stft` / `mag_pha_istft` STFT helpers — `dataset.py` — existing
- ✓ Inference script (PyTorch, batched / per-utterance) — `inference.py` — existing
- ✓ Sweep runner + analysis notebook (`run_sweep.sh`, `analyze_sweep.ipynb`) — existing
- ✓ HuggingFace checkpoint hosting (`claroche1/sparse-nsnet2-checkpoints`) — existing

### Active

<!-- v1 hypotheses for the quantization milestone — building toward these -->

- [ ] **Streaming forward pass** — NSNet2 trunk runs one STFT frame at a time with a `[frame_in, states] → [mask, states]` PyTorch interface (GRU stack unrolled to per-step cell)
- [ ] **FP32 streaming parity** — unrolled streaming forward must match the batched cuDNN-GRU forward to within tight numerical tolerance on the same input (gates the rest of the work)
- [ ] **Static int8 ONNX export** — torch → ONNX → static int8 (QDQ format) producing a single `.onnx` artifact with the same `[frame_in, states] → [mask, states]` IO signature
- [ ] **Calibration pipeline** — pulls a configurable number of utterances from the VBD train split, runs the streaming forward to gather activation ranges, produces the calibrated quantized model
- [ ] **In-training quantized eval hook** — at every FP32 validation pass, calibrate + quantize the current generator and run PESQ on the validation set; both metrics logged to TensorBoard
- [ ] **ONNX inference script** — `inference_onnx.py` (or equivalent) that loads the int8 model and produces enhanced WAVs frame-by-frame, used as the final-mile validator
- [ ] **PESQ-delta tracking** — FP32 vs int8 PESQ delta logged per validation; no hard threshold this milestone, the gap is the artifact

### Out of Scope

- **GPU / mobile / NNAPI / QNNPACK targets** — CPU x86 only this milestone, to keep validation tight. Multi-target export deferred until the CPU path is proven.
- **Quantization of the Butterfly / Monarch / structured-GRU variants** — int8 export of structured layers is non-trivial (custom ops not in stock ONNX op-set). Phase 1 covers only `linear.kind = "linear"` + `gru.kind = "gru"` (dense baseline). Structured int8 is explicitly a future milestone.
- **Resurrection of `*_standalone.py` variants (SMR / EncDec / SeNSNet2 / LRU)** — those variants live on the unmerged `research/enc-dec-study` and `research/lru-integration` branches; they stay there for this milestone.
- **Quantization-aware training (QAT)** — only post-training static quantization (PTQ) this milestone. QAT is a future follow-up if the PTQ gap is too large.
- **Full ONNX Runtime mobile / ARM validation** — generic CPU x86 only.
- **Strict PESQ-degradation threshold** — explicit user decision: track the gap, don't gate on it.

## Context

- **Research codebase, not production.** Code style is single-author, pragmatic Python (pyproject + uv, single device, single-author training script). Quantization work should follow that ethos — additive scripts and config flags, not framework rewrites.
- **Existing model is config-swappable, not class-swappable.** `NSNet2(h)` is the only trunk model; variants are selected via `linear.kind` / `gru.kind` in JSON. This is convenient for FP32 sweeps but the GRU dispatch (`make_gru`) returns either cuDNN `nn.GRU` (kind="gru") or `StructuredGRU` (a Python time-loop). For ONNX, both paths need a per-step "cell" view, but the cuDNN path is the only one in scope this milestone.
- **PESQ is the headline metric.** TensorBoard scalar `Validation/PESQ Score` is the load-bearing signal. New scalars must follow the same convention so `analyze_sweep.ipynb` keeps working without rework.
- **Checkpoint dirs are self-contained artifacts.** Each `cp_<name>/` carries its own `config.json` + best/rolling generator. Quantization output (`.onnx`, calibration-cache) should live alongside as `cp_<name>/g_best.onnx` to preserve that contract.
- **Reproducibility caveats already exist on `main`** (`best_pesq` reset to 0 on resume, `cudnn.benchmark = True`, no `np.random.seed`/`random.seed`). The quantization work should not make these worse, and ideally adds a determinism flag for the calibration step.
- **Submodule pinned.** `torch-structured` is pinned to v0.4.0 — Butterfly / Monarch primitives stay frozen during this milestone.
- **No existing test infrastructure.** Correctness today is validated via sweeps + PESQ delta vs `cp_baseline`. Streaming-parity will be the first place that introduces an actual numerical-equality test.

## Constraints

- **Tech stack**: Python 3.12 + PyTorch + uv, ONNX + onnxruntime, `pesq` package — must run via `uv run python ...` like everything else in the repo.
- **Deployment target**: ONNX Runtime CPU x86 (`CPUExecutionProvider`), QDQ format for static int8.
- **Model scope**: NSNet2 trunk only. Phase 1 narrows further to dense baseline (`linear.kind = "linear"`, `gru.kind = "gru"`).
- **Streaming contract**: forward signature is `(frame_in, states) -> (mask, states)`. Spectrogram-loop simulation in Python is the validation harness.
- **Calibration data**: random subset of VoiceBank-DEMAND-16k train split, sample count is a config field.
- **Quality**: no hard PESQ threshold; the FP32-vs-int8 PESQ delta is logged and tracked but does not gate the phase.
- **Branch**: all milestone work lands on `feat/quantization`; PRs to `main` only after the milestone closes.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Branch name `feat/quantization` | Conventional `feat/*` prefix; reserves `research/*` for the unmerged enc-dec / LRU branches | — Pending |
| NSNet2 trunk only (defer standalone variants) | Standalone source lives only on unmerged research branches; resurrecting them is a separate milestone | — Pending |
| ONNX Runtime CPU x86 only (defer mobile / ARM) | Tight validation loop; one backend's QDQ gotchas at a time | — Pending |
| In-training quantized eval at every FP32 validation | Catches quant regressions early at the cost of validation-step wall-clock — explicit user preference | — Pending |
| Calibration source = VBD train subset, configurable count | Matches training distribution; exposes calibration size as a sweepable hyperparameter | — Pending |
| Streaming entry points = `inference.py --streaming` + `inference_onnx.py` | PyTorch parity path lives in inference.py for debug; ONNX path is its own script for clean dependency boundary | — Pending |
| FP32 streaming-parity gate before quantization | Surfaces unrolling bugs before they hide under quant noise | — Pending |
| No hard PESQ-degradation threshold this milestone | Track-and-learn first; bake a gate into the next milestone once the typical gap is known | — Pending |
| Static PTQ only (no QAT this milestone) | Simpler scope; QAT is a future follow-up if PTQ gap is too large | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-27 after initialization*
