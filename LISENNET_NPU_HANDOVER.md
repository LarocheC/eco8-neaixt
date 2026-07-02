# LiSenNet → STM32N6 NPU — Deployment Handover

Context for continuing the LiSenNet NPU-deployment / efficiency effort. Read this
first, then `RESULTS_LISENNET.md` (model + host results) and `deploy/stm32n6/README.md`
(the compile/flash scaffold).

## Goal & current status

**Goal:** run the LiSenNet conv variant on the STM32N6 **Neural-ART NPU** (not just the
Cortex-M55 CPU) — the NPU is far more efficient, and LiSenNet is the best-quality
real-time model in the repo (host int8 PESQ **2.855**, clears the 2.85 deploy gate).

**Status: TRAINED AND QUALITY-CLOSED — the deploy artifact is ready.** All four
`atonn` (Neural-ART compiler) blockers are diagnosed and fixed in code; the stateless
windowed deploy graph **compiles to the NPU** (verified with random weights). The
hardened model is now **trained and swept over capacity** (nc20/24/28, 100-epoch CMGAN
each): the winner is **nc24** (36,288 params) at FP32 PESQ **3.013** / real-time int8
PESQ **2.998** on the full 824-utt split — above the GRU quality reference (3.006 /
2.930) and far clear of the 2.85/2.90 gates. The windowed **signed-int8** deploy
artifact is produced and verified
(`cp_lisennet_conv_hardened_nc24/g_best_windowed_fp32.int8_static.onnx`). What remains
is deploy-box-only: `make generate` for the real NPU MACC/epoch report + on-board
ms/frame.

## The four NPU blockers and their fixes (the core finding)

The original conv variant (`configs/lisennet_conv_wide.json`) maps its operators but
**segfaults `atonn` (signo=11)** at codegen. Peeled by bisection:

| # | Blocker | Why it breaks Neural-ART | Fix (config knob) | Where |
|---|---------|--------------------------|-------------------|-------|
| 1 | 2-axis `CustomLayerNorm` (33 ReduceMean) | not NPU-mappable; lowers to ReduceMean/Sqrt/Div | `norm="batchnorm"` (per-channel, folds into conv) | `3f38db3` |
| 2 | `PReLU`/`Mish` (per-channel float slope) | blocks full int8 / forces M55 hybrid | `act="relu"` (parameter-free) | `3f38db3` |
| 3 | `SPConvTranspose2d` **5-D tensors** | the `view→permute→view` emits rank-5 tensors → **segfault** | `upsample="convtranspose"` (4-D, ~3× fewer decoder MACC) | `3f38db3` |
| 4 | 17-tensor **FIFO streaming state** I/O | the Slice/Pad/Concat state class → **segfault** | stateless **windowed** deploy graph (no state) | this branch |

Diagnosis evidence (all reproducible via `cp_lisennet_conv_hardened/` scratch + logs):
- Bisection: the **whole-utterance** (stateless) hardened graph compiles to the NPU, the
  **streaming** (FIFO-state) graph segfaults → blocker #4 is the FIFO state.
- OE-graph inspection: 6 rank-5 tensors shaped `[1,3,15,1,16]` → blocker #3 = SPConvTranspose.
- ConvFSENet is the working NPU template and already uses BN-fold + ReLU + a stateless
  **windowed** graph (`convfsenet/…`), which is exactly the recipe ported here.

## Proof it compiles (random weights → topology/latency, not quality)

- **whole-utterance hardened int8** → NPU `network.c`, **60/102 pure-hardware epochs** (+36 hybrid, 6 SW).
- **windowed hardened int8** → NPU `network.c`; **bit-exact parity `0.00e+00`** vs offline;
  **~2.1 M MACC / emitted frame** at `emit_T=64` (window L=68 + 64 → 2.06× recompute), 92 KB NPU activation RAM.

## What's in the code now

Committed on this branch (base hardening in `3f38db3`, pipeline in the handover commit):

- `lisennet/model.py` — config knobs `norm` (`layernorm`|`batchnorm`), `act`
  (`prelu`|`relu`), `upsample` (`subpixel`|`convtranspose`), threaded through every
  submodule. **Backward-compatible**: defaults reproduce the original; `g_best` loads 0/0.
- `configs/lisennet_conv_hardened.json` — the deploy variant (conv, batchnorm, relu,
  convtranspose, nc20, intra_kernel 11, dilations [1,2,4,8]).
- `lisennet/export_onnx.py` — `--windowed --emit_T N`: `_receptive_field`,
  `LiSenNetWindowedONNX`, `export_windowed_fp32` (graph: `feat_window (B,3,L+T,F) →
  est_mag (B,T,F)`, no state; L stamped in onnx metadata).
- `lisennet/quant_onnx.py` — signed int8 (`quantize_static_int8(signed=True)`),
  `VBDWindowedCalibrationReader`, CLI `--windowed --checkpoint --signed`. **Signed (QInt8)
  is mandatory** — the NPU rejects the old unsigned QUInt8 default.
- `common/quant_fake.py` — QAT support for `Conv2d`/`ConvTranspose2d` (weight axes 0/1 +
  both activation walkers). Additive; ConvFSENet/NSNet2 unaffected.
- `lisennet/qat_train.py` — QAT fine-tune (recon loss only, no GAN), ConvFSENet-style
  checkpoint discipline. Plumbing smoke-tested (41 weight + 41 act quant points, 0/0 reload).

All 16 `tests/test_lisennet_*` pass.

## Deploy pipeline (run once the trained hardened `g_best` exists)

```bash
# (optional) QAT to recover the int8 drop:
python -m lisennet.qat_train --init_from cp_lisennet_conv_hardened/g_best \
    --checkpoint_path cp_lisennet_conv_hardened_qat --epochs 30
# windowed export -> signed int8 -> NPU compile:
python -m lisennet.export_onnx --checkpoint_file cp_lisennet_conv_hardened/g_best --windowed --emit_T 64
python -m lisennet.quant_onnx --fp32 cp_lisennet_conv_hardened/g_best_windowed_fp32.onnx \
    --mode static --windowed --config cp_lisennet_conv_hardened/config.json \
    --checkpoint cp_lisennet_conv_hardened/g_best
cd deploy/stm32n6 && make generate \
    MODEL=$(readlink -f ../../cp_lisennet_conv_hardened/g_best_windowed_fp32.int8_static.onnx) \
    GEN_OUT=st_ai_output_lisennet
# read network_generate_report.txt: macc, epochs (HW/SW), ram; grep the log for signo=11/E103 (must be absent)
```

## TODOs (updated after the training study — see outcomes)

- [x] **Stage 2 — train:** done for nc20/nc24/nc28 (100 epochs each,
  `configs/lisennet_conv_hardened{,_nc24,_nc28}.json` → `cp_lisennet_conv_hardened{,_nc24,_nc28}/`).
  The norm-semantics risk did **not** materialize — BN+ReLU trains cleanly. The
  hardened decoder came out much smaller than budgeted (ConvTranspose swap: nc20 =
  25.7 K, not ~41 K), so the sweep *grew* capacity back: **nc24 (36.3 K) wins** with
  FP32 **3.013**; nc28 (48.7 K) trains worse (2.927) — capacity is non-monotonic here.
- [x] **Stage 2 — eval + artifact:** windowed export (`emit_T=64`) → signed int8
  (verified QInt8-only) → full-split PESQ: **nc24 real-time int8 2.998** (gate ≥ 2.85 ✓,
  stretch 2.90 ✓). Artifact: `cp_lisennet_conv_hardened_nc24/g_best_windowed_fp32.int8_static.onnx`.
- [x] **Stage 3 — QAT: NEGATIVE result, dropped.** 30-epoch w8/a8 QAT on nc20 *lost*
  0.085 PESQ (2.862 → 2.777 best): `qat_train`'s recon-only loss pulls weights off the
  MetricGAN(PESQ) optimum, and the hardened PTQ gap (−0.016…−0.042) leaves nothing to
  recover. **PTQ is the deploy path**; a future QAT would need the GAN loss in the loop.
- [x] **Stage 4 — capacity study:** done as a *grow* sweep instead of a shrink (the
  hardened model was under-budget, and nc20's margin over the 2.85 gate was inside the
  ±0.02 calibration noise). If NPU latency/MACC ever needs a cheaper model, nc20
  (2.853) is the fallback; dilations `[1,2,4,8]→[1,2,4]` (RF 68→~40) is the next knob.
- [ ] **Deploy box — `make generate`** on the nc24 windowed int8 artifact: record real
  NPU MACC, HW/SW epochs, activation RAM (grep the log: no signo=11/E103).
- [ ] **On-board ms/frame:** needs Arm GNU **13.3** + STM32CubeProgrammer (see `config.mk`)
  to flash + measure. Until then, latency is the stedgeai report estimate only.

## Key decisions / knobs

- **`emit_T` (latency vs recompute):** window = L(68) + emit_T. `emit_T=1` → ~69× recompute,
  ~16 ms latency; `emit_T=16` → ~5×, ~256 ms; `emit_T=64` → ~2.06×, ~1 s. Pick per latency
  budget. All compile to the NPU; per-frame MACC is trivially real-time on the NPU regardless.
- **Norm-semantics risk — RESOLVED:** `CustomLayerNorm` normalizes **per-frame** over
  (channel,freq); `BatchNorm2d` uses **fixed per-channel** stats. Training showed the swap
  costs nothing at the right capacity (nc24 FP32 3.013 vs the un-hardened conv_wide 2.970) —
  the power-compressed input (`compress_factor 0.3`) evidently makes per-frame norm
  unnecessary, as hoped.
- **Receptive field = 68 frames** (architectural; dilations 1/2/4/8 × 2 blocks). Sets the
  window warmup length. Stage-4 dilation reduction shrinks it.

## Environment

- **stedgeai 4.0.1** at `/home/claroche/stedgeai/install/4.0`; compile via `deploy/stm32n6`
  `make generate` (paths in `config.mk`).
- **This box = export/quant/compile only.** Training runs on a **separate box** (pull the
  hardening commit `3f38db3` there; note `origin/stm32n6-deploy` diverged after a rebase, so a
  `git push --force-with-lease` is needed before it can be pulled).
- **Local scratch (gitignored, `cp_*/`):** `cp_lisennet_conv_hardened/` (hardened artifacts +
  compile logs: `gen_windowed.log`, `st_ai_output_windowed/`), `cp_lisennet_conv_wide/` (the HF
  conv_wide download + the earlier crash logs that diagnosed blockers #1–#4).
- **Reference (the NPU template):** `convfsenet/` — `model.py` `*_QuantFriendly` (conv→bn→relu),
  `streaming.py:_fold_bn_into_conv` + `ConvFSENetWindowedONNX`, `quant.py` (signed QDQ recipe),
  `qat_train.py`. HF weights: `claroche1/LiSenNet` (`gru/` + `conv/`).
