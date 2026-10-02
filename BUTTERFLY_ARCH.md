# Butterfly-native NSNet2 for the FlexMan butterfly engine

**Goal.** Find an architecture that uses butterfly matrices *by design* and maximises VBD
PESQ, under the constraint that it runs on the FlexMan butterfly engine
(`jabra-private`, branch `butterfly-engine`, `frontends/nsnet2/BUTTERFLY_ENGINE.md`). The
engine already runs the published `butterfly_full` / `butterfly_ortho` in integer arithmetic
at fp32 quality (−0.006 / −0.000 PESQ), so **fp32 quality is the target**; int8 is the
engine's problem and is solved for this layer type.

Code: `nsnet2/bfly_arch.py` (all arms), `tests/test_bfly_arch.py`, configs `configs/ba_*.json`,
driver `run_bfly_arch_wave.sh`. Branch `butterfly-arch` (off `butterfly-a1`).

## What the engine runs (the design space)

- **Butterfly tasks**: radix-2 stages of 2×2 twiddles, n = 2…1024 tested, **any stride order
  with repeats**, several output stacks per task (§8). A stride *subset* is one task with
  fewer stages.
- **Epilogue per output** (§9): + bias, + a preloaded *seed* vector, then RAW / ReLU-requant /
  4096-entry LUT (sigmoid, tanh, or any curve). K = 0 tasks do element-wise versions.
- **Hadamard unit**: element-wise products (`r ⊙ CN`, `h' = z ⊙ (h − n) + n`).
- **FFT pair words** (§14): 16-bit complex FFT stages for the STFT / iSTFT.
- **Not available**: data-dependent normalisation (LayerNorm), permutations inside the net.
  BatchNorm is fine at inference (a fixed affine that folds into the butterfly next to it).

## Evidence the designs start from (branch `butterfly-a1`)

1. Long butterfly products collapse: the published layers are near-singular (fc_in cond
   3e5–2e8 vs 24 dense; 99 %-energy rank of fc_out 14–89 of 257 vs 104). Forcing
   orthogonality fixed conditioning but cost ~0.07 PESQ → keep the layers full-rank with
   **identity paths**, not constraints.
2. Butterflies are very sensitive to input conditioning: A1 (input-mean centring + warmup)
   gave +0.06–0.07, far more than its dead-unit mechanism predicted.
3. A butterfly is a **frequency-shaped operator**: stage k mixes elements 2^k apart, a
   local-to-global, frequency-specific dilated stack; NSNet2 ignores this (flattens 257 bins,
   pads to 512, no frequency axis after fc_in).
4. More butterfly factors (nblocks 2) helped +0.03; 400/600 → 512 was free.

## Arms (stage 1)

Shared: 256 bins (Nyquist gain = bin 255's, so fc_in is an exact 256-point butterfly), A1
(frozen per-bin input mean, 200-step warmup), lr 3e-3 ×0.99/epoch, batch 256, MP-SENet loss
+ MetricGAN, seed 1234. Engine pairs/frame = butterfly pair-issue cycles (~80 % of engine
time; 36.6K ≈ 0.43 ms at 85 MHz, budget 16 ms).

| arm | idea | change vs R | params | engine pairs/frame | hypothesis |
|---|---|---|---:|---:|---|
| `ba_R` | reference | NSNet2 topology on butterflies, 512 wide | 154k | 36.6k | — |
| `ba_R2b` | capacity control | nblocks 2 everywhere | 301k | 73.2k | more factors, no new structure (the bar for A and F) |
| `ba_A_res` | A | fc1/fc2 → 3 residual blocks `x + α·B₂(ReLU(BN(B₁x)))`, α = 0 at init | 196k | 45.8k | identity paths keep depth full-rank → beats R and R2b |
| `ba_B_cep` | B | input `[x ; DCT x]`, output `s + iDCT(q)` | 156k | 36.9k + 2 fixed DCTs | a cepstral view lets the net emit envelope / comb gains cheaply |
| `ba_C_lru` | C | GRU → LRU (complex diagonal recurrence, butterfly B/C), ReLU between layers | 117k | 27.4k | recurrence needs no recurrent matrix |
| `ba_C_mingru` | C (engine-native) | GRU → minGRU, ReLU between layers | 77k | 18.2k | same, with the engine's existing Hadamard update |
| `ba_D_grid` | D | 4 ch × 256 bins kept through the net; freq strides spread over layers, channel strides in each; per-bin GRU over channels | 150k | 29.7k | matching the butterfly's structure to the frequency axis beats flattening |
| `ba_E_unet` | E | D + skips between encoder/decoder levels | 150k | 29.7k | multi-resolution skips add to D |
| `ba_F_wide` | F | hidden 2048 | 740k | 177k | width at n log n cost pays |

Engine mapping notes (to check with the engine owner before deploying a winner):
- A: B₁ with BN folded + ReLU epilogue; B₂ with the residual as the epilogue seed (needs the
  seed at B₂'s accumulator scale). α folds into B₂.
- B: the DCT-II is fixed and exact. **A real butterfly cannot hold it**: fitting a 256-point
  butterfly (1–3 blocks, either stride order, with Makhoul input reorder and/or bit-reversed
  output) leaves ≥ 48 % relative Frobenius error. The engine computes it with its FFT pair
  words (Makhoul reorder at the feature write, a post-twiddle on the Hadamard unit).
- C-minGRU: one butterfly task with 2 stacks (sigmoid LUT on z, RAW on h̃) + the existing
  Hadamard op. C-LRU: the complex diagonal recurrence is 4 real Hadamard products per step.
- D/E: stride-subset tasks on n = 1024 (the packer must emit an explicit stride list; the RTL
  already accepts any order). Skip adds are epilogue seeds.
- F: 2048-point tasks (RTL tested to 1024; ping-pong banks 2 × n × 16 bit).

## Protocol (pre-registered)

**Stage 1 — screen.** All nine arms, seed 1234, 100 epochs, validation every 10 epochs on the
full 824-utterance test split. No resume: an arm that dies is rerun from scratch.
- Primary metric: **last-3** = mean of the last three validations. Secondary: best.
  *Correction (2026-10-02, before any reference result existed):* validation runs every 450 steps and
  100 epochs end at step 4,499, so there is no epoch-100 validation; last-3 is epochs **70, 80, 90**.
- Reference noise: an identical-config, identical-seed rerun differed by ≤ 0.02 per validation
  between epochs 40 and 90 (0.06 at epoch 10), so single-seed differences < ~0.03 are not
  readable.
- **Advance** to stage 2 if last-3 ≥ R − 0.01. **Promising** if ≥ R + 0.03.
- Report every arm on PESQ vs engine pairs/frame and params.

**Stage 2 — confirm.** Advanced arms + R, 200 epochs, seeds 1234 and 2345. An idea **helps**
if its 2-seed mean last-5 is ≥ R + 0.02 and each seed beats R's run with the same seed.
Combinations of winners (e.g. A + B) are stage 3.

## Caveats

- Validation is on the test split, so "best" is a best-of-N on the eval set; last-3/last-5
  are the less biased numbers.
- Dropping Nyquist changes R vs the `butterfly-a1` models slightly; R is the reference here,
  not the earlier wave.
- 100 epochs ranks designs; it does not give final numbers (the clean butterfly control set
  its best at epoch 180).

## Bugs found while building this (in the user's libraries)

- **gru_qat Triton butterfly GRU races.** Same input, same weights: outputs differ run to run
  by up to ~0.9 at batch 64, sometimes at batch 2–8, with or without grad. Replaced here by
  `ButterflyGRU` (fused projections on torch_structured's butterfly, input projection hoisted):
  matches `StructuredGRU` to 1e-6, deterministic, ~2× faster than it.
- **torch_structured Triton butterfly grid limit.** `grid = (n_row_tiles, batch_size * nstacks)`;
  CUDA caps grid y at 65535, so > 2 stacks at the training batch (32,256 rows) fails with
  "invalid argument". Worked around by splitting rows (`bfly()`); the fix upstream is to put
  `batch * nstacks` on grid x or loop over grid-y chunks.
- **Kernel side-table leak** (fixed on `butterfly-a1`, commit cafa798): the Triton backward
  registers every launch in PyTorch's `kernel_side_table`; ~8 KiB host RAM per call,
  ~4 MiB/step. The trainer clears it each step.
- The Triton butterfly forward itself is exact: 1e-7 vs the pure-PyTorch backend, deterministic,
  at every shape used here.
