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
- **Not available**: data-dependent normalisation (LayerNorm). BatchNorm is fine at inference (a
  fixed affine that folds into the butterfly next to it).
- *Correction (2026-10-03, from the literature review):* this section first said permutations were
  not available. They are, as butterfly tasks: a Beneš network (BB* stride order, ~2·log₂n − 1
  stages) realises **any** permutation with twiddles that are only I or the 2×2 swap — zero
  multiplies, ~1.9K pairs at n = 256. To confirm with the engine owner: the 0/1 twiddle encoding
  and a RAW epilogue on routing stages so requant passes values through exactly.

## Evidence the designs start from (branch `butterfly-a1`)

1. Long butterfly products collapse: the published layers are near-singular (fc_in cond
   3e5–2e8 vs 24 dense; 99 %-energy rank of fc_out 14–89 of 257 vs 104). Forcing
   orthogonality fixed conditioning but cost ~0.07 PESQ → keep the layers full-rank with
   **identity paths**, not constraints.
   *Corrections (2026-10-03):* (i) the collapse is present **at initialisation** — torch_structured's
   randn twiddles (N(0, ½) per entry) compose to cond ~1e10 and a 99 %-energy rank of ~94/512 for a
   512-point butterfly (cond ~1e12, rank 27–43 with nblocks 2); training *raises* the rank (to
   114–123, resp. 60–64, measured on the stage-1 checkpoints), while ortho-initialised layers
   (W_hh) start at cond 1.0 and drift only to 25–51. Initialisation is the main lever; AdamW's
   default weight decay (0.01, applied to the twiddles in every run so far) is at most secondary.
   (ii) the orthogonality penalty forced orthogonal factors **without diagonal scalings**; an
   orthogonal-butterfly + diagonal hierarchy (Kaleidoscope's OBB) is as expressive as BB* and was
   not tested, so "constraints cost 0.07" is not established.
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
- B: the DCT-II is fixed and exact. A real butterfly of 1–3 blocks cannot hold it: fitting a
  256-point butterfly (either stride order, with Makhoul input reorder and/or bit-reversed output)
  leaves ≥ 48 % relative Frobenius error. *Correction (2026-10-03):* that is a depth / fixed-
  permutation limit, not a butterfly limit — butterfly–permutation products (BP)² contain the DCT,
  DST and convolution exactly (Dao et al. 2019, Prop. 1), and with Beneš routing (above) an exact
  real DCT-II is reachable on the engine at roughly 5–6·log₂n stages. Arm B may have lost for its
  side-path design rather than for the transform. The engine computes it with its FFT pair
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

## Stage 1 results (2026-10-02, all nine arms, seed 1234, 100 epochs, no crash or resume)

Last-3 = mean of the epoch 70 / 80 / 90 validations (full 824-utterance test split). Pareto =
not beaten by an arm that is both better and cheaper on the engine.

| arm | last-3 | Δ vs R | best | params | engine pairs/frame | verdict (pre-registered rule) | Pareto |
|---|---:|---:|---:|---:|---:|---|:---:|
| `ba_R2b` | **2.818** | +0.062 | 2.842 | 301k | 73.2k | promising | ✓ |
| `ba_A_res` | 2.808 | +0.052 | 2.830 | 196k | 45.8k | promising | ✓ |
| `ba_F_wide` | 2.796 | +0.041 | 2.810 | 740k | 177.2k | promising | ✗ (R2b, A) |
| `ba_D_grid` | 2.783 | +0.027 | 2.792 | 150k | 29.7k | advance | ✓ |
| `ba_E_unet` | 2.768 | +0.012 | 2.776 | 150k | 29.7k | advance | ✗ (D) |
| `ba_R` | 2.756 | — | 2.759 | 154k | 36.6k | reference | ✗ (D) |
| `ba_B_cep` | 2.744 | −0.012 | 2.761 | 156k | 36.9k | drop (< R − 0.01) | ✗ (D) |
| `ba_C_lru` | 2.741 | −0.015 | 2.745 | 116k | 27.4k | drop | ✓ |
| `ba_C_mingru` | 2.733 | −0.023 | 2.764 | 77k | 18.2k | drop | ✓ |

Readings (single seed; differences < ~0.03 are not resolved):
- **Rule 1 holds**: identity paths (A, +0.052) and more butterfly factors (R2b, +0.062) are the two
  big wins; they are within noise of each other, A at 0.63× R2b's engine cost.
- **Rule 3 partly**: keeping a frequency axis (D) beats R (+0.027) at 0.81× R's engine cost, the
  best PESQ per engine cycle. Multi-resolution skips (E) did not add to it (−0.015 vs D).
- Width (F) helps but is dominated by A and R2b at 2.4–3.9× their engine cost.
- Cepstral side path (B) and diagonal recurrences (LRU, minGRU) do not beat the butterfly GRU.

## Stage 2 protocol (pre-registered 2026-10-02, before any stage-2 run)

**Purpose.** Confirm the stage-1 winners at full length with two seeds, and pilot the two
combinations of winners.

**Scope decided with Clément (2026-10-02).** The stage-1 rule advances five arms (R2b, A, F, D,
E). Stage 2 confirms four of them and drops two on cost/benefit, not on the rule:
- `ba_F_wide` dropped: beaten by both R2b and A at 2.4–3.9× their engine cost (Pareto-dominated).
- `ba_E_unet` dropped: −0.015 vs the D it extends; skips added nothing.
Their 2-seed configs exist (`configs/s2_F_wide_s*`, `configs/s2_E_unet_s*`) and can be appended
to the queue without disturbing the rest.

**Runs (10).** 200 epochs, everything else exactly as stage 1 (recipe, 256 bins, A1, validation
every 10 epochs on the full test split, no resume — a run that dies is rerun from scratch).

| run | arm | seeds | role |
|---|---|---|---|
| `s2_R_s*` | R | 1234, 2345 | reference |
| `s2_R2b_s*` | R2b (nblocks 2) | 1234, 2345 | confirm |
| `s2_A_res_s*` | A (residual blocks) | 1234, 2345 | confirm |
| `s2_D_grid_s*` | D (frequency grid) | 1234, 2345 | confirm |
| `s2_A2b_s1234` | **A + R2b**: residual blocks, nblocks 2 everywhere (380k params, 91.6k pairs/frame) | 1234 | pilot |
| `s2_Dres_s1234` | **D + identity paths**: each square grid level is `x + ReLU(BN(B x))`, no extra butterflies (150k, 29.7k pairs/frame) | 1234 | pilot |

Stage-1 runs are not reused: they stopped at 100 epochs and the schedule (×0.99/epoch) makes a
200-epoch run a different trajectory after epoch 100.

**Metrics.** Validations exist at epochs 10…190 (none at 200; see the stage-1 correction).
- Primary: **last-5** = mean of the epoch 150–190 validations.
- Secondary: best (a best-of-19 on the test split, biased up), params, engine pairs/frame.
- Same-seed noise floor measured here: ≤ 0.02 per validation late in training.

**Decision rules.**
1. **Confirmed improvement**: an arm's 2-seed mean last-5 ≥ R's 2-seed mean last-5 + 0.02,
   **and** for each seed its last-5 > R's last-5 with that seed.
2. **Ranking among confirmed arms** is by PESQ at matched engine cost (pairs/frame); two arms
   within 0.02 of each other are reported as tied, and the cheaper one is preferred.
3. **Pilots** (one seed): a pilot earns a 2-seed confirmation (stage 3) if its last-5 ≥ the
   better of its parents' seed-1234 last-5 + 0.02 (A2b vs A and R2b; Dres vs D). Otherwise
   the combination is reported as not additive at this precision and stopped.
4. No rule is changed after a stage-2 result exists; corrections that are needed for a rule to
   be measurable are recorded with their date, as in stage 1.

**Scheduling.** First-fit on measured per-arm training peaks (+10 %) under 22.5 GiB, at most 4
jobs; the reference R goes first so comparisons are available early. Throughput in stage 1 was
~62 run-epochs/hour, so 10 × 200 epochs ≈ 32 h (F and E would add ~15 h).

**Scope change (2026-10-02 19:40, Clément):** the F and E confirmations are added back
(`s2_E_unet_s1234/2345`, `s2_F_wide_s1234/2345`), same rules. They are queued behind the ten runs
above (a second driver sharing the first one's memory ledger, started once the first has launched
everything), so the core results arrive on the original schedule; F cannot share the GPU with E
(14.1 + 9.7 GiB > the cap), which puts the end of the wave at ~64 h after launch.

**Noise note (2026-10-02, observed, rules unchanged):** `s2_R_s1234` and stage-1 `ba_R` are the
same config and seed, yet differ by +0.022 / +0.038 / +0.046 at epochs 60 / 80 / 90 — larger than the
≤ 0.02 measured on the butterfly-a1 control rerun. Training is not reproducible run to run (the
Triton butterfly backward accumulates the twiddle gradient with atomics), and trajectories diverge.
Stage-1 single-seed deltas below ~0.05 should be read with that in mind; the 2-seed rule of stage 2
is the guard.
