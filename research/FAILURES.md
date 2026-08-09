# Failure ledger

Append-only. Never delete an entry because it is embarrassing — the entries most
worth keeping are the ones that would otherwise be re-proposed in six months by
someone (or something) with no memory of them.

Format: what was tried, what happened, **why**, and what would have caught it earlier.

---

## einsum-int8 — "int8" numbers measured on FP32 weights

**Period:** until 2026-07-11. **Blast radius:** every structured-NSNet2 int8 PESQ
number published before that date; general-purpose streaming ONNX only.

The structure-preserving export lowers each block-diagonal / Monarch matmul to an
`Einsum`. onnxruntime ships no QDQ handler for `Einsum`, so `quantize_static`
silently **skipped those nodes**, quantizing only activations and the residual
dense MatMuls. The models were labelled int8 and were not.

**Why it survived so long:** the numbers looked *right*. Block-diagonal int8 PESQ
really is loss-free, so the corrected re-measurement broadly reproduced the
original table — the claim was accidentally true while entirely unevidenced. A
plausible result is the most dangerous kind of unaudited result.

**Fix:** `nsnet2/qdq_einsum_quantizer.py` registers `QDQRegistry["Einsum"]`
(per-channel, axis=1), and `nsnet2/quant.py` now **fails** if any `Einsum` operand
is still a raw FLOAT initializer.

**Would have caught it earlier:** a post-export assertion on the dtype of every
weight initializer — cheap, deterministic, and independent of whether the PESQ
looked sensible. Generalised into `AGENTS.md` rule 9.

---

## monarch-mislabel — block-diagonal published as Monarch

**Period:** until 2026-06. **Blast radius:** run names, config names, README,
`RESULTS_NSNET2.md`, and the Hugging Face model card.

Every run named `monarch_*` was a **single block-diagonal factor** — no
cross-block mixing, no permutation, not Monarch. The name came from an early
intent and was never re-checked against what the layer factory actually built.

**Consequence:** an external reader comparing against the Monarch paper would
have been comparing against a different construction with a different parameter
count.

**Fix:** renamed to `blockdiag_*` throughout; genuine two-factor Monarch trained
and measured separately; corrected checkpoints re-published; a correction notice
left in place at the top of `RESULTS_NSNET2.md` rather than a silent edit.

**Would have caught it earlier:** asserting the structural property (does the
weight have cross-block mixing?) rather than trusting the config key's name.
Generalised into `AGENTS.md` rule 15.

---

## npu-blocker-4-misdiagnosis — the symptom was not the mechanism

**Period:** 2026-06 to 2026-07-03.

The LiSenNet streaming graph segfaulted `atonn` (signo=11). The first diagnosis
was "the 17-tensor FIFO streaming state I/O class is not supported". Five
bisection rounds later — encoder section → single DSConv stage → feature
permutations → 19-node repro → single-delta flips — the actual trigger was
`Pad` with an **empty** optional `constant_value` input, and state I/O was
innocent. The trigger also needs the dual-branch sub-band context, which is why
the windowed / whole-graph proofs never caught it.

**Why the wrong diagnosis was attractive:** state I/O was the newest, most
complicated thing in the graph, so it was the natural suspect. Novelty is not
evidence of causation.

**Would have caught it earlier:** bisecting to a minimal repro *before* forming a
narrative. The fix is one line (`_strip_empty_pad_value_inputs`).

---

## capacity-scaling — more parameters did not buy quality

**Status:** a negative result, recorded as a result.

Scaling structured NSNet2 across ~10x parameters (0.36 M → 3.64 M) and three
structure families moved FP32 PESQ only within ~2.81–2.88, non-monotonically:
0.553 M `monarch_8` (2.861) beats both 1.099 M `monarch_full` (2.838) and
2.379 M `monarch_fc` (2.843).

**Why it belongs here:** it closes a direction (buy quality with capacity in this
regime) and it is a standing prior against any future proposal whose mechanism
is "spend more compute where it matters" — including dynamic routing. See
`research/hypotheses/dynse-oracle-001.yaml`, which exists specifically to test
that prior before any router is trained.

---

## planning-drift — the agent memory went stale and nobody noticed

**Period:** 2026-04-27 to 2026-08-09.

`.planning/PROJECT.md` described the NSNet2 int8-quantization milestone as the
current objective long after ConvFSENet, LiSenNet, the perceptual-metric suite
and the STM32N6 deployment had landed. `.planning/codebase/TESTING.md` asserted
"there is no `tests/` directory" while `tests/` held 28 files. An agent reading
those files as current state would have planned against a world that stopped
existing months earlier.

**Why:** the documents mixed three things — stable rules, current state, and
history — in one place, with no owner and no freshness check. Nothing was wrong
when written; nothing forced an update afterwards.

**Fix:** `AGENTS.md` (rules) / `research/NOW.md` (current) / `research/` +
`.planning/` (history) separation; `.planning/` explicitly frozen; a freshness
check in `tools/research_lint.py` running in CI.
