# NOW

**Last updated:** 2026-08-09
**Update this file at the end of every task.** If it is more than ~30 days
stale, `tools/research_lint.py` warns.

## Current objective

Two tracks, in priority order.

1. **Efficiency on STM32N6** — the LiSenNet `nc24` conv variant is deployed and
   measured on silicon in both graph shapes. The remaining cost is not the NPU:
   for ConvFSENet the per-epoch profile puts ~70 % of the frame budget on the
   Cortex-M55 software share, dominated by 9 `Gather` ops (the dilation
   tap-select), not by the float state round-trip originally assumed. See
   `deploy/stm32n6/TODO.md` (Lever 2).
2. **Dynamic / budget-conditioned speech enhancement** — the next research
   question, not yet started. Gate order is fixed and non-negotiable:
   oracle opportunity ceiling → compute-matched statics → router → hardware
   truth. Cards: `dynse-oracle-001` (the ceiling — must clear first) and
   `dynse-router-002` (blocked on it). Both `proposed`, awaiting critique.

## State of the world (one line each, all `[measured]` unless marked)

- Three model families ship: NSNet2 (structured), ConvFSENet, LiSenNet.
- Best streaming quality on N6 silicon: LiSenNet `nc24` streaming, int8 PESQ
  2.963, 2.791 ms/frame, RTF 0.174, fully on-chip.
- Cross-family perceptual scores (PESQ/DNSMOS/NISQA/SCOREQ) exist for every
  published checkpoint, with per-utterance scores committed.
- Structured NSNet2: genuine two-factor Monarch beats block-diagonal by
  +0.011…+0.038 FP32 PESQ, at a parameter cost. Both are int8-loss-free.
- `[measured, negative]` Quality saturates: ~10× parameters across three
  structure families spans only ~2.81–2.88 PESQ, and is not monotonic in capacity.

## Immediate next actions

- [ ] Get `dynse-oracle-001` critiqued by a sceptic pass; either accept or refute
      it before any GPU time is spent. The capacity-saturation result is a strong
      prior that it will be refuted — that is a perfectly good outcome.
- [ ] Backfill `research/experiments/` for the runs that currently exist only as
      markdown tables — see the `evidence_gap` entries in `CLAIMS.yaml`. Highest
      value: the NSNet2 int8 re-measurement and the N6 on-board timings.
- [ ] Wire `tools/run_manifest.py` into `run_sweep.sh` and the `stedgeai`
      measurement path so new runs produce manifests without being asked.
- [ ] Land the first agent-eval baseline (`evals/agent/`) so instruction changes
      can be measured instead of guessed at.

## Explicitly not being worked on

- Multi-agent orchestration, autonomous idea loops, automatic large sweeps.
  Revisit only after the eval baseline exists.
- Migration to Hydra / DVC / a vector store. JSON configs + HF artifacts work.
- Any novelty claim. Human-gated.
