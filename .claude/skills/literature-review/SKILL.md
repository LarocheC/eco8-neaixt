---
name: literature-review
description: Survey prior art for a proposed direction. Use when writing a related-work section, checking whether an idea already exists, or grounding a design choice in published results.
---

# Literature review

This is the one task in this repo where breadth-first parallel search genuinely
pays: many independent lookups, shallow coupling, cheap to verify. It is also
the task where an agent most easily produces confident fiction, because a
plausible citation looks exactly like a real one.

## Rules

1. **Every claim about a paper comes from the paper.** Not from an abstract, not
   from another paper's summary of it, not from memory. If you have not opened
   it, mark it `[unverified-citation]`.
2. **A citation carries its numbers with it.** "X reports PESQ 3.07 on VBD
   (Table 2)" — with the table. A claim you cannot locate in the source is a
   claim the source may not make.
3. **Check the setup matches before comparing.** VoiceBank-DEMAND at 16 kHz vs
   48 kHz, full test split vs a subset, offline vs causal, with or without
   phase reconstruction — these routinely differ by more than the effects under
   discussion. State the mismatch or do not compare.
4. **Retrieval is a deterministic tool, not a reasoning task.** Prefer an exact
   search (arXiv id, DOI, exact title, the authors' repo) over asking a model to
   recall. Where a deterministic retrieval layer exists, use it — agents are
   markedly less reliable at assembling a corpus by reasoning than at reading one
   that was assembled deterministically.
5. **Never assess novelty.** Report what exists and how close it is. "This
   appears novel" is a human-gated conclusion (`AGENTS.md`).

## Output

One entry per paper:

```
[cited] Yan et al., LiSenNet, arXiv:2409.13285
  claim used for : lightweight sub-band + dual-path SE, ~37 K params
  their numbers  : PESQ ~3.07 on VBD test (paper Table 2)
  their setup    : 16 kHz, full test split, Griffin-Lim phase
  matches ours?  : yes -- our port reaches 3.006, within ~0.06
  verified       : yes (paper read) / no (abstract only) / unverified
```

End with what the literature says the project should stop doing, not only what
it permits. A review that finds no obstacles has usually not looked for any.

## Where the repo already has prior art

`README.md` "Acknowledgements" and the reference list at the top of each
`RESULTS_*.md` — check those before searching, and reuse their setups.
