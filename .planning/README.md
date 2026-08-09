# .planning/ — FROZEN ARCHIVE (2026-04-27)

**Nothing in this directory describes the current state of the repository.**

These documents were written when the project was still `sparse-nsnet2` and the
active milestone was int8 ONNX export for NSNet2. Since then the repo grew two
more model families (ConvFSENet, LiSenNet), a perceptual-metric suite, a
committed benchmark corpus and a full STM32N6 deployment path.

Known-false statements preserved here (deliberately — see `AGENTS.md` rule 5,
correct rather than overwrite):

- `PROJECT.md` presents the NSNet2 quantization milestone as the current goal.
- `codebase/TESTING.md` says "there is no `tests/` directory"; there are now 28
  files in `tests/`, with pytest configured in `pyproject.toml`.
- `codebase/STRUCTURE.md` describes a root-level `train.py` / `models/` layout
  that no longer exists.

## Where to look instead

| you want                  | read                                        |
| ------------------------- | ------------------------------------------- |
| rules for agents          | [`AGENTS.md`](../AGENTS.md)                 |
| what is being worked on   | [`research/NOW.md`](../research/NOW.md)     |
| what we claim, and why    | [`research/CLAIMS.yaml`](../research/CLAIMS.yaml) |
| what did not work         | [`research/FAILURES.md`](../research/FAILURES.md) |
| how the code is organised | [`README.md`](../README.md) and the code    |

## Why it is kept

The rationale is still good even where the facts have moved: the Key Decisions
table in `PROJECT.md`, the pitfall write-ups, and the reasoning about why the
streaming-parity gate had to come before quantization. Rewriting it to match
today would destroy the record of what was believed when the decisions were made.

`tools/research_lint.py` fails if any file here loses its ARCHIVE banner.
