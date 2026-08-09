# Schema reference

Checked mechanically by `tools/research_lint.py`. If you change a schema here,
change the linter in the same commit.

## Hypothesis card — `research/hypotheses/<id>.yaml`

The card is a **contract written before the result exists**. Its job is to stop
a compelling post-hoc explanation from being mistaken for a finding.

| field              | required            | meaning |
| ------------------ | ------------------- | ------- |
| `id`               | always              | matches the filename stem; `^[a-z0-9]+(-[a-z0-9]+)*$` |
| `question`         | always              | one sentence, answerable yes/no |
| `status`           | always              | `proposed` → `critiqued` → `accepted` → `running` → `complete` / `refuted` / `abandoned` |
| `created`          | always              | `YYYY-MM-DD` |
| `mechanism`        | always              | *why* it should work. "It might help" is not a mechanism. |
| `prediction`       | always              | the expected effect, with a number |
| `falsify_if`       | from `accepted`     | the result that kills the idea. Must be reachable by the experiment as specified. |
| `accept_if`        | from `accepted`     | the result that would justify a claim |
| `baselines`        | from `accepted`     | ≥1, and at least one **compute-matched** where a cost claim is involved |
| `controls`         | from `accepted`     | what is held equal (data, optimisation budget, calibration set, box) |
| `primary_metric`   | from `accepted`     | exactly one |
| `secondary_metrics`| optional            | list |
| `split`            | from `accepted`     | which split selects things. **Never `test`.** |
| `seeds`            | from `accepted`     | ≥1; ≥3 for any quality comparison inside the saturation band |
| `budget`           | from `accepted`     | e.g. `12_gpu_hours`, `1_board_session` — an approval gate, not a wish |
| `critique`         | from `accepted`     | the sceptic pass: confounders, prior art, simpler explanations |
| `experiments`      | from `running`      | ids under `research/experiments/`; each must have a manifest |
| `outcome`          | from `complete`     | `accepted_hypothesis` / `refuted` / `inconclusive`, plus one paragraph |
| `depends_on`       | optional            | card ids that must complete first. This is how a gate order (decision 0003) is enforced rather than merely written down. |

Copy `research/hypotheses/TEMPLATE.yaml`.

Two rules the linter enforces because they are the ones that get skipped:

- `split: test` is rejected outright, under every spelling
  (`Test`, `TEST`, `vbd_test`, `test-split`, …);
- a card at `accepted` or beyond must have a `falsify_if` that is not a bare
  comparator flip of `accept_if`.

Be clear about the limit of the second one: it is a tripwire for the laziest
form ("accept if ≥ 0.15" / "falsify if ≤ 0.15"), not a judgement that a
falsifier is any good. A real falsifier names a **rival explanation** — "a
compute-matched static matches it at the same measured cost" — and no regex can
check for that. The sceptic pass in `.claude/skills/design-experiment/` is what
checks it, and a card that has not had one is not ready to be `accepted`.

## Claim — an entry in `research/CLAIMS.yaml`

| field            | required | meaning |
| ---------------- | -------- | ------- |
| `id`             | yes      | stable; papers cite these |
| `statement`      | yes      | what we assert, in one sentence |
| `kind`           | yes      | `measured` / `derived` / `cited` / `hypothesis` |
| `status`         | yes      | see below |
| `evidence`       | yes      | repo-relative paths that a reader can open. `[]` only when `status: unverified` |
| `counterevidence`| yes      | `[]` if none — an empty list is an assertion that you looked |
| `scope`          | yes      | what the claim does **not** cover (host-only, one box, one split, …) |
| `evidence_gap`   | no       | present when the evidence is prose rather than a generated artifact |

`status` values:

| status         | means |
| -------------- | ----- |
| `measured`     | end-to-end measurement on the thing being claimed about |
| `compute-only` | shown in FLOPs/params/host-ONNX; **no hardware claim licensed** |
| `pilot`        | one seed / a subset; not publishable |
| `corrected`    | previously published wrong; the correction is recorded next to the original |
| `unverified`   | asserted somewhere in the repo, not yet backed |
| `refuted`      | kept deliberately, so it is not re-proposed |

## Experiment manifest — `research/experiments/<id>/manifest.json`

Emitted by `tools/run_manifest.py`, never hand-written. Immutable once
committed: a re-run gets a new id. Records git SHA + dirty state, the exact
command, the resolved config, dataset repo/revision/split, seeds, environment
hash, checkpoint and calibration-set identity, toolchain versions (compiler,
firmware, board, measurement tool), the host it ran on, and the hypothesis/claim
ids it serves.
