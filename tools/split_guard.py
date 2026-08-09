"""One definition of "this split must not be used to select anything".

This existed twice and the two copies disagreed. `run_manifest.py` compared
``split == "test"`` exactly; `research_lint.py` lowercased and knew a couple of
aliases. So `--split Test` walked straight past the tool while the linter would
have rejected the same value on a card — and the eval suite's flagship
deterministic grader tested only the one spelling that happened to work.

A guard with four spellings needs a test with four spellings, and a rule with
two implementations needs one implementation. This is that implementation.

Stdlib only: `run_manifest.py` has to run inside a board-measurement venv or a
CI container with nothing installed.
"""

from __future__ import annotations

# Spellings that mean "the held-out test set" in this project. Selecting
# anything on these — architecture, checkpoint, threshold, calibration set,
# routing policy — destroys the only unbiased estimate we have (AGENTS.md
# rule 7). Reporting one final number on them is fine and is not this check.
#
# Keep this list tight. `train_holdout`, `val`, `dev` and anything else are
# legitimate selection splits and must stay allowed.
FORBIDDEN_SELECTION_SPLITS = frozenset({
    "test",
    "tests",
    "test_set",
    "test_split",
    "testing",
    "vbd_test",
    "vbd_demand_test",
    "voicebank_test",
    "voicebank_demand_test",
    "eval_test",
})


def normalise_split(value: object) -> str:
    """Fold the spellings that differ only cosmetically.

    Case, surrounding whitespace, quotes, and hyphen-vs-underscore are all
    cosmetic. Anything else is a genuinely different name and stays different.
    """
    if value is None:
        return ""
    text = str(value).strip().strip("\"'").lower()
    return text.replace("-", "_").replace(" ", "_")


def is_forbidden_selection_split(value: object) -> bool:
    """True if `value` names the test split under any accepted spelling."""
    return normalise_split(value) in FORBIDDEN_SELECTION_SPLITS
