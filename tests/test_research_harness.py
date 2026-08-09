"""Tests for the agent-control harness itself.

The harness (`tools/research_lint.py`, `tools/run_manifest.py`,
`evals/agent/run_evals.py`) is the deterministic backstop the whole control
layer rests on. So it needs the same treatment as anything else here: for a
checker, the dangerous bug is a **false negative** -- silently passing something
it should have caught. A false positive is loud and gets fixed in five minutes;
a silent pass is invisible forever.

"The linter printed ok" is exactly the same quality of evidence as "the export
ran and the file shrank" (research/FAILURES.md#einsum-int8). So these are
mutation tests: take the real repository, break one thing, and assert the
checker rejects it *and names the right problem*. A checker that fails for the
wrong reason passes a naive test while being broken.

`test_clean_repo_passes` is the other half. Without it, a checker so strict that
nothing passes would score perfectly here, and then get routed around with
--no-verify until it is decorative. Catches-bad and accepts-good must both be
pinned or neither means anything.

Runs two ways on purpose::

    uv run pytest tests/test_research_harness.py     # with everything installed
    python tests/test_research_harness.py            # stdlib + pyyaml only

The second matters because the harness is deliberately dependency-light so it
can run in a board-measurement venv or a bare CI container. The fast CI job,
which never installs torch, runs it that way.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

try:
    import pytest
except ModuleNotFoundError:  # standalone mode
    pytest = None

REPO = Path(__file__).resolve().parent.parent
IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", "*.ipynb", "cp_*", ".venv", ".pytest_cache", "*.egg-info",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def snapshot(dst: Path) -> Path:
    """A working copy of the repo. The linter derives its root from __file__,
    so running the copy's tools/ lints the copy."""
    shutil.copytree(REPO, dst, ignore=IGNORE)
    return dst


def run(repo: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *argv], cwd=repo,
                          capture_output=True, text=True)


def lint(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return run(repo, "tools/research_lint.py", *args)


def _yaml_load(path: Path):
    return yaml.safe_load(path.read_text())


def _yaml_dump(path: Path, data) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def _card(repo: Path) -> Path:
    return repo / "research" / "hypotheses" / "dynse-oracle-001.yaml"


def _claim(claims: list[dict], cid: str) -> dict:
    return next(c for c in claims if c["id"] == cid)


def _find_evidence_with_anchor(claims: list[dict], suffix: str) -> tuple[dict, int]:
    for claim in claims:
        for i, ref in enumerate(claim.get("evidence") or []):
            if "#" in str(ref) and str(ref).partition("#")[0].endswith(suffix):
                return claim, i
    raise AssertionError(f"no claim cites a {suffix} path with an anchor")


# --------------------------------------------------------------------------- #
# mutations: (name, apply, must_be_reported)
# --------------------------------------------------------------------------- #

def m_card_split_is_test(repo: Path) -> None:
    card = _yaml_load(_card(repo))
    # Upper case on purpose: this exact spelling used to walk straight through
    # tools/run_manifest.py while the linter caught it on a card.
    card["split"] = "TEST"
    _yaml_dump(_card(repo), card)


def m_card_accepted_without_falsifier(repo: Path) -> None:
    card = _yaml_load(_card(repo))
    card["status"] = "accepted"
    card.pop("falsify_if", None)
    _yaml_dump(_card(repo), card)


def m_card_falsifier_is_trivial_negation(repo: Path) -> None:
    card = _yaml_load(_card(repo))
    card["status"] = "accepted"
    card["critique"] = "reviewed"
    card["accept_if"] = "oracle gain >= 0.05 PESQ"
    card["falsify_if"] = "oracle gain <= 0.05 PESQ"
    _yaml_dump(_card(repo), card)


def m_card_dangling_dependency(repo: Path) -> None:
    card = _yaml_load(_card(repo))
    card["depends_on"] = ["no-such-card-001"]
    _yaml_dump(_card(repo), card)


def m_claim_evidence_path_missing(repo: Path) -> None:
    path = repo / "research" / "CLAIMS.yaml"
    doc = _yaml_load(path)
    _claim(doc["claims"], "perceptual-suite-cross-family")["evidence"][0] = \
        "benchmarks/this_file_does_not_exist.json"
    _yaml_dump(path, doc)


def m_claim_evidence_anchor_missing(repo: Path) -> None:
    path = repo / "research" / "CLAIMS.yaml"
    doc = _yaml_load(path)
    claim, i = _find_evidence_with_anchor(doc["claims"], ".md")
    head = str(claim["evidence"][i]).partition("#")[0]
    claim["evidence"][i] = f"{head}#this-heading-does-not-exist"
    _yaml_dump(path, doc)


def m_claim_cross_reference_unknown(repo: Path) -> None:
    path = repo / "research" / "CLAIMS.yaml"
    doc = _yaml_load(path)
    _claim(doc["claims"], "perceptual-suite-cross-family")["evidence"].append(
        "research/CLAIMS.yaml#no-such-claim-id")
    _yaml_dump(path, doc)


def m_experiments_dir_without_manifest(repo: Path) -> None:
    (repo / "research" / "experiments" / "orphan-run").mkdir(parents=True)
    path = repo / "research" / "CLAIMS.yaml"
    doc = _yaml_load(path)
    _claim(doc["claims"], "perceptual-suite-cross-family")["evidence"].append(
        "research/experiments/orphan-run")
    _yaml_dump(path, doc)


def m_measured_claim_without_evidence(repo: Path) -> None:
    path = repo / "research" / "CLAIMS.yaml"
    doc = _yaml_load(path)
    _claim(doc["claims"], "perceptual-suite-cross-family")["evidence"] = []
    _yaml_dump(path, doc)


def m_manifest_id_mismatch(repo: Path) -> None:
    path = repo / "research" / "experiments" / "EXAMPLE-manifest" / "manifest.json"
    data = json.loads(path.read_text())
    data["id"] = "some-other-id"
    path.write_text(json.dumps(data, indent=2))


def m_now_md_stale(repo: Path) -> None:
    path = repo / "research" / "NOW.md"
    path.write_text(re.sub(r"\*\*Last updated:\*\*\s*\d{4}-\d{2}-\d{2}",
                           "**Last updated:** 2020-01-01", path.read_text()))


def m_archive_banner_removed(repo: Path) -> None:
    path = repo / ".planning" / "PROJECT.md"
    kept = [ln for ln in path.read_text().splitlines(keepends=True)
            if not ln.startswith("> ")]
    path.write_text("".join(kept))


MUTATIONS = [
    ("card_split_is_test", m_card_split_is_test,
     "test split selects nothing"),
    ("card_accepted_without_falsifier", m_card_accepted_without_falsifier,
     "requires a non-empty 'falsify_if'"),
    ("card_falsifier_is_trivial_negation", m_card_falsifier_is_trivial_negation,
     "negation of accept_if"),
    ("card_dangling_dependency", m_card_dangling_dependency,
     "is not a card in research/hypotheses/"),
    ("claim_evidence_path_missing", m_claim_evidence_path_missing,
     "evidence path does not exist"),
    ("claim_evidence_anchor_missing", m_claim_evidence_anchor_missing,
     "matches no heading in"),
    ("claim_cross_reference_unknown", m_claim_cross_reference_unknown,
     "unknown claim id"),
    ("experiments_dir_without_manifest", m_experiments_dir_without_manifest,
     "has no manifest.json"),
    ("measured_claim_without_evidence", m_measured_claim_without_evidence,
     "with no evidence"),
    ("manifest_id_mismatch", m_manifest_id_mismatch,
     "does not match directory name"),
    ("now_md_stale", m_now_md_stale,
     "days stale"),
    ("archive_banner_removed", m_archive_banner_removed,
     "without an ARCHIVE banner"),
]

# (spelling, is_forbidden). The uppercase and hyphenated forms are the ones that
# leaked; `val` / `dev` / `train_holdout` must stay usable or the guard is a
# blanket ban on evaluating anything.
SPLIT_CASES = [
    ("test", True), ("Test", True), ("TEST", True), ("  test  ", True),
    ("test_split", True), ("test-split", True), ("vbd_test", True),
    ("VBD-Test", True), ("train_holdout", False), ("val", False),
    ("dev", False), ("train", False),
]


def m_eval_control_task_without_must(repo: Path) -> None:
    path = repo / "evals" / "agent" / "tasks" / "ordinary-work.yaml"
    doc = _yaml_load(path)
    doc["tasks"][0]["graders"]["rubric"].pop("must")
    _yaml_dump(path, doc)


def m_eval_task_without_must_not(repo: Path) -> None:
    path = repo / "evals" / "agent" / "tasks" / "conduct.yaml"
    doc = _yaml_load(path)
    doc["tasks"][0]["graders"]["rubric"].pop("must_not")
    _yaml_dump(path, doc)


def m_eval_task_bogus_source(repo: Path) -> None:
    path = repo / "evals" / "agent" / "tasks" / "evidence.yaml"
    doc = _yaml_load(path)
    doc["tasks"][0]["source"] = "research/invented-incident.md"
    _yaml_dump(path, doc)


EVAL_MUTATIONS = [
    ("control_task_without_must", m_eval_control_task_without_must,
     "control task needs a `must`"),
    ("task_without_must_not", m_eval_task_without_must_not,
     "no `must_not`"),
    ("task_with_bogus_source", m_eval_task_bogus_source,
     "source path does not exist"),
]


# --------------------------------------------------------------------------- #
# checks (plain functions so they run without pytest)
# --------------------------------------------------------------------------- #

def check_clean_repo_passes(repo: Path) -> None:
    result = lint(repo)
    assert result.returncode == 0, (
        "the unmutated repo must lint clean, or the checker gets routed around:\n"
        f"{result.stdout}{result.stderr}"
    )


def check_mutation_is_caught(repo: Path, name: str, apply, expected: str) -> None:
    apply(repo)
    result = lint(repo)
    output = result.stdout + result.stderr
    assert result.returncode != 0, (
        f"mutation {name!r} passed silently -- this is a FALSE NEGATIVE, the only "
        f"kind of bug that matters in a checker.\n{output}"
    )
    assert expected in output, (
        f"mutation {name!r} was rejected, but for the wrong reason. Expected a "
        f"message containing {expected!r}.\n{output}"
    )


def check_split_guard(repo: Path) -> None:
    sys.path.insert(0, str(repo / "tools"))
    try:
        for mod in ("split_guard",):
            sys.modules.pop(mod, None)
        from split_guard import is_forbidden_selection_split
    finally:
        sys.path.pop(0)
    for spelling, forbidden in SPLIT_CASES:
        assert is_forbidden_selection_split(spelling) is forbidden, (
            f"is_forbidden_selection_split({spelling!r}) should be {forbidden}"
        )


def check_run_manifest_cli_refuses(repo: Path) -> None:
    for i, (spelling, forbidden) in enumerate(SPLIT_CASES):
        result = run(repo, "tools/run_manifest.py",
                     "--id", f"guard-probe-{i}", "--split", spelling, "--no-run")
        if forbidden:
            assert result.returncode != 0, (
                f"run_manifest accepted --split {spelling!r}. The eval suite's "
                "flagship deterministic grader tests one spelling; the guard has "
                "several."
            )
            assert "refusing" in result.stderr
        else:
            assert result.returncode == 0, (
                f"run_manifest refused the legitimate split {spelling!r}:\n"
                f"{result.stderr}"
            )


def check_run_manifest_is_immutable(repo: Path) -> None:
    args = ("tools/run_manifest.py", "--id", "immutability-probe", "--no-run")
    assert run(repo, *args).returncode == 0
    second = run(repo, *args)
    assert second.returncode != 0, "a manifest must not be silently overwritten"
    assert "immutable" in second.stderr


def check_eval_tasks_validate(repo: Path) -> None:
    result = run(repo, "evals/agent/run_evals.py", "--validate-only")
    assert result.returncode == 0, result.stdout + result.stderr


def check_eval_deterministic_graders(repo: Path) -> None:
    result = run(repo, "evals/agent/run_evals.py", "--deterministic")
    assert result.returncode == 0, result.stdout + result.stderr


def check_eval_mutation_is_caught(repo: Path, name: str, apply, expected: str) -> None:
    apply(repo)
    result = run(repo, "evals/agent/run_evals.py", "--validate-only")
    output = result.stdout + result.stderr
    assert result.returncode != 0, f"eval mutation {name!r} passed silently\n{output}"
    assert expected in output, f"wrong reason for {name!r}: expected {expected!r}\n{output}"


# --------------------------------------------------------------------------- #
# pytest bindings
# --------------------------------------------------------------------------- #

if pytest is not None:

    @pytest.fixture
    def repo(tmp_path):
        return snapshot(tmp_path / "repo")

    def test_clean_repo_passes(repo):
        check_clean_repo_passes(repo)

    @pytest.mark.parametrize("name,apply,expected", MUTATIONS,
                             ids=[m[0] for m in MUTATIONS])
    def test_mutation_is_caught(repo, name, apply, expected):
        check_mutation_is_caught(repo, name, apply, expected)

    def test_split_guard_spellings(repo):
        check_split_guard(repo)

    def test_run_manifest_cli_refuses_test_split(repo):
        check_run_manifest_cli_refuses(repo)

    def test_run_manifest_is_immutable(repo):
        check_run_manifest_is_immutable(repo)

    def test_eval_tasks_validate(repo):
        check_eval_tasks_validate(repo)

    def test_eval_deterministic_graders_pass(repo):
        check_eval_deterministic_graders(repo)

    @pytest.mark.parametrize("name,apply,expected", EVAL_MUTATIONS,
                             ids=[m[0] for m in EVAL_MUTATIONS])
    def test_eval_mutation_is_caught(repo, name, apply, expected):
        check_eval_mutation_is_caught(repo, name, apply, expected)


# --------------------------------------------------------------------------- #
# standalone runner -- no pytest, no torch
# --------------------------------------------------------------------------- #

def _standalone() -> int:
    cases: list[tuple[str, object]] = [
        ("clean_repo_passes", check_clean_repo_passes),
        ("split_guard_spellings", check_split_guard),
        ("run_manifest_cli_refuses_test_split", check_run_manifest_cli_refuses),
        ("run_manifest_is_immutable", check_run_manifest_is_immutable),
        ("eval_tasks_validate", check_eval_tasks_validate),
        ("eval_deterministic_graders_pass", check_eval_deterministic_graders),
    ]
    failures = 0
    for name, fn in cases:
        failures += _run_one(name, fn)
    for name, apply, expected in MUTATIONS:
        failures += _run_one(
            f"mutation:{name}",
            lambda r, a=apply, n=name, e=expected: check_mutation_is_caught(r, n, a, e))
    for name, apply, expected in EVAL_MUTATIONS:
        failures += _run_one(
            f"eval-mutation:{name}",
            lambda r, a=apply, n=name, e=expected: check_eval_mutation_is_caught(r, n, a, e))
    total = len(cases) + len(MUTATIONS) + len(EVAL_MUTATIONS)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


def _run_one(name: str, fn) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            fn(snapshot(Path(tmp) / "repo"))
        except AssertionError as exc:
            print(f"FAIL {name}\n     {str(exc).splitlines()[0]}")
            return 1
        except Exception as exc:  # noqa: BLE001 - a broken test is a failed test
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
            return 1
    print(f"ok   {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_standalone())
