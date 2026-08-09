#!/usr/bin/env python3
"""Agent-eval harness for eco8-neaixt.

What this does and does not do, stated plainly so nobody mistakes the second for
the first:

  * it VALIDATES the task definitions (CI runs `--validate-only`);
  * it RUNS the deterministic graders that are pure repo/tool checks;
  * it EMITS a scoring sheet for a trial run;
  * it AGGREGATES recorded trial outcomes into the metrics worth tracking.

It does NOT drive an agent. Running the tasks means giving each `prompt` to the
agent under test in a fresh session on a clean checkout, then recording what
happened. Automating that is worth doing once the task set has proved itself;
automating it first would mostly measure the harness.

    python evals/agent/run_evals.py --validate-only
    python evals/agent/run_evals.py --list
    python evals/agent/run_evals.py --deterministic
    python evals/agent/run_evals.py --sheet runs/2026-08-09.md
    python evals/agent/run_evals.py --score runs/2026-08-09.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    print("run_evals: needs pyyaml", file=sys.stderr)
    raise SystemExit(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TASKS_DIR = HERE / "tasks"

CATEGORIES = {"evidence", "comparison", "repo-invariant", "memory", "conduct"}
DIFFICULTIES = {"low", "medium", "high"}


def load_tasks() -> list[dict]:
    tasks: list[dict] = []
    for path in sorted(TASKS_DIR.glob("*.yaml")):
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            where = f" (line {mark.line + 1})" if mark else ""
            raise SystemExit(
                f"ERROR {path.relative_to(REPO)}{where}: unparseable YAML: "
                f"{getattr(exc, 'problem', exc)}"
            ) from None
        for task in doc.get("tasks", []):
            task["_file"] = str(path.relative_to(REPO))
            tasks.append(task)
    return tasks


def validate(tasks: list[dict]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    if not tasks:
        problems.append("no tasks found under evals/agent/tasks/")
    for task in tasks:
        where = f"{task.get('_file')}:{task.get('id', '<no id>')}"
        for field in ("id", "title", "category", "difficulty", "source", "prompt",
                      "graders", "trials"):
            if not task.get(field):
                problems.append(f"{where}: missing {field!r}")
        tid = task.get("id")
        if tid in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(tid)
        if task.get("category") not in CATEGORIES:
            problems.append(f"{where}: category {task.get('category')!r} not in {sorted(CATEGORIES)}")
        if task.get("difficulty") not in DIFFICULTIES:
            problems.append(f"{where}: difficulty {task.get('difficulty')!r} not in {sorted(DIFFICULTIES)}")
        if not isinstance(task.get("trials"), int) or task.get("trials", 0) < 1:
            problems.append(f"{where}: trials must be an int >= 1 (agent behaviour is stochastic)")

        graders = task.get("graders") or {}
        rubric = graders.get("rubric") or {}
        det = graders.get("deterministic") or []
        if not rubric and not det:
            problems.append(f"{where}: no graders")
        # A rubric with only `must` grades an agent that says the right words
        # while doing the wrong thing. Every task needs at least one must_not.
        if rubric and not rubric.get("must_not"):
            problems.append(f"{where}: rubric has no `must_not` — the failure mode "
                            "is what this task is for")
        for item in det:
            if not item.get("id") or not item.get("description"):
                problems.append(f"{where}: deterministic grader needs id + description")
            if "command" in item and item.get("expect") not in {"pass", "fail"}:
                problems.append(f"{where}: deterministic grader with a command needs "
                                "expect: pass|fail")

        # The source is what keeps this suite grounded in real incidents rather
        # than in invented ones.
        source = str(task.get("source", "")).split("#")[0]
        if source and not (REPO / source).exists():
            problems.append(f"{where}: source path does not exist: {source}")
    return problems


def run_deterministic(tasks: list[dict]) -> int:
    failures = 0
    ran = 0
    for task in tasks:
        for item in (task.get("graders") or {}).get("deterministic") or []:
            cmd = item.get("command")
            if not cmd:
                continue
            ran += 1
            proc = subprocess.run(cmd, shell=True, cwd=REPO,
                                  capture_output=True, text=True)
            passed = proc.returncode == 0
            want_pass = item["expect"] == "pass"
            ok = passed == want_pass
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {task['id']}/{item['id']}: "
                  f"exit={proc.returncode} expect={item['expect']}")
            if not ok and proc.stderr.strip():
                print(f"      {proc.stderr.strip().splitlines()[-1]}")
    print(f"\ndeterministic graders: {ran} run, {failures} failed")
    return 1 if failures else 0


def emit_sheet(tasks: list[dict], out: Path) -> None:
    lines = [
        "# Agent-eval scoring sheet",
        "",
        "Run each prompt in a **fresh session on a clean checkout**. One row per",
        "trial — agent behaviour is stochastic, so a single trial measures little.",
        "",
        "Record for every trial: `pass` / `fail` against the rubric, and separately",
        "whether a **scientific invariant** was violated (a `must_not`) and whether",
        "the agent asserted something **false** about the repo or the results.",
        "",
        "| task | trial | pass | invariant violated | false claim | human minutes | notes |",
        "| ---- | ----- | ---- | ------------------ | ----------- | ------------- | ----- |",
    ]
    for task in tasks:
        for i in range(1, task["trials"] + 1):
            lines.append(f"| {task['id']} | {i} |  |  |  |  |  |")
    lines += ["", "## Task reference", ""]
    for task in tasks:
        rubric = (task.get("graders") or {}).get("rubric") or {}
        lines += [
            f"### {task['id']} — {task['title']}",
            f"*{task['category']} / {task['difficulty']} / source: `{task['source']}`*",
            "",
            "**Prompt**",
            "",
            "```",
            task["prompt"].rstrip(),
            "```",
            "",
            "**Must**",
            *[f"- {m}" for m in rubric.get("must", [])],
            "",
            "**Must not**",
            *[f"- {m}" for m in rubric.get("must_not", [])],
            "",
        ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"sheet: {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")


def score(tasks: list[dict], results_path: Path) -> int:
    """Aggregate recorded trials.

    Expected JSON: {"agent": "...", "date": "...", "trials": [
      {"task": "id", "trial": 1, "pass": true, "invariant_violated": false,
       "false_claim": false, "human_minutes": 4, "notes": "..."}, ...]}
    """
    data = json.loads(results_path.read_text())
    trials = data.get("trials", [])
    by_task: dict[str, list[dict]] = {}
    for t in trials:
        by_task.setdefault(t["task"], []).append(t)

    known = {t["id"] for t in tasks}
    unknown = sorted(set(by_task) - known)
    missing = sorted(known - set(by_task))

    counts = Counter()
    for t in trials:
        counts["trials"] += 1
        counts["pass"] += bool(t.get("pass"))
        counts["invariant"] += bool(t.get("invariant_violated"))
        counts["false_claim"] += bool(t.get("false_claim"))
    minutes = sum(t.get("human_minutes") or 0 for t in trials)

    n = counts["trials"] or 1
    print(f"agent: {data.get('agent', '?')}   date: {data.get('date', '?')}")
    print(f"trials                    {counts['trials']}")
    print(f"task success              {counts['pass'] / n:.0%}")
    print(f"invariant violation rate  {counts['invariant'] / n:.0%}   <- the one that matters")
    print(f"false-claim rate          {counts['false_claim'] / n:.0%}")
    print(f"human review time         {minutes} min ({minutes / n:.1f}/trial)")

    print("\nper task (pass rate over trials):")
    for task in tasks:
        ts = by_task.get(task["id"], [])
        if not ts:
            continue
        p = sum(bool(t.get("pass")) for t in ts)
        flag = "  <-- regression candidate" if p < len(ts) else ""
        print(f"  {p}/{len(ts)}  {task['id']}{flag}")

    if missing:
        print(f"\nnot run: {', '.join(missing)}")
    if unknown:
        print(f"unknown task ids in results: {', '.join(unknown)}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--deterministic", action="store_true",
                    help="run the deterministic graders that need no agent")
    ap.add_argument("--sheet", type=Path, help="write a scoring sheet")
    ap.add_argument("--score", type=Path, help="aggregate a recorded results JSON")
    a = ap.parse_args(argv)

    tasks = load_tasks()
    problems = validate(tasks)
    for p in problems:
        print(f"ERROR {p}")
    if problems:
        print(f"\nrun_evals: {len(problems)} problem(s) in the task definitions")
        return 1

    if a.validate_only:
        cats = Counter(t["category"] for t in tasks)
        total_trials = sum(t["trials"] for t in tasks)
        print(f"run_evals: {len(tasks)} tasks ok, {total_trials} trials, "
              f"{dict(sorted(cats.items()))}")
        return 0
    if a.list:
        for t in tasks:
            print(f"{t['id']:<28} {t['difficulty']:<7} {t['category']:<14} {t['title']}")
        return 0
    if a.deterministic:
        return run_deterministic(tasks)
    if a.sheet:
        emit_sheet(tasks, a.sheet)
        return 0
    if a.score:
        return score(tasks, a.score)

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
