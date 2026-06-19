#!/usr/bin/env python3
"""Benchmark an ONNX model on ST's Edge AI Developer Cloud board farm (incl. STM32N6570-DK)
without a locally-wired board. Prints NPU macc / cycles / latency / RAM / ROM as JSON.

Auth: a MyST account (the same login you used in the web GUI). DO NOT hard-code the
password — put it in deploy/stm32n6/cloud/.env (gitignored). This script loads that file,
then the ST client caches a refresh-token JWT in ~/.stmai_token so the password is only
used on first run.

The ST Python client ships inside the model-zoo repo (Apache-2.0): clone
github.com/STMicroelectronics/stm32ai-modelzoo-services and point STM32AI_MODELZOO at it
(or clone it as a sibling of this repo). We add its `common/` to sys.path.

Usage:
  python dev_cloud_bench.py --model ../../cp_convfsenet/g_best.onnx --board STM32N6570-DK
"""
import argparse
import json
import os
import sys
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Lines: KEY=VALUE, '#' comments."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def find_modelzoo_common() -> Path:
    """Locate the stm32ai-modelzoo-services checkout that holds common/stm32ai_dc."""
    candidates = []
    env = os.environ.get("STM32AI_MODELZOO")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    candidates += [
        here.parents[4] / "stm32ai-modelzoo-services",   # sibling of the repo root
        here.parent / "stm32ai-modelzoo-services",        # cloned right here
        Path.home() / "stm32ai-modelzoo-services",
    ]
    for c in candidates:
        if c and (c / "common" / "stm32ai_dc").is_dir():
            return c
    sys.exit(
        "ERROR: cannot find stm32ai-modelzoo-services (need common/stm32ai_dc).\n"
        "  git clone --depth 1 https://github.com/STMicroelectronics/stm32ai-modelzoo-services\n"
        "  then re-run, or set STM32AI_MODELZOO=/path/to/stm32ai-modelzoo-services"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--board", default="STM32N6570-DK")
    ap.add_argument("--timeout", type=int, default=900, help="board-farm is queue-based; allow minutes")
    args = ap.parse_args()

    load_dotenv(Path(__file__).with_name(".env"))
    user = os.environ.get("STM32AI_USERNAME")
    pwd = os.environ.get("STM32AI_PASSWORD")
    if not user or not pwd:
        sys.exit("ERROR: set STM32AI_USERNAME / STM32AI_PASSWORD (see cloud/.env.example).")

    sys.path.insert(0, str(find_modelzoo_common()))
    try:
        # Class names per the model-zoo client (common/stm32ai_dc). If a future client
        # version renames these, check that repo's common/stm32ai_dc/README.md.
        from common.stm32ai_dc import Stm32Ai, CloudBackend, CliParameters  # type: ignore
    except Exception as e:  # pragma: no cover - import-time env issue
        sys.exit(f"ERROR importing ST cloud client: {e}")

    model_path = str(Path(args.model).resolve())
    model_name = Path(model_path).name

    print(f"[cloud] auth as {user} (JWT cached in ~/.stmai_token)")
    ai = Stm32Ai(CloudBackend(user, pwd, version=None))  # version=None -> latest cloud stedgeai

    boards = [b.name for b in ai.get_benchmark_boards()]
    print(f"[cloud] farm boards: {boards}")
    if args.board not in boards:
        print(f"WARNING: '{args.board}' not currently in the farm (maintenance/outage?). Submitting anyway.")

    print(f"[cloud] uploading {model_name}")
    ai.upload_model(model_path)

    print(f"[cloud] benchmarking on {args.board} (timeout {args.timeout}s; queue-based, be patient)…")
    res = ai.benchmark(CliParameters(model=model_name), args.board, args.timeout)

    out = {
        k: getattr(res, k, None)
        for k in ("macc", "cycles", "duration_ms", "cycles_by_macc",
                  "rom_size", "ram_size", "rom_n_macc", "total_ram", "total_flash")
    }
    print(json.dumps({"board": args.board, "model": model_name, "result": out}, indent=2, default=str))

    try:
        ai.delete_model(model_name)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
