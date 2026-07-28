"""Load a .tflite through the host interpreter *in a subprocess*.

`convfsenet_win_streaming.tflite` takes the interpreter down with SIGSEGV during
construction.  A segfault is delivered to the process, not to Python, so::

    try:
        Interpreter(model_path=p)      # <-- never returns, never raises
    except Exception:
        ...

does not help: the auditor dies with the model and reports nothing.  The only
safe signal is a **child process exit code**, so that is what this module uses.
A crash therefore becomes an ordinary, reportable gate failure.

Nothing in this module is imported at module scope by the gates -- the child
source is a string, executed by a fresh interpreter.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from typing import Optional

_MARKER = "@@LANE_LOAD@@"

# Runs in the child.  Kept dependency-light and defensive: any Python-level
# failure is reported as JSON, and anything worse shows up as the return code.
_CHILD_SRC = r"""
import json, sys
marker = %(marker)r
path = sys.argv[1]
out = {"loaded": False, "allocated": False, "error": None,
       "n_inputs": None, "n_outputs": None}
try:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    interp = Interpreter(model_path=path)
    out["loaded"] = True
    interp.allocate_tensors()
    out["allocated"] = True
    out["n_inputs"] = len(interp.get_input_details())
    out["n_outputs"] = len(interp.get_output_details())
except BaseException as exc:
    out["error"] = "%%s: %%s" %% (type(exc).__name__, exc)
sys.stdout.write(marker + json.dumps(out) + "\n")
sys.stdout.flush()
sys.exit(0 if (out["allocated"] and out["error"] is None) else 3)
""" % {"marker": _MARKER}


def _signal_name(returncode: int) -> Optional[str]:
    if returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except (ValueError, AttributeError):
        return "SIG%d" % (-returncode)


def load_isolated(path: str, timeout: float = 60) -> dict:
    """Construct + allocate the model in a child process.

    Returns::

        {"ok": bool, "returncode": int, "signal": str|None, "stderr": str,
         "n_inputs": int|None, "n_outputs": int|None}

    `ok` is True only if the child exited 0 *and* reported a successful
    allocate_tensors().  A SIGSEGV shows up as `returncode == -11`,
    `signal == "SIGSEGV"`.  `returncode == -signal.SIGKILL` with a "timed out"
    stderr means the child hung.
    """
    result = {
        "ok": False,
        "returncode": -1,
        "signal": None,
        "stderr": "",
        "n_inputs": None,
        "n_outputs": None,
    }

    if not os.path.isfile(path):
        result["stderr"] = "no such file: %s" % path
        return result

    env = dict(os.environ)
    # Keep the child from importing this repo's conftest/sitecustomize noise and
    # from spending 10s spinning up TF's oneDNN banner.
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_SRC, os.path.abspath(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        result["returncode"] = -int(getattr(signal, "SIGKILL", 9))
        result["signal"] = "SIGKILL"
        result["stderr"] = "child timed out after %.1fs: %s" % (timeout, exc)
        return result
    except OSError as exc:
        result["stderr"] = "could not spawn interpreter child: %s" % exc
        return result

    stdout = proc.stdout.decode("utf-8", "replace")
    stderr = proc.stderr.decode("utf-8", "replace")

    result["returncode"] = int(proc.returncode)
    result["signal"] = _signal_name(int(proc.returncode))

    payload = None
    for line in stdout.splitlines():
        if line.startswith(_MARKER):
            try:
                payload = json.loads(line[len(_MARKER) :])
            except ValueError:
                payload = None

    if payload:
        result["n_inputs"] = payload.get("n_inputs")
        result["n_outputs"] = payload.get("n_outputs")
        if payload.get("error"):
            stderr = (stderr + "\n" + payload["error"]).strip()

    result["ok"] = bool(proc.returncode == 0 and payload and payload.get("allocated"))
    # Trim: TF/litert children are chatty and the whole thing ends up in JSON.
    result["stderr"] = stderr.strip()[-4000:]
    return result


__all__ = ["load_isolated"]
