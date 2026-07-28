"""G-LOAD-ISOLATED: does the model survive construction + tensor allocation?

The host interpreter is not the deploy target, so a clean load proves nothing
about the RT595.  A *dirty* load, however, proves plenty: a flatbuffer that
segfaults a hardened desktop runtime is not going to behave on an M33 with no
MMU.  `convfsenet_win_streaming.tflite` does exactly that (SIGSEGV, rc -11), and
because the crash is delivered to the process rather than raised as an
exception, the check only works at all because
`lane.runtime.litert_host.load_isolated` runs it in a child process.
"""

from __future__ import annotations

from . import Gate, GateResult, HARD_FAIL, skip

G_LOAD_ISOLATED = Gate(
    id="G-LOAD-ISOLATED",
    title="Loads + allocates in an isolated subprocess",
    severity=HARD_FAIL,
)


def check_load_isolated(ctx: dict) -> GateResult:
    """Hard-fail if the subprocess load crashed, hung or errored."""
    res = ctx.get("load_result")
    if res is None:
        return skip(
            G_LOAD_ISOLATED,
            "isolated load not run (use --load / do not pass --no-load)",
        )

    model = ctx.get("model")
    evidence = {
        "returncode": res.get("returncode"),
        "signal": res.get("signal"),
        "n_inputs": res.get("n_inputs"),
        "n_outputs": res.get("n_outputs"),
        "stderr_tail": (res.get("stderr") or "")[-800:],
    }

    if res.get("ok"):
        detail = "loaded and allocated in a child process (%s in / %s out)" % (
            res.get("n_inputs"),
            res.get("n_outputs"),
        )
        # Cross-check the interpreter's view against our own flatbuffer parse.
        if model is not None:
            mismatch = []
            if res.get("n_inputs") not in (None, len(model.inputs)):
                mismatch.append(
                    "inputs %s vs flatbuffer %d" % (res.get("n_inputs"), len(model.inputs))
                )
            if res.get("n_outputs") not in (None, len(model.outputs)):
                mismatch.append(
                    "outputs %s vs flatbuffer %d"
                    % (res.get("n_outputs"), len(model.outputs))
                )
            evidence["io_mismatch"] = mismatch
            if mismatch:
                return GateResult(
                    gate=G_LOAD_ISOLATED,
                    passed=False,
                    detail=(
                        "loaded, but the interpreter's IO count disagrees with the "
                        "flatbuffer: " + "; ".join(mismatch)
                    ),
                    evidence=evidence,
                )
        return GateResult(
            gate=G_LOAD_ISOLATED, passed=True, detail=detail, evidence=evidence
        )

    sig = res.get("signal")
    if sig:
        detail = (
            "the interpreter child died on %s (rc=%s) while loading this model. "
            "A fatal signal is not catchable in-process -- anything that loads this "
            "file directly takes the whole process down with it."
            % (sig, res.get("returncode"))
        )
    else:
        tail = (res.get("stderr") or "").strip().splitlines()
        why = tail[-1] if tail else "no diagnostic"
        detail = "the interpreter child exited %s without allocating: %s" % (
            res.get("returncode"),
            why,
        )

    return GateResult(
        gate=G_LOAD_ISOLATED, passed=False, detail=detail, evidence=evidence
    )


check_load_isolated.gate = G_LOAD_ISOLATED  # type: ignore[attr-defined]

__all__ = ["G_LOAD_ISOLATED", "check_load_isolated"]
