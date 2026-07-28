"""lane -- a standalone, flatbuffer-only deployment auditor for the NXP RT595 lane.

`lane` answers one question about a .tflite artifact: *would this thing actually
run on the RT595's TFLite-Micro runtime, and would it be numerically sane if it
did?*  It answers it without torch, without a dataset, without the board, and
without importing anything from the rest of this repo.

Everything structural is read straight out of the flatbuffer (see
`lane.runtime.flatbuffer`).  The litert interpreter is only ever touched inside a
subprocess (`lane.runtime.litert_host`) because at least one of the artifacts
under test takes the host process down with SIGSEGV, which no `try/except` can
catch.

Public surface::

    from lane.runtime.flatbuffer import load_model, ModelInfo, TensorInfo
    from lane.gates import Gate, GateResult, run_gates, GATES
    python -m lane audit <model.tflite> [--resolver ops.cpp] ...
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
