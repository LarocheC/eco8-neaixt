# tests/

Test suite for the sparse-nsnet2 streaming + ONNX export pipeline. Phase 1
gates the FP32 ground truth (cuDNN-vs-unrolled parity) and the ONNX export
path (no opaque GRU op survives) — every downstream phase compares against
these gates.

## How to run

```bash
uv run pytest -xvs                                  # full suite
uv run pytest tests/test_streaming_parity.py -xvs   # parity gate only (STR-01..03)
uv run pytest tests/test_streaming_export.py -xvs   # ONNX export smoke (STR-04)
```

The `python -m pytest` form is equivalent and matches the ROADMAP success
criterion phrasing for Phase 1 — either form invokes the same pytest entry:

```bash
uv run python -m pytest tests/test_streaming_parity.py -xvs   # equivalent to the focused parity command above
```

The session-scoped autouse fixture in `tests/conftest.py` enables
`torch.use_deterministic_algorithms(True)`, sets `cudnn.benchmark=False` /
`cudnn.deterministic=True`, and seeds `torch.manual_seed(0)`. The
`CUBLAS_WORKSPACE_CONFIG=:4096:8` environment variable is set at the top of
`conftest.py` BEFORE `import torch` — see "If a test fails with
CUBLAS_WORKSPACE_CONFIG error" below.

## Tolerance buckets (parity test)

The parity test (`tests/test_streaming_parity.py::test_parity_strict_tolerance`)
hard-fails at `max_abs_err < 1e-5`. On failure, the assertion message names a
bucket. Do NOT loosen the tolerance — investigate using the checklist below.

| `max_abs_err`   | Verdict                                      | Diagnostic checklist |
|-----------------|----------------------------------------------|----------------------|
| `< 1e-5`        | **1e-5 pass**                                | — |
| `1e-5 .. 1e-4`  | **1e-4 suspect Pitfall 1** (`linear_before_reset`) | (a) Re-check the `n` formula in `NSNet2Streaming.forward_step` in `nsnet2/streaming.py`: `n = tanh(x @ W_in.T + b_in + r * (h_prev @ W_hn.T + b_hn))` — the reset gate `r` multiplies the **whole** `W_hn` projection AND its bias. Do NOT use the textbook formula `tanh(... + W_hn @ (r * h_prev) + b_hn)` (which is what `nsnet2.layers.StructuredGRUCell` uses — that diverges by ~1.4e-1). (b) Confirm the parity test's bias-override helper ran (every `bias_ih_l[k]` and `bias_hh_l[k]` should have stddev ~0.5, not zero — zero biases mask Pitfall 1). (c) Confirm `h0` dtype matches the input dtype. |
| `1e-4 .. 1e-3`  | **1e-3 suspect Pitfall 2** (gate ordering)   | (a) Verify `chunk(3, dim=0)` on `weight_ih_l[k]` / `weight_hh_l[k]` / `bias_ih_l[k]` / `bias_hh_l[k]` yields PyTorch `[r | z | n]` order — NOT ONNX `[z | r | h]`. Add an inline shape comment at every `chunk(3, dim=0)` site naming the gate order explicitly. (b) Run the parity test with `T=1` (single-step) — if `T=1` also fails, the bug is in the input projection slice, not the recurrence. (c) Spot-check the wrapper's `W_ir_0` / `W_iz_0` / `W_in_0` against `base.gru.weight_ih_l0.chunk(3, dim=0)[i]` for `i ∈ {0, 1, 2}` — they must be element-equal. |
| `>= 1e-3`       | **gross failure** (more than gate-ordering)  | Re-read `01-RESEARCH.md` §"cuDNN GRU Arithmetic" and `01-PATTERNS.md` §"nsnet2/streaming.py". Likely a structural error: the unrolled cell may be feeding the wrong layer-1 input dim (must be `H`, not `n_freq`), or `forward_full` may not be threading `states` correctly between iterations (run `tests/test_streaming_parity.py::test_forward_full_drives_forward_step` — that's its job). |

If the parity test passes but `test_parity_failure_message_format` fails:
the failure-message format has been edited and a bucket string is no longer
verbatim. Restore the strings `1e-5 pass`, `1e-4 suspect Pitfall 1`, and
`1e-3 suspect Pitfall 2` in the assertion message of
`test_parity_strict_tolerance` — and update the table above to match (this
README and the assertion are intentionally a single source of truth; if you
change one, change both).

## ONNX export smoke-test failure modes

The export smoke test (`tests/test_streaming_export.py`) fails in two
distinct ways that map to documented pitfalls.

### Pitfall 10: `nn.GRUCell` blocks ONNX export

**Symptom:** `test_onnx_export_smoke` raises
`OnnxExporterError: ... aten::_thnn_fused_gru_cell` (or similar
`aten::*gru*` symbolic-not-found error) at export time.

**Cause:** `NSNet2Streaming.forward_step` in `nsnet2/streaming.py` is using
`nn.GRUCell`, `nn.RNNCell`, or `nn.LSTMCell`. The legacy
`torch.onnx.export(dynamo=False)` exporter has no symbolic for
`aten::_thnn_fused_gru_cell`.

**Fix:** Replace the cell call with the per-gate inline MatMul + Add +
Sigmoid + Tanh sequence using the twelve `nn.Parameter` tensors per layer
that `NSNet2Streaming.__init__` slices once from `base.gru.weight_ih_l[k]`
/ `weight_hh_l[k]` / `bias_ih_l[k]` / `bias_hh_l[k]`.

### Pitfall 11: Opaque `GRU` op survives export

**Symptom:** `test_onnx_export_smoke` passes, but `test_onnx_no_opaque_gru`
fails with `Pitfall 11: N opaque GRU op(s) survived export`.

**Cause:** `NSNet2Streaming.forward_step` is calling `self.base.gru(...)`
(the cuDNN `nn.GRU` instance) or constructing an `nn.GRU(seq_len=1)` to
"unroll" per-frame. Either path emits an opaque ONNX `GRU` op that Phase 4
`quantize_static` cannot rewrite — the GRU compute would run FP32 internally
even after quant.

**Fix:** Remove all `self.base.gru(...)` calls (and any `nn.GRU(...)` calls)
from `forward_step` in `nsnet2/streaming.py`. Use ONLY the per-gate
`nn.Parameter` tensors (sliced once at `__init__`) and inline `torch.sigmoid`
/ `torch.tanh` / `torch.relu` / `@` matmul / `+` add. The `forward = forward_step`
class-level alias is the trace target — verify it is present at the bottom
of `NSNet2Streaming` so `torch.onnx.export(streaming, ...)` traces
`forward_step` directly.

### If a test fails with CUBLAS_WORKSPACE_CONFIG error

**Symptom:** Any test fails with
`RuntimeError: ... CUBLAS_WORKSPACE_CONFIG=:16:8 or :4096:8 must be set ...`
even though "the env var is set in conftest.py".

**Cause:** The `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")`
line in `tests/conftest.py` has been moved INSIDE an autouse fixture (or
otherwise placed AFTER `import torch`). cuBLAS reads the env var at the
first CUDA op, which can occur during `torch` import itself — by the time
the fixture runs, it is too late.

**Fix:** Move the `os.environ.setdefault(...)` call back to the top of
`tests/conftest.py`, BEFORE the `import torch` line.

## File map

| File                          | Purpose                                                           | Phase |
|-------------------------------|-------------------------------------------------------------------|-------|
| `conftest.py`                 | Session-scoped determinism fixture; pre-import CUBLAS env var      | 1     |
| `test_streaming_parity.py`    | STR-01 / STR-02 / STR-03 — shape, parity, time-loop equivalence    | 1     |
| `test_streaming_export.py`    | STR-04 — ONNX export smoke + no-opaque-GRU + onnx.checker          | 1     |

Future phases (2–6) extend this directory; this README adds sections as
each new test file lands.
