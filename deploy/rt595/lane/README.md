# `lane audit` — deploy auditor for RT595 tflite artifacts

Answers one question about a converted `.tflite`: **would this actually run on the board, and is
it structurally sound?** Flatbuffer-only — no torch, no dataset, no board, no simulator. Runs in
about a second per model.

It exists because every conversion bug this project has hit was produced by code that printed a
cheerful success message. The int8 ConvFSENet export carried 182 NaN quantization scales through
conversion, quantization and an ISS run without a single warning; it only surfaced when someone
went looking. These gates go looking, every time.

## Run it

```bash
cd deploy/rt595
S=/path/to/tflm_rt500          # dir holding tflite-micro/ and libtflm_rt500.a

~/.venvs/rt595-export/bin/python -m lane audit --all \
    --tflm-tree $S/tflite-micro --lib $S/libtflm_rt500.a

# one model, against the resolver it will actually link
~/.venvs/rt595-export/bin/python -m lane audit host_out/relu6deep_streaming_int8.tflite \
    --resolver iss/resolvers/lisennet_ops.cpp \
    --tflm-tree $S/tflite-micro --lib $S/libtflm_rt500.a
```

**Exit codes:** `0` clear or warnings only · `1` at least one hard-fail · `2` usage/IO error ·
`3` a deploy-blocking gate could not run. `3` matters: a gate that cannot run must never be
mistaken for a gate that passed.

`--json out.json` for machine-readable output. `-v` dumps each gate's evidence.

## The gates

Severity is the contract — it decides the exit code, so the split is deliberate.

| gate | severity | catches |
|---|---|---|
| `G-SCALE-FINITE` | hard-fail | NaN/Inf quantization scales (the 182-tensor ConvFSENet bug) |
| `G-SCALE-POSITIVE` | hard-fail | zero or negative scales |
| `G-QUANT-STRUCT` | hard-fail | malformed quantization blocks, per-axis/axis mismatches |
| `G-SCALE-FLOOR` | warn | ranges collapsed onto the calibrator floor — the signature of a calibrator that never saw real activations |
| `G-OPSET` | hard-fail | an op with no TFLM kernel (POW, REDUCE_ALL). Names the tree it consulted |
| `G-RESOLVER-EXACT` | hard-fail | a graph op the resolver does **not** register → `AllocateTensors()` fails at boot |
| `G-RESOLVER-LEAN` | warn | a registration the graph never uses → wasted slot + dead kernel |
| `G-RESOLVER-ARITY` | hard-fail | `MicroMutableOpResolver<N>` too small → registrations silently dropped |
| `G-RESOLVER-FIRMWARE` | hard-fail | same, against `app/model_ops_micro.cpp` — the resolver that *ships* |
| `G-IO-SYMMETRY` | hard-fail | a streaming export that lost a state output |
| `G-IO-LAYOUT` | warn | `model_io_layout.h` describes a different model than the artifact |
| `G-SINGLE-SUBGRAPH` | hard-fail | extra subgraphs the audit did not cover |
| `G-ACCEL-COVERAGE` | warn | ops falling back to portable C instead of an xa_nnlib HiFi4 kernel |
| `G-LOAD-ISOLATED` | hard-fail | the model kills the interpreter |

Two design notes worth knowing:

- **`G-RESOLVER-EXACT` fails on the missing direction only.** A surplus registration is not a boot
  blocker, and folding both directions into one hard-fail made the only healthy artifact in the
  tree come back BLOCKED over a single dead `AddTanh`. A gate that cries wolf on the healthy case
  is a gate people learn to ignore.
- **`G-LOAD-ISOLATED` runs the interpreter in a child process and reads the return code.** One
  artifact here (`convfsenet_win_streaming.tflite`) takes the host down with SIGSEGV, which no
  `try/except` can catch. In-process loading is not an option.

## Degraded mode

The TFLM tree and `libtflm_rt500.a` are build artifacts, not repo contents (see `../tflm/`).
Without them, the gates that need them **SKIP visibly** rather than passing — and if a skipped
gate is deploy-blocking, the verdict is `INCOMPLETE` and the exit code is `3`.

## Tests

```bash
~/.venvs/rt595-export/bin/python -m pytest lane/tests -q      # 167 tests
```

Gates take a plain `ctx` dict, so every one is unit-testable with no `.tflite` on disk. Two
golden self-tests are load-bearing: the resolver-header parse must brace-match method bodies (a
naive per-line regex loses `FULLY_CONNECTED` and `TRANSPOSE_CONV`, which would fail NSNet2 and
LiSenNet on a phantom op wall), and the gate id→severity map is asserted whole.

## What it does not do

No numerical parity against torch, no PESQ, no cycle measurement. Those need the model, the
dataset and the simulator respectively — this tool deliberately needs none of them. See
`../BENCHMARKS.md` for cycles and the LANE proposal for the rest of the intended pipeline.
