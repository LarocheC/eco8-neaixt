# Coding Conventions

> **ARCHIVE — frozen 2026-04-27. Not current state.**
> A snapshot analysis of the codebase as it was in April 2026, under its former
> `sparse-nsnet2` layout (root-level `train.py`, `models/`, no `tests/`). The
> current layout is per-family packages (`nsnet2/`, `convfsenet/`, `lisennet/`,
> `common/`, `benchmarks/`) with a pytest suite. **Where this file and the code
> disagree, the code wins.** Kept for the rationale, not the facts.
>
> Current state: [`research/NOW.md`](../../research/NOW.md). Rules:
> [`AGENTS.md`](../../AGENTS.md).

**Analysis Date:** 2026-04-27

This is a research codebase. Conventions are split into two tiers:

1. **Core pipeline** — `train.py`, `inference.py`, `dataset.py`, `models/*.py`,
   `utils.py`, `env.py`. Inherited from the MP-SENet training recipe. Plain,
   un-typed, lowercase-snake_case Python; minimal docstrings.
2. **Standalone variants** — `*_standalone.py` files at the repo root (e.g.
   `nsnet2_encdec_v4_standalone.py`, `smr_nsnet_standalone.py`,
   `sensnet2_ch64_la4_FDbidir_freqnorm_grp4_standalone.py`). These are the
   *new* code added by research iterations and follow a stricter, fully
   typed, heavily documented "paste-and-go" style. New architectural
   variants MUST follow the standalone style.

There is no enforced linter or formatter (no `.ruff.toml`, no `.flake8`,
no `pyproject` lint section). Style is enforced by review.

## Naming Patterns

**Files:**
- Core pipeline: short lowercase nouns — `train.py`, `dataset.py`, `env.py`,
  `utils.py`, `inference.py`.
- Variant files: `<name>_standalone.py` at repo root. Examples that have
  existed: `nsnet2_encdec_v4_standalone.py`,
  `nsnet2_encdec_v5_tfcm_standalone.py`, `smr_nsnet_standalone.py`,
  `sensnet2_ch64_la4_FDbidir_freqnorm_grp4_standalone.py`. The stem
  encodes the variant identity and matches the corresponding `cp_<stem>/`
  checkpoint directory and `configs/<stem>.json`.
- Configs: `configs/<run_name>.json`, one file per sweep run.
- Sweep logs: `<family>_sweep.log` at repo root (e.g. `sweep.log`,
  `encdec_sweep.log`, `lru_sweep.log`, `smr_sweep.log`). Per-run stdout
  is mirrored to `cp_<run>/train.log`.

**Functions / methods:**
- `snake_case` everywhere (including factories: `make_linear`, `make_gru`,
  `mag_pha_stft`, `load_checkpoint`, `scan_checkpoint`).
- Test/sanity helpers in standalones use the `test_` prefix
  (`test_shapes`, `test_streaming_equivalence`, `test_state_passing`,
  `test_smr_decay_bounds`) — see `git show b1e8e63:smr_nsnet_standalone.py`.
- Underscore prefix for private helpers used only inside a single module
  (e.g. `_make_model`, `_GRUSequence`, `_channel_shuffle` in the
  standalones).

**Variables:**
- Short, domain-flavoured: `h` (hyperparams `AttrDict`), `a` (argparse
  `Namespace`), `cfg` (config dict in standalones), `cp_g` / `cp_do`
  (generator / disc-optim checkpoint paths), `mag` / `pha` / `com`
  (magnitude / phase / complex), `B`, `T`, `F`, `C` (tensor dims).
- The pair `h` and `a` is load-bearing and crosses files
  (`train.train(rank, a, h)`); do not rename.

**Types / classes:**
- `PascalCase` for `nn.Module` subclasses (`NSNet2`, `NSNet2EncDec`,
  `MetricDiscriminator`, `StructuredGRU`, `StructuredGRUCell`,
  `SMRBlock`, `GRUBlock`, `TFGRUBlock`, `SeNSNet2`, `CausalDenseBlock`,
  `CausalTFCMBlock`, `FreqInstanceNorm`, `LearnableSigmoid1d`,
  `SPConvTranspose2d`).
- `UPPER_CASE` for module-level constants — `CONFIG` (canonical
  hyperparams in every standalone), `LINEAR_KINDS`, `GRU_KINDS`,
  `HF_DATASET_NAME`, `HAVE_BUTTERFLY`, `HAVE_MONARCH`, `RUNS` (notebook).

**Checkpoint files (literal naming, parsed by `utils.scan_checkpoint`):**
- `g_<8-digit step>` — generator state. Pattern `g_????????` is required;
  `scan_checkpoint(cp_dir, 'g_')` does an 8-`?` glob.
- `do_<8-digit step>` — discriminator + optim state.
- `g_best` — generator-only state for the best validation PESQ seen so
  far. Saved with key `{'generator': ...}` so any loader can do
  `state['generator']`.
- `config.json` — copy of the training config, written by
  `env.build_env`.
- `logs/` — TensorBoard event files.
- `train.log` — per-run stdout mirror (created by `run_sweep.sh`).

## Code Style

**Formatting:**
- 4-space indent, no trailing semicolons.
- Line length: ~100 chars in core (some lines longer in `train.py`),
  ~80–100 chars in standalones.
- Hanging-indent calls and dict literals when wrapped (see
  `train.py:181-182` for a multi-arg `print` and any `CONFIG = {...}`
  block in a standalone).
- f-strings preferred in standalones; `.format()` style preferred in
  core (matches the upstream MP-SENet recipe — keep using `.format()`
  inside `train.py`/`inference.py` to avoid mixed style there).

**Linting:**
- None enforced. Pre-commit / CI lint is not configured.

**Typing:**
- Core pipeline: untyped (`def train(rank, a, h):`,
  `def make_linear(in_size, out_size, bias=True, *, cfg=None):` — note
  cfg-only is typed).
- `models/layers.py`: typed with `from __future__ import annotations`,
  `Optional[dict]`, return annotations on factories.
- Standalone variants: **always** typed. Use `from __future__ import
  annotations` and `from typing import Optional, Tuple` at the top,
  type every public arg (`x: torch.Tensor`, `cfg: Optional[dict] = None`,
  `device: str = "cuda"`) and every return.

## Import Organization

**Order (observed in `train.py`, standalones, `models/model.py`):**
1. `from __future__ import annotations` (standalones, `models/layers.py`).
2. `import warnings` + `warnings.simplefilter(...)` if needed
   (only `train.py`).
3. `import sys; sys.path.append("..")` if the file lives under a
   subdirectory and needs repo-root imports (`models/model.py`,
   `models/discriminator.py`, `inference.py`). The core scripts at repo
   root do not need this; do not add it to new standalones.
4. Stdlib imports (`os`, `time`, `json`, `argparse`, `math`, `random`,
   `glob`, `re`, `pathlib`).
5. Third-party (`torch`, `torch.nn as nn`, `torch.nn.functional as F`,
   `numpy as np`, `soundfile as sf`, `librosa`, `matplotlib.pyplot as
   plt`, `pesq`, `joblib`, `rich.progress`, `huggingface_hub`,
   `datasets`).
6. Local (`from env import AttrDict`, `from dataset import ...`,
   `from models.model import ...`, `from models.layers import ...`,
   `from utils import ...`).

**Path aliases:** none. The repo root is on `sys.path` either implicitly
(scripts run from root) or explicitly via `sys.path.append("..")` from
files inside `models/`.

**Guarded optional imports:**
Use `try / except ImportError` with a `HAVE_X` flag for optional CUDA
extensions (see `models/layers.py:33-45`):

```python
try:
    from torch_structured import Butterfly
    HAVE_BUTTERFLY = True
except ImportError:
    Butterfly = None
    HAVE_BUTTERFLY = False
```

Then raise a clear `ImportError` from the factory only when the user
actually selects that backend.

## Error Handling

**Patterns:**
- **Validation in `__init__`:** raise `ValueError` with a message that
  echoes the offending value, e.g. `SMRBlock`:
  ```python
  if channels % groups != 0:
      raise ValueError(f"channels ({channels}) must be divisible by groups ({groups})")
  if not (0.0 <= a_min < a_max < 1.0):
      raise ValueError(f"need 0 <= a_min < a_max < 1, got [{a_min}, {a_max}]")
  ```
- **Unknown kind:** include the allowed set
  (`models/layers.py:80`):
  `raise ValueError(f"Unknown linear kind: {kind!r} (expected one of {LINEAR_KINDS})")`
- **PESQ failures** (`models/model.py:75-79`,
  `models/discriminator.py:13-19`): return sentinel `-1` and let the
  caller skip the step. `train.py:139-143` checks `batch_pesq_score is
  not None` and prints `"pesq is None!"` to stdout instead of raising —
  this is intentional (silent test windows commonly fail PESQ).
- **`assert` is used as a hard precondition** for I/O paths
  (`utils.py:19`: `assert os.path.isfile(filepath)`) and as a smoke-test
  assertion in standalone tests (`assert y.shape == x.shape`,
  `assert max_err < 1e-5`). Do not strip asserts.
- **No `try/except Exception` swallowing** in production code paths.
  The only bare `except` is `models/discriminator.py:cal_pesq` and is
  scoped to the PESQ call only.

## Logging

**Framework:** `print()` only. No `logging` module. TensorBoard
(`torch.utils.tensorboard.SummaryWriter`) for scalar/curve telemetry.

**stdout patterns (load-bearing — parsed downstream):**
- Train step (every `stdout_interval`):
  `Steps : {steps:d}, Gen Loss: {:4.3f}, Disc Loss: {:4.3f}, Metric loss: {:4.3f}, Magnitude Loss : {:4.3f}, Complex Loss : {:4.3f}, Time Loss : {:4.3f}, STFT Loss : {:4.3f}, s/b : {:4.3f}`
  (`train.py:181-182`).
- Validation:
  `Steps : {steps:d}, PESQ Score: {:4.3f}, s/b : {:4.3f}` — parsed by
  the analysis notebook (`analyze_sweep.ipynb`, cell `parse_pesq` uses
  the regex `r'Steps : (\d+), PESQ Score: ([\d.]+)'`). Do not change
  this format; it is the canonical comparator across runs.
- Checkpoint saves: `Saving checkpoint to {path}` (`utils.py:24`).
- Run banners in sweep logs:
  `=== [HH:MM:SS] Run: <name> -> cp_<name> (epochs=N, val=K) ===` and
  `=== [HH:MM:SS] Done: <name> (cp size: …) ===` (`run_sweep.sh:50,70`).
- Total parameter count at startup, e.g. `Total Parameters: 2.784M`
  (`train.py:38-39`). Standalone smoke tests print
  `"... OK. params=…M  in=… out=…"`.

**TensorBoard scalar names (stable identifiers consumed across runs):**
- `Training/Generator Loss`, `Training/Discriminator Loss`,
  `Training/Metric Loss`, `Training/Magnitude Loss`,
  `Training/Complex Loss`, `Training/Time Loss`,
  `Training/Consistency Loss`.
- `Validation/PESQ Score`, `Validation/Magnitude Loss`,
  `Validation/Complex Loss`, `Validation/Consistency Loss`.

## Comments / Docstrings

**Core pipeline:** sparse — typically one-line method docstrings only
(`dataset.py:Dataset` has the longest docstring in core). Inline
comments mark loss components (`# L2 Magnitude Loss`,
`# Discriminator`, `# Generator` in `train.py:133-152`). Keep this
minimal style in core.

**Standalone variants:** **mandatory rich docstrings.** Every standalone
opens with a multi-section module docstring covering:

1. Variant name + a one-line summary.
2. A "results table" or "config" block giving the exact PESQ + param
   count vs. the baseline (e.g. `nsnet2_encdec_v4_standalone.py:5-9`,
   `nsnet2_encdec_v5_tfcm_standalone.py:8-13`).
3. An ASCII pipeline diagram with tensor shapes annotated at every
   stage. This is non-optional — it's how reviewers verify the
   architecture without reading every block. See
   `nsnet2_encdec_v4_standalone.py:17-39` and
   `sensnet2_ch64_la4_FDbidir_freqnorm_grp4_standalone.py:14-29`.
4. A "Dependencies:" line (always `torch only` for standalones — no
   import of internal repo modules).
5. Pointer to the canonical checkpoint name (`cp_<name>/g_best`).

**Class-level docstrings** in standalones describe the math
(equations in plain text, not LaTeX) and the constructor knobs in an
`Args:` block. Example: `SMRBlock` documents its update equation
`h_t = a * h_{t-1} + g_t * u_t`, then explains every kwarg
(`groups`, `tau_min`, `tau_max`, `a_min`, `a_max`, `gated`,
`decay_param`, `use_layernorm`).

**Inline shape comments** on every tensor mutation — the core forward
contract. Example from `nsnet2_encdec_v4_standalone.py`:
```python
x = noisy_mag.unsqueeze(1).transpose(2, 3)   # (B, 1, T, F)
x = self.in_conv(x)                          # (B, C, T, F)
x = self.enc_dense(x)                        # (B, C, T, F)
...
mask = torch.sigmoid(x.squeeze(1).transpose(1, 2))  # (B, F, T)
```
Always include the shape comment for any reshape, permute, transpose,
or stride-changing conv.

**Section banners** separate logical groups in standalones (~75-char
underscore rule):
```python
# ---------------------------------------------------------------------------
# Model components.
# ---------------------------------------------------------------------------
```
Common sections: `Config`, `Model components`, `Main model`,
`STFT helpers`, `Convenience: load + enhance end-to-end`,
`Smoke test / CLI`, `Sanity tests`, `Benchmark utilities`.

## Function Design

**Size:** keep `nn.Module` `forward` methods short (most are 5–20
lines); push reusable transforms into helpers (`mag_pha_stft`,
`mag_pha_istft`, `_channel_shuffle`).

**Parameters:**
- Core training entry points (`train`, `main`) take the `(rank, a, h)`
  trio. Don't break this.
- Standalone constructors take a single `cfg: dict` and read keys via
  `cfg.get("key", default)` with explicit casts (`int(cfg.get(...))`,
  `bool(cfg.get(...))`). This mirrors how `train.py` reads from `h`
  (`getattr(h, "hidden_dim", 400)`).
- Factories accept positional in/out sizes and a keyword-only `cfg=None`
  (`make_linear(in_size, out_size, bias=True, *, cfg=None)`).
- Helper functions accept tensors plus the STFT params explicitly
  (`mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0,
  center=True)`) — never an `h`/`cfg` object directly. This keeps them
  reusable from notebooks and standalones.

**Return values:**
- Models return tuples — `(denoised_mag, denoised_pha, denoised_com)`
  is the canonical 3-tuple shape (`models/model.py:NSNet2.forward`,
  every standalone). New variants MUST match this contract or wrap to
  it. The standalone docstring explicitly notes this:
  `"Forward contract matches the other standalones so the call site
  stays uniform."` (`sensnet2_ch64_la4_FDbidir_freqnorm_grp4_standalone.py`).
- Streaming-capable variants additionally accept and return a `state`
  dict (or per-block tensor); the offline forward without state is
  byte-identical to chunked forward with state (verified by
  `test_streaming_equivalence`).

## Module Design

**Exports:** standalone variants are self-contained — they re-implement
`mag_pha_stft` / `mag_pha_istft` / `load_model` / `enhance_wav`
internally so a third party can copy a single file and run it. Do NOT
factor a standalone's helpers back into a shared utility module — this
defeats the "paste-and-go" promise.

**Barrel files (`__init__.py`):** the `models/` package has none. Import
from leaf modules directly (`from models.model import NSNet2`).

## Variant / Architecture Pattern

This is the central pattern of the repo. New variants follow it.

**Two ways to add a variant:**

1. **Pluggable backend in core `NSNet2`** — when the variant is just a
   layer swap (linear → butterfly/monarch). Add a `kind` to
   `models/layers.py` (`LINEAR_KINDS`, `GRU_KINDS`), wire it into
   `make_linear` / `make_gru`, and ship a config in `configs/<name>.json`
   that sets `"linear": {"kind": "..."}` and `"gru": {"kind": "..."}`.
   The model file (`models/model.py`) does not change — `NSNet2.__init__`
   reads `cfg["kind"]` from `h.linear` and `h.gru` and dispatches via
   the factory.

2. **Standalone architecture variant** — when the model topology changes
   (encoder/decoder, recurrence, TFCM, etc.). This is the dominant
   pattern. The standalone is the single source of truth for the
   variant; nothing in `models/` is touched. The standalone is
   *pasted* into the training loop later via a tracking PR (e.g.
   `e959a7f Standalone file for v5_tfcm`, `639c619 Integrate seNSNet2
   variant`, `b1e8e63 Add SMR-NSNet variant`).

**Common interface every variant satisfies (duck-typed):**

- `__init__(cfg: dict | h: AttrDict)` — accept a config object; read
  keys with `.get(key, default)` or `getattr(h, key, default)`.
- `forward(noisy_mag, noisy_pha) -> (denoised_mag, denoised_pha,
  denoised_com)` — three-tuple. Shape contract:
  `noisy_mag, noisy_pha: (B, F, T)`; `denoised_com: (B, F, T, 2)`.
- Module-level `CONFIG: dict` constant in standalones, with the *exact*
  hyperparameters used to train the published checkpoint.
- Module-level `load_model(checkpoint_path, cfg=None, device="cuda")`
  helper that calls `torch.load(..., weights_only=False)`, peels
  `state["generator"]` if present, and returns an eval-mode model.
- Module-level `enhance_wav(model, wav, cfg=None)` decorated with
  `@torch.no_grad()`, that RMS-normalises the input, runs STFT →
  forward → iSTFT, and unscales by the same RMS factor.
- `if __name__ == "__main__":` block that does either a 3-arg CLI
  (`ckpt_path, in_wav, out_wav` — load + enhance + write) or a
  zero-arg smoke test (random tensor forward + parameter print).

**RMS normalisation (load-bearing):** training and every standalone
multiply the waveform by `sqrt(L / (sum(wav²) + 1e-8))` before the STFT
and divide by it after the iSTFT (`dataset.py:72-74`,
`inference.py:30-31`, every standalone's `enhance_wav`). New variants
must preserve this scale convention or PESQ will collapse.

## Reproducibility

- **Single seed** in every config (`"seed": 1234`). Set in `train.py:285`
  via `torch.manual_seed(h.seed)` then `torch.cuda.manual_seed(h.seed)`
  per rank. The Dataset shuffle is seeded with the same `h.seed`
  (`dataset.py:54`).
- `torch.backends.cudnn.benchmark = True` (`train.py:22`) — bit-exact
  reproducibility is not required; epoch-level reproducibility is.
- Configs are immutable artifacts: `env.build_env` copies the training
  config into `cp_<run>/config.json` at run start so the run is
  self-describing. `inference.py:76` reads `config.json` from the
  checkpoint's directory; do not rely on the `configs/` source file at
  inference time.
- Standalones embed the canonical `CONFIG` dict in the file itself,
  redundantly with `cp_<name>/config.json` — this makes the file
  reproducible without the repo.

---

*Convention analysis: 2026-04-27*
