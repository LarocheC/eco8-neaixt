# Codebase Structure

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

## Directory Layout

```
sparse-nsnet2/
├── train.py                     # Training entry point (GAN loop, DDP-aware)
├── inference.py                 # Single-checkpoint inference entry point
├── dataset.py                   # HF VoiceBank-DEMAND-16k Dataset + STFT helpers
├── env.py                       # AttrDict + build_env (config copy)
├── utils.py                     # scan/load/save_checkpoint, LearnableSigmoid1d
├── run_sweep.sh                 # Sequential per-config sweep runner
├── analyze_sweep.ipynb          # Results / weight-pattern analysis notebook
├── pyproject.toml               # uv-managed deps; torch pinned to cu118 wheels
├── uv.lock                      # uv lockfile
├── .python-version              # 3.12 (per .python-version)
├── config.json                  # Default top-level config (small, baseline-shape)
├── README.md                    # Setup / training / sweep / checkpoints docs
├── LICENSE                      # MIT
├── .gitmodules                  # Pins torch-structured submodule
│
├── configs/                     # Per-run sweep configs (one JSON per cp_<name>)
│   ├── baseline.json
│   ├── butterfly_2blocks.json
│   ├── butterfly_fc.json
│   ├── butterfly_full.json
│   ├── butterfly_ortho.json
│   ├── monarch_8.json
│   ├── monarch_fc.json
│   ├── monarch_full.json
│   └── wide_monarch.json
│
├── models/                      # Trunk model package
│   ├── model.py                 # NSNet2 (~50 lines, pure wiring)
│   ├── layers.py                # make_linear, make_gru, StructuredGRU(Cell)
│   └── discriminator.py         # MetricDiscriminator + batch_pesq
│
├── torch-structured/            # Git submodule: structured-matrix primitives
│   ├── torch_structured/
│   │   ├── __init__.py          # Re-exports Butterfly, LRU, make_linear
│   │   ├── factory.py
│   │   ├── butterfly/           # Butterfly + special transforms (FFT/DCT/...)
│   │   ├── monarch/             # BlockdiagLinear, monarch products, flash_mm
│   │   ├── structured/          # LDR / Toeplitz / Fastfood / Hadamard / ...
│   │   ├── recurrent/           # LRU
│   │   └── *.so                 # Compiled C++/CUDA extensions
│   ├── csrc/                    # C++/CUDA source for the .so extensions
│   ├── tests/                   # Submodule's own pytest suite
│   ├── experiments/             # Submodule's reference experiments
│   ├── pyproject.toml
│   └── setup.py                 # Builds CUDA extensions on uv sync
│
├── cp_<run>/                    # Per-run output dirs (one per sweep config)
│   ├── config.json              # Copy of launching configs/<run>.json
│   ├── train.log                # tee'd stdout from run_sweep.sh
│   ├── logs/                    # Tensorboard event files
│   ├── g_<step>                 # Latest rolling generator checkpoint
│   ├── do_<step>                # Latest rolling discriminator + optim state
│   └── g_best                   # Best-PESQ generator (after best_start_epoch)
│
├── *.log                        # sweep-level stdout aggregations (gitignored or stale)
└── .planning/codebase/          # GSD codebase analysis docs (this directory)
```

## Directory Purposes

**Repo root (flat by design):**
- Purpose: All entry points and small support modules live as bare `.py` files at the repo root - the project is small enough not to warrant a package layout.
- Contains: training / inference / dataset / config-helpers / utils / sweep-runner / analysis notebook / project metadata.
- Key files: `train.py`, `inference.py`, `dataset.py`, `env.py`, `utils.py`, `run_sweep.sh`, `analyze_sweep.ipynb`, `pyproject.toml`, `README.md`.
- Standalone variants (per project memory): new alternative architectures (`SMRNSNet`, `NSNet2EncDec`, `SeNSNet2`, LRU-based, etc.) live as `*_standalone.py` files **at this level**, each carrying its own `train(...)` and `main()` so they can be launched directly via `uv run python <variant>_standalone.py`. None present in the working tree at HEAD - the legacy `cp_lru_*`, `cp_smr_*`, `cp_encdec_v[1-8]*`, `cp_sensnet2_*` directories were produced by such files.

**`configs/`:**
- Purpose: Snapshot of each run's hyperparameters in the trunk-model sweep. One JSON per `cp_<name>/`.
- Contains: 9 JSON files, one per name in `run_sweep.sh`'s `ORIGINAL_RUNS` + `NEW_RUNS`. They differ almost exclusively in the `linear` / `gru` blocks (and `hidden_dim` / `fc_hidden_dim` for `wide_monarch`).
- Key files: `configs/baseline.json` (dense reference), `configs/butterfly_full.json`, `configs/monarch_full.json`, `configs/wide_monarch.json` (768/1024-dim monarch).

**`models/`:**
- Purpose: The trunk model and its discriminator.
- Contains: 3 modules. No `__init__.py` re-exports (every consumer imports the leaf module path).
- Key files: `models/model.py` (`NSNet2`), `models/layers.py` (factories + `StructuredGRU`), `models/discriminator.py` (`MetricDiscriminator`, `batch_pesq`).

**`torch-structured/` (git submodule):**
- Purpose: Vendored structured-matrix library (Butterfly / Monarch / LDR / LRU). Pinned to v0.4.0 (per recent commit `aa69d53 Pin torch-structured submodule to v0.4.0`).
- Contains: Python package `torch_structured/`, C++/CUDA sources `csrc/`, its own `tests/`, `experiments/`, `pyproject.toml`, `setup.py`.
- Generated: yes - `.so` extension modules are built on `uv sync` (also rebuild on submodule pointer changes).
- Committed: source yes; `.so` no.

**`cp_<run>/`:**
- Purpose: Self-contained record of one experiment.
- Contains: config copy, stdout log, tensorboard event dir, latest rolling checkpoints, `g_best`.
- Generated: yes (created by `train.py` / `run_sweep.sh`); never edited by hand.
- Committed: no (the directory pattern is in `.gitignore`); the trained generators are mirrored externally on HuggingFace at `claroche1/sparse-nsnet2-checkpoints`.
- Key sub-files: `config.json` (single source of truth for re-running this experiment), `g_best` (used by `inference.py` and `analyze_sweep.ipynb`), `logs/events.out.tfevents.*`.

**`.planning/codebase/`:**
- Purpose: GSD codebase-mapping analysis docs (consumed by `/gsd-plan-phase` and `/gsd-execute-phase`).
- Contains: `ARCHITECTURE.md`, `STRUCTURE.md` (this file), and any other focus-area docs the orchestrator writes.
- Committed: typically yes (kept in repo for cross-session continuity).

**`__pycache__/`, `.venv/`, `.git/`:** standard tooling caches; not part of the source.

## Key File Locations

**Entry Points:**
- `train.py`: training entry point (`main` at `train.py:259`, training loop at `train.py:25`).
- `inference.py`: single-checkpoint inference entry point (`main` at `inference.py:65`).
- `run_sweep.sh`: sequential sweep over `configs/<name>.json` (`run_sweep.sh:39` - the loop).
- `analyze_sweep.ipynb`: results notebook (loads each `cp_<run>/g_best`, plots PESQ + weight matrices).

**Configuration:**
- `configs/<run>.json`: per-run hyperparameter file consumed by `train.py --config`.
- `cp_<run>/config.json`: auto-copied snapshot used by `inference.py` (it reads the sibling of the checkpoint file).
- `pyproject.toml`: dependency manifest; pins `torch` to the cu118 wheel index, declares `torch-structured` as an editable submodule path dep.
- `.python-version`: pins the Python interpreter version for `uv`.

**Core Logic:**
- `models/model.py`: `NSNet2` (the only generator class in the trunk).
- `models/layers.py`: `make_linear`, `make_gru`, `StructuredGRUCell`, `StructuredGRU` (the variant-dispatch layer).
- `models/discriminator.py`: `MetricDiscriminator`, `batch_pesq`, `cal_pesq`.
- `dataset.py`: `Dataset`, `mag_pha_stft`, `mag_pha_istft`, `load_voicebank_demand`.
- `env.py`: `AttrDict`, `build_env`.
- `utils.py`: `scan_checkpoint`, `load_checkpoint`, `save_checkpoint`, `LearnableSigmoid1d`.

**Submodule (when adding a new structured layer):**
- `torch-structured/torch_structured/__init__.py`: top-level re-exports.
- `torch-structured/torch_structured/butterfly/butterfly.py`: `Butterfly`.
- `torch-structured/torch_structured/monarch/blockdiag_linear.py`: `BlockdiagLinear`.
- `torch-structured/torch_structured/recurrent/lru.py`: `LRU`.
- `torch-structured/torch_structured/structured/`: LDR / Toeplitz / Fastfood / Hadamard / Krylov.

**Testing:**
- No tests in the trunk repo. The submodule has its own pytest suite at `torch-structured/tests/` (not run by the trunk's CI - there is no CI configured).

## Naming Conventions

**Files:**
- Trunk Python modules: `snake_case.py` (one word where possible: `train.py`, `dataset.py`, `utils.py`, `env.py`, `inference.py`).
- Standalone variants: `<variant>_standalone.py` at repo root (e.g. `lru_standalone.py`, `smr_standalone.py`, `encdec_standalone.py`, `sensnet2_standalone.py` per project memory). The `_standalone.py` suffix is the marker that the file is a self-contained training script for an alternative architecture.
- Configs: `configs/<run>.json` where `<run>` is also the suffix of the matching `cp_<run>/` directory.
- Checkpoint dirs: `cp_<run>/` always prefixed with `cp_`; subgroups use underscores: `cp_butterfly_2blocks`, `cp_lru_deep_narrow`, `cp_encdec_v3_unet`, `cp_sensnet2_ch64_la4_FDbidir_freqnorm_grp4`.
- Checkpoint files: `g_<step:08d>` (generator) and `do_<step:08d>` (discriminator + optimizer state) for rolling checkpoints; `g_best` for the best-PESQ snapshot. The `g_/do_` glob in `utils.scan_checkpoint` requires exactly 8 digits.
- Sweep logs: `<group>_sweep.log` at repo root (e.g. `sweep.log`, `lru_sweep.log`, `smr_sweep.log`, `encdec_sweep.log`); `cp_<run>/train.log` per-run.
- Shell scripts: lowercase `_` separated with `.sh` extension (`run_sweep.sh`).

**Directories:**
- All lowercase, snake-case-ish (`cp_butterfly_full`, `torch-structured`). One hyphenated exception: the submodule directory uses a hyphen (`torch-structured`) to match its GitHub repo name, while the Python package inside uses an underscore (`torch_structured`).

**Identifiers (Python):**
- Modules / functions / variables: `snake_case` (`make_linear`, `mag_pha_stft`, `noisy_mag`).
- Classes: `PascalCase` (`NSNet2`, `MetricDiscriminator`, `StructuredGRUCell`, `StructuredGRU`, `LearnableSigmoid1d`, `AttrDict`, `Dataset`).
- Constants: `UPPER_SNAKE` (`HF_DATASET_NAME`, `HAVE_BUTTERFLY`, `HAVE_MONARCH`, `LINEAR_KINDS`, `GRU_KINDS`).
- Config keys: `lower_snake` (`hidden_dim`, `fc_hidden_dim`, `num_gru_layers`, `compress_factor`, `dist_config`).
- Variant tags inside configs: short lowercase (`"linear"`, `"butterfly"`, `"monarch"`, `"gru"`, `"lru"`, `"smr"`, `"encdec"`, `"sensnet2"`).

**STFT tensors (recurring):**
- `mag` / `pha` / `com` for magnitude / phase / complex (real-imag-stacked) representation. Suffix `_g` for generator output, `_g_hat` for re-STFT'ed generator output, `clean_*` / `noisy_*` for ground truth / input.

## Where to Add New Code

**New trunk-model variant (dense -> structured swap):**
- Add a new `kind` branch in `make_linear` / `make_gru` in `models/layers.py`.
- Add a corresponding `configs/<name>.json` with `linear.kind` / `gru.kind` set to the new value.
- Append `<name>` to `NEW_RUNS` in `run_sweep.sh:21`.
- Re-export the new primitive from `torch-structured/torch_structured/__init__.py` if it lives in the submodule.

**New whole-architecture variant (different model class):**
- Per project memory: create `<variant>_standalone.py` **at the repo root** (sibling of `train.py`).
- Vendor whatever is needed for the variant's training loop: dataset use, STFT helpers, model class, loss schedule. The standalone file is *expected* to repeat scaffolding from `train.py` rather than subclass it - that is a deliberate trade for keeping the trunk `NSNet2` clean.
- Use the same JSON config shape (with a `model_kind` field and a variant-specific sub-block, see `cp_smr_baseline/config.json`, `cp_encdec_v1_pw/config.json`, `cp_sensnet2_*/config.json` for the existing schemas to reuse).
- Run via `uv run python <variant>_standalone.py --config configs/<variant>.json --checkpoint_path cp_<variant>`.
- Output goes into a fresh `cp_<variant>/` directory following the same `config.json` / `train.log` / `logs/` / `g_<step>` / `do_<step>` / `g_best` layout so `analyze_sweep.ipynb` can still load it.

**New loss / training tweak in the trunk:**
- Edit `train.py` (the loss-weighting block at `train.py:164` and the validation logging block at `train.py:194` are the two main extension points).
- Add any new metric to the tensorboard scalar block to keep it in `analyze_sweep.ipynb`'s reach.

**New dataset / preprocessing:**
- Extend `dataset.py`: add a new loader function next to `load_voicebank_demand` and a new `Dataset`-style class if the schema differs. Keep `mag_pha_stft` / `mag_pha_istft` as the shared STFT primitives.

**New utility:**
- Single-purpose helpers go in `utils.py`. Anything model-specific (custom activations, blocks) goes in `models/layers.py`.

**New structured primitive in the submodule:**
- Add module under `torch-structured/torch_structured/<subpackage>/`, re-export from the subpackage's `__init__.py` and (if it should be top-level) from `torch-structured/torch_structured/__init__.py`. Then expose it from the trunk via a new branch in `models/layers.py:make_linear` / `make_gru`.

**New analysis output:**
- Append cells to `analyze_sweep.ipynb`. The notebook's existing pattern is: run-name list -> per-run `cp_<run>/config.json` + `g_best` -> derived metrics / plots.

## Special Directories

**`torch-structured/`:**
- Purpose: Git submodule pointing at `git@github.com:LarocheC/torch-structured.git` (per `.gitmodules`), pinned to v0.4.0.
- Generated: source no, compiled `.so` extensions yes (built by `uv sync` via `setup.py`). Listed in `pyproject.toml` as `no-build-isolation-package = ["torch-structured"]` and resolved as an editable path dependency.
- Committed: source committed in the submodule repo; this repo only stores the commit pointer.

**`cp_<run>/`:**
- Purpose: Per-run experiment output (see above).
- Generated: yes by `train.py`.
- Committed: no - matches the `cp_*` pattern in `.gitignore`.

**`.venv/`:**
- Purpose: uv-managed virtualenv; activated by `run_sweep.sh:37` and used as the interpreter for `uv run`.
- Generated: yes by `uv sync`.
- Committed: no.

**`__pycache__/`:**
- Purpose: CPython bytecode cache.
- Generated: yes.
- Committed: no.

**`.planning/`:**
- Purpose: GSD planning artifacts. `codebase/` holds long-lived analysis docs; sibling subdirectories hold per-task plans / progress.
- Generated: by GSD commands.
- Committed: typically yes.

**`.claude/`:**
- Purpose: Claude Code project-local config (settings, command overrides, optional skills).
- Generated: yes (currently untracked per `git status`).
- Committed: optional.

**Top-level `*.log` files (`sweep.log`, `lru_sweep.log`, `smr_sweep.log`, `encdec_sweep.log`):**
- Purpose: Aggregated stdout from full sweep invocations (`./run_sweep.sh > sweep.log 2>&1 &`).
- Generated: by manual sweep invocations.
- Committed: no (matched by `.gitignore`).

---

*Structure analysis: 2026-04-27*
