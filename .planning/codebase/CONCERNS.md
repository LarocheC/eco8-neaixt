# Codebase Concerns

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

This is research code. The audit below distinguishes between "intentional research-grade tradeoffs" (acceptable for an exploratory NSNet2 sweep repo) and "actual risks" — things that will silently bite a collaborator, reviewer, or future-self trying to publish, reproduce, or extend the work.

---

## Tech Debt

### Orphaned checkpoints from deleted research variants  *(HIGH — actual risk)*
- Issue: 17 of the 27 `cp_*` directories on disk reference architectural variants whose source code is **not on `main`**. The configs persist in each `cp_*/config.json` but the matching `*_standalone.py` files and `models/encdec.py`, `models/smr.py`, `models/sensnet2.py` only live on the unmerged branches `research/enc-dec-study` and `research/lru-integration`.
- Evidence:
  - `cp_encdec_v1_pw` ... `cp_encdec_v8a_fpe` (8 dirs) reference `"model_kind": "encdec"` — handler not on `main`.
  - `cp_lru_baseline`, `cp_lru_deep_narrow`, `cp_lru_monarch` reference `"gru": {"kind": "lru"}` — `lru` is not in `LINEAR_KINDS` or `GRU_KINDS` in `models/layers.py:52,170`.
  - `cp_smr_baseline`, `cp_smr_gru_match`, `cp_smr_long_mem`, `cp_smr_no_conv`, `cp_smr_pure_lru` reference `"model_kind": "smr"`.
  - `cp_sensnet2_ch64_la4_FDbidir_freqnorm_grp4` references `"model_kind": "sensnet2"`.
  - `__pycache__/nsnet2_encdec_v4_standalone.cpython-312.pyc` and `__pycache__/nsnet2_encdec_v5_tfcm_standalone.cpython-312.pyc` are leftover compiled artifacts of deleted source files.
  - Branches `research/enc-dec-study` and `research/lru-integration` contain `models/encdec.py`, `models/smr.py`, `models/sensnet2.py`, `nsnet2_encdec_v4_standalone.py`, `nsnet2_encdec_v5_tfcm_standalone.py`, `sensnet2_ch64_la4_FDbidir_freqnorm_grp4_standalone.py`, `smr_nsnet_standalone.py`, plus `configs/encdec_*.json`, `configs/lru_*.json`, `configs/smr_*.json`, `configs/sensnet2_*.json` — none on `main`.
- Files: `cp_encdec_*/`, `cp_lru_*/`, `cp_smr_*/`, `cp_sensnet2_*/`, `__pycache__/*standalone*.pyc`
- Impact: Anyone running `inference.py --checkpoint_file cp_smr_baseline/g_best` from `main` gets a silent state-dict mismatch (`inference.py:39` instantiates `NSNet2(h)`, ignores `model_kind`, then `load_state_dict` will explode with missing/unexpected keys). The `g_best` weights for these runs are stranded — irreproducible from `main`.
- Fix approach: pick one of (a) merge the research branches into `main` and reinstate the architecture dispatch, (b) move the orphaned `cp_*` dirs to a `.archive/` location off-tree (they're already ignored by `.gitignore` so deletion is local-only), or (c) keep the configs but add a banner in README pointing collaborators at the correct branch for each variant. Option (a) is the right one if the encdec/SMR/LRU work is ever going to be published.

### Stale `__pycache__/` at repo root  *(LOW — cleanup)*
- Issue: `__pycache__/nsnet2_encdec_v4_standalone.cpython-312.pyc` and `__pycache__/nsnet2_encdec_v5_tfcm_standalone.cpython-312.pyc` (Apr 21, 16-17 KB each) are byte-compiled snapshots of source files no longer on `main`. Confusing forensic evidence and a (very mild) license/IP concern if the repo is ever shipped as-is.
- Files: `__pycache__/nsnet2_encdec_v4_standalone.cpython-312.pyc`, `__pycache__/nsnet2_encdec_v5_tfcm_standalone.cpython-312.pyc`, `__pycache__/dataset.cpython-312.pyc`, `__pycache__/env.cpython-312.pyc`, `__pycache__/utils.cpython-312.pyc`
- Impact: None at runtime — `__pycache__/` is in `.gitignore` so untracked. But `pyc` files for absent `.py` files are misleading.
- Fix approach: `rm -rf __pycache__/`. Optional: add a `make clean` target.

### Duplicate `config.json` at repo root  *(LOW — research-grade)*
- Issue: `config.json` at the repo root is a **near-duplicate** of `configs/baseline.json` with different hyperparameters (`batch_size: 4` vs `256`, `lr: 0.0005` vs `0.003`, `n_fft: 400` vs `512`). It's not loaded by any code path on `main` (`train.py:268` requires `--config` and `inference.py:76` reads from the checkpoint dir).
- Files: `config.json`
- Impact: Confusing dead config — looks like a default but isn't.
- Fix approach: delete `config.json`, or move it to `configs/dev_smoke.json` if it's actually used as a quick local sanity-check setup.

### Hardcoded `n_jobs=30` in PESQ scoring  *(LOW — research-grade)*
- Issue: `models/model.py:67` hardcodes `Parallel(n_jobs=30)`. `models/discriminator.py:23` uses `n_jobs=-1`. Inconsistent and not exposed via config.
- Files: `models/model.py:67`, `models/discriminator.py:23`
- Impact: Wastes cores or oversubscribes depending on host. On a 4-core CI box, `n_jobs=30` is wasteful but not broken.
- Fix approach: thread `n_jobs` through config (`h.pesq_n_jobs`) or default to `os.cpu_count()`.

### `sys.path.append("..")` boilerplate  *(LOW — cosmetic)*
- Issue: `train.py:4`, `inference.py:3`, `models/discriminator.py:2` all append `..` to `sys.path`. Holdover from the pre-`uv`/non-package layout. The package now installs cleanly via `uv sync` so these lines are no-ops.
- Files: `train.py:4`, `inference.py:3`, `models/discriminator.py:2`
- Impact: None functional — just dead lines.
- Fix approach: delete; verify imports still resolve.

### `from utils import *` in discriminator  *(LOW — style)*
- Issue: `models/discriminator.py:10` uses `from utils import *` to pull in `LearnableSigmoid1d`. Star-imports hurt readability and obscure what `utils.py` exposes.
- Files: `models/discriminator.py:10`
- Impact: None functional.
- Fix approach: replace with `from utils import LearnableSigmoid1d`.

### Blanket `FutureWarning` suppression  *(LOW — research-grade)*
- Issue: `train.py:2` silences **all** `FutureWarning`s globally. Hides upcoming PyTorch / numpy / datasets API removals.
- Files: `train.py:1-2`
- Impact: Will mask the pre-3.0 deprecation of `torch.load(weights_only=False)` default and similar API churn during the torch 2.4–2.6 window pinned in `pyproject.toml:7`.
- Fix approach: remove the blanket suppression; address warnings as they appear, or scope the filter narrowly.

---

## Known Bugs

### `best_pesq` resets to 0 on resume  *(MEDIUM — silent quality regression)*
- Symptoms: When resuming from a checkpoint, `best_pesq` is re-initialized to `0` at `train.py:102`, before the loop. The first validation pass after resume that beats `0` (always) overwrites `g_best` — even if it's worse than the pre-resume best.
- Files: `train.py:102, 242-246`
- Trigger: `python train.py --config ... --checkpoint_path <existing>` resumes from latest `g_*`/`do_*`, but the previous best PESQ is lost.
- Workaround: don't resume into a directory you care about, or copy `g_best` aside first.
- Fix approach: persist `best_pesq` inside the `do_*` checkpoint and reload it at `train.py:54-59`.

### `loss_disc_g = 0` (Python int, not tensor) on PESQ failure  *(LOW — cosmetic, no NaN)*
- Symptoms: `train.py:143` sets `loss_disc_g = 0` (a Python int) when `batch_pesq_score` is `None`. The next line `loss_disc_all = loss_disc_r + loss_disc_g` works because `tensor + int` broadcasts, but `.backward()` then propagates only through `loss_disc_r`. This is fine functionally — just unusual idiom.
- Files: `train.py:139-145`
- Trigger: Whenever PESQ scoring fails on every utterance in the batch (silent clip, etc.); first epochs typically hit it (`encdec_sweep.log` shows several `pesq is None!` lines in early epochs).
- Workaround: none needed.
- Fix approach: cosmetic — `loss_disc_g = torch.zeros((), device=device)` for clarity.

### `validation_loader` undefined when `validation_interval` never hit  *(LOW — edge case)*
- Symptoms: `validation_loader` is built only on `rank == 0` (`train.py:91`). The validation block at `train.py:203` is also `rank == 0`-gated, so this is fine in practice. But the `validset = Dataset(...)` constructor (`train.py:88`) loads the full HF test split into the dataloader regardless of whether validation will ever run — wasted memory if `training_epochs * len(loader) // validation_interval == 0`.
- Files: `train.py:87-95`
- Trigger: Smoke runs with very few steps.
- Workaround: none.
- Fix approach: lazy-construct the validation loader on first use.

---

## Security Considerations

### `torch.load(weights_only=False)` default  *(MEDIUM — supply-chain)*
- Risk: `utils.py:20` and `inference.py:23` both call `torch.load(filepath, map_location=device)` without `weights_only=True`. PyTorch 2.4 still defaults to `weights_only=False`, which executes arbitrary pickle on load — meaning a malicious checkpoint can run code at import.
- Files: `utils.py:17-20`, `inference.py:20-25`
- Current mitigation: All checkpoints are produced by this repo locally. The README's HuggingFace example (`README.md:117`) explicitly sets `weights_only=False`.
- Recommendations: switch to `weights_only=True` for the local load paths. The state-dicts here are pure tensor dicts so there's nothing pickle-rich to lose. PyTorch 2.6+ flips the default — pinning `torch<2.6` (`pyproject.toml:7`) means this concern stays live until the pin is bumped.

### `.env` / secrets — not present  *(NONE — clean)*
- No `.env`, no API keys, no credentials in tracked files. HuggingFace Hub access is anonymous read on `JacobLinCool/VoiceBank-DEMAND-16k`. The README mention of `claroche1/sparse-nsnet2-checkpoints` is a public HF repo.

### `dist_url: tcp://localhost:54321` in every config  *(LOW — single-host only)*
- Risk: Hardcoded multi-GPU rendezvous URL across all `configs/*.json`. Two simultaneous training runs on the same machine collide.
- Files: `configs/baseline.json:27`, all 9 `configs/*.json` carry the same line.
- Current mitigation: `run_sweep.sh:39` runs configs sequentially.
- Recommendations: parameterize via env var or config interpolation if multi-job training is ever wanted. Single-GPU runs (the entire current sweep is `num_gpus: 0` in config + auto-set from `torch.cuda.device_count()` at `train.py:288`) don't hit this code path.

---

## Performance Bottlenecks

### Python-loop `StructuredGRU` time recurrence  *(KNOWN — by design)*
- Problem: `models/layers.py:147-163` runs a Python `for t in range(T)` loop calling `StructuredGRUCell` per timestep. With `T = 32000 / 256 ≈ 126` frames per training segment and `B = 256`, this is ~126 Python-level launches per minibatch — orders of magnitude slower than cuDNN's fused GRU.
- Files: `models/layers.py:121-163`
- Cause: Structured projections (`Butterfly`, `BlockdiagLinear`) can't be packed into cuDNN's recurrence kernel. The Python loop is the price of pluggable backends.
- Improvement path: (a) `torch.jit.script` the cell + loop, (b) write a CUDA fused kernel in `torch-structured`, (c) accept it — the docstring at `models/layers.py:129-130` already calls this out as intentional. **For a research repo, this is fine.**

### `cudnn.benchmark = True`  *(KNOWN — appropriate)*
- `train.py:22` enables benchmark autotuning. Good for fixed-shape training. Comes at the cost of non-determinism — interacts with the seeding gap below.

### PESQ joblib pool overhead per validation  *(LOW — research-grade)*
- Problem: `pesq_score()` at `models/model.py:66-72` spawns a fresh `Parallel(n_jobs=30)` pool **every validation call** (every 5000 steps by default, every 200 in the sweep script). Pool startup overhead dominates for small validation sets.
- Files: `models/model.py:66`, `models/discriminator.py:22`
- Cause: No persistent worker pool.
- Improvement path: use `Parallel(..., backend='loky')` with a context manager scoped to the training loop, or `torch.multiprocessing.Pool` reused across calls.

---

## Fragile Areas

### Coupling between `NSNet2(h)` and config-driven backend switching  *(MEDIUM — by design but easy to break)*
- Files: `models/model.py:27-45`, `models/layers.py:55-198`, `inference.py:39-41`
- Why fragile: `NSNet2` reads `h.linear` and `h.gru` config blocks at construction time. Any new backend kind (e.g. the `lru` kind referenced in `cp_lru_*` configs) must be added to **both** `LINEAR_KINDS`/`GRU_KINDS` and the `make_linear`/`make_gru` factory dispatch. If a config references an unknown kind, you only find out at construction (`models/layers.py:80, 190` raise `ValueError`) — and `inference.py:39` instantiates `NSNet2` *before* `load_state_dict`, so the error surfaces with no checkpoint context.
- Safe modification: when adding a backend, update the `LINEAR_KINDS` / `GRU_KINDS` tuples for early validation, the README backend table, and `models/layers.py` factory together. Treat the constants as the source of truth.
- Test coverage: **none on main** — no `tests/` dir, no smoke test that walks every `configs/*.json`, instantiates `NSNet2`, and runs one forward pass.

### `inference.py` doesn't honor `model_kind`  *(MEDIUM — see also "Orphaned checkpoints" above)*
- Files: `inference.py:39`
- Why fragile: `inference.py` always builds `NSNet2(h)`. There is no dispatch on `h.model_kind`. Loading any encdec/smr/sensnet2 checkpoint from the orphaned `cp_*` dirs will fail with state-dict-key errors.
- Safe modification: when (if) the research branches merge, add a `build_model(h)` helper that dispatches on `h.get("model_kind", "nsnet2")` and use it in both `train.py:33` and `inference.py:39`.

### Submodule pin: `torch-structured @ v0.4.0`  *(LOW — just-fixed)*
- Files: `.gitmodules`, top-level `torch-structured` gitlink
- Why fragile *was*: prior to commit `aa69d53` the submodule was unpinned (just tracked HEAD of the upstream main). Now pinned to tag `v0.4.0` at commit `ceb76e0`. Verified via `git submodule status` — clean, detached HEAD at the tag.
- Remaining risk: if the upstream repo `LarocheC/torch-structured` is ever moved or the tag re-pointed, fresh `git clone --recursive` will silently get different code. Tags are mutable in git.
- Fix approach: optionally also commit a `requirements-frozen.txt` snapshot of the built `torch_structured` `__version__` for cross-checking. Or rely on `uv.lock` (already tracked, 332 KB).

### `train.py` resume logic + `cudnn.benchmark` + missing `python random.seed`  *(MEDIUM — reproducibility)*
- Files: `train.py:22, 30, 285-287`
- Why fragile: `torch.manual_seed`/`torch.cuda.manual_seed` are set, but **`numpy.random.seed` and `random.seed` are not**. `dataset.py:54` uses a `random.Random(seed)` *local* RNG for the index shuffle (good), but `dataset.py:79`'s `random.randint` for segment-start cropping uses the **global** `random` module — uncontrolled across workers. Also `cudnn.benchmark = True` (`train.py:22`) introduces non-determinism in cuDNN algo selection.
- Impact: Two reruns of the same config with the same seed will produce slightly different PESQ trajectories. Not a problem for the headline-results sweep (numbers are stable across seeds for these architectures) but a problem if anyone tries to *exactly* reproduce a published number.
- Fix approach: add `np.random.seed(h.seed); random.seed(h.seed)` and a `worker_init_fn` to the DataLoader; document the cuDNN-benchmark non-determinism caveat in README.

### `env.py` is misnamed  *(LOW — naming)*
- Files: `env.py`
- Why fragile: `env.py` does **not** handle environment variables — it defines `AttrDict` and `build_env` (which copies a config file). Confusing for new readers who expect a 12-factor-style env loader. No `os.environ`, no env-specific assumptions.
- Safe modification: rename to `config_utils.py` if a refactor pass happens; otherwise leave alone.

### Path assumption: `inference.py` reads `config.json` from checkpoint dir  *(LOW — by design)*
- Files: `inference.py:76`
- Why fragile: `os.path.split(a.checkpoint_file)[0]` + `'config.json'`. Works because `train.py:283` calls `build_env` which copies the config there. Breaks if someone moves `g_best` elsewhere.
- Safe modification: consider an explicit `--config` flag on `inference.py` as a fallback.

---

## Scaling Limits

### Sequential sweep runner  *(LOW — research-grade)*
- Current capacity: `run_sweep.sh:39` runs configs **sequentially** on one GPU. The 9-run sweep at 50 epochs takes a wall-clock day or so on a single Pascal/Ampere GPU.
- Limit: wall clock; nothing parallel.
- Scaling path: trivial to fan out across GPUs by replacing the `for name in $RUNS` loop with `parallel -j` or per-GPU `CUDA_VISIBLE_DEVICES` assignment.

### `joblib n_jobs=30` PESQ pool  *(LOW)*
- Current capacity: 30 worker processes for PESQ on validation.
- Limit: hosts with <30 cores oversubscribe; hosts with >30 cores leave headroom.
- Scaling path: see "Hardcoded `n_jobs=30`" above.

---

## Dependencies at Risk

### `torch < 2.6` pin  *(MEDIUM — time-bound)*
- Files: `pyproject.toml:7`
- Risk: PyTorch 2.6 (released late 2025) flips `torch.load(weights_only=...)` default. Once the pin is lifted, the existing `torch.load` calls (`utils.py:20`, `inference.py:23`) will fail to load any checkpoint that wasn't saved as a pure tensor dict. They probably are pure, so this should be fine — but untested.
- Impact: easy fix when bumping; easy to miss.
- Migration plan: when bumping to `torch>=2.6`, audit all `torch.load` sites and add explicit `weights_only=True` (and verify checkpoint contents are pickle-safe).

### `torch-structured` upstream  *(LOW — pinned)*
- Files: `.gitmodules`, `pyproject.toml:30` (`editable = true` from local submodule path)
- Risk: Submodule pinned to `v0.4.0`. The package builds CUDA kernels on `uv sync`, so any environment without `gcc-11`/CUDA dev headers fails install (README.md:48-53 calls this out clearly). The `FORCE_CPU=1` fallback is documented but `models/layers.py:36-38, 43-45` only guards imports — actual `Butterfly()` / `BlockdiagLinear()` construction will still fail without compiled extensions.
- Impact: collaborator on a CPU-only laptop can `uv sync` but not actually run the structured configs.
- Migration plan: documented in README. Acceptable for a research repo. Long-term, a pure-PyTorch fallback inside `torch-structured` would help.

### `pesq` package  *(LOW — fragile native lib)*
- Files: `pyproject.toml:14`
- Risk: `pesq>=0.0.4` is a thin wrapper over a 1990s ITU C reference implementation. Builds break on novel platforms (Apple Silicon historically). Non-numeric returns are caught broadly (`models/model.py:78` `except Exception: return -1`).
- Impact: validation breaks silently if `pesq` import or call fails.
- Migration plan: none currently; the PESQ-as-discriminator-target loss (MetricGAN) makes this a hard dep.

### `datasets >= 2.14, < 4` (HuggingFace)  *(LOW)*
- Files: `pyproject.toml:15`
- Risk: HF `datasets` 3.0 changed audio decoding semantics in subtle ways. Pinned `<4` for now. `dataset.py:62-63` does `np.asarray(item["clean"]["array"], dtype=np.float32)` — relies on the dict-with-`array`-key shape, stable across 2.x and 3.x.
- Migration plan: track HF release notes when bumping the cap.

---

## Missing Critical Features

### No automated test suite  *(MEDIUM — for shareability)*
- Problem: zero tests on `main`. The `torch-structured/tests/` directory is upstream library tests, not this repo's. No smoke test that constructs each config, no checkpoint-roundtrip test.
- Blocks: any CI; any safe refactor; any "hand this to a collaborator" use case.
- Fix approach: minimal `tests/test_smoke.py` that for each `configs/*.json`: builds `NSNet2(h)`, runs one forward pass on a random `(B=1, T=16000)` waveform, asserts shapes. Plus a `test_roundtrip` for `save_checkpoint` -> `load_checkpoint`.

### No `make_model(h)` dispatch  *(MEDIUM — see Fragile Areas)*
- Problem: `train.py:33` and `inference.py:39` both hardcode `NSNet2(h)`. No place to dispatch on `h.model_kind` if/when the research-branch architectures land.
- Blocks: merging `research/enc-dec-study` and `research/lru-integration` cleanly.
- Fix approach: extract a `build_model(h)` factory in `models/__init__.py` or a new `models/registry.py`.

### No checkpoint-format version stamp  *(LOW)*
- Problem: `save_checkpoint` (`utils.py:23`) saves whatever dict is passed. No `version` key. If state-dict layout changes, old checkpoints silently mismatch.
- Blocks: clean schema migration when `model_kind` lands.
- Fix approach: add `obj["__schema_version__"] = 1` and check on load.

---

## Test Coverage Gaps

### All training/inference paths  *(MEDIUM)*
- What's not tested: every line of `train.py`, `inference.py`, `dataset.py`, `models/*.py`, `utils.py`.
- Files: entire repo source.
- Risk: refactors break silently; reviewers can't get confidence cheap.
- Priority: Medium for a research repo; High if/when this becomes a public reference implementation.

### Checkpoint compatibility across `linear/gru` kinds  *(MEDIUM)*
- What's not tested: that a checkpoint trained with `{"linear": {"kind": "monarch", "nblocks": 4}}` loads cleanly into a fresh `NSNet2(h)` built from the same config.
- Files: `models/model.py:27`, `models/layers.py`
- Risk: silent state-dict-key mismatches like the orphaned `cp_*` issue above. The bug template is "config schema drifted but checkpoint didn't get re-trained."
- Priority: Medium.

### Reproducibility regression tests  *(LOW)*
- What's not tested: that the same seed + same config produces the same first-batch loss within tolerance.
- Files: `train.py:285-287`, `dataset.py:54, 79`
- Risk: see "missing seeds" above. Without this test, seed regressions go unnoticed.
- Priority: Low (research code, eval is by sweep result not bit-exactness).

### `run_sweep.sh` checkpoint-pruning  *(LOW)*
- What's not tested: the bash regex at `run_sweep.sh:66-67` (`find -regex ".*/\(g\|do\)_[0-9]+"`) that deletes everything but the latest rolling checkpoint and `g_best`.
- Files: `run_sweep.sh:62-68`
- Risk: a typo here silently nukes `g_best`. There's a `! -path "$latest_g"` guard but no `! -path "*/g_best"` — fortunately `g_best` doesn't match the `[0-9]+` suffix regex, so it's safe by accident, not by design.
- Priority: Low.

---

## Repository Hygiene

### Large committed history-only artifacts  *(NONE — clean)*
- All `cp_*/` directories and `*.log` files are correctly ignored by `.gitignore:18-21` (`cp_*/`, `*.log`). `git ls-files` confirms only 28 tracked files; nothing big or binary in tree. The local working copy carries ~700 MB of `cp_*/` data + ~480 KB of sweep logs but none is tracked. **This is correct.**

### Local-only `.claude/` directory  *(NONE — local config)*
- `git status` shows `.claude/` as untracked. Per `.claude/settings.local.json` it's a local Claude-Code workspace; not in `.gitignore` but not committed either.
- Recommendation: add `.claude/` to `.gitignore` to prevent accidental staging.

### `REVIEW.md` on research branches but not main  *(NONE — informational)*
- `research/enc-dec-study` carries a `REVIEW.md`. `main` does not. If those branches merge, decide whether `REVIEW.md` rolls forward.

---

## Summary by Severity

**HIGH (blocks reproducibility):**
- Orphaned checkpoints from deleted research variants (encdec/smr/lru/sensnet2 cp_* dirs).

**MEDIUM (real risk, time-bound):**
- `best_pesq` resets to 0 on resume (`train.py:102`).
- `torch.load(weights_only=False)` supply-chain risk (`utils.py:20`, `inference.py:23`).
- Missing seed coverage (no `np.random.seed`, no `random.seed`) plus `cudnn.benchmark=True`.
- `inference.py` doesn't dispatch on `model_kind` — couples to the orphaned-checkpoint issue.
- No automated tests of any kind.
- `torch < 2.6` pin will eventually need a coordinated bump.

**LOW (cleanup / cosmetic):**
- Stale `__pycache__/` for deleted standalone files.
- Duplicate root `config.json`.
- Hardcoded `n_jobs=30`.
- `sys.path.append("..")` boilerplate.
- Star imports in discriminator.
- Blanket FutureWarning suppression.
- `env.py` is misnamed.
- `dist_url` hardcoded across configs.
- Python-loop `StructuredGRU` (intentional).
- `.claude/` not in `.gitignore`.

---

*Concerns audit: 2026-04-27*
