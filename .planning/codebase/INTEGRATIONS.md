# External Integrations

**Analysis Date:** 2026-04-27

This is a speech-enhancement research repo: there is no web service, no auth provider, no application database, no webhook surface. The only external systems it touches are (a) the HuggingFace Hub (datasets + model checkpoints), (b) NVIDIA CUDA on the local machine, (c) PyPI/PyTorch wheel mirrors at install time, and (d) a local TensorBoard log directory. This document catalogues those.

## APIs & External Services

**HuggingFace Hub — datasets:**
- `JacobLinCool/VoiceBank-DEMAND-16k` (the resampled VoiceBank-DEMAND speech-enhancement corpus, 16 kHz, paired clean/noisy)
- SDK: `datasets` `4.8.4` (`from datasets import load_dataset`)
- Used in `dataset.py:load_voicebank_demand` (`load_dataset(HF_DATASET_NAME, cache_dir=cache_dir)`) where `HF_DATASET_NAME = "JacobLinCool/VoiceBank-DEMAND-16k"`
- Consumers: `train.py` (train + test splits), `inference.py` (test split fallback), `analyze_sweep.ipynb` (test split for spectrogram comparisons)
- Cache: `--hf_cache_dir` CLI flag on both `train.py` and `inference.py`; defaults to the standard HF cache (`~/.cache/huggingface/`)
- Auth: none required (public dataset)

**HuggingFace Hub — model checkpoints:**
- `claroche1/sparse-nsnet2-checkpoints` — mirrors the 9 best-PESQ generators (`g_best`) plus their `config.json`
- SDK: `huggingface-hub` `1.11.0` (`from huggingface_hub import hf_hub_download`)
- Used in `analyze_sweep.ipynb` (`resolve_paths` with `SOURCE = "hf"`, calling `hf_hub_download(HF_REPO, f"{name}/config.json")` and `hf_hub_download(HF_REPO, f"{name}/g_best")`) and in the README Python snippet under "Trained checkpoints"
- Auth: none required (public repo); no token plumbing in the code
- `hf-xet 1.4.3` is pulled transitively for chunked transfer

**NVIDIA CUDA runtime:**
- Required at import time for `torch_structured`'s C++/CUDA extensions (`_butterfly.cpython-312-x86_64-linux-gnu.so`, `_diag_mult_cuda.cpython-312-x86_64-linux-gnu.so`, `_hadamard_cuda.cpython-312-x86_64-linux-gnu.so`)
- `torch.cuda.is_available()` gates GPU vs CPU selection in `train.py:main` and `inference.py:main`
- Distributed training over multiple GPUs: `train.py` uses `torch.multiprocessing.spawn` + `torch.distributed.init_process_group` with `dist_backend = "nccl"` (NCCL inter-GPU collectives, configured in `configs/*.json:dist_config`)

**PyTorch wheel index (install-time only):**
- `https://download.pytorch.org/whl/cu118` declared in `pyproject.toml` `[[tool.uv.index]]` as `pytorch-cu118`, marked `explicit = true`
- `torch` is sourced exclusively from this index (`tool.uv.sources.torch = { index = "pytorch-cu118" }`)

**Git (install-time only):**
- `torch-structured` is fetched as a git submodule from `git@github.com:LarocheC/torch-structured.git` (`.gitmodules`), pinned to commit `ceb76e0` (tag `v0.4.0`)

## Data Storage

**Databases:**
- None. No SQL, no key-value store, no ORM in the codebase.

**File Storage (local filesystem only):**
- Checkpoint directories `cp_<run>/` (gitignored, per `.gitignore` `cp_*/`) — one per sweep config; contain:
  - `config.json` (copy of the run's config, written by `env.py:build_env`)
  - `g_<N>` rolling generator checkpoint, `do_<N>` rolling discriminator+optimizer checkpoint (auto-pruned in `run_sweep.sh` to keep only the latest)
  - `g_best` — best-PESQ generator (saved in `train.py` once `epoch >= a.best_checkpoint_start_epoch`)
  - `logs/` — TensorBoard event files
  - `train.log` — captured stdout (`tee` in `run_sweep.sh`)
- Inference output: `--output_dir` (default `generated_files/`, gitignored) — `inference.py` writes `PCM_16` WAVs via `soundfile.write`
- HuggingFace dataset cache: default `~/.cache/huggingface/`, override via `--hf_cache_dir`
- HuggingFace hub cache (for downloaded checkpoints in the notebook): default `~/.cache/huggingface/hub/`

**Caching:**
- HuggingFace `datasets` arrow cache (set by `cache_dir` argument)
- `torch` JIT/CUDA kernel caches under `~/.cache/torch_extensions/` (implicit, used when `torch-structured` builds extensions)

## Authentication & Identity

- Not applicable. No user accounts, no auth provider, no API keys are read by the codebase. HuggingFace public assets only.

## Monitoring & Observability

**Error Tracking:**
- None. No Sentry, Bugsnag, or equivalent.

**Logs:**
- Plain stdout from training/inference, captured to `cp_<run>/train.log` by the `2>&1 | tee` pipeline in `run_sweep.sh`
- Sweep-level logs at the repo root: `sweep.log`, `lru_sweep.log`, `encdec_sweep.log`, `smr_sweep.log` (all gitignored via `.gitignore` `*.log`)
- `print()` statements in `train.py:train` (per-step loss summary, per-epoch wall-clock, validation PESQ score)

**Experiment tracking (TensorBoard):**
- `train.py` writes scalars via `torch.utils.tensorboard.SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))`:
  - `Training/Generator Loss`, `Training/Discriminator Loss`, `Training/Metric Loss`, `Training/Magnitude Loss`, `Training/Complex Loss`, `Training/Time Loss`, `Training/Consistency Loss`
  - `Validation/PESQ Score`, `Validation/Magnitude Loss`, `Validation/Complex Loss`, `Validation/Consistency Loss`
- README documents multi-run comparison via `tensorboard --logdir_spec=baseline:cp_baseline/logs,butterfly_fc:cp_butterfly_fc/logs,...`
- No remote experiment tracker (no wandb, mlflow, comet, neptune, aim — verified by absence of imports)

## CI/CD & Deployment

**Hosting / deployment:**
- Not applicable. Models are trained on a local CUDA box; results are shared as HuggingFace Hub artifacts and a results table in `README.md`.

**CI Pipeline:**
- None present in repo (`.github/`, `.gitlab-ci.yml`, etc. are absent).

**Sweep orchestration:**
- `run_sweep.sh` is a sequential bash driver that activates `.venv` and runs `python train.py` for each name in `RUNS`. Resumability is built into `train.py:train` via `scan_checkpoint(a.checkpoint_path, 'g_')` / `'do_'`.

## Environment Configuration

**Required env vars at runtime:**
- None required for normal training/inference (everything is read from JSON config files passed via `--config`)

**Optional env vars consumed by `run_sweep.sh`:**
- `RUNS` — space-separated list of run names (defaults to the full 9-run sweep `"baseline monarch_fc butterfly_fc monarch_full butterfly_full monarch_8 butterfly_ortho butterfly_2blocks wide_monarch"`)
- `EPOCHS` (default `30`), `VAL_INTERVAL` (`200`), `BEST_START` (`5`), `CHECKPOINT_INTERVAL` (`500`), `STDOUT_INTERVAL` (`50`)
- `PYTHONUNBUFFERED=1` (set inline before `python -u train.py`)

**Build-time env vars (documented in `README.md` for `torch-structured` compilation):**
- `TORCH_CUDA_ARCH_LIST` (e.g. `"6.1"` for sm_61)
- `CC`, `CXX` (e.g. `gcc-11`, `g++-11`)
- `NVCC_FLAGS` (e.g. `"-ccbin gcc-11"`)
- `FORCE_CPU=1` — skip CUDA kernels (CPU-only path)

**Secrets location:**
- No secrets. No `.env*` files exist (`ls .env*` returns empty). No `os.environ` / `os.getenv` calls in any `.py` file in the repo.

## Webhooks & Callbacks

**Incoming:**
- None. The repo runs no server.

**Outgoing:**
- None. The only outbound network calls are HTTPS reads from `huggingface.co` (datasets + checkpoints) at runtime, plus PyPI/PyTorch wheel index reads at `uv sync` time.

---

*Integration audit: 2026-04-27*
