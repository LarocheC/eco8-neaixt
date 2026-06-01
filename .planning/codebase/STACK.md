# Technology Stack

**Analysis Date:** 2026-04-27

## Languages

**Primary:**
- Python `>=3.10,<3.13` (pinned to 3.12 in `.python-version`) — all training, inference, dataset, and model code in `train.py`, `inference.py`, `dataset.py`, `models/*.py`, `env.py`, `utils.py`, and `analyze_sweep.ipynb`

**Secondary (vendored via the `torch-structured` submodule):**
- C++ (C++14) — custom PyTorch ops in `torch-structured/csrc/butterfly.cpp`, `torch-structured/csrc/cpu/`, etc.
- CUDA C++ — GPU kernels in `torch-structured/csrc/cuda/`, `torch-structured/csrc/diag_mult/`, `torch-structured/csrc/hadamard/`, `torch-structured/csrc/flashmm/`
- Bash — sweep driver in `run_sweep.sh`

## Runtime

**Environment:**
- CPython 3.12 (declared in `.python-version`; `pyproject.toml` accepts 3.10–3.12)
- Linux-only practical target (CUDA 11.8 wheels and `dist_backend = "nccl"` in `configs/*.json`)

**Package Manager:**
- `uv` (drives `uv sync` in `README.md`; `pyproject.toml` declares `[tool.uv]` with `package = false` and `no-build-isolation-package = ["torch-structured"]`)
- Lockfile: `uv.lock` present (~340 KB, 102 pinned packages)
- Project itself is non-installable (`tool.uv.package = false`); it is a script repo run via `uv run python …` or `source .venv/bin/activate`

**Virtual environment:**
- `.venv/` (gitignored) — created by `uv sync`; activated explicitly by `run_sweep.sh`

## Frameworks

**Core deep-learning stack:**
- `torch` `2.5.1+cu118` (resolved from `https://download.pytorch.org/whl/cu118`, pinned via `pyproject.toml` `>=2.4,<2.6`)
- `triton` `3.1.0` (transitive from torch)
- NVIDIA CUDA 11.8 user-space wheels (transitive): `nvidia-cublas-cu11 11.11.3.6`, `nvidia-cudnn-cu11 9.1.0.70`, `nvidia-cufft-cu11`, `nvidia-curand-cu11`, `nvidia-cusolver-cu11`, `nvidia-cusparse-cu11`, `nvidia-nccl-cu11`, `nvidia-nvtx-cu11`, `nvidia-cuda-{cupti,nvrtc,runtime}-cu11`

**Structured-matrix primitives (vendored submodule):**
- `torch-structured` `0.4.0` (git submodule at `torch-structured/`, pinned to commit `ceb76e0`, Apache-2.0). Exposes `Butterfly`, `ButterflyBmm`, `ButterflyBase4`, `ButterflyUnitary`, `LRU`, `make_linear`, and `monarch.blockdiag_linear.BlockdiagLinear` — used by `models/layers.py`.
- Built locally via `setuptools` + `torch.utils.cpp_extension` + `ninja`; produces `_butterfly.cpython-312-x86_64-linux-gnu.so`, `_diag_mult_cuda.cpython-312-x86_64-linux-gnu.so`, `_hadamard_cuda.cpython-312-x86_64-linux-gnu.so` inside `torch-structured/torch_structured/`.

**Audio / signal processing:**
- `librosa` `0.11.0` — used in `inference.py` (`librosa.load`) and `analyze_sweep.ipynb` (`librosa.stft`)
- `soundfile` `0.13.1` — WAV writes in `inference.py` (`sf.write(..., 'PCM_16')`)
- `scipy` `1.15.3` / `1.17.1` — declared dependency, used transitively by `librosa` and `torch-structured`
- `audioread` `3.1.0`, `soxr` `1.0.0`, `pooch` `1.9.0`, `lazy-loader 0.5` — librosa transitives
- `numba` `0.65.0` + `llvmlite` `0.47.0` — librosa transitives

**Numerics / utilities:**
- `numpy` `2.2.6` / `2.4.4` (multi-version resolution by Python tag)
- `einops` `0.8.2`, `opt-einsum` `3.4.0` — declared in `pyproject.toml`, used inside `torch-structured`

**Speech-enhancement metric:**
- `pesq` `0.0.4` (perceptual evaluation of speech quality) — used in `models/model.py:eval_pesq` (validation) and `models/discriminator.py:cal_pesq` (PESQ-based metric discriminator)
- `joblib` `1.5.3` — `Parallel(n_jobs=...)` in `models/model.py:pesq_score` (`n_jobs=30`) and `models/discriminator.py:batch_pesq` (`n_jobs=-1`) to parallelize PESQ scoring

**Datasets / model hub:**
- `datasets` `4.8.4` — HuggingFace `datasets`; `dataset.py:load_voicebank_demand` calls `load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")`
- `huggingface-hub` `1.11.0` — `analyze_sweep.ipynb` uses `hf_hub_download(REPO, …)` to pull `g_best` checkpoints from `claroche1/sparse-nsnet2-checkpoints`
- `hf-xet` `1.4.3`, `pyarrow` `23.0.1`, `multiprocess` `0.70.19`, `dill` `0.4.1`, `fsspec` `2026.2.0`, `aiohttp` `3.13.5` — datasets transitives

**Experiment tracking:**
- `tensorboard` `2.20.0` — `train.py` uses `torch.utils.tensorboard.SummaryWriter` writing to `cp_<run>/logs/`
- No wandb / mlflow / comet usage anywhere in `*.py` or `*.ipynb`

**CLI / progress:**
- `rich` `15.0.0` — `inference.py` uses `rich.progress.track`
- `tqdm` `4.67.3` — transitive (datasets, huggingface-hub)
- `typer` `0.24.1`, `click` `8.3.2`, `shellingham` `1.5.4` — transitives

**Notebook (declared but unresolved in lockfile):**
- `jupyter>=1.0` and `matplotlib>=3.7` are listed in `pyproject.toml` `[project].dependencies` but do **not** appear in `uv.lock` (suggesting the lockfile was generated under a different resolver pass). `analyze_sweep.ipynb` imports `matplotlib.pyplot` and `IPython.display`, so these must be installed separately or via an extra `uv add`.

## Key Dependencies

**Critical (referenced directly in app code):**
- `torch` (with CUDA 11.8) — entire training/inference pipeline; `torch.stft` / `torch.istft` in `dataset.py`, optimizers + AMP-free GAN loop in `train.py`, distributed primitives (`init_process_group`, `DistributedDataParallel`, `DistributedSampler`) in `train.py`
- `torch-structured` — `models/layers.py` imports `Butterfly` and `BlockdiagLinear` to back the `make_linear` / `make_gru` factories
- `datasets`, `huggingface_hub` — sole data + checkpoint distribution mechanism
- `pesq` — both validation metric (`models/model.py`) and discriminator training signal (`models/discriminator.py`)
- `librosa`, `soundfile` — audio I/O for `inference.py` and the analysis notebook

**Infrastructure (transitive but load-bearing):**
- `nvidia-*-cu11` wheels — supply CUDA runtime; required for `torch.cuda.is_available()` and the C++/CUDA extensions in `torch-structured`
- `setuptools 82.0.1`, `ninja` (declared in `torch-structured/pyproject.toml` build-system, not pinned in main lockfile) — needed to compile `torch-structured` from source on `uv sync`

## Configuration

**Environment:**
- No `.env` / `.envrc` / dotenv files in repo. No `os.environ` / `os.getenv` calls anywhere in `*.py` / `models/*.py`.
- Build-time env vars (documented in `README.md`, consumed by `torch-structured/setup.py`):
  - `TORCH_CUDA_ARCH_LIST` (e.g. `"6.1"`)
  - `CC`, `CXX` (e.g. `gcc-11`, `g++-11`)
  - `NVCC_FLAGS` (e.g. `"-ccbin gcc-11"`)
  - `FORCE_CPU=1` to skip CUDA kernels
- Runtime env vars consumed by `run_sweep.sh`: `RUNS`, `EPOCHS`, `VAL_INTERVAL`, `BEST_START`, `CHECKPOINT_INTERVAL`, `STDOUT_INTERVAL`, plus `PYTHONUNBUFFERED=1`

**Application config:**
- Per-run JSON config files under `configs/`: `baseline.json`, `butterfly_2blocks.json`, `butterfly_fc.json`, `butterfly_full.json`, `butterfly_ortho.json`, `monarch_8.json`, `monarch_fc.json`, `monarch_full.json`, `wide_monarch.json`
- A top-level template `config.json` (smaller batch + `n_fft=400`) lives at the repo root
- Loaded as `AttrDict(json.load(...))` in `env.py:AttrDict` — a `dict` subclass that exposes keys as attributes
- Schema (consumed by `train.py` and `models/model.py`): `num_gpus`, `batch_size`, `learning_rate`, `adam_b1`, `adam_b2`, `lr_decay`, `seed`, `hidden_dim`, `fc_hidden_dim`, `num_gru_layers`, `compress_factor`, `linear: {kind, nblocks?, init?}`, `gru: {kind, nblocks?, x_init?, h_init?}`, `sampling_rate` (16000), `segment_size` (32000), `n_fft` (400/512), `hop_size`, `win_size`, `num_workers`, `dist_config: {dist_backend: "nccl", dist_url, world_size}`
- `env.py:build_env` copies the chosen config into `cp_<run>/config.json` so each checkpoint directory is self-describing

**Build:**
- `pyproject.toml` (PEP 621 + `[tool.uv]`) — repo root, declares deps + the cu118 PyTorch index
- `torch-structured/pyproject.toml` (setuptools backend, version `0.4.0`)
- `torch-structured/setup.py` — invokes `torch.utils.cpp_extension` to compile C++/CUDA ops
- `.gitmodules` — pins `torch-structured` to `git@github.com:LarocheC/torch-structured.git`

## Platform Requirements

**Development:**
- Linux + NVIDIA GPU (Pascal sm_61 onwards, per `README.md`) is the supported path
- CUDA 11.8 dev headers for first `uv sync`: `libcublas-dev-11-8`, `libcusparse-dev-11-8`
- `gcc-11` / `g++-11` host compiler for nvcc
- `uv` installed
- CPU-only fallback exists (`FORCE_CPU=1 uv sync`) but is documented as "only useful for inspecting weights, not training"

**Production / Deployment:**
- Not applicable — research codebase. "Deployment" = training runs on a single multi-GPU box (`torch.multiprocessing.spawn` + `nccl`) and inference via `inference.py` against the HF test split or a local directory of noisy WAVs
- Trained checkpoints are published to HuggingFace Hub (`claroche1/sparse-nsnet2-checkpoints`) rather than packaged into a service

---

*Stack analysis: 2026-04-27*
