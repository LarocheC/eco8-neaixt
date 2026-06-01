# Architecture

**Analysis Date:** 2026-04-27

## Pattern Overview

**Overall:** Single-model GAN training script with a config-driven structured-layer factory + a flat sweep-runner pattern.

The repo is a small NSNet2 speech-enhancement (SE) research codebase. A single
generator (`models.model.NSNet2`) is trained against a metric (PESQ-mimicking)
discriminator (`models.discriminator.MetricDiscriminator`). The interesting
research dimension lives entirely inside the layer-construction step: every
linear and the GRU stack are built via factory functions
(`models.layers.make_linear`, `models.layers.make_gru`) that dispatch to
dense / butterfly / monarch backends from the `torch-structured` git submodule.
Variants are selected by editing the `linear` / `gru` blocks in the config JSON
- the model code itself never branches on variant.

A second pattern (per `MEMORY.md` and the historical commit log -
`b1e8e63 Add SMR-NSNet variant`, `639c619 Integrate seNSNet2`,
`e959a7f Standalone file for v5_tfcm`, `327b60d Standalone paste-and-transfer
file for v4_fdown`) is **standalone variant files at the repo root**:
`*_standalone.py` files that vendor an entire alternative architecture
(SMRNSNet, NSNet2EncDec, SeNSNet2, LRU-NSNet) with their own `train(...)` /
`main()` and are launched directly via `uv run python <variant>_standalone.py`.
None are present in the working tree as of 2026-04-27 - they were the
short-lived runners that produced the legacy `cp_lru_*`, `cp_smr_*`,
`cp_encdec_v1..v8*`, `cp_sensnet2_*` checkpoint directories. The `config.json`
inside each of those checkpoint dirs still carries the variant-specific schema
(`model_kind`, `smr.*`, `encdec.*`, `sensnet2.*`, `gru.kind = "lru"`) and is
the authoritative reference for re-creating that variant.

**Key Characteristics:**
- Magnitude-mask predictor over STFT (mag, phase, complex) - phase is passed through unchanged in the trunk `NSNet2`.
- GAN training with two losses: generator (mag MSE + complex MSE + STFT-consistency MSE + L1 time + metric MSE) and PESQ-discriminator (MSE against per-batch normalized PESQ scores).
- Config is a flat JSON consumed via `env.AttrDict` (dict-as-attrs) + `env.build_env` (copies the chosen config into the checkpoint dir as `config.json`).
- Sweep is a flat bash loop (`run_sweep.sh`) over named runs, each pointing at one `configs/<name>.json` and writing to `cp_<name>/`.
- All variant selection is data (config), not code, for the trunk model. Standalone variants are vendored copies of the *whole* training script, not subclasses.
- Distributed-data-parallel scaffolding is present in `train.py` but configs default to `num_gpus: 0` (single-process / CPU or single-GPU).

## Layers

**Data layer (HuggingFace + STFT):**
- Purpose: Load paired clean/noisy audio at 16 kHz and convert to STFT magnitude, phase, complex tensors.
- Location: `dataset.py`
- Contains: `load_voicebank_demand` (HF `datasets.load_dataset` wrapper for `JacobLinCool/VoiceBank-DEMAND-16k`), `Dataset` (per-utterance random-crop to `segment_size` during training, full utterance for validation, RMS-style norm by `sqrt(N / sum(x^2))`), `mag_pha_stft` / `mag_pha_istft` (compress factor applied to magnitude).
- Depends on: `datasets`, `torch`, `numpy`.
- Used by: `train.py` (both train + validation loaders), `inference.py` (test split fallback).

**Model layer:**
- Purpose: Map noisy STFT to enhanced STFT.
- Location: `models/model.py`, `models/layers.py`, `models/discriminator.py`
- Contains: `NSNet2` (FC -> stacked GRU -> FC -> FC -> FC -> sigmoid mask), `MetricDiscriminator` (4-stage spectral-norm Conv2d with `LearnableSigmoid1d` head), `pesq_score` / `eval_pesq` (joblib-parallelized PESQ for validation), `make_linear` / `make_gru` factories, `StructuredGRUCell` / `StructuredGRU` (Python time-loop GRU using two `make_linear` projections per cell, packed across r/z/n gates).
- Depends on: `torch_structured.Butterfly`, `torch_structured.monarch.blockdiag_linear.BlockdiagLinear`, `pesq`.
- Used by: `train.py`, `inference.py`, `analyze_sweep.ipynb`.

**Training layer:**
- Purpose: Drive the GAN training loop, validation, checkpointing.
- Location: `train.py`, `utils.py`
- Contains: `train(rank, a, h)` (DDP-aware loop with explicit generator + discriminator step), `main()` (arg parsing, config load, `mp.spawn` for multi-GPU), `scan_checkpoint` / `load_checkpoint` / `save_checkpoint` (file-glob based latest-checkpoint discovery), `LearnableSigmoid1d` (used by the discriminator).
- Depends on: model layer, data layer, `torch.distributed`, `torch.utils.tensorboard`.
- Used by: invoked directly via `uv run python train.py --config <cfg> --checkpoint_path <dir>` or via `run_sweep.sh`.

**Inference layer:**
- Purpose: Single-checkpoint enhancement of either a directory of WAVs or the HF test split.
- Location: `inference.py`
- Contains: `enhance(model, noisy_wav)` (RMS-norm -> STFT -> model -> ISTFT -> denorm), `inference(a)` (per-file or per-HF-row loop with `rich.progress.track`), `main()` (loads config from `<checkpoint_path_dir>/config.json`).
- Depends on: model layer, data layer, `librosa`, `soundfile`.
- Used by: invoked directly via `uv run python inference.py --checkpoint_file <path>`.

**Sweep / analysis layer:**
- Purpose: Run the full per-config grid and analyze results.
- Location: `run_sweep.sh`, `analyze_sweep.ipynb`
- Contains: sequential bash loop with auto-prune (keeps only latest rolling `g_/do_` + `g_best`), tensorboard-launch hint; notebook loads each `cp_<run>/g_best`, plots PESQ trajectories, visualizes equivalent-dense weight matrices for every linear and GRU projection, runs inference on a few test items.
- Depends on: training layer + model layer + tensorboard event files in `cp_<run>/logs/`.

**Structured-primitives layer (submodule):**
- Purpose: Provide the low-rank / structured matrix building blocks the factory dispatches to.
- Location: `torch-structured/torch_structured/`
- Contains: `butterfly/` (Butterfly + variants, exact transforms FFT/DCT/Hadamard/circulant/Toeplitz), `monarch/` (`BlockdiagLinear`, butterfly-monarch product, low-rank, hyena utils, flash matmul), `structured/` (LDR, Toeplitz, Hankel, Vandermonde, Fastfood, Circulant), `recurrent/` (`LRU` linear-recurrent unit), `factory.py` (its own `make_linear`).
- Depends on: precompiled `_butterfly`, `_diag_mult_cuda`, `_hadamard_cuda` C++/CUDA extensions.
- Used by: `models.layers` only (the rest of the trunk does not import torch_structured directly).

## Data Flow

**Training step (per batch in `train.py:112`):**

1. `DataLoader` yields `(clean_audio, noisy_audio)` from `dataset.Dataset` (RMS-normalized, randomly cropped to `segment_size`).
2. `mag_pha_stft(...)` produces `(mag, pha, com)` for both clean and noisy at `n_fft` / `hop_size` / `win_size` from config.
3. `generator(noisy_mag, noisy_pha)` -> `(mag_g, pha_g, com_g)`. Inside `NSNet2.forward`: `mag.transpose(1,2)` -> `fc_in` -> ReLU -> `gru` -> `fc1` -> ReLU -> `fc2` -> ReLU -> `fc_out` -> sigmoid -> mask in [0,1] -> `denoised_mag = noisy_mag * mask`, phase passes through.
4. `mag_pha_istft(mag_g, pha_g, ...)` -> `audio_g`, then re-STFT to `(mag_g_hat, pha_g_hat, com_g_hat)` for the consistency loss.
5. `batch_pesq(clean_audio, audio_g)` (joblib-parallel `pesq.pesq`) yields normalized PESQ in [0,1] (or `None` if any sample errored).
6. Discriminator step: `D(clean_mag, clean_mag)` matched to ones, `D(clean_mag, mag_g_hat.detach())` matched to `batch_pesq`; sum and `optim_d.step()`.
7. Generator step: weighted sum of `mse(clean_mag, mag_g) * 0.9 + mse(clean_com, com_g) * 0.1 * 2 + mse(com_g, com_g_hat) * 0.1 * 2 + mse(D(clean_mag, mag_g_hat), 1) * 0.05 + l1(clean_audio, audio_g) * 0.2`; `optim_g.step()`.
8. Logging: stdout every `stdout_interval` steps; tensorboard scalars every `summary_interval`; rolling `g_<step>` + `do_<step>` checkpoints every `checkpoint_interval`; PESQ-tracked `g_best` after `best_checkpoint_start_epoch`.

**Inference flow (`inference.py:38`):**

1. Load `config.json` from `os.path.split(checkpoint_file)[0]`.
2. Build `NSNet2(h)`, `load_state_dict(state_dict['generator'])`, `.eval()`.
3. Per WAV / per HF row: `enhance(...)` -> `sf.write(...)` to `output_dir`.

**Sweep flow (`run_sweep.sh`):**

1. Iterate over space-separated `RUNS` env var (default = 9-run sweep).
2. For each `name`: `python -u train.py --config configs/<name>.json --checkpoint_path cp_<name> ...` with stdout `tee` to `cp_<name>/train.log`.
3. After each run completes, prune all rolling `g_/do_` checkpoints except the latest pair (keeps `g_best`).

**Variant selection:**
- Trunk variants: edit `linear.kind` / `gru.kind` in the config (no code change). `models.layers.make_linear` and `make_gru` dispatch to dense `nn.Linear` / cuDNN `nn.GRU`, `torch_structured.Butterfly`, or `torch_structured.monarch.BlockdiagLinear`. For structured GRU, the cell's `W_ih` / `W_hh` (each producing `3 * hidden`, packed r/z/n gates) are themselves `make_linear` calls; recurrent `W_hh` defaults to `init='ortho'` for stability (`models/layers.py:197`).
- Standalone variants (LRU / SMR / EncDec / SeNSNet2): historically lived in `*_standalone.py` files at repo root, each with its own dataset/train/main and a model class matching the `model_kind` field in their `config.json`. Currently absent from the tree; only their checkpoint outputs (`cp_lru_*`, `cp_smr_*`, `cp_encdec_v[1-8]*`, `cp_sensnet2_*`) and configs survive. To resurrect a variant, recreate `<variant>_standalone.py` at repo root (run via `uv run python <variant>_standalone.py --config cp_<variant>/config.json --checkpoint_path cp_<variant>`).

**State Management:**
- Model + optimizer state: PyTorch `state_dict` snapshots (`save_checkpoint` -> `torch.save(obj)` in `utils.py`). Generator-only payload `{generator: state_dict}` written to `g_<step>` and `g_best`. Discriminator + both optimizers + `steps` + `epoch` written to `do_<step>`.
- Resume: on startup `train.py` calls `scan_checkpoint(cp_dir, 'g_')` / `'do_'` (glob `g_????????` then sorted -> latest); silently starts from scratch if no rolling checkpoint found. `g_best` is *not* used for resume - only for inference / publishing.
- Config copy: `env.build_env` copies the launching JSON to `<checkpoint_path>/config.json`, so each `cp_*` is self-contained.

## Key Abstractions

**Config object (`AttrDict`):**
- Purpose: Flat JSON dict accessed as attributes (`h.batch_size`, `h.linear`, `h.dist_config`).
- Location: `env.py:4` (`class AttrDict(dict)` setting `self.__dict__ = self`).
- Pattern: Loaded once in `train.main` / `inference.main` from `--config`, augmented at runtime with `h.num_gpus` and a divided `h.batch_size`. Passed wholesale into `NSNet2(h)`, where `getattr(h, "...", default)` is used to keep configs backwards-compatible.

**Linear / GRU factory:**
- Purpose: One construction-site for every weight matrix in the network, so the dense / butterfly / monarch choice is a one-line change in JSON.
- Examples: `models/layers.py:55` (`make_linear`), `models/layers.py:173` (`make_gru`), used at `models/model.py:40-44`.
- Pattern: `cfg = dict(cfg or {}); kind = cfg.get("kind", ...)`; per-kind branch builds the correct module forwarding the remaining cfg keys (`nblocks`, `init`, `bias`). Unknown kind raises `ValueError`. `HAVE_BUTTERFLY` / `HAVE_MONARCH` import-guards let the file load even when the C++/CUDA extensions are unbuilt (failure is deferred to first construction).

**Structured GRU cell:**
- Purpose: Drop-in replacement for `nn.GRU` whose two projections (`W_ih` and `W_hh`) are themselves structured.
- Examples: `models/layers.py:87` (`StructuredGRUCell`), `models/layers.py:121` (`StructuredGRU`).
- Pattern: standard PyTorch GRU equations (r, z, n gates, n-gate uses `r * h_n` not `r * (W_ih_n @ x)`); single Python time-loop in `StructuredGRU.forward`; `(num_layers, B, H)` hidden state stacked at exit. Slow vs cuDNN, only used when `kind != "gru"`.

**Run / checkpoint dir as artifact:**
- Purpose: Atomic, self-contained record of one experiment.
- Examples: `cp_baseline/`, `cp_butterfly_2blocks/`, `cp_smr_long_mem/`.
- Pattern: every `cp_<name>/` contains `config.json` (copy of the launching config), `train.log` (tee'd stdout), `logs/` (tensorboard event files), `g_best` (best-PESQ generator after `best_checkpoint_start_epoch`), and the latest rolling `g_<step>` + `do_<step>` pair (older ones pruned by `run_sweep.sh`). Inference / notebook code locates `config.json` by `os.path.split(checkpoint_file)[0]`.

**STFT helper pair:**
- Purpose: Single source of truth for the (mag, pha, com) representation used everywhere.
- Examples: `dataset.py:11` (`mag_pha_stft`), `dataset.py:23` (`mag_pha_istft`).
- Pattern: shared Hann window built on the fly on the input device; `mag = sqrt(real^2 + imag^2 + eps)`, `pha = atan2(imag + eps, real + eps)` (the per-axis epsilons keep the gradient defined when real==0); compression `mag = mag^c` with `c = h.compress_factor` (typically 0.3); `com = stack(mag*cos pha, mag*sin pha)`.

## Entry Points

**Training:**
- Location: `train.py` (`main` at line 259, `train` at line 25)
- Triggers: `uv run python train.py --config <cfg> --checkpoint_path <dir>` (single run) or `run_sweep.sh` (loop).
- Responsibilities: parse args, load + AttrDict-wrap config, copy config into checkpoint dir, seed RNGs, count GPUs / divide batch size, `mp.spawn` if `num_gpus > 1`, instantiate generator + discriminator + optimizers + schedulers, build train/val loaders, run the GAN loop with periodic stdout / tensorboard / rolling checkpoints / `g_best` updates.

**Inference:**
- Location: `inference.py` (`main` at line 65)
- Triggers: `uv run python inference.py --checkpoint_file <path> [--input_noisy_wavs_dir <dir>] [--output_dir <dir>]`.
- Responsibilities: read sibling `config.json`, build model, load `state_dict['generator']`, iterate test inputs (either a WAV directory via `librosa.load` or the HF test split), write enhanced WAVs as 16-bit PCM at `h.sampling_rate`.

**Sweep runner:**
- Location: `run_sweep.sh`
- Triggers: `EPOCHS=50 ./run_sweep.sh > sweep.log 2>&1` or `RUNS="baseline monarch_fc" ./run_sweep.sh`.
- Responsibilities: source `.venv`, loop over `$RUNS`, exec `train.py` per name, post-prune rolling checkpoints, print tensorboard launch hint at the end. Resumable - re-running the same name continues from latest rolling checkpoint.

**Analysis:**
- Location: `analyze_sweep.ipynb`
- Triggers: `jupyter lab analyze_sweep.ipynb`.
- Responsibilities: load each `cp_<run>/g_best` (default: from HuggingFace `claroche1/sparse-nsnet2-checkpoints`), plot PESQ trajectories, render equivalent-dense weight heatmaps for every structured projection, run inference on a handful of test utterances with side-by-side spectrograms + audio players.

**Standalone variants (historical / removable):**
- Location: would-be `<variant>_standalone.py` at repo root (none committed at HEAD).
- Triggers: `uv run python <variant>_standalone.py --config <cfg> --checkpoint_path <dir>`.
- Responsibilities: full training pipeline for an alternative model architecture (`SMRNSNet`, `NSNet2EncDec`, `SeNSNet2`, LRU-based variants); each one is intentionally self-contained so the trunk `NSNet2` does not need a `model_kind` switch.

## Error Handling

**Strategy:** "fail fast in setup, swallow per-sample failures during training."

**Patterns:**
- Setup-time: `assert os.path.isfile(filepath)` in `utils.load_checkpoint` and `inference.load_checkpoint` (`utils.py:18`, `inference.py:21`); `argparse` `required=True` on `--checkpoint_file` (`inference.py:73`); JSON `.read()` will raise on a missing config.
- Factory-time: `make_linear` / `make_gru` raise `ImportError` with a "rebuild torch-butterfly" hint when the structured C++/CUDA modules are unavailable (`models/layers.py:65`, `74`); `ValueError` on unknown `kind`.
- Per-batch: `pesq` failures (e.g. silent segments) are caught and replaced with `-1` (`models/model.py:78`, `models/discriminator.py:17`); when *any* sample in a batch returns `-1`, `batch_pesq` returns `None` and the discriminator generator-loss is skipped for that step with a `print('pesq is None!')` (`train.py:142`).
- Distributed: `init_process_group` / `DistributedDataParallel` only when `h.num_gpus > 1`; otherwise a plain single-process path (rank 0).
- No try/except around the model forward / optimizer step (a NaN or shape error will crash the run, by design).

## Cross-Cutting Concerns

**Logging:**
- Stdout via plain `print(...)` (`train.py:181`, etc.). Sweep runs `tee` stdout to `cp_<name>/train.log` (`run_sweep.sh:59`).
- Tensorboard via `torch.utils.tensorboard.SummaryWriter` writing to `cp_<name>/logs/`. Scalars: `Training/Generator Loss`, `Training/Discriminator Loss`, `Training/Metric Loss`, `Training/Magnitude Loss`, `Training/Complex Loss`, `Training/Time Loss`, `Training/Consistency Loss`, `Validation/PESQ Score`, `Validation/Magnitude Loss`, `Validation/Complex Loss`, `Validation/Consistency Loss` (`train.py:194-239`).
- No structured logger (no `logging` module use), no log file rotation - the on-disk size is bounded by the sweep auto-prune.

**Validation:** Pure JSON via the `argparse` defaults + `getattr(h, key, default)` pattern in the model. There is no schema validation of the config beyond "AttrDict will return None on missing top-level keys."

**Authentication:** None for training/inference. Optional HuggingFace Hub access for downloading `claroche1/sparse-nsnet2-checkpoints` (anonymous public repo) and the `JacobLinCool/VoiceBank-DEMAND-16k` dataset (anonymous public dataset). `--hf_cache_dir` argument lets the user point at a pre-populated cache.

**Reproducibility:** `torch.manual_seed(h.seed)` and `torch.cuda.manual_seed(h.seed)` set in `main()`. `Dataset` shuffles its index list with a `random.Random(seed)` in single-GPU mode. `cudnn.benchmark = True` (`train.py:22`) trades determinism for throughput.

**Devices:** Single device chosen as `cuda:rank` if available else `cpu`; no MPS / ROCm path. The `dist_config` block in JSON is wired through `init_process_group` but the sweep uses `num_gpus: 0` and the configs do not exercise it.

---

*Architecture analysis: 2026-04-27*
