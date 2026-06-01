# Testing Patterns

**Analysis Date:** 2026-04-27

## Test Framework

**This is a research codebase with no formal test framework.**

- No `pytest`, no `unittest`, no `nose`. There is no `tests/` directory,
  no `conftest.py`, no `pytest.ini` or `[tool.pytest]` section in
  `pyproject.toml`.
- `pyproject.toml` has no `dev` / `test` extras. CI is not configured.
- `grep -rn "def test_\|pytest\|unittest" *.py models/*.py` over the
  current tree returns nothing.

Correctness is validated empirically through four mechanisms, listed in
order of authority:

1. **Sweep PESQ** — the canonical "did this work?" test.
2. **Standalone smoke / sanity tests** — embedded `test_*` functions
   that run from `if __name__ == "__main__":`.
3. **Analysis notebook** — `analyze_sweep.ipynb` reloads checkpoints,
   plots PESQ trajectories, materialises weight matrices, and compares
   spectrograms / audio across runs.
4. **Comparison to the dense baseline** — `cp_baseline/g_best` is the
   reference. Every variant's worth is its delta-PESQ vs. baseline at
   matched parameter count.

## How New Variants Are Validated

### 1. Sweep training run

Each variant is added to a sweep family driven by `run_sweep.sh`. The
sweep:

- Iterates over `RUNS` (env-overridable; default is the union of the
  family's `ORIGINAL_RUNS` and `NEW_RUNS`).
- For each run name, looks up `configs/<name>.json` and trains via
  `python -u train.py --config configs/<name>.json --checkpoint_path
  cp_<name> --training_epochs 50 ...`.
- Mirrors stdout to `cp_<name>/train.log` and tees it into the
  family-level sweep log at the repo root
  (`sweep.log`, `encdec_sweep.log`, `lru_sweep.log`, `smr_sweep.log`).
- Auto-prunes rolling `g_<step>` / `do_<step>` checkpoints after each
  run, keeping only the latest pair plus `g_best`
  (`run_sweep.sh:62-68`).
- Resumes automatically if the same `cp_<name>/` already contains
  rolling checkpoints — `train.py:44-59` calls `scan_checkpoint` on
  the directory and reloads.

A run is considered "passing" if its `cp_<name>/g_best` validation PESQ
(reported every `validation_interval` steps as
`Steps : N, PESQ Score: X.XXX`) is at or above the dense baseline at
comparable parameter count, or if the variant's design intent (e.g.
~10× compression) is met within an acceptable PESQ delta. The
README "Headline results" table is the public scoreboard.

**Run banners** in sweep logs are the test boundaries:

```
=== [HH:MM:SS] Run: <name> -> cp_<name> (epochs=N, val=K) ===
...
=== [HH:MM:SS] Done: <name> (cp size: …) ===
```

### 2. Standalone smoke / sanity tests

Every standalone variant ships with an `if __name__ == "__main__":`
block that runs a *forward smoke test* on random input. Two flavours
have been used:

**Minimal smoke test** (the `nsnet2_encdec_v4_standalone.py`,
`nsnet2_encdec_v5_tfcm_standalone.py`,
`sensnet2_ch64_la4_FDbidir_freqnorm_grp4_standalone.py` style):

```python
if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if len(sys.argv) > 1:
        # Load a trained checkpoint and run a one-shot enhancement on a wav.
        ckpt_path, in_wav, out_wav = sys.argv[1], sys.argv[2], sys.argv[3]
        import soundfile as sf
        model = load_model(ckpt_path, device=device)
        noisy, sr = sf.read(in_wav, dtype="float32")
        assert sr == CONFIG["sampling_rate"], f"expected {CONFIG['sampling_rate']} Hz"
        enh = enhance_wav(model, torch.from_numpy(noisy).to(device))
        sf.write(out_wav, enh.cpu().numpy(), sr, "PCM_16")
        print(f"wrote {out_wav}")
    else:
        # Build + random-forward smoke test.
        model = NSNet2EncDec(CONFIG).to(device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        wav = torch.randn(2, 16000, device=device)
        enh = enhance_wav(model, wav)
        print(f"NSNet2EncDec v4 standalone OK. params={n_params/1e6:.3f}M  "
              f"in={tuple(wav.shape)}  out={tuple(enh.shape)}")
```

This double-purpose CLI is the convention: zero args = build + random
forward + print param count, three args = real-checkpoint enhancement.
Run it with `uv run python <variant>_standalone.py`. The "OK" line is
the pass signal.

**Multi-test sanity block** (the `smr_nsnet_standalone.py` style — the
strictest variant tested in this repo):

```python
@torch.no_grad()
def test_shapes():
    for kind in ("smr", "gru"):
        m = _make_model(kind).eval()
        x = torch.randn(2, 257, 50)
        y, _ = m(x)
        assert y.shape == x.shape, f"{kind}: got {y.shape}"
    print("test_shapes: ok")


@torch.no_grad()
def test_streaming_equivalence(chunk: int = 7, atol: float = 1e-5):
    """Full causal forward must equal frame/chunk-by-frame forward."""
    torch.manual_seed(0)
    for kind in ("smr", "gru"):
        m = _make_model(kind).eval()
        x = torch.randn(2, 257, 41)

        y_full, _ = m(x)

        state = m.init_state(batch_size=2)
        outs = []
        T = x.size(-1)
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            y_c, state = m.step(x[..., s:e], state)
            outs.append(y_c)
        y_stream = torch.cat(outs, dim=-1)

        max_err = (y_full - y_stream).abs().max().item()
        assert max_err < atol, f"{kind}: streaming mismatch, max_err={max_err}"
        print(f"test_streaming_equivalence ({kind}, chunk={chunk}): max_err={max_err:.2e}")


@torch.no_grad()
def test_state_passing():
    """Continuing a sequence via state must equal a single full pass."""
    ...


@torch.no_grad()
def test_smr_decay_bounds():
    """SMR decays must always sit inside (a_min, a_max), no matter the input."""
    ...


def run_tests():
    test_shapes()
    test_streaming_equivalence(chunk=1)
    test_streaming_equivalence(chunk=7)
    test_state_passing()
    test_smr_decay_bounds()


if __name__ == "__main__":
    run_tests()
    ...                                   # then bench / report
```

**Conventions for embedded tests:**
- Decorate with `@torch.no_grad()` — these are forward-only invariants.
- Seed with `torch.manual_seed(0)` inside the test for tests that
  compare two forward paths.
- Prefer numerical-equivalence assertions
  (`assert max_err < 1e-5`) over isclose helpers — explicit tolerance.
- Print one terminal line per test on success
  (`print(f"test_streaming_equivalence (...): max_err={max_err:.2e}")`).
- Provide a `_make_model(kind, **kwargs)` factory inside the file so
  each test can instantiate small variants quickly.
- Tests live alongside the model in the same `*_standalone.py`. Do
  *not* extract them to a `tests/` folder — that breaks the
  "paste-and-go" promise (see CONVENTIONS.md).

**Useful invariants to check in new variants:**
- Shape preservation through `forward` (`test_shapes`).
- Streaming/offline equivalence for any model with hidden state
  (`test_streaming_equivalence`, `test_state_passing`).
- Parameter range invariants for parameterised constraints
  (`test_smr_decay_bounds` checks `a ∈ [a_min, a_max]` after random
  perturbation of the underlying free param).

### 3. Analysis notebook

`analyze_sweep.ipynb` is the primary cross-run validator and the
publish-quality reporter. Run from the project root after activating
the venv:

```bash
source .venv/bin/activate
jupyter lab analyze_sweep.ipynb
```

Or via `uv run jupyter lab analyze_sweep.ipynb`.

What it checks:

- **Loadability.** Every checkpoint in `RUNS` is loaded
  (`hf_hub_download` from `claroche1/sparse-nsnet2-checkpoints`, or
  `cp_<run>/g_best` locally if `SOURCE = "local"`). Failure to load is
  a regression.
- **PESQ trajectory parsing.** `parse_pesq` greps each
  `cp_<run>/train.log` for `r'Steps : (\d+), PESQ Score: ([\d.]+)'`
  and plots the curves on one axis. The shape of the curve is the
  qualitative health signal (monotonic-ish, no late-stage collapse).
- **Weight-structure visualisation.** `materialize_dense(module)` runs
  an identity matrix through the layer to recover the equivalent
  dense weight; `plot_grid` shows `|W|` per layer. This is how we
  confirm a butterfly is actually butterfly-structured and a monarch
  is actually block-diagonal — i.e. the structured-layer plumbing
  hasn't silently collapsed back to dense.
- **Audio + spectrogram comparison.** Loads three test utterances
  (`EXAMPLE_IDX = [0, 50, 200]`), runs each model, plots
  clean/noisy/enhanced spectrograms side-by-side, and embeds audio
  players for subjective comparison.

The notebook does not assert anything programmatically; the human
reviewer is the assertion target. Treat regressions in the
visualisations (collapsed weight structure, PESQ curve goes flat) as
test failures.

### 4. Comparison to the dense baseline

Every config family pins a "baseline" run:

- `cp_baseline/` — dense NSNet2, the canonical reference.
- `cp_lru_baseline/` — dense GRU replaced by a single LRU stack.
- `cp_smr_baseline/` — SMR-NSNet with default kwargs.

The baseline is trained on the *same* sweep with the *same* epoch
budget so the comparison is apples-to-apples. The README headline
table reports `Δ baseline` per run; new variants are expected to
report this delta in their PR / commit message.

## Test File Organization

There is no test directory. The conventions above mean:

- **Per-variant tests live in the standalone file itself**
  (`<variant>_standalone.py`), under a `# Sanity tests` banner near
  the bottom.
- **Cross-run / regression checks live in `analyze_sweep.ipynb`** at
  the repo root.
- **End-to-end behaviour is the sweep log** (`*_sweep.log` at repo
  root, `cp_<run>/train.log` per run).

## Run Commands

```bash
# (ad-hoc) per-variant smoke test on random input
uv run python nsnet2_encdec_v4_standalone.py
uv run python smr_nsnet_standalone.py            # also runs all `test_*`

# (ad-hoc) per-variant real enhancement test (3-arg CLI)
uv run python nsnet2_encdec_v4_standalone.py \
    cp_encdec_v4_fdown/g_best in.wav out.wav

# Train a single config from scratch (this is the integration test)
uv run python train.py --config configs/baseline.json \
    --checkpoint_path cp_baseline

# Resume a run (idempotent — picks up latest g_<step>/do_<step>)
uv run python train.py --config configs/baseline.json \
    --checkpoint_path cp_baseline

# Full sweep — produces cp_<name>/ and a top-level <family>_sweep.log
EPOCHS=50 ./run_sweep.sh > sweep.log 2>&1 &

# Restricted sweep — env-override RUNS to only re-train one variant
RUNS="butterfly_2blocks" EPOCHS=50 ./run_sweep.sh

# Inference on the HF VoiceBank-DEMAND test split
uv run python inference.py --checkpoint_file cp_baseline/g_best \
    --output_dir generated_files

# Cross-run analysis
uv run jupyter lab analyze_sweep.ipynb

# Live monitoring during a sweep
tensorboard --logdir_spec=baseline:cp_baseline/logs,wide_monarch:cp_wide_monarch/logs
```

## Mocking

**No mocking framework is used or needed.** Tests run real `torch`
forward/backward against random tensors. The standalone smoke tests
deliberately avoid mocking so the path being exercised is the same
path used in production (training + inference).

When the variant has a hidden state and streaming API, the "mock" is
the state object itself: pass `state=None` for offline, pass the
returned state back in for streaming. See the `test_state_passing`
pattern above.

## Fixtures and Test Data

**Random tensors** are the default fixture in standalone tests:

```python
torch.manual_seed(0)
x = torch.randn(2, 257, 50)        # (B, F, T) for blocks
wav = torch.randn(2, 16000, ...)   # (B, L) for end-to-end
```

Shapes used:
- `(B=2, F=257, T=50)` for STFT-domain block tests at `n_fft=512`.
- `(B=2, F=257, T=41)` to exercise an odd time length for chunk
  tests.
- `(B=2, L=16000)` for end-to-end (1 second of audio at 16 kHz).

**Real audio** for end-to-end / notebook tests comes from the
HuggingFace `JacobLinCool/VoiceBank-DEMAND-16k` dataset
(`dataset.py:HF_DATASET_NAME`). Specific test indices used in
`analyze_sweep.ipynb`: `EXAMPLE_IDX = [0, 50, 200]` — index `200` is a
load-bearing "tricky" example for subjective comparison.

There are no committed fixture files. Reproducibility comes from the
HF dataset hash + seeds in the configs.

## Coverage

**Not measured.** No `coverage.py`, no `pytest --cov`, no Codecov
integration. The notional "coverage" target is:

- Every variant ships a working `*_standalone.py` that passes its
  embedded smoke / sanity block when run with `uv run python`.
- Every config in `configs/<name>.json` has a corresponding `cp_<name>/`
  directory with a converged `g_best` whose PESQ is within the
  variant's expected band (recorded in the standalone's docstring
  results table).

## Test Types

**Unit tests:** none formal. Closest analogues are the standalone
`test_shapes`, `test_streaming_equivalence`, `test_state_passing`,
`test_smr_decay_bounds` style sanity tests, which exercise individual
`nn.Module` sub-blocks in isolation.

**Integration tests:** the standalone `if __name__ == "__main__":`
smoke block that builds the full model and runs `enhance_wav` on a
random or real waveform end-to-end (STFT → forward → iSTFT →
unscale).

**End-to-end / system tests:** the sweep run itself
(`run_sweep.sh` → `train.py` → `cp_<name>/train.log` →
`Validation/PESQ Score`). A successful 50-epoch sweep that produces a
`g_best` whose PESQ matches the published table is the system-level
pass condition.

**Regression tests:** comparison to the previous `g_best` PESQ for the
same config name. Concretely: re-run the sweep with the same epochs/
batch/seed and check the new `g_best` PESQ matches the README table to
within ~0.01.

**Property / invariant tests:** only inside `smr_nsnet_standalone.py`
(streaming equivalence, decay bounds). New variants with non-trivial
parameter constraints or streaming behaviour MUST add equivalent
property tests in their own standalone.

## Common Patterns

**Async testing:** not applicable — everything is synchronous PyTorch.

**Error testing:** rare. The closest example is the validation
exception path inside `MetricDiscriminator`'s `cal_pesq`, which is
exercised implicitly by the training loop on silent windows
(`train.py:139-143` prints `pesq is None!` when `batch_pesq` returns
`None`). New variants should not add explicit error tests; rely on
constructor `ValueError` raises (see CONVENTIONS.md "Error Handling")
and let the smoke test surface them.

**Numerical equivalence:**

```python
max_err = (y_full - y_stream).abs().max().item()
assert max_err < 1e-5, f"{kind}: streaming mismatch, max_err={max_err}"
```

Always compute the scalar error explicitly and include it in the
assertion message. Use `1e-5` as the default float32 tolerance for
`@torch.no_grad()` equivalence tests.

**Random seeding for tests:**

```python
torch.manual_seed(0)            # inside the test, not at module scope
```

Per-test seeding (not module-scope) so tests don't accidentally
depend on each other's RNG state.

**Smoke-test reporting:**
End every smoke run with a one-line summary that includes the
parameter count and tensor shapes:

```python
print(f"NSNet2EncDec v4 standalone OK. params={n_params/1e6:.3f}M  "
      f"in={tuple(wav.shape)}  out={tuple(enh.shape)}")
```

This line is the canonical "test passed" signal — keep the `OK` token
and the `params=` field, downstream scripts grep for them.

---

*Testing analysis: 2026-04-27*
