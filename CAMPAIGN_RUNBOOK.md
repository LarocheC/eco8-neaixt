# MP-SENet retraining campaign — runbook

Retrain every model on the shared MP-SENet loss (`common/losses.py`), branch
`mpsenet-loss`. 28 runs: 12 LiSenNet + 3 ConvFSENet + 13 NSNet2.

Read [Why the numbers move](#why-the-numbers-move) before you interpret results —
only two of the four families actually changed objective.

---

## 1. Get the code there

```bash
git clone git@github.com:LarocheC/eco8-neaixt.git     # or: git fetch
cd eco8-neaixt
git checkout mpsenet-loss
git log --oneline -1        # expect: mpsenet: carry the LiSenNet hybrid bottleneck onto the shared loss
```

- [ ] On branch `mpsenet-loss`

## 2. Environment

`uv.lock` already pins the versions the branch needs (`gru-qat` 0.5.0,
`torch-structured` 1.3.0). A stale venv is the single most likely way to lose
time here — the four genuine-Monarch runs will not even build without them.

```bash
uv sync
uv run python -c "import importlib.metadata as m; print('gru-qat', m.version('gru-qat')); print('torch-structured', m.version('torch-structured'))"
```

- [ ] `gru-qat >= 0.5.0` and `torch-structured >= 1.3.0`

To move to newer deps than the lock pins (optional, and it changes the
environment the published numbers were produced under): `uv lock --upgrade && uv sync`.

## 3. Preflight — 30 seconds, saves hours

The script parses every `loss` block **and builds every model** before it touches
the GPU. Nothing trains; it either prints one `ok` line per run or names exactly
what is broken.

```bash
DRY_RUN=1 ./run_mpsenet_campaign.sh
```

- [ ] 28 `ok` lines, ending `preflight: all 28 runs parse and build`

If `monarch_*` fails with `MonarchLinear unavailable` → step 2 didn't take.
If a config fails with `unknown loss weight(s) ['adv', 'mag']` → that config
predates the loss port and needs its `loss` block migrated.

## 4. Capacity check

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
nproc
df -h .
```

- [ ] **VRAM.** Peak per concurrent run at the shipped batch sizes:
  LiSenNet ~2 GB · ConvFSENet ~6 GB · NSNet2 ~10 GB · BASENet ~13 GB.
  On 96 GB, `JOBS=5` is comfortable; `JOBS=8` fits if BASENet stays out.
- [ ] **CPU — the non-obvious one.** PESQ is the campaign's hidden cost and it
  does *not* scale with the GPU: `batch_pesq` runs on **every training step**
  with `n_jobs=-1`. The script caps each job's joblib pool to `nproc / JOBS`
  (`CORES_PER_JOB` overrides). Confirm the header prints a sane split — if
  `nproc` is small relative to `JOBS`, lower `JOBS` rather than starving every
  trainer.
- [ ] **Disk:** budget ~50 GB. Rolling checkpoints are kept (not pruned), and
  ConvFSENet at 200 epochs is the bulk of it.
- [ ] **Network:** first run downloads VoiceBank-DEMAND (~2.2 GB) and caches it.

## 5. Launch

```bash
DETACH=1 JOBS=5 ./run_mpsenet_campaign.sh
```

`DETACH=1` runs it in a detached tmux session, so an SSH drop doesn't kill it.

- [ ] Session started; `tmux attach -t mpsenet_campaign` shows runs in flight

Useful variants:

```bash
MODELS="lisennet" ./run_mpsenet_campaign.sh         # one family
MODELS="lisennet convfsenet nsnet2 basenet" ...     # add BASENet (off by default)
RUNS="lisennet_hybrid_nc24 lisennet_conv_hardened_nc24" ...   # an explicit pair
```

## 6. Watch

```bash
tail -f campaign.log                                  # campaign-level START/DONE/FAILED
tail -f cp_mpsenet_<name>/train.log                   # one run
./run_mpsenet_campaign.sh --summary                   # PESQ table, any time
tensorboard --logdir_spec=...                         # printed when the campaign ends
```

- [ ] **First 20 minutes, check one log for `nan`.** A LiSenNet run that prints
  `G: nan` means a loss term is backpropping through an unguarded
  `power_compress` — that exact bug was found and fixed on this branch, and the
  regression test pins it, but `nan` is the one failure that quietly wastes the
  whole campaign because training "completes" and every PESQ is garbage.
  Steady state looks like `G: 0.33, D: 0.01, Mag: 0.16` and descending.

## 7. Expect

Extrapolated from real 1-epoch runs on a **4090**, so treat as ±50%, and the RTX
6000 Pro is meaningfully faster:

| family | epochs | steps/epoch | ≈ per run | runs | ≈ GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| LiSenNet | 100 | 723 | ~8.5 h | 12 | ~100 |
| ConvFSENet | 200 | 723 | ~23 h | 3 | ~70 |
| NSNet2 | 200 | 45 | ~3 h | 13 | ~40 |
| | | | | **28** | **~210** |

At `JOBS=5` that is roughly **1.5–2 days** wall-clock. ConvFSENet at 200 epochs
dominates per-run cost; NSNet2 is cheap (batch 256 → only 45 steps/epoch).

## 8. Interrupted?

Just re-run the same command. Completed runs are skipped (`.done` marker) and a
partial run resumes from its latest rolling checkpoint.

---

## Why the numbers move

**Only LiSenNet and ConvFSENet changed objective.**

- **LiSenNet** gained the time and STFT-consistency terms, and its discriminator
  now sees the round-tripped magnitude instead of the raw network output.
- **ConvFSENet**'s `DynCompMSE` (active-speech-level normalisation, α=0.3
  mag/complex split, no time or consistency term) was replaced outright.
- **NSNet2** and **BASENet** were already exact MP-SENet ports. Their refactor
  onto the shared module is numerically a no-op — pinned by
  `tests/test_mpsenet_loss_parity.py` — so retraining them should *reproduce*
  RESULTS_*.md, not move it. They are in the campaign for uniform provenance,
  not because the loss will change them. **If an NSNet2 number moves materially,
  that is a signal something is wrong, not a result.**

**Read PESQ deltas against the noise floor, not against zero.**
`lisennet_conv_hardened_nc24_s2` is `..._nc24` at a different seed. The gap
between those two *is* the run-to-run variance of this recipe. A loss-driven
gain smaller than that gap is not a gain. Do not drop `_s2` to save a slot.

**The hybrid runs are paired ablations.** Each `lisennet_hybrid_*` is a
single-variable swap (conv-on-time → GRU-on-time) against the conv model of the
same name. Train the pair or the comparison means nothing:

| hybrid | baseline |
| --- | --- |
| `lisennet_hybrid_nc24` | `lisennet_conv_hardened_nc24` |
| `lisennet_hybrid_nc24_deep_relu6` | `lisennet_conv_hardened_nc24_deep_relu6` |

**Checkpoints land in `cp_mpsenet_<name>/`, never `cp_<name>/`.** The trainers
auto-resume from any checkpoint they find, so pointing them at the existing dirs
would silently continue an *old-loss* run and destroy the baseline you are trying
to compare against. Do not "tidy" this.

**Published numbers in RESULTS_*.md predate the loss change** for LiSenNet and
ConvFSENet (banners in those files say so). They are the A/B baseline.

## Scoring beyond PESQ

`--summary` reports PESQ, read back from the train logs. The `benchmarks/`
harness (DNSMOS / NISQA / SCOREQ) deliberately scores **from the HF Hub**, not
from local `cp_*` dirs — `benchmarks/published.py` treats the Hub repos as the
artefacts of record. So the retrained models cannot be scored on the perceptual
metrics until they are published, or until the harness is taught to read a local
checkpoint dir. Decide which before you need the numbers.
