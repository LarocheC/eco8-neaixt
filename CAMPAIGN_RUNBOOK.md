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

- [ ] **GPU.** `nvidia-smi` — if another job is on the box, note which device is
  free and pass `GPU=<id>` at launch (step 5).
- [ ] **VRAM.** Peak per concurrent run at the shipped batch sizes:
  LiSenNet ~2 GB · ConvFSENet ~6 GB · NSNet2 ~10 GB · BASENet ~13 GB.
  On 96 GB, `JOBS=5` is comfortable; `JOBS=8` fits if BASENet stays out.
- [ ] **CPU.** The per-step MetricGAN target now runs on the GPU (torch-pesq),
  so PESQ is no longer what limits concurrency — see
  [The PESQ target](#the-pesq-target). The CPU still runs the *reported* PESQ at
  each validation (`n_jobs=30`), so the script caps each job's joblib pool to
  `nproc / JOBS` (`CORES_PER_JOB` overrides). Confirm the header prints a sane
  split.
- [ ] **Disk:** budget ~50 GB. Rolling checkpoints are kept (not pruned), and
  ConvFSENet at 200 epochs is the bulk of it.
- [ ] **Network:** first run downloads VoiceBank-DEMAND (~2.2 GB) and caches it.

## 5. Launch

```bash
DETACH=1 JOBS=5 ./run_mpsenet_campaign.sh
```

`DETACH=1` detaches the campaign so an SSH drop doesn't kill it. It uses tmux if
the box has it, and otherwise falls back to a plain detached process (`setsid`,
its own session, so SIGHUP can't reach it) — **no tmux required**. Force either
with `DETACH_MODE=tmux` / `DETACH_MODE=nohup`.

Without tmux there's no session to attach to, so watch it through the log:

```bash
tail -f campaign.log                       # same output you'd see in tmux
./run_mpsenet_campaign.sh --summary        # PESQ table
./run_mpsenet_campaign.sh --stop           # stop the campaign AND its trainers
```

`--stop` signals the whole process group, so it takes the in-flight trainers down
with it. That matters: killing only the campaign script would orphan them and
they'd keep holding the GPU, and the next launch would fight them for VRAM.

- [ ] Header shows the GPU and CPU split you expect
- [ ] Session started; `tmux attach -t mpsenet_campaign` shows runs in flight

**Sharing the box with another job?** Pin the campaign to a free device:

```bash
GPU=1 DETACH=1 JOBS=5 ./run_mpsenet_campaign.sh    # GPU 0 is busy elsewhere
GPU=1,2 DETACH=1 JOBS=6 ./run_mpsenet_campaign.sh  # spread runs across 1 and 2
```

Use `GPU=` rather than exporting `CUDA_VISIBLE_DEVICES` yourself. With
`DETACH=1` the campaign re-execs inside tmux, and `tmux new-session` inherits the
tmux *server's* environment, not your shell's — so if a tmux server was already
running, an exported `CUDA_VISIBLE_DEVICES` is silently dropped and you land on
GPU 0, on top of the job you were trying to avoid. `GPU=` is forwarded through
the re-exec explicitly. A non-existent index is a hard error, not a fallback.

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
./run_mpsenet_campaign.sh --stop                      # stop everything (resumable)
tensorboard --logdir_spec=...                         # printed when the campaign ends
```

Everything above works whether or not the box has tmux.

- [ ] **First 20 minutes, check one log for `nan`.** A LiSenNet run that prints
  `G: nan` means a loss term is backpropping through an unguarded
  `power_compress` — that exact bug was found and fixed on this branch, and the
  regression test pins it, but `nan` is the one failure that quietly wastes the
  whole campaign because training "completes" and every PESQ is garbage.
  Steady state looks like `G: 0.33, D: 0.01, Mag: 0.16` and descending.

## 7. Expect

Extrapolated from real 1-epoch runs on a **4090** with the torch-pesq target, so
treat as ±50%; the RTX 6000 Pro is meaningfully faster.

| family | epochs | steps/epoch | ≈ per run | runs | ≈ GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| LiSenNet | 100 | 723 | ~2.4 h | 12 | ~28 |
| ConvFSENet | 200 | 723 | ~11 h | 3 | ~33 |
| NSNet2 | 200 | 45 | ~2 h | 13 | ~26 |
| | | | | **28** | **~90** |

At `JOBS=5`, roughly **half a day to a day** wall-clock. ConvFSENet at 200 epochs
dominates per-run cost; NSNet2 is cheap (batch 256 → only 45 steps/epoch).

Measured, 1 LiSenNet epoch on the 4090 — note the CPU target is what *punished*
concurrency, and the GPU one barely notices it:

| | JOBS=1 | JOBS=3 |
| --- | ---: | ---: |
| ITU PESQ target (CPU) | 104 s | 175 s |
| torch-pesq target (GPU) | 77 s | 85 s |
| speedup | 1.35x | **2.05x** |

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

## The PESQ target

PESQ appears in the campaign in two completely different roles, and they use two
different implementations on purpose.

| | implementation | why |
| --- | --- | --- |
| **MetricGAN target**, every training step | `torch-pesq` on the GPU (`common.discriminator.batch_pesq_torch`) | It is only a regression target for the discriminator, which needs a signal that *orders* quality correctly. Measured against ITU on 200 real VBD pairs: Pearson 0.999, Spearman 0.997, mean err 0.043. |
| **Reported PESQ**, at each validation | `pesq` — the ITU-T P.862 reference (`common.metrics.pesq_score`) | This is the number of record. It gates `g_best` and is what RESULTS_*.md compares against published baselines and the MP-SENet paper. Swapping it would silently redefine the metric. **Do not "optimise" this one.** |

torch-pesq is *not* the reference implementation — it skips time alignment and
does level alignment with IIR filtering. That is acceptable for a training
target and unacceptable for a reported score, which is exactly the split above.

To fall back to the reference target (bit-identical to what the published
checkpoints were trained against — use it to reproduce them exactly, or to rule
the approximation out if a run looks off):

```json
"gan": { "enabled": true, "metric_loss_lambda": 0.05, "pesq_backend": "itu" }
```

## Scoring beyond PESQ

`--summary` reports PESQ, read back from the train logs. The `benchmarks/`
harness (DNSMOS / NISQA / SCOREQ) deliberately scores **from the HF Hub**, not
from local `cp_*` dirs — `benchmarks/published.py` treats the Hub repos as the
artefacts of record. So the retrained models cannot be scored on the perceptual
metrics until they are published, or until the harness is taught to read a local
checkpoint dir. Decide which before you need the numbers.
