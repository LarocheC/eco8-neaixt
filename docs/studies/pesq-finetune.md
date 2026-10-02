# Fine-tuning with a differentiable PESQ loss

**Status: set up, not yet run.** No numbers on this page until the GPU runs below are done.

**Question.** If a trained enhancer is fine-tuned with PESQ itself as an extra loss term, does
quality go up, and does it go up for the listener (DNSMOS, NISQA, SCOREQ), not only for PESQ?

**Why now.** [triton-pesq](https://github.com/LarocheC/triton-pesq) makes the PESQ loss cheap
enough to train with: 60x faster forward and up to 181x forward+backward than audiolabs'
torch-pesq (GTX 1080 Ti). It had never been wired into a trainer here. The closest experiment so
far is dnsmos_exported's *on-device* gradient loop, which lost on all six metrics. That was
adaptation on the STM32N6 against DNSMOS, and it leaves offline fine-tuning against PESQ open.

## Protocol

| | |
| --- | --- |
| Models | LiSenNet `conv-hardened` nc24, the STM32N6 deploy model (FP32 PESQ 3.013). Then optionally `conv-hardened-deep` and ConvFSENet. |
| Base | The published checkpoint (`hf:conv-hardened` = `claroche1/LiSenNet/conv-hardened/g_best`). Its own `config.json` defines the architecture and the base losses. |
| Recipe | 10 epochs, AdamW lr 1e-4, the base config's schedule and losses. `--init_from` loads the generator; the metric discriminator starts fresh in both arms. |
| Arms | **control**: PESQ weight 0. **pesq**: weight 0.2. Same seed, same data order. |
| Scoring | `finetune_pesq.py compare` on all 824 VBD test utterances: PESQ, DNSMOS (OVRL/SIG/BAK/P.808), NISQA (5 dims), SCOREQ (NR and REF). Output at the dataset's level, paired per-utterance deltas with 95 % CIs. |

The control arm is the point of the design. Fine-tuning moves the numbers by itself (more
steps, a lower learning rate, a fresh discriminator). Without a control, that movement would be
credited to the PESQ term.

### Decision rule (written before any result)

- **Adopt** the PESQ term if `pesq − control` on PESQ is at least **+0.03**, with a 95 % CI that
  excludes 0, and none of DNSMOS OVRL, NISQA MOS or SCOREQ-REF gets significantly worse. "Worse"
  means a CI entirely on the wrong side of 0; SCOREQ-REF is a distance, so lower is better.
- **Metric gaming**: PESQ goes up and at least one of those three judges gets significantly
  worse. Report it and do not adopt.
- **Null**: the PESQ CI includes 0. The term does not help at this weight; try 0.5 once before
  dropping the idea.

### Known caveats

- **Checkpoint selection.** `g_best` is chosen by validation PESQ on the VBD *test* split, which
  is this repo's convention for every model. Both arms are selected the same way, but it favours
  whichever arm overfits PESQ. Also compare the selection-free last rolling checkpoints, e.g.
  `control_last=cp_ft/…__control…/g_00007000 pesq_last=cp_ft/…__pesq…/g_00007000`.
- **Offline condition.** LiSenNet is scored with its Griffin-Lim phase (FP32), as `_validate` does.
  The deployed condition is int8 with the noisy phase (`int8_rt`). A positive result has to be
  confirmed there after export, through the normal `benchmarks/` pipeline.
- **Digital silence.** The PESQ model's backward is NaN on exactly-zero samples, which every
  zero-padded training segment contains. `common/pesq_loss.py` adds a −100 dB dither (1e-5 on
  RMS-normalised audio), which keeps gradients finite without moving the loss on ordinary audio.
  See its docstring and `tests/test_pesq_finetune.py`.

## How to run

```bash
uv sync --group pesq-loss                           # triton-pesq (pinned in pyproject.toml)
./run_finetune_pesq.sh lisennet hf:conv-hardened    # control + pesq arms, then the comparison
# variants: a weight as 3rd argument; EPOCHS / LR / METRICS / ROOT in the environment
```

The steps it chains, if you want them one by one:

```bash
python finetune_pesq.py train --model lisennet --base hf:conv-hardened --out cp_ft/ch__control --pesq_weight 0
python finetune_pesq.py train --model lisennet --base hf:conv-hardened --out cp_ft/ch__pesq0.2 --pesq_weight 0.2
python finetune_pesq.py compare --model lisennet --metrics all \
    base=hf:conv-hardened control=cp_ft/ch__control/g_best pesq=cp_ft/ch__pesq0.2/g_best \
    --json results/pesq_finetune/ch.json --md results/pesq_finetune/ch.md
```

The JSON keeps every per-utterance score and a provenance block: git commit, metric backend
versions, device. Any delta can then be re-derived without re-running.

**Where the code is.** `common/pesq_loss.py` holds the loss and backend choice.
`lisennet/train.py` and `convfsenet/train.py` read an optional `pesq_loss` config block (weight 0
by default, so existing configs train exactly as before). `finetune_pesq.py` is the runner and the
comparison. The tests are in `tests/test_pesq_finetune.py`.

**Sanity check of the scoring path.** On 2026-10-02, `compare` on the unmodified published
`conv-hardened` checkpoint, across all 824 test utterances (CPU), gave PESQ **3.013**. That is the
documented FP32 figure (`docs/models/lisennet.md`), so the enhancement and scoring path here
matches the one the published number came from.

## Results

_Pending the GPU runs._
