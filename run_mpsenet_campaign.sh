#!/usr/bin/env bash
# Retrain every model on the MP-SENet loss (branch `mpsenet-loss`).
#
# WHY THIS EXISTS
#   common/losses.py is now the single training objective for all four models.
#   For LiSenNet and ConvFSENet that is a REAL change of objective, so their
#   published numbers are stale until retrained. For BASENet and NSNet2 the
#   refactor was numerically a no-op — retraining them reproduces what is
#   already in RESULTS_*.md, and is only worth it for uniform provenance.
#
# CHECKPOINT DIRS ARE FRESH ON PURPOSE
#   Every run writes to cp_mpsenet_<name>/, NOT the existing cp_<name>/. The
#   trainers auto-resume from any rolling checkpoint they find, so pointing them
#   at the old dirs would silently CONTINUE an old-loss run and destroy the very
#   baseline you need to compare against. Do not "helpfully" repoint these.
#
# PARALLELISM
#   Runs execute JOBS-at-a-time on one GPU. Every shipped batch_size is left
#   exactly as-is so the new numbers stay a clean A/B against the old ones —
#   the big card buys wall-clock through concurrency, not through a changed
#   recipe. Tune JOBS to your VRAM (see the table under "Usage").
#
# Usage
#   ./run_mpsenet_campaign.sh                    # default campaign, 3 at a time
#   JOBS=5 ./run_mpsenet_campaign.sh             # 5 concurrent runs
#   MODELS="lisennet" ./run_mpsenet_campaign.sh  # one model group
#   RUNS="lisennet_conv_hardened_nc24 convfsenet_win" ./run_mpsenet_campaign.sh
#   DETACH=1 ./run_mpsenet_campaign.sh           # run inside tmux, survives SSH drop
#   DRY_RUN=1 ./run_mpsenet_campaign.sh          # print the plan, train nothing
#   ./run_mpsenet_campaign.sh --summary          # PESQ table for whatever has run
#
#   Rough peak VRAM per concurrent run (shipped batch sizes, 32000-sample segments):
#     lisennet    ~2 GB    (37-46k params, batch 16)
#     convfsenet  ~6 GB    (0.7-1.5M params, batch 16)
#     nsnet2      ~10 GB   (batch 256)
#     basenet     ~13 GB   (batch 4 — activation-heavy, 7.1 GMACs)
#   On a 96 GB card JOBS=5 is comfortable; JOBS=8 fits if you exclude basenet.
#
# Resume: just re-run. Finished runs are skipped (cp_mpsenet_<name>/.done);
# a partial run continues from its latest rolling checkpoint.

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# What to train
# ---------------------------------------------------------------------------

# LiSenNet — objective REALLY changed (gained the time + consistency terms).
# nc24 is the current deploy winner; nc24_s2 is the same recipe at a different
# seed and is what tells you the run-to-run noise floor — without it you cannot
# tell a real loss-driven PESQ gain from variance. Keep it.
LISENNET_RUNS="lisennet lisennet_conv lisennet_conv_wide lisennet_conv_hardened
               lisennet_conv_hardened_nc24 lisennet_conv_hardened_nc28
               lisennet_conv_hardened_nc24_deep lisennet_conv_hardened_nc24_deep_relu6
               lisennet_conv_hardened_nc24_dil16 lisennet_conv_hardened_nc24_s2"

# ConvFSENet — objective REALLY changed (DynCompMSE fully replaced).
#   convfsenet_win           192/384, n_fft 512 — the deployed backbone
#   convfsenet_128_256       128/256, n_fft 512 — the HF `128-256/` capacity variant
#   convfsenet_win_smallstft 192/384, n_fft 256 — Track 2 (smaller STFT)
# `configs/convfsenet.json` is deliberately NOT here: its `windowed`/`calibration`
# blocks are export metadata the trainer never reads, so it is training-identical
# to convfsenet_win and would just burn a GPU slot producing the same weights.
CONVFSENET_RUNS="convfsenet_win convfsenet_128_256 convfsenet_win_smallstft"

# NSNet2 — loss UNCHANGED (the refactor is a numerical no-op). 13 runs post-merge:
# the old mislabeled `monarch_*` are now `blockdiag_*`, and `monarch_*` is now
# genuine 2-factor Monarch.
NSNET2_RUNS="baseline
             blockdiag_fc blockdiag_full blockdiag_8 wide_blockdiag
             monarch_fc monarch_full monarch_8 wide_monarch
             butterfly_fc butterfly_full butterfly_ortho butterfly_2blocks"

# BASENet — loss UNCHANGED. Off by default; add "basenet" to MODELS to include.
# (basenet_wide has never been trained at all — it is a staged capacity test.)
BASENET_RUNS="basenet_causal basenet3_noncausal basenet_eff8_cosine basenet_wide"

MODELS="${MODELS:-lisennet convfsenet nsnet2}"
JOBS="${JOBS:-3}"
DRY_RUN="${DRY_RUN:-0}"
DETACH="${DETACH:-0}"

# Epochs per model — the counts the published runs actually used, so the new
# numbers are comparable to the old ones.
LISENNET_EPOCHS="${LISENNET_EPOCHS:-100}"
CONVFSENET_EPOCHS="${CONVFSENET_EPOCHS:-200}"
NSNET2_EPOCHS="${NSNET2_EPOCHS:-200}"
BASENET_EPOCHS="${BASENET_EPOCHS:-100}"

CKPT_PREFIX="${CKPT_PREFIX:-cp_mpsenet_}"

# Logging / checkpoint cadence (same knob names as run_sweep.sh).
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2000}"
STDOUT_INTERVAL="${STDOUT_INTERVAL:-50}"
BEST_START="${BEST_START:-1}"

# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

model_of() {   # run name -> trainer module
    case "$1" in
        lisennet*)   echo lisennet   ;;
        convfsenet*) echo convfsenet ;;
        basenet*)    echo basenet    ;;
        *)           echo nsnet2     ;;   # baseline / blockdiag_* / monarch_* / butterfly_*
    esac
}

epochs_of() {
    case "$(model_of "$1")" in
        lisennet)   echo "$LISENNET_EPOCHS"   ;;
        convfsenet) echo "$CONVFSENET_EPOCHS" ;;
        basenet)    echo "$BASENET_EPOCHS"    ;;
        nsnet2)     echo "$NSNET2_EPOCHS"     ;;
    esac
}

build_run_list() {
    local out=""
    for m in $MODELS; do
        case "$m" in
            lisennet)   out="$out $LISENNET_RUNS"   ;;
            convfsenet) out="$out $CONVFSENET_RUNS" ;;
            nsnet2)     out="$out $NSNET2_RUNS"     ;;
            basenet)    out="$out $BASENET_RUNS"    ;;
            *) die "unknown model group '$m' (want: lisennet convfsenet nsnet2 basenet)" ;;
        esac
    done
    echo $out
}

# --summary: PESQ table for whatever has run so far. Safe to call any time.
if [[ "${1:-}" == "--summary" ]]; then
    printf '%-42s %10s %8s %s\n' RUN "BEST PESQ" STATE CKPT
    for cp in ${CKPT_PREFIX}*/; do
        [[ -d "$cp" ]] || continue
        name="${cp#"$CKPT_PREFIX"}"; name="${name%/}"
        best="—"
        if [[ -f "$cp/train.log" ]]; then
            best=$(grep -oP 'PESQ Score: \K[0-9.]+' "$cp/train.log" 2>/dev/null \
                   | sort -g | tail -1 || true)
            [[ -n "$best" ]] || best="—"
        fi
        if   [[ -f "$cp/.done" ]];   then state=done
        elif [[ -f "$cp/.failed" ]]; then state=FAILED
        else                              state=partial
        fi
        printf '%-42s %10s %8s %s\n' "$name" "$best" "$state" "$cp"
    done
    exit 0
fi

RUNS="${RUNS:-$(build_run_list)}"

# ---------------------------------------------------------------------------
# Preflight — fail before burning hours, not after
# ---------------------------------------------------------------------------

[[ -d .venv ]] || die "no .venv found. Run 'uv sync' first."
PY=".venv/bin/python"

for name in $RUNS; do
    [[ -f "configs/${name}.json" ]] || die "no configs/${name}.json (run '$name')"
done

# Preflight every run before committing the GPU to any of them: parse the loss
# block AND actually build the model. This catches, in seconds rather than hours:
#   - a config still on the pre-port `mag`/`adv` loss keys (loss_weights raises),
#   - a stale venv — the blockdiag_*/monarch_* runs need gru-qat>=0.5.0 and
#     torch-structured>=1.3.0, and an old venv fails only once the run starts,
#   - any config/architecture mismatch.
PYTHONPATH=. "$PY" - "$RUNS" <<'PYEOF' || die "preflight failed (see above). If a structured NSNet2 run failed to build, run 'uv sync'."
import json, sys, traceback
from common.env import AttrDict
from common.losses import loss_weights

def build(name, h):
    if name.startswith("lisennet"):
        from lisennet.model import build_lisennet;      return build_lisennet(h)
    if name.startswith("convfsenet"):
        from convfsenet.model import build_causal_model; return build_causal_model(h)
    if name.startswith("basenet"):
        from basenet.model import build_basenet;         return build_basenet(h)
    from nsnet2.model import NSNet2;                     return NSNet2(h)

bad = 0
for name in sys.argv[1].split():
    h = AttrDict(json.load(open(f"configs/{name}.json")))
    try:
        loss_weights(h)                      # raises on unknown/stale loss keys
        m = build(name, h)
        print(f"  ok  {name:42s} {sum(p.numel() for p in m.parameters())/1e6:7.3f}M params")
    except Exception as e:
        bad += 1
        print(f"  !!  {name:42s} {type(e).__name__}: {e}")
        traceback.print_exc(limit=1)
sys.exit(1 if bad else 0)
PYEOF
log "preflight: all $(echo $RUNS | wc -w) runs parse and build"

if [[ "$DETACH" == "1" && -z "${TMUX:-}" ]]; then
    SESSION="mpsenet_campaign"
    tmux has-session -t "$SESSION" 2>/dev/null && \
        die "tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
    tmux new-session -d -s "$SESSION" \
        "MODELS='$MODELS' RUNS='$RUNS' JOBS='$JOBS' DETACH=0 ./run_mpsenet_campaign.sh 2>&1 | tee campaign.log"
    echo "Campaign launched in tmux session '$SESSION'."
    echo "  attach:  tmux attach -t $SESSION      (detach: Ctrl-b then d)"
    echo "  log:     tail -f campaign.log"
    echo "  summary: ./run_mpsenet_campaign.sh --summary"
    exit 0
fi

n_runs=$(echo $RUNS | wc -w)
log "MP-SENet retraining campaign"
log "  models : $MODELS"
log "  runs   : $n_runs, $JOBS at a time"
log "  ckpts  : ${CKPT_PREFIX}<name>/   (fresh dirs — the old cp_<name>/ are left untouched)"
echo
printf '  %-42s %-11s %7s  %s\n' RUN TRAINER EPOCHS CKPT
for name in $RUNS; do
    printf '  %-42s %-11s %7s  %s\n' \
        "$name" "$(model_of "$name")" "$(epochs_of "$name")" "${CKPT_PREFIX}${name}"
done
echo

if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1 — nothing trained."
    exit 0
fi

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

train_one() {
    local name="$1"
    local model cp_dir epochs
    model="$(model_of "$name")"
    cp_dir="${CKPT_PREFIX}${name}"
    epochs="$(epochs_of "$name")"

    mkdir -p "$cp_dir"
    rm -f "$cp_dir/.failed"

    # Append, don't truncate: a resumed run must not lose the earlier PESQ history
    # that --summary reads back.
    if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
       "$PY" -u -m "${model}.train" \
            --config "configs/${name}.json" \
            --checkpoint_path "$cp_dir" \
            --training_epochs "$epochs" \
            --stdout_interval "$STDOUT_INTERVAL" \
            --summary_interval "$STDOUT_INTERVAL" \
            --checkpoint_interval "$CHECKPOINT_INTERVAL" \
            --validation_interval "$VAL_INTERVAL" \
            --best_checkpoint_start_epoch "$BEST_START" \
            >> "$cp_dir/train.log" 2>&1
    then
        touch "$cp_dir/.done"
        log "DONE   $name  (best PESQ $(grep -oP 'PESQ Score: \K[0-9.]+' "$cp_dir/train.log" | sort -g | tail -1))"
    else
        touch "$cp_dir/.failed"
        log "FAILED $name — see $cp_dir/train.log"
        return 1
    fi
}

failed=""
started=0
for name in $RUNS; do
    cp_dir="${CKPT_PREFIX}${name}"

    if [[ -f "$cp_dir/.done" ]]; then
        log "SKIP   $name (already complete; rm $cp_dir/.done to force)"
        continue
    fi
    # Two trainers on one dir corrupt each other's checkpoints.
    if pgrep -fa "${name}.json" | grep -q -- "--checkpoint_path[ =]*$cp_dir"; then
        log "SKIP   $name (a trainer is already running on $cp_dir)"
        continue
    fi

    # Block until a slot frees up.
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || true; done

    log "START  $name -> $cp_dir ($(model_of "$name"), $(epochs_of "$name") epochs)"
    train_one "$name" &
    started=$((started + 1))
done

# Drain.
while (( $(jobs -rp | wc -l) > 0 )); do wait -n || true; done

for name in $RUNS; do
    [[ -f "${CKPT_PREFIX}${name}/.failed" ]] && failed="$failed $name"
done

echo
log "Campaign finished ($started launched)."
[[ -n "$failed" ]] && log "FAILED:$failed"
echo
"$0" --summary
echo
echo "Compare in tensorboard:"
echo "  tensorboard --logdir_spec=$(for n in $RUNS; do printf '%s:%s%s/logs,' "$n" "$CKPT_PREFIX" "$n"; done | sed 's/,$//')"
[[ -z "$failed" ]]
