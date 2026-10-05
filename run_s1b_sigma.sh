#!/bin/bash
# Stage 1b: the seed-variance measurement, plus the asymmetric-multiplier arm
# that the stage-1 conditioning diagnostics suggested.
#
# WHY sigma is the deliverable: single-seed rankings in this setup have been
# shown not to reproduce -- stage 1 (100 ep) ranked R2b > D_grid > R and
# stage 2 (200 ep) ranked R > D_grid > R2b, fully inverted, with R's two
# stage-2 seeds 0.047 apart. Every later stage's power depends on sigma, and
# no measurement of PESQ seed variance on VoiceBank-DEMAND has ever been
# published, so this is also a small contribution in its own right.
#
# sigma arm  : s1_twlr10_wdkeep settings (twiddle_lr_mult 10, decay inherited),
#              seeds 1235..1243. Seed 1234 already exists as cp_s1_twlr10_wdkeep
#              and is repeat #1 -- with deterministic:true a re-run would be
#              bitwise identical, so regenerating it would waste 2.5 h.
# asym arm   : the same, but the ortho-initialised recurrent twiddles are held
#              at the base rate while the randn feed-forward ones take 10x.
#              Tests the stage-1 observation that a global 10x helps fcs.0
#              (rank 109-112 -> 130-135) and hurts rnn.h_proj.0 (438-447 -> 230-282).
#              3 seeds against the sigma arm's 10 is underpowered on purpose:
#              sigma itself will say how many more are needed.
#
# 12 runs x ~2.5 h = ~30 h sequential on a GTX 1080 Ti.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
LOGDIR=logs_s1b; mkdir -p "$LOGDIR"
MANIFEST="$LOGDIR/manifest.tsv"
[ -f "$MANIFEST" ] || printf 'run\tconfig\tstatus\tstarted\tfinished\n' > "$MANIFEST"

for cfg in configs/s1b_sigma_s*.json configs/s1b_asym_s*.json; do
  name=$(basename "$cfg" .json)
  if grep -qP "^$name\tdone" "$MANIFEST" 2>/dev/null; then
    echo "[skip] $name already done"; continue
  fi
  echo "[start] $name  $(date -Is)"
  printf '%s\t%s\trunning\t%s\t\n' "$name" "$cfg" "$(date -Is)" >> "$MANIFEST"
  $PY -m nsnet2.train \
      --config "$cfg" --checkpoint_path "cp_$name" \
      --max_steps 2251 --validation_interval 450 --checkpoint_interval 450 \
      --summary_interval 100 --stdout_interval 45 --training_epochs 100 \
      --best_checkpoint_start_epoch 1000 \
      > "$LOGDIR/$name.log" 2>&1
  rc=$?
  st=$([ $rc -eq 0 ] && echo done || echo "failed_rc$rc")
  sed -i "s|^$name\t$cfg\trunning\(.*\)\t$|$name\t$cfg\t$st\1\t$(date -Is)|" "$MANIFEST"
  echo "[$st] $name  $(date -Is)"
done
echo "=== S1B COMPLETE ==="
