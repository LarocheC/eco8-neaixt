#!/bin/bash
# Stage 1 of the butterfly-recurrent study: fix the recipe before anything else.
#
# A 2x3 factorial over the twiddle optimiser settings, NOT two separate sweeps:
#   twiddle_lr_mult      in {1, 10, 100}   (muP says a radix-2 factor has fan-in 2
#                                           regardless of layer width, so its LR
#                                           prescription differs from the dense
#                                           layer it replaced -- INFERRED, hence swept)
#   twiddle_weight_decay in {inherit 0.01, 0.0}  (decay on factors bounds the
#                                           NUCLEAR norm of the composed matrix)
# Each main effect is then estimated with the other factor averaged over, which is
# the whole point: single-seed one-factor-at-a-time rankings in this setup have
# already been shown not to reproduce (stage 1 vs stage 2 inverted).
#
# 50 epochs x 45 steps = 2250 steps; validation every 450 steps = every 10 epochs,
# giving 5 validations at epochs 10/20/30/40/50. ~2.6 h per run on a GTX 1080 Ti,
# ~15.5 h for the sweep. Runs are sequential: one 7.2 GB job fits in 11 GB.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
LOGDIR=logs_s1; mkdir -p "$LOGDIR"
MANIFEST="$LOGDIR/manifest.tsv"
[ -f "$MANIFEST" ] || printf 'run\tconfig\tstatus\tstarted\tfinished\n' > "$MANIFEST"

for cfg in configs/s1_twlr*.json; do
  name=$(basename "$cfg" .json)
  ckpt="cp_$name"
  if grep -qP "^$name\tdone" "$MANIFEST" 2>/dev/null; then
    echo "[skip] $name already done"; continue
  fi
  echo "[start] $name  $(date -Is)"
  printf '%s\t%s\trunning\t%s\t\n' "$name" "$cfg" "$(date -Is)" >> "$MANIFEST"
  $PY -m nsnet2.train \
      --config "$cfg" --checkpoint_path "$ckpt" \
      --max_steps 2251 --validation_interval 450 --checkpoint_interval 450 \
      --summary_interval 100 --stdout_interval 45 --training_epochs 100 \
      --best_checkpoint_start_epoch 1000 \
      > "$LOGDIR/$name.log" 2>&1
  rc=$?
  st=$([ $rc -eq 0 ] && echo done || echo "failed_rc$rc")
  # exact_resume is on, so a failed run can be relaunched and will pick up
  # bitwise where it stopped rather than restarting the epoch.
  sed -i "s|^$name\t$cfg\trunning\(.*\)\t$|$name\t$cfg\t$st\1\t$(date -Is)|" "$MANIFEST"
  echo "[$st] $name  $(date -Is)"
done
echo "=== S1 SWEEP COMPLETE ==="
