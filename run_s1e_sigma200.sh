#!/bin/bash
# Stage 1e: a SECOND, independent 200-epoch sigma -- on this box.
#
# Why. The 4090's stage-2 data hints that sigma at 200 epochs (pooled 0.0184,
# df 6) is far below this box's 50-epoch 0.0427 (df 9). F = 5.41, p = 0.053 --
# suggestive, not significant, and the two values imply OPPOSITE strategic
# conclusions: at 0.043 the architecture question is unaffordable to resolve,
# at 0.018 A_res's +0.026 needs only ~4 seeds. The 4090 is settling its own
# 200-epoch sigma on the twlr10 arm (n=6). This box settles whether the same
# reduction happens here.
#
# Cross-box PESQ comparison is invalid (identical config and seed diverge by
# 0.217 ~ 5 sigma between these machines), but sigma is a WITHIN-box quantity,
# so comparing this box's sigma against theirs is legitimate and is the point.
#
# Design. Extend the SAME SIX SEEDS already run to 50 epochs, so the 50-vs-200
# comparison is PAIRED on seed rather than between independent groups. This is
# only sound because exact_resume makes a resumed run bitwise identical to an
# uninterrupted one -- otherwise extending would not measure the same thing as
# a fresh 200-epoch run. It also costs 150 incremental epochs instead of 200,
# saving ~17 h across the six.
#
# ~7.5 h per run (150 epochs x 169 s + 15 validations), ~45 h total, serial.
# Epoch-50 validation lives in the pre-extension event file; epochs 60-200 in
# the new one. EventAccumulator merges both from the same logs dir.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
LOGDIR=logs_s1e; mkdir -p "$LOGDIR"
MANIFEST="$LOGDIR/manifest.tsv"
[ -f "$MANIFEST" ] || printf 'run\tckpt\tstatus\tstarted\tfinished\n' > "$MANIFEST"

# wait for the stage-1b queue to release the GPU (one 7.2 GB job fits in 11 GB)
while ! grep -q "S1B COMPLETE" logs_s1b/sweep.out 2>/dev/null; do sleep 60; done
echo "[s1e] stage 1b finished, starting $(date -Is)"

for seed in 1234 1235 1236 1237 1238 1239; do
  if [ "$seed" = 1234 ]; then ck=cp_s1_twlr10_wdkeep; cfg=configs/s1_twlr10_wdkeep.json
  else ck=cp_s1b_sigma_s$seed; cfg=configs/s1b_sigma_s$seed.json; fi
  name=s1e_e200_s$seed
  if grep -qP "^$name\tdone" "$MANIFEST" 2>/dev/null; then echo "[skip] $name"; continue; fi
  echo "[start] $name ($ck) $(date -Is)"
  printf '%s\t%s\trunning\t%s\t\n' "$name" "$ck" "$(date -Is)" >> "$MANIFEST"
  $PY -m nsnet2.train --config "$cfg" --checkpoint_path "$ck" \
      --max_steps 9001 --validation_interval 450 --checkpoint_interval 450 \
      --summary_interval 100 --stdout_interval 45 --training_epochs 400 \
      --best_checkpoint_start_epoch 10000 \
      > "$LOGDIR/$name.log" 2>&1
  rc=$?
  st=$([ $rc -eq 0 ] && echo done || echo "failed_rc$rc")
  sed -i "s|^$name\t$ck\trunning\(.*\)\t$|$name\t$ck\t$st\1\t$(date -Is)|" "$MANIFEST"
  echo "[$st] $name $(date -Is)"
done
echo "=== S1E COMPLETE ==="
