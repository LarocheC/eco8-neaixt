#!/usr/bin/env bash
# Butterfly NSNet2, first wave: does the A1 block design (fixed input-mean subtraction + 200-step
# LR warmup) lift the butterfly-structured family, and do butterfly-specific fixes add to it?
#   bf_full_old            butterfly_full exactly as published (400/600, randn FC, ortho W_hh), current code
#   bf_full_a1             + A1                                        -> what the recipe alone buys
#   bf_full_a1_w512        + hidden = fc = 512                         -> no pad/truncate waste, 155k params
#   bf_2b_a1_w512          + nblocks 2 on every butterfly (BB^T)       -> capacity arm, 303k params
#   bf_full_a1_w512_ortho  + butterfly_ortho_lambda 0.05               -> conditioning arm
# lambda 0.05 balances |grad penalty| against |grad task loss| over the twiddles at init (ratio 0.051).
# Every arm uses the Python StructuredGRU (fused W_ih/W_hh butterflies), the backend the HF
# butterfly_* checkpoints were trained with: triton_butterfly is per-gate, a different model at
# 400 wide (identical function class at 512). Seed 1234, 200 epochs, validation every 10 epochs on
# the full test split, same flags as the NSNet2 block-design waves, so best / last-5 compare.
# ~3 GiB GPU each; all five run at once (the per-step CPU batch_pesq dominates epoch time).
#   DRY=1 ./run_bfly_wave.sh    print the commands, launch nothing
set -u
cd "$(dirname "$0")"
PY="${PY:-/home/clement/eco8-neaixt/.venv/bin/python}"
export PYTHONPATH="$PWD"
DRY="${DRY:-0}"
NS=(--training_epochs 200 --stdout_interval 45 --validation_interval 450 --checkpoint_interval 1800 --best_checkpoint_start_epoch 0)
JOBS=("bf_2b_a1_w512" "bf_full_old" "bf_full_a1" "bf_full_a1_w512" "bf_full_a1_w512_ortho")

for arm in "${JOBS[@]}"; do   # never resume or overwrite a stale run
  [ -f "configs/${arm}.json" ] || { echo "refusing: configs/${arm}.json missing"; exit 1; }
  if compgen -G "cp_${arm}/g_*" >/dev/null || compgen -G "cp_${arm}/do_*" >/dev/null; then echo "refusing: cp_${arm} has checkpoints"; exit 1; fi
  if [ -s "cp_${arm}.log" ] && [ "$DRY" != 1 ]; then echo "refusing: cp_${arm}.log exists (move it away first)"; exit 1; fi
done

MAX=5; running=0
for arm in "${JOBS[@]}"; do
  if [ "$DRY" = 1 ]; then
    echo "$PY -m nsnet2.train --config configs/${arm}.json --checkpoint_path cp_${arm} ${NS[*]} > cp_${arm}.log 2>&1"
    continue
  fi
  if [ "$running" -ge "$MAX" ]; then wait -n; running=$((running - 1)); fi
  echo "$(date '+%F %H:%M:%S')  start  $arm"
  ( $PY -m nsnet2.train --config "configs/${arm}.json" --checkpoint_path "cp_${arm}" "${NS[@]}" > "cp_${arm}.log" 2>&1
    echo "$(date '+%F %H:%M:%S')  end    $arm  rc=$?" ) &
  running=$((running + 1))
done
[ "$DRY" = 1 ] && exit 0
wait

echo "=== butterfly wave: best / last-5 validation PESQ ==="
for arm in "${JOBS[@]}"; do
  [ -f "cp_${arm}.log" ] || { echo "$arm  no log"; continue; }
  $PY -c "import re,statistics as s,sys; t=[float(x) for x in re.findall(r'PESQ Score: ([\d.]+)', open('cp_'+sys.argv[1]+'.log').read())]; print(f'{sys.argv[1]:<24} n={len(t):>2} best={max(t):.3f} last5={s.mean(t[-5:]):.3f}' if len(t)>=5 else sys.argv[1]+' too few validations')" "$arm"
done
echo "$(date '+%F %H:%M:%S')  done"
