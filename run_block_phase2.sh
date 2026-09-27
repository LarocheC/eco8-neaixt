#!/usr/bin/env bash
# Phase 2 of the block-design study: 200-epoch dense parents, old recipes, new block.
# NSNet2 new design = A1 (fixed input-mean subtraction + 200-step generator LR warmup,
# no BatchNorm) -- the pre-registered minimal design after the pilot (A1 passed: fc_in
# 0% dead vs 66% in the control). ConvFSENet new design = B1 (input-mean subtraction +
# frontend conv->BN->ReLU + 2000-step warmup; frontend 0/96 dead vs 29/96 in the control).
# Flags match each old counterpart's run, so last-5 is comparable to SPARSE_MATMUL_RUNS.csv.
# At most 4 concurrent (the pilot ran 6 at 20.6 GB); longest (ConvFSENet) first.
set -u
cd "$(dirname "$0")"
PY="${PY:-/home/clement/eco8-neaixt/.venv/bin/python}"
export PYTHONPATH="$PWD"
NS_SQ=(--training_epochs 200 --stdout_interval 45 --validation_interval 450 --checkpoint_interval 1800 --best_checkpoint_start_epoch 0)
NS_H68=(--training_epochs 200 --stdout_interval 45 --validation_interval 200 --checkpoint_interval 1800 --best_checkpoint_start_epoch 0)
CF=(--training_epochs 200 --stdout_interval 723 --validation_interval 7230 --checkpoint_interval 20657 --best_checkpoint_start_epoch 10)
JOBS=("cf:p2_cf_c96_b1" "ns_sq:p2_ns_sq192_a1" "ns_sq:p2_ns_sq192_a1_s2345" "cf:p2_cf_c96_b1_s2345"
      "cf:p2_cf_c96_old_s2345" "ns_sq:p2_ns_sq192_old_s2345" "ns_h68:p2_ns_h68_a1" "ns_h68:p2_ns_h68_a1_s2345")
for j in "${JOBS[@]}"; do   # never resume a stale run
  arm="${j##*:}"; if ls "cp_${arm}"/g_* >/dev/null 2>&1; then echo "refusing: cp_${arm} has checkpoints"; exit 1; fi
done
MAX=4; running=0
for j in "${JOBS[@]}"; do
  kind="${j%%:*}"; arm="${j##*:}"
  if [ "$running" -ge "$MAX" ]; then wait -n; running=$((running - 1)); fi
  case "$kind" in
    cf)     mod=convfsenet.train; flags=("${CF[@]}") ;;
    ns_sq)  mod=nsnet2.train;     flags=("${NS_SQ[@]}") ;;
    ns_h68) mod=nsnet2.train;     flags=("${NS_H68[@]}") ;;
  esac
  echo "$(date '+%F %H:%M:%S')  start  $arm"
  $PY -m "$mod" --config "configs/${arm}.json" --checkpoint_path "cp_${arm}" "${flags[@]}" > "cp_${arm}.log" 2>&1 &
  running=$((running + 1))
done
wait
echo "=== best / last-5 validation PESQ ==="
for j in "${JOBS[@]}"; do
  arm="${j##*:}"
  $PY -c "import re,statistics as s,sys; t=[float(x) for x in re.findall(r'PESQ Score: ([\d.]+)', open('cp_${arm}.log').read())]; print(f'{sys.argv[1]:<26} n={len(t):>2} best={max(t):.3f} last5={s.mean(t[-5:]):.3f}' if len(t)>=5 else sys.argv[1]+' too few validations')" "$arm"
done
echo "$(date '+%F %H:%M:%S')  done"
