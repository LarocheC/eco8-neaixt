#!/usr/bin/env bash
# Pareto front of the NEW block designs: dense parents at every width, for the 2:4@c1
# (+ channel permutation + least-squares refit) conversion that Phase 3 applies afterwards.
#   NSNet2 A1 (fixed input-mean subtraction + 200-step warmup), square family H = hidden = fc_hidden:
#     64, 96, 128, 256, 384   (192 = p2_ns_sq192_a1 / _s2345, already trained)
#   ConvFSENet B1 (input-mean subtraction + frontend conv->BN->ReLU + 2000-step warmup), res = C, conv = 2C:
#     64, 128, 160, 192       (96 = p2_cf_c96_b1 / _s2345, already trained)
# Recipes are the Phase-2 configs with only the widths changed; seed 1234; flags identical to
# run_block_phase2.sh per family, so last-5 is comparable across the whole front.
# Epoch time does not depend on width (generator fwd+bwd is a flat 3.3 ms/step from c64 to c192;
# the per-step CPU batch_pesq with n_jobs=-1 dominates), so jobs are CPU-bound and share the box.
# At most 4 concurrent, longest first: ConvFSENet (widest first), then NSNet2 (widest first).
# Estimated wall-clock from the Phase-2 logs: ~22 h (range ~20-25 h).
#   DRY=1 ./run_block_pareto.sh    print the commands, launch nothing
set -u
cd "$(dirname "$0")"
PY="${PY:-/home/clement/eco8-neaixt/.venv/bin/python}"
export PYTHONPATH="$PWD"
DRY="${DRY:-0}"
NS=(--training_epochs 200 --stdout_interval 45 --validation_interval 450 --checkpoint_interval 1800 --best_checkpoint_start_epoch 0)
CF=(--training_epochs 200 --stdout_interval 723 --validation_interval 7230 --checkpoint_interval 20657 --best_checkpoint_start_epoch 10)
JOBS=("cf:p3_cf_c192_b1" "cf:p3_cf_c160_b1" "cf:p3_cf_c128_b1" "cf:p3_cf_c64_b1"
      "ns:p3_ns_sq384_a1" "ns:p3_ns_sq256_a1" "ns:p3_ns_sq128_a1" "ns:p3_ns_sq96_a1" "ns:p3_ns_sq64_a1")
# Points of the front that Phase 2 already trained (summary only, never relaunched).
DONE=("p2_ns_sq192_a1" "p2_ns_sq192_a1_s2345" "p2_cf_c96_b1" "p2_cf_c96_b1_s2345")

for j in "${JOBS[@]}"; do   # never resume or overwrite a stale run
  arm="${j##*:}"
  [ -f "configs/${arm}.json" ] || { echo "refusing: configs/${arm}.json missing"; exit 1; }
  if compgen -G "cp_${arm}/g_*" >/dev/null || compgen -G "cp_${arm}/do_*" >/dev/null; then echo "refusing: cp_${arm} has checkpoints"; exit 1; fi
  if [ -s "cp_${arm}.log" ] && [ "$DRY" != 1 ]; then echo "refusing: cp_${arm}.log exists (move it away first)"; exit 1; fi
done

MAX=4; running=0
for j in "${JOBS[@]}"; do
  kind="${j%%:*}"; arm="${j##*:}"
  case "$kind" in
    cf) mod=convfsenet.train; flags=("${CF[@]}") ;;
    ns) mod=nsnet2.train;     flags=("${NS[@]}") ;;
  esac
  if [ "$DRY" = 1 ]; then
    echo "$PY -m $mod --config configs/${arm}.json --checkpoint_path cp_${arm} ${flags[*]} > cp_${arm}.log 2>&1"
    continue
  fi
  if [ "$running" -ge "$MAX" ]; then wait -n; running=$((running - 1)); fi
  echo "$(date '+%F %H:%M:%S')  start  $arm"
  ( $PY -m "$mod" --config "configs/${arm}.json" --checkpoint_path "cp_${arm}" "${flags[@]}" > "cp_${arm}.log" 2>&1
    echo "$(date '+%F %H:%M:%S')  end    $arm  rc=$?" ) &
  running=$((running + 1))
done
[ "$DRY" = 1 ] && exit 0
wait

echo "=== new-design dense front: best / last-5 validation PESQ ==="
for arm in "${DONE[@]}" "${JOBS[@]##*:}"; do
  [ -f "cp_${arm}.log" ] || { echo "$arm  no log"; continue; }
  $PY -c "import re,statistics as s,sys; t=[float(x) for x in re.findall(r'PESQ Score: ([\d.]+)', open('cp_'+sys.argv[1]+'.log').read())]; print(f'{sys.argv[1]:<24} n={len(t):>2} best={max(t):.3f} last5={s.mean(t[-5:]):.3f}' if len(t)>=5 else sys.argv[1]+' too few validations')" "$arm"
done
echo "$(date '+%F %H:%M:%S')  done"
