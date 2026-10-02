#!/usr/bin/env bash
# Butterfly-native architecture screen, stage 1 (BUTTERFLY_ARCH.md): nine arms, seed 1234,
# 100 epochs, validation every 10 epochs on the full test split, no resume -- an arm that
# dies is rerun from scratch (a resume cost the butterfly control ~0.07 PESQ, 2026-10-01).
# Scheduling is a GPU-memory ledger: each arm declares its measured training peak (one real
# epoch incl. validation, expandable segments, + margin) and starts only while the declared
# total of the running arms stays under CAP_MIB and fewer than MAX_JOBS run (the per-step
# CPU batch_pesq is the shared bottleneck). nvidia-smi free memory is NOT used: every
# validation's empty_cache() makes a running job look small.
#   DRY=1 ./run_bfly_arch_wave.sh        print the plan
#   ARMS="ba_R ba_A_res" ./...           a subset
set -u
cd "$(dirname "$0")"
PY="${PY:-/home/clement/eco8-neaixt/.venv/bin/python}"
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DRY="${DRY:-0}"; EPOCHS="${EPOCHS:-100}"; CAP_MIB="${CAP_MIB:-22500}"; MAX_JOBS="${MAX_JOBS:-4}"; POLL="${POLL:-30}"
NS=(--training_epochs "$EPOCHS" --stdout_interval 45 --validation_interval 450 --checkpoint_interval 1800 --best_checkpoint_start_epoch 0)
# arm:MiB in launch-priority order. MiB = measured training peak + 10 %: one real epoch incl.
# a full-test validation, expandable segments, nvidia-smi sampled every 2 s (2026-10-01).
# First-fit: whenever memory frees, the first pending arm that fits starts. The order puts
# the GPU-heavy arms first (F, D, E) so the CPU-bound small ones fill around them (~15 h
# simulated makespan vs ~18 h for D/E first).
PLAN="${PLAN:-ba_F_wide:14124 ba_D_grid:9379 ba_E_unet:9665 ba_A_res:6708 ba_C_lru:6398 ba_B_cep:5982 ba_R2b:5852 ba_R:5828 ba_C_mingru:4706}"
declare -A NEED
JOBS=()
for j in $PLAN; do JOBS+=("${j%%:*}"); NEED[${j%%:*}]="${j##*:}"; done
[ -n "${ARMS:-}" ] && read -r -a JOBS <<< "$ARMS"
RUN=.arch_running; mkdir -p "$RUN"
exec 3>&1

for arm in "${JOBS[@]}"; do
  [ -f "configs/${arm}.json" ] || { echo "refusing: configs/${arm}.json missing"; exit 1; }
  [ -n "${NEED[$arm]:-}" ] || { echo "refusing: no memory budget for $arm"; exit 1; }
  if compgen -G "cp_${arm}/g_*" >/dev/null || compgen -G "cp_${arm}/do_*" >/dev/null; then echo "refusing: cp_${arm} has checkpoints"; exit 1; fi
  if [ -s "cp_${arm}.log" ] && [ "$DRY" != 1 ]; then echo "refusing: cp_${arm}.log exists"; exit 1; fi
done

ledger() { local t=0 f; for f in "$RUN"/*; do [ -e "$f" ] && t=$((t + $(cat "$f"))); done; echo $t; }
running() { ls "$RUN" | wc -l; }

if [ "$DRY" = 1 ]; then
  for arm in "${JOBS[@]}"; do echo "$arm  needs ${NEED[$arm]} MiB  ->  $PY -m nsnet2.train --config configs/${arm}.json --checkpoint_path cp_${arm} ${NS[*]}"; done
  exit 0
fi
rm -f "$RUN"/*
pending=("${JOBS[@]}")
while [ ${#pending[@]} -gt 0 ]; do
  started=0
  for i in "${!pending[@]}"; do                       # first fit, in priority order
    arm=${pending[$i]}; need=${NEED[$arm]}
    [ $(( $(ledger) + need )) -le "$CAP_MIB" ] && [ "$(running)" -lt "$MAX_JOBS" ] || continue
    echo "$need" > "$RUN/$arm"
    echo "$(date '+%F %H:%M:%S')  start  $arm  (needs $need MiB, ledger now $(ledger) MiB, $(running) running)"
    ( exec > "cp_${arm}.log" 2>&1
      $PY -m nsnet2.train --config "configs/${arm}.json" --checkpoint_path "cp_${arm}" "${NS[@]}"
      rc=$?; rm -f "$RUN/$arm"; echo "$(date '+%F %H:%M:%S')  end    $arm  rc=$rc" >&3 ) &
    unset 'pending[i]'; pending=("${pending[@]}"); started=1
    sleep "${LAUNCH_GAP:-20}"; break
  done
  [ "$started" = 1 ] || sleep "$POLL"
done
wait

echo "=== stage 1: best / last-3 (last three validations: epochs 70, 80, 90) PESQ, engine pairs/frame ==="
for arm in "${JOBS[@]}"; do
  $PY - "$arm" <<'PYEOF'
import json, re, statistics as st, sys
from common.env import AttrDict
from nsnet2.bfly_arch import build_generator, engine_pairs
arm = sys.argv[1]
v = dict((int(a), float(b)) for a, b in re.findall(r"Steps : (\d+), PESQ Score: ([\d.]+)", open(f"cp_{arm}.log").read()))
t = [v[k] for k in sorted(v)]
m = build_generator(AttrDict(json.load(open(f"configs/{arm}.json"))))
npar, pairs = sum(p.numel() for p in m.parameters()), engine_pairs(m)
print(f"{arm:<14} n={len(t):>2} best={max(t) if t else float('nan'):.3f} last3={st.mean(t[-3:]) if len(t) >= 3 else float('nan'):.4f} "
      f"params={npar/1e3:.0f}k pairs/frame={pairs/1e3:.1f}k")
PYEOF
done 2>&1 | grep -v -i 'warn\|routing\|from \.\|@torch'
echo "$(date '+%F %H:%M:%S')  done"
