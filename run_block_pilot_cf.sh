#!/usr/bin/env bash
# ConvFSENet block-design pilot: does GEMM -> BN -> ReLU in the frontend, a frozen
# per-bin input mean, and a generator lr warmup change early training?
#
# Two arms, 10 epochs each from scratch, run concurrently on the same recipe as
# configs/cfs_c96_dense.json (96/192, 9 TCM blocks, lr 8e-4 cosine, metric-GAN):
#   pilot_cf_B0   unchanged copy of cfs_c96_dense (control; flags off -> the
#                 original model and trainer, byte-identical)
#   pilot_cf_B1   + frontend_norm "batch"   frontend = Conv1d -> BatchNorm1d -> ReLU
#                 + input_norm mean         subtract the frozen training-set mean
#                   (configs/stats/convfsenet_vbd_train_magc0.3_nfft512.json,
#                    built by `python -m convfsenet.input_stats`)
#                 + warmup_steps 2000       linear generator-lr warmup
#
# Note: the cosine T_max is --training_epochs (10), so the lr reaches lr_min at
# the end of the pilot; both arms share that.
# Export of a B1 checkpoint: python -m convfsenet.fold --checkpoint_file
# cp_pilot_cf_B1/<ckpt> --output_dir <dir>, then the usual export_onnx on <dir>/g_best.
#
# Usage: ./run_block_pilot_cf.sh            (from the repo root; PY overrides the interpreter)
set -u
cd "$(dirname "$0")"

if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=/home/clement/eco8-neaixt/.venv/bin/python; fi
fi
# 723 steps/epoch => 10 epochs are steps 0..7229, so a 7230 validation interval would
# never fire (no PESQ, no g_best). Validate every epoch instead: training-neutral here
# (eval mode under no_grad, both loaders carry their own seeded generator).
FLAGS=(--training_epochs 10 --stdout_interval 723 --validation_interval 723
       --checkpoint_interval 1446 --best_checkpoint_start_epoch 0)
ARMS=(pilot_cf_B0 pilot_cf_B1)

STATS=configs/stats/convfsenet_vbd_train_magc0.3_nfft512.json
if [ ! -f "$STATS" ]; then
  echo "missing $STATS -- build it with: $PY -m convfsenet.input_stats --output $STATS"; exit 1
fi
for arm in "${ARMS[@]}"; do
  if compgen -G "cp_${arm}/g_*" > /dev/null; then
    echo "cp_${arm} already holds checkpoints; the trainer would RESUME, not start from scratch. Move it away first."
    exit 1
  fi
done
# Fail fast on a config that does not build (e.g. stats/feature mismatch) before any arm starts.
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" $PY - "${ARMS[@]}" <<'PY' || exit 1
import json, sys
from common.env import AttrDict
from convfsenet.model import build_causal_model
for arm in sys.argv[1:]:
    m = build_causal_model(AttrDict(json.load(open(f"configs/{arm}.json"))))
    print(f"{arm}: builds, {sum(p.numel() for p in m.parameters()):,} params, "
          f"frontend_norm={m.frontend_norm}, input_centering={m.input_centering}")
PY

pids=()
for arm in "${ARMS[@]}"; do
  echo "$(date +%H:%M:%S)  start  $arm"
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" $PY -m convfsenet.train --config "configs/${arm}.json" \
    --checkpoint_path "cp_${arm}" "${FLAGS[@]}" > "cp_${arm}.log" 2>&1 &
  pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "$(date +%H:%M:%S)  FAILED ${ARMS[$i]}"; rc=1; }
done
for arm in "${ARMS[@]}"; do
  printf '  %-14s PESQ: %s\n' "$arm" "$(grep -o 'PESQ Score: [0-9.]*' "cp_${arm}.log" | sed 's/PESQ Score: //' | tr '\n' ' ')"
done
echo "$(date +%H:%M:%S)  done"
exit $rc
