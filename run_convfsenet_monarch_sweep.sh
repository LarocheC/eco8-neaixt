#!/usr/bin/env bash
# ConvFSENet Monarch pointwise block-count sweep — MAC-matched against plain
# narrower dense models. FROM SCRATCH, full recipe (lr 8e-4 cosine, 200 epochs,
# metric-GAN on), no --init_from anywhere.
#
# The question, the same one RESULTS_NSNET2.md's block-count sweep asked of
# NSNet2: when you take MACs out of the pointwise convolutions, is it better to
# spend the remaining budget on Monarch's cross-channel structure at full width,
# or on a plain dense model that is simply narrower? Two arms, matched on MACs
# per frame (within 1.34%), one knob each:
#
#   A  cfs_mon_nb{4,8,16,32}   192/384 channels, TCM pointwise convs replaced by
#                              two-factor Monarch, only `nblocks` varies.
#   B  cfs_dense_r{146,108,83,67}  no structure, only the channel width varies.
#
#   nblocks  MACs/frame  params   |  width    MACs/frame  params   | MAC delta
#         4     855,552  873,281  |  146/292    850,304  863,847   |  -0.61%
#         8     482,304  500,033  |  108/216    481,248  491,333   |  -0.22%
#        16     295,680  313,409  |   83/166    295,148  302,958   |  -0.18%
#        32     202,368  220,097  |   67/134    199,660  206,014   |  -1.34%
#
# The grid starts at 4 on purpose: Monarch's first factor scales with in**2, so
# the contracting 384->192 layer only compresses nblocks/3, and at nblocks=2 the
# "compressed" model is 1.12x LARGER than the dense one.
#
# The dense 192/384 anchor is NOT retrained: cp_cfs_dense (2.877) already ran
# this exact recipe for 200 epochs. Every config here is derived from
# cp_cfs_dense/config.json so the only differences are the swept knobs — note
# that means num_workers=3, not the 4 in configs/convfsenet.json.
#
# Waves (4-up on one GPU, ~12 h each — measured at 215 s/epoch when four ran
# together for the blockdiag sweep):
#   WAVE=1  the two decisive pairings: nb8/r108 (crossover) and nb32/r67 (small)
#   WAVE=2  the rest of the trend: nb4/r146 and nb16/r83
#   WAVE=3  repeat seeds, set SEED=2345 or 3456 and ARMS to the chosen pairing
# Override freely:  WAVE=2 ./run_convfsenet_monarch_sweep.sh
#                   ARMS="cfs_mon_nb8" ./run_convfsenet_monarch_sweep.sh
set -u

cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
EPOCHS="${EPOCHS:-200}"
WAVE="${WAVE:-1}"
SEED="${SEED:-}"                 # empty = the config's own seed (1234)

case "$WAVE" in
    1) DEFAULT_ARMS="cfs_mon_nb8 cfs_dense_r108 cfs_mon_nb32 cfs_dense_r67" ;;
    2) DEFAULT_ARMS="cfs_mon_nb4 cfs_dense_r146 cfs_mon_nb16 cfs_dense_r83" ;;
    *) DEFAULT_ARMS="" ;;
esac
ARMS="${ARMS:-$DEFAULT_ARMS}"
if [[ -z "$ARMS" ]]; then
    echo "ERROR: no arms. Set WAVE=1|2 or ARMS=\"...\"." >&2
    exit 1
fi

suffix=""
if [[ -n "$SEED" ]]; then suffix="_s${SEED}"; fi

# ---- preflight: every arm builds, and lands on the parameter count the
# results table reports. Twelve hours is too long to find out a config typo
# silently produced a dense model.
echo "=== preflight ==="
for arm in $ARMS; do
    cfg="configs/${arm}.json"
    [[ -f $cfg ]] || { echo "ERROR: missing $cfg" >&2; exit 1; }
    $PY - "$cfg" <<'EOF' || exit 1
import json, sys
from torch import nn
from convfsenet.model import build_causal_model
from convfsenet.layers import MonarchPointwise, BlockdiagPointwise
h = json.load(open(sys.argv[1]))
m = build_causal_model(h)
macs = 0
for mod in m.modules():
    if isinstance(mod, nn.Conv1d):
        macs += (mod.in_channels // mod.groups) * mod.out_channels * mod.kernel_size[0]
    elif isinstance(mod, MonarchPointwise):
        macs += mod.w1.numel() + mod.w2.numel()
    elif isinstance(mod, BlockdiagPointwise):
        macs += mod.weight.numel()
pw = h.get("pointwise", {"kind": "conv"})
structured = sum(isinstance(b.conv1x1, (MonarchPointwise, BlockdiagPointwise)) for b in m.tcm)
if pw.get("kind", "conv") != "conv" and structured != len(m.tcm):
    raise SystemExit(f"FAIL {sys.argv[1]}: {structured}/{len(m.tcm)} blocks structured")
print(f"  {sys.argv[1]:32s} {pw.get('kind','conv'):9s} "
      f"res={h['n_channels_res']:3d} params={sum(p.numel() for p in m.parameters()):9,d} "
      f"macs/frame={macs:9,d}")
EOF
done

# PREFLIGHT=1 stops here: check the configs without committing the box to
# ~12 h of training. (Running the driver just to see the table starts the
# whole wave otherwise — it does, ask me how I know.)
if [[ "${PREFLIGHT:-0}" == "1" ]]; then
    echo "PREFLIGHT=1 — configs check out, not launching."
    exit 0
fi

# Wait for any in-flight convfsenet training before adding load — this box is
# shared, and four concurrent runs already saturate it.
# The bracket trick alone is not enough — commit 05afbcc fixed this exact
# stall. An unescaped dot matches "/", so a stray `vim convfsenet/train.py`
# counts as in-flight training, and so does any shell whose command line merely
# mentions the module. Anchor on the real launch shape and escape the dot; cap
# the wait so a false positive is loud instead of an indefinite hang.
waited=0
while pgrep -f "python.* -m [c]onvfsenet\.train" > /dev/null 2>&1; do
    if (( waited >= 7200 )); then
        echo "ERROR: still waiting after 2 h. Check: pgrep -af 'python.* -m convfsenet\.train'" >&2
        exit 1
    fi
    echo "$(date +%H:%M:%S)  waiting for in-flight convfsenet training to finish..."
    sleep 120; waited=$((waited + 120))
done
echo "$(date +%H:%M:%S)  box is clear, starting wave ${WAVE}: $ARMS"

pids=()
names=()
for arm in $ARMS; do
    cfg="configs/${arm}.json"
    cp_dir="cp_${arm}${suffix}"
    if [[ -n "$SEED" ]]; then
        cfg="/tmp/${arm}${suffix}.json"
        $PY -c "import json,sys;c=json.load(open('configs/${arm}.json'));c['seed']=${SEED};json.dump(c,open('${cfg}','w'),indent=4)"
    fi
    echo "$(date +%H:%M:%S)  start  ${arm}${suffix} -> ${cp_dir}"
    # Append rather than truncate: train.py auto-resumes from the rolling
    # checkpoint, so relaunching after a crash is the recovery path — and it
    # would otherwise destroy the pre-crash validations the summary below greps.
    echo "=== $(date +%F\ %T) launch ${arm}${suffix} ===" >> "${cp_dir}.log"
    # PYTHONUNBUFFERED: stdout to a file is block-buffered, and at one line per
    # epoch that means ~70 epochs of silence per 8 KiB flush — a 12 h run you
    # cannot watch. Costs nothing.
    PYTHONUNBUFFERED=1 $PY -m convfsenet.train \
        --config "$cfg" \
        --checkpoint_path "$cp_dir" \
        --training_epochs "$EPOCHS" --stdout_interval 723 \
        --validation_interval 7230 --checkpoint_interval 28920 \
        --best_checkpoint_start_epoch 10 >> "${cp_dir}.log" 2>&1 &
    pids+=($!)
    names+=("${arm}${suffix}")
done

for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || echo "$(date +%H:%M:%S)  FAILED ${names[$i]} — see cp_${names[$i]}.log"
done

echo
echo "=== best validation PESQ per arm (dense 192/384 anchor: cp_cfs_dense 2.877) ==="
for n in "${names[@]}"; do
    best=$(grep 'PESQ Score' "cp_${n}.log" 2>/dev/null | sed 's/.*PESQ Score: //;s/,.*//' | sort -g | tail -1)
    printf '%-24s %s\n' "$n" "${best:-none}"
done
echo "$(date +%H:%M:%S)  wave ${WAVE} done"
