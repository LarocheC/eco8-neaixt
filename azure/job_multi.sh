#!/usr/bin/env bash
# Several run_init_sweep.sh runs side by side on one Azure ML node (one GPU).
#   bash job_multi.sh OUT COMMIT ARM:SEED [ARM:SEED ...]
# Same steps as job.sh (clone eco8-neaixt at COMMIT on $BRANCH, default
# nsnet2-mixed-radix-all, frozen lock, clean checkout, restore partial runs,
# mirror to OUT every 10 minutes), but the runs share the GPU: the small
# NSNet2 variants leave most of an A100 idle. Each run is the same command as
# a job.sh run and writes its own cp_<ARM>_s<SEED>; VAL_SPEAKERS and the other
# run_init_sweep.sh variables pass through. Exits non-zero if any run failed.
set -euo pipefail
OUT="$1"; COMMIT="$2"; shift 2; RUNS=("$@"); BRANCH="${BRANCH:-nsnet2-mixed-radix-all}"
echo "node $(hostname), $(nproc) cpus, start $(date -Is), runs: ${RUNS[*]}"; free -g | head -2
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if ! command -v git >/dev/null || ! command -v rsync >/dev/null; then
    apt-get update -qq && apt-get install -y -qq git rsync curl >/dev/null
fi
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH="$HOME/.local/bin:$PATH"
W=/tmp/w; rm -rf "$W"; mkdir -p "$W"; cd "$W"
git clone -q --branch "$BRANCH" https://github.com/LarocheC/eco8-neaixt eco8
cd eco8
git checkout -q "$COMMIT"
uv sync --frozen
if [[ -n "$(git status --porcelain)" ]]; then git status --porcelain; echo "checkout not clean" >&2; exit 1; fi
for r in "${RUNS[@]}"; do
    NAME="cp_${r%%:*}_s${r##*:}"
    for d in "$OUT/$NAME" "$OUT/$NAME".aborted*; do
        if [[ -e $d ]]; then cp -a "$d" .; echo "restored $(basename "$d") from a previous attempt"; fi
    done
done
( while true; do sleep 600; rsync -a --exclude 'g_0*' --exclude 'do_0*' ./cp_* "$OUT"/ 2>/dev/null || true; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; done ) &
MIRROR=$!
pids=()
for r in "${RUNS[@]}"; do
    ARMS="${r%%:*}" SEEDS="${r##*:}" GPU=0 ./run_init_sweep.sh > "sweep_${r%%:*}_s${r##*:}.log" 2>&1 &
    pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "${RUNS[$i]} ok"; else echo "${RUNS[$i]} FAILED"; tail -20 "sweep_${RUNS[$i]%%:*}_s${RUNS[$i]##*:}.log"; rc=1; fi
done
kill "$MIRROR" 2>/dev/null || true
rsync -a ./cp_* ./sweep_*.log "$OUT"/
echo "end $(date -Is) rc=$rc"
exit $rc
