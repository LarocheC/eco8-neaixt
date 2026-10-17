#!/usr/bin/env bash
# One run_init_sweep.sh run on an Azure ML node (one GPU).
#   bash job.sh OUT ARM SEED COMMIT [BRANCH]
# Clones eco8-neaixt at COMMIT (on BRANCH, default nsnet2-mixed-radix-all),
# installs the committed lock file, checks the checkout is clean, and trains
# cp_<ARM>_s<SEED> with the sweep's flags (200 epochs). The run directory is
# mirrored to OUT every 10 minutes (rolling checkpoints left out) and copied at
# the end. A partial run left in OUT by an earlier attempt is restored first, so
# the sweep moves it to <dir>.aborted<k> and restarts from the same weights.
# Same steps as the training grid's job script (rnn-study
# scripts/init_grid/azure/job.sh), with the commit as an argument.
set -euo pipefail
OUT="$1"; ARM="$2"; SEED="$3"; COMMIT="$4"; BRANCH="${5:-nsnet2-mixed-radix-all}"
NAME="cp_${ARM}_s${SEED}"
echo "node $(hostname), $(nproc) cpus, start $(date -Is)"; free -g | head -2
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
for d in "$OUT/$NAME" "$OUT/$NAME".aborted*; do
    if [[ -e $d ]]; then cp -a "$d" .; echo "restored $(basename "$d") from a previous attempt"; fi
done
( while true; do sleep 600; rsync -a --exclude 'g_0*' --exclude 'do_0*' ./"$NAME"* "$OUT"/ 2>/dev/null || true; done ) &
MIRROR=$!
rc=0
ARMS="$ARM" SEEDS="$SEED" GPU=0 ./run_init_sweep.sh || rc=$?
kill "$MIRROR" 2>/dev/null || true
rsync -a ./"$NAME"* "$OUT"/
echo "end $(date -Is) rc=$rc"
exit $rc
