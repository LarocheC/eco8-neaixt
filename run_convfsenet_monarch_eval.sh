#!/usr/bin/env bash
# FP32 + int8 evaluation for the ConvFSENet Monarch sweep arms.
#
# Per arm: FP32 streaming ONNX export -> static int8 PTQ (QDQ, per-channel,
# MinMax over `NUM_CALIB` VBD train utterances) -> dual FP32/int8 PESQ on the
# full 824-utterance VBD test split, plus the int8 RTF.
#
# The Monarch arms lower to grouped Conv nodes, so quantize_static handles their
# weights natively — no Einsum/QDQ registry patching — and convfsenet/quant.py's
# audit (common/quant_audit.py) fails the run if ANY compute node is left
# reading an FP32 weight. That check is the point: counting QuantizeLinear nodes
# cannot see a skipped weight, which is how the pre-2026-07-11 structured NSNet2
# int8 table turned out to be hybrid-precision.
#
# Run AFTER training: ORT eval and the training runs compete for the same cores.
# The guard below refuses to start while any convfsenet.train is alive.
#
#   ./run_convfsenet_monarch_eval.sh                      # all 8 arms + the anchor
#   ARMS="cfs_mon_nb8 cfs_dense_r108" ./run_...sh         # a subset
#   MAX_UTTERANCES=20 ./run_...sh                         # fast plumbing smoke
set -u

cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
NUM_CALIB="${NUM_CALIB:-200}"
MAX_UTTERANCES="${MAX_UTTERANCES:-0}"          # 0 = full 824-utterance test split
ARMS="${ARMS:-cfs_mon_nb4 cfs_mon_nb8 cfs_mon_nb16 cfs_mon_nb32 cfs_dense_r146 cfs_dense_r108 cfs_dense_r83 cfs_dense_r67 cfs_dense}"

max_args=()
if [[ "$MAX_UTTERANCES" != "0" ]]; then
    max_args=(--max_utterances "$MAX_UTTERANCES")
    echo "NOTE: MAX_UTTERANCES=$MAX_UTTERANCES — a smoke run, NOT a reportable PESQ."
fi

if pgrep -f "python.* -m [c]onvfsenet\.train" > /dev/null 2>&1; then
    echo "ERROR: convfsenet training is running; eval would contend for the same" >&2
    echo "       cores and mis-time the RTF. Wait for the sweep to finish." >&2
    exit 1
fi

status=0
for arm in $ARMS; do
    cp_dir="cp_${arm}"
    log="cp_${arm}_eval.log"
    if [[ ! -f "${cp_dir}/g_best" ]]; then
        echo "SKIP ${arm} (no ${cp_dir}/g_best)"
        continue
    fi
    echo "=== [$(date +%H:%M:%S)] ${arm} ==="
    {
        echo "=== $(date +%F\ %T) eval ${arm} ==="
        PYTHONUNBUFFERED=1 $PY -m convfsenet.export_onnx --checkpoint_file "${cp_dir}/g_best" &&
        PYTHONUNBUFFERED=1 $PY -m convfsenet.quant --checkpoint_dir "${cp_dir}" \
            --num_utterances "$NUM_CALIB" &&
        PYTHONUNBUFFERED=1 $PY -m convfsenet.inference_onnx \
            --checkpoint_file "${cp_dir}/g_best.onnx" \
            --output_dir "${cp_dir}/enhanced" "${max_args[@]}"
    } >> "$log" 2>&1 || { echo "  FAILED — see $log"; status=1; continue; }
    echo "  done -> $log"
done

echo
echo "=== FP32 / int8 PESQ per arm ==="
# inference_onnx prints its own summary line; parse that rather than a JSON
# sidecar, which it only writes when extra perceptual metrics are requested.
# Dual (FP32 sidecar present, the normal case):
#   Mean: PESQ FP32=2.931, primary=2.911, delta=0.020; RTF primary=0.017; ...
# Single:
#   Mean: PESQ primary=2.911; RTF=0.017; ...
# NB "primary=" appears twice in the dual line (PESQ and RTF), which a greedy
# regex silently gets wrong — hence a parser with a test rather than sed.
$PY - $ARMS <<'EOF'
import re, sys, os

PESQ_DUAL = re.compile(r"PESQ FP32=([\d.]+),\s*primary=([\d.]+),\s*delta=(-?[\d.]+)")
PESQ_SINGLE = re.compile(r"PESQ primary=([\d.]+)")
RTF = re.compile(r"RTF(?:\s+primary)?=([\d.]+)")


def parse(line):
    """-> (fp32, int8, delta, rtf); None for anything the line does not carry."""
    m = PESQ_DUAL.search(line)
    if m:
        fp32, int8, delta = float(m.group(1)), float(m.group(2)), float(m.group(3))
    else:
        m = PESQ_SINGLE.search(line)
        if not m:
            return None
        fp32, int8, delta = None, float(m.group(1)), None
    r = RTF.search(line)
    return fp32, int8, delta, (float(r.group(1)) if r else None)


_dual = ("Mean: PESQ FP32=2.931, primary=2.911, delta=0.020; "
         "RTF primary=0.017; PESQ failures=0")
assert parse(_dual) == (2.931, 2.911, 0.020, 0.017), parse(_dual)
_neg = ("Mean: PESQ FP32=2.850, primary=2.870, delta=-0.020; "
        "RTF primary=0.0140; PESQ failures=1")
assert parse(_neg) == (2.850, 2.870, -0.020, 0.0140), parse(_neg)
assert parse("Mean: PESQ primary=2.911; RTF=0.017; PESQ failures=0") == (
    None, 2.911, None, 0.017)
assert parse("nothing here") is None


def fmt(v, w, p):
    return f"{v:{w}.{p}f}" if isinstance(v, float) else f"{'n/a':>{w}}"


print(f"{'arm':<18}{'FP32':>9}{'int8':>9}{'delta':>9}{'int8 RTF':>10}")
for arm in sys.argv[1:]:
    path = f"cp_{arm}_eval.log"
    if not os.path.exists(path):
        continue
    got = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("Mean: PESQ"):
                got = parse(line) or got
    if got is None:
        print(f"{arm:<18}{'no-result':>9}")
        continue
    fp32, int8, delta, rtf = got
    print(f"{arm:<18}{fmt(fp32,9,3)}{fmt(int8,9,3)}{fmt(delta,9,3)}{fmt(rtf,10,4)}")
EOF
exit $status
