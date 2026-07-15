#!/usr/bin/env bash
# Download the trained LiSenNet checkpoint used by the RT595 firmware from HuggingFace.
# Default: conv-hardened (the deploy target; matches the committed model_data.h).
# Needs `huggingface_hub` (pip install huggingface_hub). Public repo, no token needed.
set -euo pipefail
cd "$(dirname "$0")/../../.."                     # repo root
VARIANT="${1:-conv-hardened}"                     # gru | conv | conv-hardened | conv-hardened-deep
DST="cp_lisennet_${VARIANT//-/_}"
PY="${PYTHON:-python3}"

"$PY" - "$VARIANT" "$DST" <<'EOF'
import sys, os, shutil
from huggingface_hub import hf_hub_download
variant, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in ("config.json", "g_best"):
    p = hf_hub_download("claroche1/LiSenNet", f"{variant}/{f}")
    shutil.copy(p, os.path.join(dst, f))
    print(f"  {f}: {os.path.getsize(os.path.join(dst, f))} bytes")
print("downloaded", variant, "->", dst)
EOF
