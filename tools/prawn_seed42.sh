#!/usr/bin/env bash
set -Eeuo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(dirname "$REPO")
PY=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
export LC_ALL=C LANG=C PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=${DDVC_GPU:-0}
export DDVC_C1_WORK="$WORK" HF_HOME="$WORK/hf-cache"
export PYTHONPATH="$REPO/tools:$REPO:$REPO/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$WORK/tools/java/bin:$PATH"
test -x "$PY"
cd "$REPO"
exec "$PY" -u tools/prawn_seed42.py "$@"
