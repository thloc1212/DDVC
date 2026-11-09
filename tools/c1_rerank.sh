#!/usr/bin/env bash
# Run alongside the existing C1 checkout; never checkout/pull inside DDVC_C1.
set -Eeuo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE=${DDVC_C1_WORK:-/marimo/DDVC_C1}
PY=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
export LC_ALL=C LANG=C PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=${DDVC_GPU:-0}
export PYTHONPATH="$REPO:$REPO/tools:$BASE/source:$BASE/source/tools:$BASE/source/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$BASE/tools/java/bin:$PATH"
export HF_HOME=${DDVC_HF_HOME:-$BASE/hf-cache}
if [[ -f "$BASE/cuda130_env.sh" ]]; then source "$BASE/cuda130_env.sh"; fi
test -x "$PY"
cd "$REPO"
exec "$PY" -u "$REPO/tools/c1_rerank.py" "$@" --base "$BASE"
