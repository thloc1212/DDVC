#!/usr/bin/env bash
# One private config restore, then independent dataset workers. No notebook cells.
set -Eeuo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg LC_ALL=C LANG=C
export CUDA_VISIBLE_DEVICES=${DDVC_GPU:-0}
if [[ ! -x "$PY" ]]; then
  echo "Missing native Molab Python: $PY; select the GPU runtime first." >&2
  exit 1
fi
exec "$PY" -u "$REPO/tools/qts_rclone.py" "$@"
