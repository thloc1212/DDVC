#!/usr/bin/env bash
set -Eeuo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
export LC_ALL=C LANG=C PYTHONUNBUFFERED=1 PYTHONHASHSEED=777
export TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg PRAWN_ABLATION_VARIANT=full
export PYTHONPATH="$REPO/tools:$REPO:$REPO/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"
test -x "$PY"
cd "$REPO"
exec "$PY" -u tools/prawn_ablation.py "$@"
