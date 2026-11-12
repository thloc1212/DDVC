#!/usr/bin/env bash
set -Eeuo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(dirname "$REPO")
PY=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
export LC_ALL=C LANG=C PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$REPO:$REPO/tools${PYTHONPATH:+:$PYTHONPATH}"
COMMAND=${1:-help}
DATASET=${2:-yc2}
case "$DATASET" in yc2|anet) ;; *) echo 'Use yc2 or anet'; exit 2;; esac
cd "$REPO"
case "$COMMAND" in
 auth|restore|start|backup|export)
   exec "$PY" -u tools/mask_quality_beta_restore.py "$COMMAND" "$DATASET" --work "$WORK";;
 logs) exec tail -n 80 -F "$WORK/logs/$DATASET.beta.log";;
 results) exec cat "$WORK/runs/$DATASET/beta_grid/summary.tsv";;
 *) echo 'mask_quality_beta.sh auth|restore|start|logs|results|backup|export yc2|anet'; exit 2;;
esac
