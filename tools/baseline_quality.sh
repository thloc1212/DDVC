#!/usr/bin/env bash
# Two stages, with separate verified Drive recovery for the baseline and head.
set -Eeuo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(dirname "$REPO")
export DDVC_RUN_VARIANT=baseline_quality
export DDVC_C1_WORK="$WORK"
export LC_ALL=C LANG=C PYTHONUNBUFFERED=1
COMMAND=${1:-help}
DATASET=${2:-yc2}
case "$DATASET" in yc2) EPOCHS=15;; anet) EPOCHS=6;; *) echo 'Use yc2 or anet'; exit 2;; esac
if [[ $# -ge 2 ]]; then shift 2; elif [[ $# -ge 1 ]]; then shift; fi
case "$COMMAND" in
 run) exec "${DDVC_PYTHON:-/tmp/uv-venv/bin/python}" -u "$REPO/tools/baseline_quality_chain.py" start "$DATASET";;
 retry) exec "${DDVC_PYTHON:-/tmp/uv-venv/bin/python}" -u "$REPO/tools/baseline_quality_chain.py" retry "$DATASET";;
 restore-data) exec "${DDVC_PYTHON:-/tmp/uv-venv/bin/python}" -u "$REPO/tools/baseline_stage_assets.py" "$DATASET";;
 auth) exec bash "$REPO/tools/c1_drive.sh" auth "$@";;
 start|status|backup|restore|export)
   exec bash "$REPO/tools/c1_drive.sh" "$COMMAND" "$DATASET" "$@";;
 quality|quality-status|quality-backup|quality-restore|quality-export)
   ACTION=${COMMAND#quality-}; if [[ "$COMMAND" == quality ]]; then ACTION=start; fi
   exec bash "$REPO/tools/c1_rerank.sh" "$ACTION" "$DATASET" --work "$WORK/frozen_quality" "$@";;
 logs)
   # tail -F also follows files that the next stage creates later.
   exec tail -n 45 -F "$WORK/logs/$DATASET.drive.log" \
     "$WORK/logs/$DATASET.baseline_quality.chain.log" \
     "$WORK/logs/${DATASET}_baseline_e${EPOCHS}_b1.launch.log" \
     "$WORK/logs/${DATASET}_baseline_e${EPOCHS}_b1.log" \
     "$WORK/frozen_quality/logs/$DATASET.rerank.log";;
 results) exec cat "$WORK/frozen_quality/runs/$DATASET/evaluation/summary.tsv";;
 *) echo 'baseline_quality.sh auth | run|retry|start|status|backup|restore|export DATASET [flags]'
    echo 'baseline_quality.sh quality|quality-status|quality-backup|quality-restore|quality-export|logs|results DATASET [flags]'
    exit 2;;
esac
