#!/usr/bin/env bash
# Managed joint training with the 30/15-epoch, effective-batch-8 recipe.
set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO; inspect the log above." >&2' ERR

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
export LC_ALL=C LANG=C PYTHONUNBUFFERED=1
export DDVC_RUN_VARIANT=online_quality DDVC_QUALITY_MODE=joint

COMMAND=${1:?Use auth/start/logs/status/resume/retry/restore/export/backup/recipe}
shift
if [[ "$COMMAND" == auth ]]; then
    exec bash "$REPO/tools/online_quality_drive.sh" auth "$@"
fi

DATASET=${1:?Dataset must be yc2 or anet}
shift
case "$DATASET" in
    yc2|anet) ;;
    *) echo 'Dataset must be yc2 or anet' >&2; exit 2 ;;
esac

case "$COMMAND" in
    start|resume|retry|restore)
        test -f /marimo/KLTN_DRIVE_PRIVATE.env
        command -v rclone >/dev/null
        bash "$REPO/tools/online_quality_drive.sh" auth
        ;;
esac

case "$COMMAND" in
    start)
        exec bash "$REPO/tools/online_quality_drive.sh" start "$DATASET" "$@"
        ;;
    resume)
        exec bash "$REPO/tools/online_quality_drive.sh" start "$DATASET" --resume "$@"
        ;;
    retry)
        exec bash "$REPO/tools/online_quality_drive.sh" start "$DATASET" --retry "$@"
        ;;
    logs|status|restore|export|backup)
        exec bash "$REPO/tools/online_quality_drive.sh" "$COMMAND" "$DATASET" "$@"
        ;;
    recipe)
        "$PY" - "$REPO" "$DATASET" <<'PY'
import json
from pathlib import Path
import sys

repo, dataset = Path(sys.argv[1]), sys.argv[2]
options = json.loads((repo / 'tools/c1_frozen_options.json').read_text())[dataset]
print(json.dumps({
    'dataset': dataset,
    'epochs': 30 if dataset == 'yc2' else 15,
    'seed': options['seed'],
    'batch_size': options['batch_size'],
    'gradient_accumulation_steps': options['gradient_accumulation_steps'],
    'effective_batch_size': options['batch_size'] * options['gradient_accumulation_steps'],
    'quality_gradient_mode': 'joint',
    'quality_loss_coef': 0.1,
    'quality_head_lr': 0.001,
    'generator_lr': options['lr'],
    'proposal_mask_loss_coef': options['proposal_mask_loss_coef'],
    'num_queries': options['num_queries'],
}, indent=2))
PY
        ;;
    *) echo 'Use auth/start/logs/status/resume/retry/restore/export/backup/recipe' >&2; exit 2 ;;
esac
