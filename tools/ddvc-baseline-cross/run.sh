#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C LANG=C PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export DDVC_RUN_VARIANT=baseline_quality CUDA_VISIBLE_DEVICES=${DDVC_GPU:-0}

bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$bundle/../.." && pwd)
work=$(dirname "$repo")
python=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
direction=${1:-both}
case "$direction" in both|yc2-to-anet|anet-to-yc2) ;; *) echo 'Usage: bash run.sh [both|yc2-to-anet|anet-to-yc2]' >&2; exit 2;; esac
test -x "$python"
command -v rclone >/dev/null
mkdir -p "$work/logs"
exec > >(tee -a "$work/logs/$direction.log") 2>&1
exec 9>"$work/cross.lock"
flock -n 9 || { echo 'Another cross evaluation is running.' >&2; exit 1; }

cd "$repo"
export PYTHONPATH="$repo/tools:$repo:$repo/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$work/hf-cache" PATH="$work/tools/java/bin:$PATH"
if [[ ! -f /marimo/secrets/rclone.conf ]]; then
  "$python" -c 'from c1_rclone import restore_auth; restore_auth("/marimo/KLTN_DRIVE_PRIVATE.env", "/marimo/secrets/rclone.conf", "kltnbackup")'
fi
rclone --config /marimo/secrets/rclone.conf lsf kltnbackup:KLTNRUNS/runs --max-depth 1 --dirs-only > /dev/null
if [[ ! -f "$work/provenance/cuda_ready" ]]; then
  bash tools/c1_molab.sh setup
fi
if [[ -f "$work/cuda130_env.sh" ]]; then source "$work/cuda130_env.sh"; fi

if [[ "$direction" != anet-to-yc2 ]]; then
  "$python" -u "$bundle/cross_eval.py" yc2 anet
fi
if [[ "$direction" != yc2-to-anet ]]; then
  "$python" -u "$bundle/cross_eval.py" anet yc2
fi
