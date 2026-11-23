#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C LANG=C PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg

bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$bundle/../.." && pwd)
work=$(dirname "$repo")
python=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
direction=${1:-both}
export PRAWN_CROSS_BASE_COMMIT=31f1eac41841f3a870bb56d9c74a52fc55057466
case "$direction" in both|yc2-to-anet|anet-to-yc2) ;; *) echo 'Usage: bash prawn_cross.sh [both|yc2-to-anet|anet-to-yc2]' >&2; exit 2;; esac
test -x "$python"
command -v rclone >/dev/null
mkdir -p "$work/logs"
exec > >(tee -a "$work/logs/$direction.log") 2>&1
exec 9>"$work/cross.lock"
flock -n 9 || { echo 'Another cross evaluation is running.' >&2; exit 1; }

cd "$repo"
git diff --quiet "$PRAWN_CROSS_BASE_COMMIT" HEAD -- . ':(exclude)tools/prawn-cross-molab' || {
  echo "Checkout must retain the PRAWN source from $PRAWN_CROSS_BASE_COMMIT." >&2
  exit 1
}
export PYTHONPATH="$repo/tools:$repo:$repo/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$work/hf-cache" PATH="$work/tools/java/bin:$PATH"
if [[ ! -f "$work/provenance/cuda_ready" ]]; then
  bash tools/c1_molab.sh setup
fi
if [[ -f "$work/cuda130_env.sh" ]]; then source "$work/cuda130_env.sh"; fi

if [[ "$direction" != anet-to-yc2 ]]; then
  "$python" -u "$bundle/prawn_cross.py" yc2 anet
fi
if [[ "$direction" != yc2-to-anet ]]; then
  "$python" -u "$bundle/prawn_cross.py" anet yc2
fi
