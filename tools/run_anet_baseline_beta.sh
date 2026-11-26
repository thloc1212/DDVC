#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=${DDVC_BETA_WORK:-$(dirname "$repo")}
python=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
if [[ $# -eq 0 ]]; then set -- 10 12 14 16 18 20; fi
export LC_ALL=C LANG=C PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$work/tools/java/bin:$PATH"

test -x "$python"
command -v rclone >/dev/null
mkdir -p "$work/tools"
exec 9>"$work/beta-sweep.lock"
flock -n 9 || { echo 'An ActivityNet beta sweep is already running here.' >&2; exit 1; }
cd "$repo"

"$python" -c 'import torch; print("TORCH_READY", torch.__version__)'
if ! "$python" -c 'import yaml, scipy, pandas, joblib, matplotlib, tqdm' > /dev/null 2>&1; then
  uv pip install --python "$python" PyYAML scipy pandas joblib matplotlib tqdm
fi
"$python" -u tools/restore_anet_baseline_beta.py --work "$work"
"$python" -u tools/molab_pipeline.py java --target "$work/tools/java"
"$python" -u tools/frozen_quality_beta_grid.py --work "$work" --dataset anet \
  --relocate-references --betas "$@"
