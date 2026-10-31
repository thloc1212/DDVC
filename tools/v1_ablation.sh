#!/usr/bin/env bash
set -Eeuo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(dirname "$REPO")
PY=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
export CUDA_VISIBLE_DEVICES=${DDVC_GPU:-0} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 MPLBACKEND=Agg LC_ALL=C
export PYTHONPATH="$REPO:$REPO/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${DDVC_HF_HOME:-$WORK/hf-cache}
export PATH="$WORK/tools/java/bin:$PATH"
if [[ -f "$WORK/cuda130_env.sh" ]]; then source "$WORK/cuda130_env.sh"; fi
cd "$REPO"
mkdir -p "$WORK/logs" "$WORK/provenance"
COMMAND=${1:?Use smoke/train/report/score/postprocess followed by a config path}
CFG=${2:?Missing config path}
RUN=$("$PY" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["id"])' "$CFG")
SIGNATURE=$(sha256sum "$CFG" | cut -d' ' -f1)
MARKER="$WORK/provenance/ablation_${RUN}_${SIGNATURE}_smoke_ready"
case "$COMMAND" in
  smoke)
    test -f "$WORK/provenance/cuda_ready"
    "$PY" -u tools/molab_pipeline.py smoke --config "$CFG" 2>&1 | tee "$WORK/logs/${RUN}.smoke.log"
    touch "$MARKER"
    ;;
  train)
    test -f "$WORK/provenance/cuda_ready"
    test -f "$MARKER" || { echo 'Run smoke for this exact config first'; exit 1; }
    "$PY" tools/training_pid.py --pid-file "$WORK/logs/$RUN.pid" --config "$CFG"
    "$PY" - "$CFG" <<'PY'
import sys,yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text())
assert not cfg.get('start_from') and not cfg.get('pretrain'), 'Fresh training only'
dest=Path(cfg['save_dir'])/cfg['id']
if dest.exists():raise FileExistsError(f'Preserve run: {dest}. Choose a new run ID.')
PY
    nohup "$PY" -u train.py --cfg_path "$CFG" > "$WORK/logs/$RUN.log" 2>&1 < /dev/null &
    echo $! > "$WORK/logs/$RUN.pid"
    echo "STARTED PID=$! LOG=$WORK/logs/$RUN.log"
    ;;
  report) "$PY" -u tools/molab_pipeline.py report --config "$CFG" ;;
  postprocess) "$PY" -u tools/v1_ablation.py postprocess --config "$CFG" --raw "${3:?raw JSON required}" --output-dir "${4:?output directory required}" ;;
  score) "$PY" -u tools/v1_ablation.py score --config "$CFG" --input-dir "${3:?selection directory required}" ;;
  *) echo 'Unknown command'; exit 1 ;;
esac
