#!/usr/bin/env bash
# Every command discovers its own paths; no WORK/RUN shell variables required.
set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO. See output above and the logs directory." >&2' ERR
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(dirname "$REPO")
PY=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
MODE=${2:-subset}
case "$MODE" in subset|full) ;; *) echo 'Mode must be subset or full'; exit 1;; esac
RUN=yc2_${MODE}_cvpr2018_kaggle_e15_b1
SMOKE_RUN=yc2_smoke_cvpr2018_e2_b1
CFG="$WORK/prepared/$RUN/config.yml"
export WORK REPO PY RUN CFG
export CUDA_VISIBLE_DEVICES=${DDVC_GPU:-0}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$REPO:$REPO/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${DDVC_HF_HOME:-$WORK/hf-cache}
export PATH="$WORK/tools/java/bin:$PATH"
if [[ -f "$WORK/cuda130_env.sh" ]]; then source "$WORK/cuda130_env.sh"; fi
mkdir -p "$WORK/logs" "$WORK/provenance" "$WORK/runs" "$WORK/prepared"
cd "$REPO"
[[ -x "$PY" ]] || { echo "Set DDVC_PYTHON to Molab's native interpreter"; exit 1; }

logged() {
  local name=$1; shift
  echo "LOG: $WORK/logs/$name.log"
  "$@" 2>&1 | tee "$WORK/logs/$name.log"
}

prepare() {
  local mode=$1 name=$2 epochs=$3
  if [[ -f "$WORK/prepared/$name/config.yml" ]]; then
    echo "Existing config retained: $WORK/prepared/$name/config.yml"; return
  fi
  logged "prepare_$mode" "$PY" -u tools/molab_pipeline.py prepare \
    --mode "$mode" --run-name "$name" --epochs "$epochs" \
    --data-root "${DDVC_DATA_ROOT:-$WORK/assets}" --prepared-root "$WORK/prepared" \
    --output-root "$WORK/runs" --log-root "$WORK/logs"
}

check_config() {
  "$PY" tools/kaggle_recipe.py --config "$CFG" --mode "$MODE"
}

start_training() {
  local resume=$1
  check_config
  if [[ -f "$WORK/logs/$RUN.pid" ]] && kill -0 "$(cat "$WORK/logs/$RUN.pid")" 2>/dev/null; then
    echo "Saved PID is still alive; inspect status before launching another process"; exit 1
  fi
  if [[ "$resume" == yes ]]; then
    "$PY" tools/molab_pipeline.py resume-config --config "$CFG"
  else
    "$PY" - <<'PY'
import os,time
from pathlib import Path
p=Path(os.environ['WORK'])/'runs'/os.environ['RUN']
if p.exists():
    if list(p.glob('*.pth')):
        raise RuntimeError('Run has checkpoints; use resume, not train')
    saved=p.with_name(p.name+'_startup_failed_'+time.strftime('%Y%m%d_%H%M%S'))
    p.rename(saved)
    print('Preserved incomplete startup:',saved)
PY
  fi
  nohup "$PY" -u train.py --cfg_path "$CFG" >> "$WORK/logs/$RUN.log" 2>&1 < /dev/null &
  echo $! > "$WORK/logs/$RUN.pid"
  echo "Started PID $(cat "$WORK/logs/$RUN.pid")"
  echo "LOG: $WORK/logs/$RUN.log"
}

case ${1:-help} in
  setup)
    "$PY" smoke_test_kaggle_recipe.py
    "$PY" smoke_test_legacy_recipe.py
    "$PY" -c 'import torch,torchvision; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda)'
    "$PY" -c 'import json,torch,torchvision; from pathlib import Path; Path("../provenance/native_torch_before.json").write_text(json.dumps([torch.__version__,torchvision.__version__]))'
    logged packages uv pip install --python "$PY" -r tools/molab-requirements.txt
    logged clip uv pip install --python "$PY" --no-build-isolation --no-deps 'git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1'
    "$PY" -c 'import json,torch,torchvision; from pathlib import Path; assert json.loads(Path("../provenance/native_torch_before.json").read_text())==[torch.__version__,torchvision.__version__],"Native Torch changed"'
    "$PY" tools/molab_pipeline.py patch
    "$PY" smoke_test_legacy_recipe.py
    logged java "$PY" -u tools/molab_pipeline.py java --target "$WORK/tools/java"
    if [[ $("$PY" -c 'import torch; print(torch.version.cuda)') == 13.0 ]]; then
      logged toolchain "$PY" -u tools/molab_cuda130.py --work "$WORK"
      source "$WORK/cuda130_env.sh"
    fi
    rm -f "$WORK/provenance/cuda_ready"
    logged build_cuda "$PY" -u tools/molab_pipeline.py build-cuda --jobs 1
    logged cuda_test "$PY" smoke_test_runtime_perf.py --cuda
    touch "$WORK/provenance/cuda_ready"
    uv pip freeze --python "$PY" > "$WORK/provenance/resolved_packages.txt"
    "$PY" tools/molab_pipeline.py env > "$WORK/provenance/runtime.json"
    ;;
  prepare-smoke) test -f "$WORK/provenance/cuda_ready"; prepare smoke "$SMOKE_RUN" 2 ;;
  smoke)
    logged smoke "$PY" -u tools/molab_pipeline.py smoke --config "$WORK/prepared/$SMOKE_RUN/config.yml"
    touch "$WORK/provenance/model_smoke_ready"
    ;;
  train-smoke) logged smoke_train "$PY" -u train.py --cfg_path "$WORK/prepared/$SMOKE_RUN/config.yml" ;;
  prepare-run) test -f "$WORK/provenance/cuda_ready"; prepare "$MODE" "$RUN" 15; check_config ;;
  train) test -f "$WORK/provenance/cuda_ready"; test -f "$WORK/provenance/model_smoke_ready"; start_training no ;;
  resume) test -f "$WORK/provenance/cuda_ready"; start_training yes ;;
  restage) logged restage "$PY" -u tools/molab_pipeline.py restage --config "$CFG" --data-root "${DDVC_DATA_ROOT:-$WORK/assets}" ;;
  smoke-resume) logged resume_smoke "$PY" -u tools/molab_pipeline.py smoke --config "$CFG" --checkpoint "$WORK/runs/$RUN/model-last.pth" ;;
  logs) exec tail -n 80 -f "$WORK/logs/$RUN.log" ;;
  report) "$PY" tools/molab_pipeline.py report --config "$CFG" ;;
  status)
    if [[ -f "$WORK/logs/$RUN.pid" ]]; then ps -p "$(cat "$WORK/logs/$RUN.pid")" -o pid,etime,%cpu,%mem,args || true; fi
    tail -n 40 "$WORK/logs/$RUN.log" ;;
  *) echo 'Usage: kaggle_molab.sh COMMAND [subset|full]; default subset. Commands: setup prepare-smoke smoke train-smoke prepare-run train logs status report restage smoke-resume resume' ;;
esac
