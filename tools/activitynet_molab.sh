#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO; inspect the log above." >&2' ERR
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(dirname "$REPO")
PY=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
MODE=${2:-full}
case "$MODE" in full|subset) ;; *) echo 'Mode must be full or subset'; exit 1;; esac
RUN=anet_${MODE}_current_model_q10_e15_b1
SMOKE_RUN=anet_smoke_current_model_q10_e2_b1
CFG="$WORK/prepared/$RUN/config.yml"
export CUDA_VISIBLE_DEVICES=${DDVC_GPU:-0}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 MPLBACKEND=Agg LC_ALL=C
export PYTHONPATH="$REPO:$REPO/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${DDVC_HF_HOME:-$WORK/hf-cache}
export PATH="$WORK/tools/java/bin:$PATH"
if [[ -f "$WORK/cuda130_env.sh" ]]; then source "$WORK/cuda130_env.sh"; fi
mkdir -p "$WORK/logs" "$WORK/runs" "$WORK/prepared" "$WORK/provenance"
cd "$REPO"
logged() { local name=$1; shift; echo "LOG: $WORK/logs/$name.log"; "$@" 2>&1 | tee "$WORK/logs/$name.log"; }
prepare() {
  local mode=$1 name=$2
  local cfg="$WORK/prepared/$name/config.yml"
  if [[ -f "$cfg" ]]; then "$PY" tools/activitynet_pipeline.py check --config "$cfg"; return; fi
  local old_name=anet_${mode}_current_model_e15_b1
  if [[ "$mode" == smoke ]]; then old_name=anet_smoke_current_model_e2_b1; fi
  if [[ -f "$WORK/prepared/$old_name/config.yml" ]]; then
    logged "prepare_$mode" "$PY" -u tools/activitynet_pipeline.py configure-q10 \
      --work "$WORK" --run-name "$name" --source-config "$WORK/prepared/$old_name/config.yml"
    return
  fi
  local feature_args=()
  if [[ -n ${DDVC_ANET_FEATURES:-} ]]; then feature_args=(--features "$DDVC_ANET_FEATURES"); fi
  logged "prepare_$mode" "$PY" -u tools/activitynet_pipeline.py prepare \
    --work "$WORK" --mode "$mode" --run-name "$name" "${feature_args[@]}"
}
start_train() {
  local resume=$1
  "$PY" tools/activitynet_pipeline.py check --config "$CFG"
  "$PY" - "$WORK" "$MODE" <<'PY'
import sys
from pathlib import Path
work=Path(sys.argv[1]);old='anet_'+sys.argv[2]+'_current_model_e15_b1'
pidfile=work/'logs'/(old+'.pid')
if pidfile.is_file():
    try:
        pid=int(pidfile.read_text().strip())
        cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except (FileNotFoundError,ValueError):pass
    else:
        if 'train.py' in cmd and str(work/'prepared'/old/'config.yml') in cmd:
            raise RuntimeError(f'Old Q50 run still active at PID {pid}. Stop that training process before Q10; Ctrl+C on tail does not stop training.')
PY
  if [[ -f "$WORK/logs/$RUN.pid" ]] && kill -0 "$(cat "$WORK/logs/$RUN.pid")" 2>/dev/null; then
    echo 'Saved PID is alive; inspect status first'; exit 1
  fi
  if [[ "$resume" == yes ]]; then
    "$PY" tools/molab_pipeline.py resume-config --config "$CFG"
  else
    "$PY" - "$WORK/runs/$RUN" "$CFG" <<'PY'
import sys
from pathlib import Path
import yaml
run=Path(sys.argv[1]);cfg=yaml.safe_load(Path(sys.argv[2]).read_text())
if run.exists():raise FileExistsError('Run directory already exists. Preserve it; use resume for existing checkpoints.')
assert not cfg.get('start_from') and not cfg.get('pretrain'), 'Fresh training requested but load/resume is configured'
assert not cfg.get('only_ft_captioner') and not cfg.get('only_ft_class_head')
print('FRESH_ACTIVITYNET_CURRENT_MODEL_PASS')
PY
  fi
  nohup "$PY" -u train.py --cfg_path "$CFG" >> "$WORK/logs/$RUN.log" 2>&1 < /dev/null &
  echo $! > "$WORK/logs/$RUN.pid"
  echo "Started PID $(cat "$WORK/logs/$RUN.pid"); log: $WORK/logs/$RUN.log"
}
case ${1:-help} in
  setup)
    logged adapter_cpu "$PY" smoke_test_activitynet_adapter.py
    bash tools/dedup_molab.sh setup full no_dedup_same_merges
    ;;
  prepare-smoke) test -f "$WORK/provenance/cuda_ready"; prepare smoke "$SMOKE_RUN" ;;
  smoke)
    logged smoke "$PY" -u tools/molab_pipeline.py smoke --config "$WORK/prepared/$SMOKE_RUN/config.yml"
    touch "$WORK/provenance/activitynet_model_q10_smoke_ready"
    ;;
  prepare-run) test -f "$WORK/provenance/cuda_ready"; prepare "$MODE" "$RUN" ;;
  train)
    test -f "$WORK/provenance/cuda_ready"; test -f "$WORK/provenance/activitynet_model_q10_smoke_ready"
    start_train no
    ;;
  logs) exec tail -n 80 -f "$WORK/logs/$RUN.log" ;;
  report) "$PY" tools/molab_pipeline.py report --config "$CFG" ;;
  restage)
    feature_args=()
    if [[ -n ${DDVC_ANET_FEATURES:-} ]]; then feature_args=(--features "$DDVC_ANET_FEATURES"); fi
    logged restage "$PY" -u tools/activitynet_pipeline.py restage --work "$WORK" --config "$CFG" "${feature_args[@]}"
    ;;
  smoke-resume)
    logged resume_smoke "$PY" -u tools/molab_pipeline.py smoke --config "$CFG" --checkpoint "$WORK/runs/$RUN/model-last.pth"
    ;;
  resume) test -f "$WORK/provenance/cuda_ready"; start_train yes ;;
  *) echo 'activitynet_molab.sh setup|prepare-smoke|smoke|prepare-run|train|logs|report|restage|smoke-resume|resume [full|subset]. Current model: 10 queries, mask ON, dedup OFF, Soft-NMS/merges ON.' ;;
esac
