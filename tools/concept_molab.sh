#!/usr/bin/env bash
# Same concept module: ActivityNet defaults to 8 epochs, YouCook2 to 15.
set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO; inspect the command/log above." >&2' ERR
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(dirname "$REPO")
PY=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
DATASET=${2:-anet}
VARIANT=${3:-full}
case "$DATASET" in anet|yc2) ;; *) echo 'Dataset must be anet or yc2'; exit 1;; esac
EPOCHS=${DDVC_CONCEPT_EPOCHS:-}
if [[ -z "$EPOCHS" ]]; then
  if [[ "$DATASET" == anet ]]; then EPOCHS=8; else EPOCHS=15; fi
fi
case "$EPOCHS" in 8|15) ;; *) echo 'DDVC_CONCEPT_EPOCHS must be 8 or 15'; exit 1;; esac
case "$VARIANT" in full) SUFFIX=event_concept;; aux) SUFFIX=concept_aux;; control) SUFFIX=concept_control;; *) echo 'Variant must be full, aux or control'; exit 1;; esac
RUN=${DATASET}_full_${SUFFIX}_e${EPOCHS}_b1
CFG="$WORK/prepared/$RUN/config.yml"
SMOKE_CFG="$WORK/prepared/$RUN/smoke/config.yml"
MARKER="$WORK/provenance/${RUN}_smoke_ready.json"
export CUDA_VISIBLE_DEVICES=${DDVC_GPU:-0}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 MPLBACKEND=Agg LC_ALL=C
export PYTHONPATH="$REPO:$REPO/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${DDVC_HF_HOME:-$WORK/hf-cache}
export PATH="$WORK/tools/java/bin:$PATH"
mkdir -p "$WORK/logs" "$WORK/prepared" "$WORK/runs" "$WORK/provenance"
cd "$REPO"
if [[ -f "$WORK/cuda130_env.sh" ]]; then source "$WORK/cuda130_env.sh"; fi
logged() { local name=$1; shift; echo "LOG: $WORK/logs/$name.log"; "$@" 2>&1 | tee "$WORK/logs/$name.log"; }
check() { "$PY" tools/concept_pipeline.py check --config "$CFG"; }
smoke_marker() {
  "$PY" - "$CFG" "$MARKER" "$1" <<'PY'
import hashlib,json,sys
from pathlib import Path
cfg,marker,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
files=[cfg,Path('ddvc/CaptioningHead/event_concepts.py'),Path('ddvc/CaptioningHead/gpt2.py'),Path('ddvc/ddvc_ret_encdec_clip_capQ.py')]
signature={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
if mode=='save':marker.write_text(json.dumps(signature))
elif not marker.is_file() or json.loads(marker.read_text())!=signature:
    raise RuntimeError('Run the concept smoke test for this exact config/source before training')
PY
}
start() {
  local resume=$1
  test -f "$WORK/provenance/cuda_ready"
  check
  "$PY" tools/training_pid.py --pid-file "$WORK/logs/$RUN.pid" --config "$CFG"
  if [[ "$resume" == yes ]]; then
    "$PY" tools/molab_pipeline.py resume-config --config "$CFG"
  else
    smoke_marker check
    "$PY" - "$WORK/runs/$RUN" "$CFG" <<'PY'
import sys,yaml
from pathlib import Path
from datetime import datetime
run,cfg=Path(sys.argv[1]),yaml.safe_load(Path(sys.argv[2]).read_text())
assert cfg['epoch'] in (8,15) and not cfg.get('start_from') and not cfg.get('pretrain')
assert not cfg.get('only_ft_captioner') and not cfg.get('only_ft_class_head')
if run.exists():
    if list(run.glob('*.pth')):raise FileExistsError('Run has weights. Use resume; do not overwrite it.')
    preserved=run.with_name(run.name+'_failed_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.rename(preserved);print('PRESERVED_FAILED_START',preserved)
print('FRESH_CONCEPT_TRAINING epochs='+str(cfg['epoch'])+'; all DDVC parameters train; no old checkpoint loaded')
PY
  fi
  nohup "$PY" -u train.py --cfg_path "$CFG" >> "$WORK/logs/$RUN.log" 2>&1 < /dev/null &
  echo $! > "$WORK/logs/$RUN.pid"
  echo "STARTED PID $(cat "$WORK/logs/$RUN.pid")"
  echo "LOG: $WORK/logs/$RUN.log"
}
case ${1:-help} in
  setup)
    # Reuse the established native-Torch/private-CUDA setup, without changing its recipe.
    bash tools/activitynet_molab.sh setup full
    logged concept_cpu "$PY" smoke_test_event_concepts.py
    logged concept_gpt2_cpu "$PY" smoke_test_concept_gpt2.py
    ;;
  reuse-runtime)
    test -n "${4:-}" || { echo 'Supply the existing Molab work directory as argument 4'; exit 1; }
    logged reuse_runtime "$PY" tools/concept_pipeline.py reuse-runtime --work "$WORK" --source-work "$4"
    if [[ -f "$WORK/cuda130_env.sh" ]]; then source "$WORK/cuda130_env.sh"; fi
    rm -f "$WORK/provenance/cuda_ready"
    logged concept_cpu "$PY" smoke_test_event_concepts.py
    logged concept_gpt2_cpu "$PY" smoke_test_concept_gpt2.py
    logged cuda_test "$PY" smoke_test_runtime_perf.py --cuda
    touch "$WORK/provenance/cuda_ready"
    ;;
  prepare)
    test -f "$WORK/provenance/cuda_ready"
    SOURCE_ARGS=()
    if [[ -n ${4:-} ]]; then SOURCE_ARGS=(--source-config "$4"); fi
    logged "prepare_${RUN}" "$PY" -u tools/concept_pipeline.py prepare \
      --work "$WORK" --dataset "$DATASET" --variant "$VARIANT" --epochs "$EPOCHS" "${SOURCE_ARGS[@]}"
    ;;
  smoke)
    test -f "$WORK/provenance/cuda_ready"; check
    logged "smoke_${RUN}" "$PY" -u tools/molab_pipeline.py smoke --config "$SMOKE_CFG"
    smoke_marker save
    ;;
  train) start no ;;
  resume) start yes ;;
  smoke-resume)
    check
    logged "resume_smoke_${RUN}" "$PY" -u tools/molab_pipeline.py smoke --config "$CFG" --checkpoint "$WORK/runs/$RUN/model-last.pth"
    ;;
  logs) exec tail -n 80 -f "$WORK/logs/$RUN.log" ;;
  report) "$PY" tools/molab_pipeline.py report --config "$CFG" ;;
  restage) logged "restage_${RUN}" "$PY" -u tools/concept_pipeline.py restage --work "$WORK" --config "$CFG" ;;
  status)
    if [[ -f "$WORK/logs/$RUN.pid" ]]; then ps -p "$(cat "$WORK/logs/$RUN.pid")" -o pid,etime,%cpu,%mem,args || true; fi
    tail -n 30 "$WORK/logs/$RUN.log"
    ;;
  *) echo 'concept_molab.sh setup|reuse-runtime|prepare|smoke|train|resume|smoke-resume|restage|logs|report|status [anet|yc2] [full|aux|control] [source-work for reuse-runtime OR source-config for prepare]' ;;
esac
