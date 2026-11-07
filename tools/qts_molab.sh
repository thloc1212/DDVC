#!/usr/bin/env bash
# Query temporal support; paths are discovered from this script, no shell WORK/RUN required.
set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO; inspect logs." >&2' ERR
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(dirname "$REPO")
PY=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
COMMAND=${1:-help}
DATASET=${2:-yc2}
case "$DATASET" in yc2) EPOCHS=15;; anet) EPOCHS=6;; *) echo 'Dataset must be yc2 or anet'; exit 2;; esac
VARIANT=${3:-set}
case "$VARIANT" in baseline|refine|support|set) ;; *) echo 'Variant must be baseline, refine, support or set'; exit 2;; esac
RUN=${DATASET}_qts_${VARIANT}_e${EPOCHS}_b1
export CUDA_VISIBLE_DEVICES=${DDVC_GPU:-0} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 MPLBACKEND=Agg LC_ALL=C LANG=C
export PYTHONPATH="$REPO:$REPO/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${DDVC_HF_HOME:-$WORK/hf-cache}
export PATH="$WORK/tools/java/bin:$PATH"
mkdir -p "$WORK/logs" "$WORK/prepared" "$WORK/runs" "$WORK/provenance"
cd "$REPO"
if [[ -f "$WORK/cuda130_env.sh" ]]; then source "$WORK/cuda130_env.sh"; fi
logged() { local name=$1; shift; echo "LOG: $WORK/logs/$name.log"; "$@" 2>&1 | tee -a "$WORK/logs/$name.log"; }
case "$COMMAND" in
 setup)
   "$PY" -c 'import torch,torchvision; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda)'
   "$PY" -c 'import json,torch,torchvision; from pathlib import Path; Path("../provenance/native_torch_before.json").write_text(json.dumps([torch.__version__,torchvision.__version__]))'
   logged packages uv pip install --python "$PY" -r tools/molab-requirements.txt
   logged clip uv pip install --python "$PY" --no-build-isolation --no-deps 'git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1'
   "$PY" -c 'import json,torch,torchvision; from pathlib import Path; assert json.loads(Path("../provenance/native_torch_before.json").read_text())==[torch.__version__,torchvision.__version__],"Native Torch changed"'
   "$PY" tools/molab_pipeline.py patch
   logged cpu_qts "$PY" smoke_test_temporal_support.py
   logged cpu_event_set "$PY" smoke_test_event_set.py
   logged caption_targets "$PY" smoke_test_caption_targets.py --gpt2
   logged model_qts "$PY" smoke_test_proposal_mask.py --qts --event-set
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
 prepare)
   test -f "$WORK/provenance/cuda_ready"
   ARGS=()
   if [[ -n ${DDVC_QTS_SOURCE_CONFIG:-} ]]; then ARGS+=(--source-config "$DDVC_QTS_SOURCE_CONFIG"); fi
   if [[ -n ${DDVC_FEATURES:-} ]]; then ARGS+=(--features "$DDVC_FEATURES"); fi
   logged "prepare_$RUN" "$PY" -u tools/qts_pipeline.py prepare --work "$WORK" --dataset "$DATASET" --variant "$VARIANT" "${ARGS[@]}"
   ;;
 smoke)
   test -f "$WORK/provenance/cuda_ready"
   "$PY" -u tools/qts_pipeline.py smoke --work "$WORK" --dataset "$DATASET" --variant "$VARIANT"
   ;;
 train|resume)
   test -f "$WORK/provenance/cuda_ready"
   # The foreground guard ignores stale or unrelated PIDs after a session restart.
   "$PY" tools/training_pid.py --pid-file "$WORK/logs/$RUN.pid" --config "$WORK/prepared/$RUN/config.yml"
   SUPERVISOR="$WORK/logs/$RUN.supervisor.pid"
   if [[ -f "$SUPERVISOR" ]]; then
     PID=$(cat "$SUPERVISOR")
     if ps -p "$PID" -o args= 2>/dev/null | grep -Fq 'tools/qts_pipeline.py train'; then
       echo "QTS_ALREADY_RUNNING PID=$PID"; exit 1
     fi
   fi
   ARGS=(); if [[ "$COMMAND" == resume ]]; then ARGS+=(--resume); fi
   nohup "$PY" -u tools/qts_pipeline.py train --work "$WORK" --dataset "$DATASET" --variant "$VARIANT" "${ARGS[@]}" >> "$WORK/logs/$RUN.launch.log" 2>&1 < /dev/null &
   echo $! > "$SUPERVISOR"
   echo "STARTED supervisor PID=$!"
   echo "LAUNCH/SMOKE LOG: $WORK/logs/$RUN.launch.log"
   echo "TRAIN LOG: $WORK/logs/$RUN.log"
   ;;
 report|export|restage|compare)
   ARGS=(); if [[ "$COMMAND" == restage && -n ${DDVC_FEATURES:-} ]]; then ARGS+=(--features "$DDVC_FEATURES"); fi
   "$PY" -u tools/qts_pipeline.py "$COMMAND" --work "$WORK" --dataset "$DATASET" --variant "$VARIANT" "${ARGS[@]}"
   ;;
 logs) exec tail -n 80 -F "$WORK/logs/$RUN.launch.log" "$WORK/logs/$RUN.log" ;;
 status)
   for FILE in "$WORK/logs/$RUN.supervisor.pid" "$WORK/logs/$RUN.pid"; do
     if [[ -f "$FILE" ]]; then ps -p "$(cat "$FILE")" -o pid,etime,%cpu,%mem,args || true; fi
   done
   ;;
 *) echo 'qts_molab.sh setup|prepare|smoke|train|resume|logs|status|report|restage|export|compare yc2|anet [baseline|refine|support|set]'; exit 2;;
esac
