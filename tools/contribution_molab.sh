#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO; inspect logs." >&2' ERR
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(dirname "$REPO")
PY=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
COMMAND=${1:-help}
DATASET=${2:-yc2}
case "$DATASET" in yc2|anet) ;; *) echo 'Dataset must be yc2 or anet'; exit 2;; esac
export CUDA_VISIBLE_DEVICES=${DDVC_GPU:-0} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export LC_ALL=C LANG=C
export PYTHONPATH="$REPO:$REPO/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${DDVC_HF_HOME:-$WORK/hf-cache}
export PATH="$WORK/tools/java/bin:$PATH"
mkdir -p "$WORK/logs" "$WORK/provenance"
cd "$REPO"
if [[ -f "$WORK/cuda130_env.sh" ]]; then source "$WORK/cuda130_env.sh"; fi
case "$COMMAND" in
 setup)
   bash tools/concept_molab.sh setup "$DATASET" control
   "$PY" smoke_test_contribution_ablation.py
   ;;
 verify-setup)
   "$PY" smoke_test_runtime_perf.py --cuda
   "$PY" smoke_test_event_concepts.py
   "$PY" smoke_test_concept_gpt2.py
   "$PY" smoke_test_contribution_ablation.py
   touch "$WORK/provenance/cuda_ready"
   ;;
 reuse-runtime)
   bash tools/concept_molab.sh reuse-runtime "$DATASET" control "${3:?Existing work directory required}"
   "$PY" smoke_test_contribution_ablation.py
   ;;
 prepare)
   test -f "$WORK/provenance/cuda_ready"
   SOURCE=()
   if [[ -n ${DDVC_CONTRIBUTION_SOURCE_CONFIG:-} ]]; then SOURCE=(--source-config "$DDVC_CONTRIBUTION_SOURCE_CONFIG"); fi
   CASES=("${@:3}")
   if [[ ${#CASES[@]} -eq 0 ]]; then CASES=(C0 C1 C2 C3 C4 C5); fi
   "$PY" -u tools/contribution_ablation.py prepare --work "$WORK" --dataset "$DATASET" \
     --cases "${CASES[@]}" "${SOURCE[@]}" 2>&1 | tee -a "$WORK/logs/prepare_${DATASET}_contribution.log"
   ;;
 queue|resume)
   test -f "$WORK/provenance/cuda_ready"
   CASES=("${@:3}")
   if [[ ${#CASES[@]} -eq 0 ]]; then echo 'Specify cases explicitly: queue yc2 C0 C1 C2 C3 C4 C5'; exit 2; fi
   for CASE in "${CASES[@]}"; do
     case "$CASE" in C0|C1|C2|C3|C4|C5) ;; *) echo "Invalid case: $CASE"; exit 2;; esac
   done
   PID_FILE="$WORK/logs/queue_${DATASET}.pid"
   if [[ -f "$PID_FILE" ]]; then
     PID=$(cat "$PID_FILE")
     if ps -p "$PID" -o args= 2>/dev/null | grep -Fq 'contribution_ablation.py queue'; then
       echo "QUEUE_ALREADY_RUNNING PID=$PID"; exit 1
     fi
   fi
   EXTRA=()
   if [[ "$COMMAND" == resume ]]; then EXTRA=(--resume); fi
   nohup "$PY" -u tools/contribution_ablation.py queue --work "$WORK" --dataset "$DATASET" \
     --cases "${CASES[@]}" "${EXTRA[@]}" >> "$WORK/logs/queue_${DATASET}.log" 2>&1 < /dev/null &
   echo $! > "$PID_FILE"
   echo "QUEUE_STARTED PID=$! CASES=${CASES[*]}"
   echo "QUEUE_LOG: $WORK/logs/queue_${DATASET}.log"
   ;;
 smoke)
   test -f "$WORK/provenance/cuda_ready"
   "$PY" -u tools/contribution_ablation.py smoke --work "$WORK" --dataset "$DATASET" --case "${3:?Case required}"
   ;;
 report|export)
   "$PY" -u tools/contribution_ablation.py "$COMMAND" --work "$WORK" --dataset "$DATASET"
   ;;
 logs) exec tail -n 80 -f "$WORK/logs/queue_${DATASET}.log" ;;
 *) echo 'contribution_molab.sh setup|verify-setup|reuse-runtime|prepare|queue|resume|smoke|report|export|logs yc2|anet [C0 ... C5]'; exit 2;;
esac
