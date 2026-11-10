#!/usr/bin/env bash
set -Eeuo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(dirname "$REPO")
PY=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
export PYTHONPATH="$REPO/tools:$REPO:$REPO/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"
export LC_ALL=C LANG=C PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export DDVC_QUALITY_MODE=${DDVC_QUALITY_MODE:-joint}
export HF_HOME=${DDVC_HF_HOME:-$WORK/hf-cache}
export PATH="$WORK/tools/java/bin:$PATH"
cd "$REPO"
if [[ -f "$WORK/cuda130_env.sh" ]]; then source "$WORK/cuda130_env.sh"; fi
if [[ ${1:-} == setup ]]; then
    env -u DDVC_RUN_VARIANT -u DDVC_QUALITY_MODE bash tools/c1_molab.sh setup
    "$PY" -u tools/test_online_quality.py
    exit 0
fi
COMMAND=${1:?command required}; DATASET=${2:?dataset required}
case "$COMMAND" in
 prepare) ARGS=(); if [[ -n ${DDVC_FEATURES:-} ]]; then ARGS+=(--features "$DDVC_FEATURES"); fi
     exec "$PY" -u tools/online_quality_pipeline.py prepare --work "$WORK" --dataset "$DATASET" "${ARGS[@]}" ;;
 smoke|report|export|restage|finalize) exec "$PY" -u tools/online_quality_pipeline.py "$COMMAND" --work "$WORK" --dataset "$DATASET" ;;
 *) echo 'Use online_quality_drive.sh start/resume for managed training and backup' >&2; exit 2 ;;
esac
