#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO; inspect logs." >&2' ERR
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(dirname "$REPO")
PY=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
COMMAND=${1:-help}
DATASET=${2:-yc2}
VARIANT=${3:-both}
MODE=${4:-full}
case "$DATASET" in yc2|anet) ;; *) exit 2;; esac
case "$VARIANT" in control|counter|duplicate|both) ;; *) exit 2;; esac
case "$MODE" in smoke|subset|full) ;; *) exit 2;; esac
EPOCHS=15
if [[ "$DATASET" == anet ]]; then EPOCHS=8; fi
if [[ "$MODE" == smoke ]]; then EPOCHS=2; fi
EPOCHS=${DDVC_EVENT_SET_EPOCHS:-$EPOCHS}
RUN=${DATASET}_${MODE}_event_set_${VARIANT}_e${EPOCHS}_b1
CFG="$WORK/prepared/$RUN/config.yml"
export CUDA_VISIBLE_DEVICES=${DDVC_GPU:-0} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export LC_ALL=C LANG=C
export PYTHONPATH="$REPO:$REPO/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${DDVC_HF_HOME:-$WORK/hf-cache}
export PATH="$WORK/tools/java/bin:$PATH"
if [[ -f "$WORK/cuda130_env.sh" ]]; then source "$WORK/cuda130_env.sh"; fi
mkdir -p "$WORK/logs" "$WORK/provenance"
cd "$REPO"
verify_setup() {
   "$PY" smoke_test_runtime_perf.py --cuda
   "$PY" smoke_test_event_set_heads.py
   touch "$WORK/provenance/cuda_ready"
   uv pip freeze --python "$PY" > "$WORK/provenance/resolved_packages.txt"
   "$PY" tools/molab_pipeline.py env > "$WORK/provenance/runtime.json"
}
case "$COMMAND" in
 setup)
   "$PY" -c 'import json,torch,torchvision; from pathlib import Path; assert torch.cuda.is_available(); Path("../provenance/native_torch_before.json").write_text(json.dumps([torch.__version__,torchvision.__version__]))'
   uv pip install --python "$PY" -r tools/molab-requirements.txt 2>&1 | tee "$WORK/logs/packages.log"
   uv pip install --python "$PY" --no-build-isolation --no-deps 'git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1' 2>&1 | tee "$WORK/logs/clip_install.log"
   "$PY" -c 'import json,torch,torchvision; from pathlib import Path; assert json.loads(Path("../provenance/native_torch_before.json").read_text())==[torch.__version__,torchvision.__version__],"Native Torch changed"'
   "$PY" tools/molab_pipeline.py patch
   "$PY" -u tools/molab_pipeline.py java --target "$WORK/tools/java" 2>&1 | tee "$WORK/logs/java.log"
   if [[ $("$PY" -c 'import torch; print(torch.version.cuda)') == 13.0 ]]; then
     "$PY" -u tools/molab_cuda130.py --work "$WORK" 2>&1 | tee "$WORK/logs/toolchain.log"
     source "$WORK/cuda130_env.sh"
   fi
   rm -f "$WORK/provenance/cuda_ready"
   "$PY" -u tools/molab_pipeline.py build-cuda --jobs 1 2>&1 | tee "$WORK/logs/build_cuda.log"
   verify_setup
   ;;
 verify-setup)
   verify_setup
   ;;
 prepare)
   if [[ -f "$CFG" ]]; then echo "Existing config retained: $CFG"; exit 0; fi
   SOURCE_ARGS=()
   if [[ -n ${DDVC_EVENT_SET_SOURCE_CONFIG:-} ]]; then SOURCE_ARGS=(--source-config "$DDVC_EVENT_SET_SOURCE_CONFIG"); fi
   "$PY" -u tools/event_set_pipeline.py --work "$WORK" --dataset "$DATASET" --variant "$VARIANT" \
      --mode "$MODE" --epochs "$EPOCHS" "${SOURCE_ARGS[@]}" 2>&1 | tee "$WORK/logs/prepare_$RUN.log"
   ;;
 smoke|train|resume|report)
   test -f "$CFG"
   SIGNATURE=$(sha256sum "$CFG" train.py ddvc/event_set_heads.py ddvc/ddvc_ret_encdec_clip_capQ.py \
     misc/proposal_selection.py eval_utils_clip.py misc/gradient_guard.py transformers/caption_targets.py \
     ddvc/temporal_pooling.py | sha256sum | cut -d' ' -f1)
   MARKER="$WORK/provenance/${RUN}_${SIGNATURE}_event_set_smoke_ready"
   if [[ "$COMMAND" == smoke ]]; then
     test -f "$WORK/provenance/cuda_ready"
     "$PY" smoke_test_event_set_heads.py
     "$PY" -u tools/molab_pipeline.py smoke --config "$CFG" 2>&1 | tee "$WORK/logs/$RUN.smoke.log"
     touch "$MARKER"
   elif [[ "$COMMAND" == report ]]; then
     "$PY" -u tools/molab_pipeline.py report --config "$CFG"
   else
     test -f "$WORK/provenance/cuda_ready"
     test -f "$MARKER" || { echo 'Run smoke for this exact config/source first'; exit 1; }
     "$PY" tools/training_pid.py --pid-file "$WORK/logs/$RUN.pid" --config "$CFG"
     if [[ "$COMMAND" == resume ]]; then
       "$PY" tools/molab_pipeline.py resume-config --config "$CFG"
     else
       "$PY" - "$CFG" <<'PY'
import sys,yaml
from pathlib import Path
c=yaml.safe_load(Path(sys.argv[1]).read_text())
assert not c.get('start_from') and not c.get('pretrain'), 'Fresh training required'
p=Path(c['save_dir'])/c['id']
if p.exists():raise FileExistsError(f'Preserve existing run: {p}')
PY
     fi
     nohup "$PY" -u train.py --cfg_path "$CFG" >> "$WORK/logs/$RUN.log" 2>&1 < /dev/null &
     echo $! > "$WORK/logs/$RUN.pid"
     echo "STARTED PID=$! LOG=$WORK/logs/$RUN.log"
   fi
   ;;
 *) echo 'Usage: event_set_molab.sh setup|verify-setup|prepare|smoke|train|resume|report yc2|anet control|counter|duplicate|both smoke|subset|full'; exit 2;;
esac
