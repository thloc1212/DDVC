#!/usr/bin/env bash
set -Eeuo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(dirname "$REPO")
PY=${DDVC_PYTHON:-/tmp/uv-venv/bin/python}
export LC_ALL=C LANG=C PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO:$REPO/tools:$REPO/densevid_eval3:$REPO/densevid_eval3/SODA${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$WORK/tools/java/bin:$PATH"
cd "$REPO"
if [[ ${1:-} == setup ]]; then
  mkdir -p "$WORK/provenance" "$WORK/logs"
  exec > >(tee -a "$WORK/logs/setup.log") 2>&1
  test -x "$PY"
  command -v rclone >/dev/null || { echo 'rclone missing from this session. Install rclone before setup.' >&2; exit 1; }
  "$PY" -c 'import torch,json; from pathlib import Path; assert torch.cuda.is_available(), "Select a Molab GPU"; Path("../provenance/native_torch_before.json").write_text(json.dumps([torch.__version__,torch.version.cuda])); print("NATIVE_GPU",torch.cuda.get_device_name(0),torch.__version__)'
  uv pip install --python "$PY" numpy scipy tqdm PyYAML
  "$PY" -c 'import torch,json; from pathlib import Path; assert json.loads(Path("../provenance/native_torch_before.json").read_text())==[torch.__version__,torch.version.cuda], "GPU Torch was replaced"'
  "$PY" -u tools/molab_pipeline.py java --target "$WORK/tools/java"
  "$PY" -c 'import sys,torch,numpy,scipy,json,platform; from pathlib import Path; Path("../provenance/native_runtime.json").write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),numpy=numpy.__version__,scipy=scipy.__version__,mode="frozen cached features; no DDVC CUDA extension"),indent=2))'
  echo 'RELATION_COUNT_SETUP_READY'
  exit 0
fi
if [[ ${1:-} == prepare ]]; then
  mkdir -p "$WORK/logs"
  "$PY" -u "$REPO/tools/relation_count.py" "$@" 2>&1 | tee -a "$WORK/logs/prepare_${2:-yc2}.log"
  exit 0
fi
exec "$PY" -u "$REPO/tools/relation_count.py" "$@"
