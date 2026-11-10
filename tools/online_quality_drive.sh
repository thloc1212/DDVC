#!/usr/bin/env bash
set -Eeuo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DDVC_RUN_VARIANT=online_quality
export DDVC_QUALITY_MODE=${DDVC_QUALITY_MODE:-joint}
export PYTHONPATH="$REPO/tools:$REPO:$REPO/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"
exec bash "$REPO/tools/c1_drive.sh" "$@"
