#!/usr/bin/env bash
set -Eeuo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$REPO/tools/prawn_content_mask.sh" "$@"
