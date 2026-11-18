# Offline HTML: YC2 + ActivityNet on one fresh Molab account

Upload your existing private `KLTN_DRIVE_PRIVATE.env` to `/marimo`. Do not paste its contents into chat or GitHub. No GPU is needed.

Run this whole block in the Molab terminal:

```bash
bash <<'BASH'
set -euo pipefail
export LC_ALL=C LANG=C PYTHONUTF8=1
test -f /marimo/KLTN_DRIVE_PRIVATE.env
PY="$(command -v python3 || command -v python)"
if [ -x /tmp/uv-venv/bin/python ]; then PY=/tmp/uv-venv/bin/python; fi
curl -fL --retry 3 \
  https://raw.githubusercontent.com/thloc1212/DDVC/prawn-results-viewer-20261118-v1/tools/prawn_results_viewer.py \
  -o /tmp/prawn_results_viewer.py
"$PY" -u /tmp/prawn_results_viewer.py --upload 2>&1 | tee /marimo/prawn_html.log
BASH
```

Outputs:

- `/marimo/PRAWN_HTML/index.html`: standalone, open offline in Chrome/Edge.
- `/marimo/PRAWN_HTML/PRAWN_YC2_ANET_HTML.zip`: HTML, compact data, provenance, verified input text, exporter source/template. Download this ZIP from Molab's file browser.
- `/marimo/prawn_html.log`: progress and exact source run IDs.
- `kltnbackup:KLTNRUNS/analysis/html/<UTC timestamp>_PRAWN_YC2_ANET_HTML.zip`: durable copy when `--upload` is used.

The command processes **both** datasets sequentially on **one** account. It does not train or run neural inference. A missing rclone executable is installed in the export tools directory using a checksum-verified official archive; no administrator permission or training packages are required. Existing rclone is reused.

## Which results are selected

For each dataset, auto mode chooses the first available generator namespace in this order:

1. `runs/prawn_caption_match_generator` (caption matching experiment).
2. `runs/prawn_content_mask_generator`.
3. `runs/prawn_original_generator`.
4. `runs/ddvc_c1` (legacy learned mask).

Within that namespace, it chooses the newest complete, verified recovery snapshot. It uses `info.json`'s **best epoch**, not the last prediction filename. Incomplete/corrupt snapshots are logged and the next snapshot is tried. An incomplete newest experiment does not silently fall back to an older architecture.

It loads at most two additional generator profiles: the actual `runs/ddvc_baseline` backup when available, plus the closest available prior mask profile. Profiles with different GT references are excluded. Missing baseline backups are reported; the exporter does not invent DDVC results or relabel same-generator selectors as DDVC.

For each generator, quality evaluation is joined only if all four agree:

- Best-checkpoint SHA256 (read from metadata only).
- Config SHA256.
- Raw prediction SHA256.
- Best epoch.

If the corresponding completed quality evaluation is absent, generator final is still viewable with an explicit note. The HTML labels the saved selector policies: Top-N without quality, Soft-NMS without quality, and frozen quality + Soft-NMS. These three selectors can share a generator; their raw proposals are identical.

To explicitly choose a known profile, add e.g. `--primary matching`, `--primary content`, `--primary original` or `--primary c1`. To export only one dataset, use `--datasets yc2` or `--datasets anet`. Defaults process both. No beta is fitted or changed.

## Download scope and verification

Only selected best-epoch raw/final JSON, per-reference validation annotations, preparation manifests, `info.json`, quality manifests and saved evaluation JSON/metrics are downloaded. Every artifact is checked against the selected READY inventory's size and SHA256. The script does not download `.pth`, `.pt`, `.npz`, video features, caption banks, training caches, toolchains or full source archives. A default 384 MiB text-download budget and 96 MiB per-file cap catch unexpected downloads. Retries reuse locally verified files.

The private env and rclone config remain outside the exported artifacts and ZIP. The downloaded authentication helper is fixed to source commit `31f1eac41841f3a870bb56d9c74a52fc55057466` and checked against its SHA256 before import. No training helper commands are called.

## How to read the HTML

- Switch dataset, GT reference, raw/final mode, and model/selector A/B.
- Search video IDs or caption words; filter overlap, short GT events, A/B gains/losses and recoverable discarded proposals.
- Reviewer A/B splits sorted IDs into disjoint halves separately for each dataset/reference, before other filters. It is deterministic for the same reference pool.
- Timeline uses a maximum-cardinality **temporal one-to-one** diagnostic at the chosen tIoU threshold. Matching colors do not imply caption correctness. This differs from the official benchmark evaluator.
- Official validation metrics are loaded from saved evaluator outputs. They do not change when the viewer filters videos. The displayed F1 is the harmonic mean of saved aggregate P/R. No new benchmark evaluation is performed.
- ANET reference sets remain separate, including their different video counts. The viewer requires both selected models to contain the video.
- Export filtered CSV, individual case JSON and timeline SVG. Source run ID, source commit, checkpoint hash, seed, query count, epoch, beta and quality linkage are recorded.

## Validation of this helper

Offline fixture tests cover blocked heavy/unsafe paths, hash corruption, wrong-generator quality rejection, incomplete training rejection and one invocation exporting both datasets without credentials in the ZIP. Browser QA used the recorded parent experiment's real 436 YC2 / 4917 ANET videos (4885 in ANET reference 2), testing search, reviewer splits, raw/final, metric invariance, overlap, maximum-cardinality matching and downloads. These QA inputs are not presented as newly fetched caption-matching results.

Live Drive access is performed on your Molab account using your private env. It was not exercised from the development workstation.
