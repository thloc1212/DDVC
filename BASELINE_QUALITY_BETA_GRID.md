# ActivityNet frozen quality beta grid

Evaluate how strongly the frozen quality head contributes to selection on the completed global DDVC baseline.

The score before Gaussian Soft-NMS is

```text
score = original_joint_score + beta * predicted_quality
```

The grid uses beta values 0, 0.25, 0.5, 1 and 2. Gaussian Soft-NMS uses sigma 0.5. The runner reads the saved validation candidate cache and the trained quality head. Each row retains the counter budget and the generated timestamp and caption.

## Required session

The completed baseline experiment lives at `/marimo/DDVC_Baseline_Quality`. It contains the prepared ANET config and references, plus `frozen_quality/runs/anet/head-best.pt`, its manifest and validation feature cache.

The original training and backup revision stays checked out. Fetch the tagged runner as a single additional file. This preserves compatibility with the existing backup state.

## Download the evaluation runner

Paste into the Molab terminal.

```bash
bash <<'BASH'
set -euo pipefail
git -C /marimo/DDVC_Baseline_Quality/source fetch origin tag baseline-quality-beta-grid-20261112
git -C /marimo/DDVC_Baseline_Quality/source show baseline-quality-beta-grid-20261112:tools/frozen_quality_beta_grid.py > /marimo/DDVC_Baseline_Quality/source/tools/frozen_quality_beta_grid.py
test -f /marimo/DDVC_Baseline_Quality/frozen_quality/runs/anet/head-best.pt
test -f /marimo/DDVC_Baseline_Quality/prepared/anet_baseline_e6_b1/config.yml
mkdir -p /marimo/DDVC_Baseline_Quality/logs
mkdir -p /marimo/DDVC_Baseline_Quality/frozen_quality/runs/anet/beta_grid
BASH
```

## Run the grid

```bash
cd /marimo/DDVC_Baseline_Quality/source
PATH="/marimo/DDVC_Baseline_Quality/tools/java/bin:$PATH" LC_ALL=C LANG=C nohup /tmp/uv-venv/bin/python -u tools/frozen_quality_beta_grid.py --work /marimo/DDVC_Baseline_Quality --dataset anet --betas 0 0.25 0.5 1 2 > /marimo/DDVC_Baseline_Quality/frozen_quality/runs/anet/beta_grid/run.log 2>&1 < /dev/null &
echo $! > /marimo/DDVC_Baseline_Quality/logs/anet.baseline_beta_grid.pid
tail -n 80 -F /marimo/DDVC_Baseline_Quality/frozen_quality/runs/anet/beta_grid/run.log
```

Each completed beta prints `BETA_RESULT`. The final marker is `BETA_GRID_COMPLETE`. Official scorer detail is appended to `frozen_quality/runs/anet/beta_grid/scorer_detail.log`.

The same command can resume after an interrupted evaluation. Completed metric files are reused after checking prediction and reference hashes. A process lock prevents concurrent workers from writing the same grid directory.

## Inspect and back up

```bash
cat /marimo/DDVC_Baseline_Quality/frozen_quality/runs/anet/beta_grid/summary.tsv
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh quality-backup anet
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh quality-export anet
```

The existing quality backup includes the grid's JSON, TSV and scorer log. Export prints the ZIP path. Source provenance is recorded by the runner SHA256 in `beta_grid/manifest.json` and this release tag.

## Interpretation

Beta zero uses the baseline joint score with Soft-NMS. Increasing beta increases the influence of the predicted quality. The counter budget and candidate generation remain fixed throughout the grid.

The quality head's original beta was selected on its development split of TRAIN. This grid is a validation ablation. Its summary reports all beta settings. The original head checkpoint, beta-selection file and original evaluation directory retain their completed experiment values.

The runner can reuse original official scores when the selected query IDs, timestamps and captions match exactly and the stored prediction/reference metric fingerprint is valid. Additional beta settings receive the full DDVC 2018 language and localization evaluation.
