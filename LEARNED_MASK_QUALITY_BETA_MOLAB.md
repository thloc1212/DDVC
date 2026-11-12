# Learned mask + frozen quality beta ablation

The experiment evaluates beta values `0, 0.25, 0.5, 1, 2, 4, 6, 8` on saved validation candidates from the learned-mask generator and its trained quality head.

The ranking score is `DDVC joint score + beta * quality`, followed by Gaussian Soft-NMS with sigma 0.5. Each beta uses the same counter budget, candidate captions and timestamps. Beta zero evaluates learned mask + Soft-NMS. The original quality weight was 1 for YouCook2 and 2 for ActivityNet.

## Pinned input runs

| Dataset | Original generator run | Original quality run | Generator epochs | Queries |
|---|---|---|---|---|
| YouCook2 | `20261109T053520Z_a2e591384d73` | `20261109T092734Z_05a32edc0bd5` | 15 | 50 |
| ActivityNet | `20261110T034832Z_b1f9f74b1cbd` | `20261110T065806Z_00646d58d36e` | 6 | 10 |

The original learned-mask + quality results are 39.5025 CIDEr on YouCook2 and 36.7905 on ActivityNet. The runner verifies stored metric fingerprints and reuses scores when the selected event identities, timestamps and captions match exactly.

## Prepare both accounts

Use a fresh Molab session on each account and upload `KLTN_DRIVE_PRIVATE.env` to `/marimo`. Paste the following into each terminal.

```bash
bash <<'BASH'
set -euo pipefail
test -f /marimo/KLTN_DRIVE_PRIVATE.env
git clone --single-branch --branch exp/learned-mask-quality-beta-grid \
  https://github.com/thloc1212/DDVC.git \
  /marimo/DDVC_Mask_Quality_Beta/source
git -C /marimo/DDVC_Mask_Quality_Beta/source checkout --detach \
  learned-mask-quality-beta-grid-20261112
uv pip install --python /tmp/uv-venv/bin/python \
  -r /marimo/DDVC_Mask_Quality_Beta/source/tools/molab-requirements.txt
LC_ALL=C LANG=C bash /marimo/DDVC_Mask_Quality_Beta/source/tools/mask_quality_beta.sh auth
BASH
```

Authentication uses the same Drive credentials on both accounts. Output paths on Drive include the dataset and original quality-run ID.

## Account 1, YouCook2

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Mask_Quality_Beta/source/tools/mask_quality_beta.sh start yc2
LC_ALL=C LANG=C bash /marimo/DDVC_Mask_Quality_Beta/source/tools/mask_quality_beta.sh logs yc2
```

## Account 2, ActivityNet Captions

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Mask_Quality_Beta/source/tools/mask_quality_beta.sh start anet
LC_ALL=C LANG=C bash /marimo/DDVC_Mask_Quality_Beta/source/tools/mask_quality_beta.sh logs anet
```

## Execution and recovery

The background worker restores the quality head, validation cache, original config and references from the existing two-slot Drive snapshots. It verifies file hashes, generator identity and the correspondence between the head and generator checkpoint fingerprint. Downloaded inputs stay under `inputs/` in the new work folder. Original config bytes retain their recorded hash, and the evaluator resolves reference paths to their restored local copies.

The validation feature cache contains approximately 82 MiB for YouCook2 and over 200 MiB for ActivityNet. Including JSON metadata and references increases the transfer size. Scoring uses this cache and the small quality checkpoint. The worker prepares Java for the official DDVC evaluator, evaluates each beta and backs up the completed results after each scored setting.

Useful log markers include `LEARNED_MASK_QUALITY_INPUTS_VERIFIED`, `CACHE_PROGRESS`, `BETA_RESULT`, `BETA_BACKUP_VERIFIED`, `BETA_GRID_COMPLETE`, `ZIP_READY` and `LEARNED_MASK_BETA_COMPLETE`. `scorer_detail.log` contains the official evaluator output.

After an interrupted worker, repeat `start DATASET` in the same session. Completed metric files are reused when their prediction/reference fingerprints match. On a new session, repeat preparation and `start DATASET`. The worker restores original evaluation inputs and the most recent completed beta backup. Source generator and quality checkpoints retain their trained values throughout the ablation.

## Inspect results

YouCook2

```bash
cat /marimo/DDVC_Mask_Quality_Beta/runs/yc2/beta_grid/summary.tsv
```

ActivityNet

```bash
cat /marimo/DDVC_Mask_Quality_Beta/runs/anet/beta_grid/summary.tsv
```

## Export and save

The worker creates an analysis ZIP automatically after evaluation. To export again or request another backup, use the corresponding dataset.

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Mask_Quality_Beta/source/tools/mask_quality_beta.sh backup yc2
LC_ALL=C LANG=C bash /marimo/DDVC_Mask_Quality_Beta/source/tools/mask_quality_beta.sh export yc2
```

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Mask_Quality_Beta/source/tools/mask_quality_beta.sh backup anet
LC_ALL=C LANG=C bash /marimo/DDVC_Mask_Quality_Beta/source/tools/mask_quality_beta.sh export anet
```

ZIP files are written under `/marimo` and include predictions, metrics, scorer logs, config, references, the quality head and validation cache. Verified Drive result snapshots are stored under

```text
kltnbackup:KLTNRUNS/analysis/learned_mask_quality_beta/yc2/20261109T092734Z_05a32edc0bd5/recovery/
kltnbackup:KLTNRUNS/analysis/learned_mask_quality_beta/anet/20261110T065806Z_00646d58d36e/recovery/
```

The backup uses two slots, writes each READY marker after upload verification, and updates LATEST last.

## Reporting

This is a validation hyperparameter-sensitivity experiment on fixed learned-mask generators and quality heads. Report the full beta curve. The original beta settings were selected using head-development data from TRAIN. Selecting a new deployment beta can use an expanded TRAIN head-dev search with a predefined objective.
