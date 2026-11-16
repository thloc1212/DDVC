# PRAWN joint training on two Molab accounts

| Setting | YouCook2 | ActivityNet Captions |
| --- | --- | --- |
| Epochs | 30 | 15 |
| Seed | 777 | 777 |
| Micro-batch | 1 video | 1 video |
| Gradient accumulation | 8 | 8 |
| Effective batch | 8 videos | 8 videos |
| Generator initial learning rate | 5e-5 | 5e-5 |
| Quality-head initial learning rate | 1e-3 | 1e-3 |
| Quality loss coefficient | 0.1 | 0.1 |
| Auxiliary mask loss coefficient | 0 | 0 |
| Queries | 50 | 10 |

Generator, learned mask and quality head train jointly. Each accumulation window
averages its video losses before an optimizer update. The last partial window
uses its actual size. The epoch-based learning-rate schedule starts decay at
epoch 8, with an interval of 3 and a factor of 0.5.

## Fresh account A, YouCook2

Select a GPU runtime. Place `KLTN_DRIVE_PRIVATE.env` in `/marimo`.
Paste this block into the terminal.

```bash
bash <<'BASH'
set -euo pipefail
export LC_ALL=C LANG=C
test -f /marimo/KLTN_DRIVE_PRIVATE.env
git clone --single-branch --branch prawn-joint-long-a8-20261115 \
  https://github.com/thloc1212/DDVC.git /marimo/PRAWN_Joint_Long/source
bash /marimo/PRAWN_Joint_Long/source/tools/prawn_joint_long.sh start yc2
bash /marimo/PRAWN_Joint_Long/source/tools/prawn_joint_long.sh logs yc2
BASH
```

## Fresh account B, ActivityNet

```bash
bash <<'BASH'
set -euo pipefail
export LC_ALL=C LANG=C
test -f /marimo/KLTN_DRIVE_PRIVATE.env
git clone --single-branch --branch prawn-joint-long-a8-20261115 \
  https://github.com/thloc1212/DDVC.git /marimo/PRAWN_Joint_Long/source
bash /marimo/PRAWN_Joint_Long/source/tools/prawn_joint_long.sh start anet
bash /marimo/PRAWN_Joint_Long/source/tools/prawn_joint_long.sh logs anet
BASH
```

The start command restores rclone authentication, launches a detached worker,
sets up the native runtime and tokenizer, stages data from the existing verified
Drive cache, prepares the exact experiment config, and runs the real-model GPU
forward/backward smoke. It then trains, calibrates beta on TRAIN head-dev,
evaluates the selected checkpoint, and verifies its Drive backup.

The existing feature bundles are approximately 872 MiB for YouCook2 and 2.52 GiB
for ActivityNet before local preparation. They are staged into
`/marimo/PRAWN_Joint_Long`. Each account stages its selected dataset.

## Watching and checking status

```bash
LC_ALL=C LANG=C bash /marimo/PRAWN_Joint_Long/source/tools/prawn_joint_long.sh logs yc2
LC_ALL=C LANG=C bash /marimo/PRAWN_Joint_Long/source/tools/prawn_joint_long.sh status yc2
```

Use `anet` on account B. Pressing Ctrl+C in the log viewer returns to the terminal.
The training worker runs as a detached process.

The epoch header includes `micro_batch=1 accumulation=8 effective_batch=8`.

## Results

YouCook2 results are written to

```text
/marimo/PRAWN_Joint_Long/runs/yc2_online_quality_joint_e30_b1_a8/quality_evaluation/summary.tsv
```

ActivityNet results are written to

```text
/marimo/PRAWN_Joint_Long/runs/anet_online_quality_joint_e15_b1_a8/quality_evaluation/summary.tsv
```

The quality-aware Soft-NMS result is the `online_quality_softnms` row. Generator
validation is logged after each epoch. The final quality-aware evaluation follows
TRAIN head-dev beta calibration after the requested training budget finishes.

## Drive backup and resume

Each worker saves logs and verified, two-slot checkpoint snapshots under

```text
KLTNRUNS/runs/prawn_joint_long_a8/joint/yc2/<run-id>/
KLTNRUNS/runs/prawn_joint_long_a8/joint/anet/<run-id>/
```

On the same account/session, resume a stopped run with

```bash
LC_ALL=C LANG=C bash /marimo/PRAWN_Joint_Long/source/tools/prawn_joint_long.sh resume yc2
LC_ALL=C LANG=C bash /marimo/PRAWN_Joint_Long/source/tools/prawn_joint_long.sh logs yc2
```

For a setup failure before the first checkpoint, use `retry yc2`.

On a fresh session, clone the same tag at the same path, upload the private env
file, then restore the run ID printed by the original start/status command.

```bash
LC_ALL=C LANG=C bash /marimo/PRAWN_Joint_Long/source/tools/prawn_joint_long.sh restore yc2 --run-id YOUR_RUN_ID
LC_ALL=C LANG=C bash /marimo/PRAWN_Joint_Long/source/tools/prawn_joint_long.sh resume yc2
LC_ALL=C LANG=C bash /marimo/PRAWN_Joint_Long/source/tools/prawn_joint_long.sh logs yc2
```

Use `anet` for the ActivityNet account.

### Restore on a work root containing runtime provenance

The recovery update tagged `prawn-joint-long-a8-recovery-20261116` restores
snapshots created by `prawn-joint-long-a8-20261115`. Compatibility is checked by
comparing every tracked file. Only recovery transport, its tests, and this guide
may differ. Model and training code must match.

Recovery checks existing files before downloading. Identical files are reused.
Different local logs or provenance files are moved into `.c1-local-conflicts`
before the verified snapshot is installed. Different local checkpoint/config
files are reported before a download starts.

Verified download files are retained under `.c1-restore-cache` after an interrupted
attempt and are reused on retry. The older temporary-directory downloader removes
its staging files when it fails, so its failed download must be fetched again once.

```bash
git -C /marimo/PRAWN_Joint_Long/source fetch origin tag prawn-joint-long-a8-recovery-20261116
git -C /marimo/PRAWN_Joint_Long/source checkout --detach prawn-joint-long-a8-recovery-20261116
bash /marimo/PRAWN_Joint_Long/source/tools/prawn_joint_long.sh restore yc2 --run-id 20261115T112053Z_602050db7fd8
bash /marimo/PRAWN_Joint_Long/source/tools/prawn_joint_long.sh resume yc2
bash /marimo/PRAWN_Joint_Long/source/tools/prawn_joint_long.sh logs yc2
```

## Export for analysis

```bash
LC_ALL=C LANG=C bash /marimo/PRAWN_Joint_Long/source/tools/prawn_joint_long.sh export yc2
```

The command prints the analysis ZIP path. Use `export anet` on account B.
