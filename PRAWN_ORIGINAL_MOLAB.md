# PRAWN original two-stage recipe

Stage 1 trains DDVC with the confidence-gated learned proposal mask for 15 epochs
on YouCook2 or 6 epochs on ActivityNet Captions. The generator batch size is one
video, gradient accumulation is one, and the seed is 777. Auxiliary mask loss has
coefficient zero. Generator checkpoint selection uses validation METEOR + SODA_c.

Stage 2 freezes the selected generator and extracts its candidate descriptors.
The quality head trains for 20 epochs with batch size eight videos, training seed
777 and head-fit/head-dev split seed 777. Final inference uses beta 1 for YouCook2
and beta 6 for ActivityNet, followed by Gaussian Soft-NMS with sigma 0.5.
The beta values come from the previous seed-777 experiments. ActivityNet beta 6
was selected through a validation sensitivity sweep.

## Account A

Select a fresh GPU session and upload the updated private environment file to
`/marimo/KLTN_DRIVE_PRIVATE.env`. Run in the terminal.

```bash
bash <<'BASH'
set -euo pipefail
export LC_ALL=C LANG=C
test -f /marimo/KLTN_DRIVE_PRIVATE.env
git clone --single-branch --branch prawn-original-b1-20261116-v2 \
  https://github.com/thloc1212/DDVC.git /marimo/PRAWN_Original/source
bash /marimo/PRAWN_Original/source/tools/prawn_original.sh start yc2
bash /marimo/PRAWN_Original/source/tools/prawn_original.sh logs yc2
BASH
```

## Account B

```bash
bash <<'BASH'
set -euo pipefail
export LC_ALL=C LANG=C
test -f /marimo/KLTN_DRIVE_PRIVATE.env
git clone --single-branch --branch prawn-original-b1-20261116-v2 \
  https://github.com/thloc1212/DDVC.git /marimo/PRAWN_Original/source
bash /marimo/PRAWN_Original/source/tools/prawn_original.sh start anet
bash /marimo/PRAWN_Original/source/tools/prawn_original.sh logs anet
BASH
```

The runner restores rclone authentication, stages verified data from the existing
Drive cache, sets up the native Molab environment, runs a real GPU model smoke
test, trains the generator, verifies its backup, then trains and evaluates the
frozen quality head. A fresh generator and optimizer are initialized.

## Results and recovery

Final summaries are written to

```text
/marimo/PRAWN_Original/frozen_quality/runs/yc2/evaluation/summary.tsv
/marimo/PRAWN_Original/frozen_quality/runs/anet/evaluation/summary.tsv
```

Use the `reranker_softnms` row. Drive backups are written to

```text
KLTNRUNS/runs/prawn_original_generator/<dataset>/<run-id>/
KLTNRUNS/runs/prawn_original_quality/<dataset>/<run-id>/
```

On the same session, `prawn_original.sh resume yc2` continues an interrupted chain.
Use `anet` for ActivityNet. The `status` command prints both stage states. `export`
creates analysis ZIPs for both stages and prints their paths.

The runner verifies checkpoint backups after each stage and monitors logs and
recovery snapshots every 120 seconds while running.
