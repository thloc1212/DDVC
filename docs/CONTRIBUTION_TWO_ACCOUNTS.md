# Contribution ablation: V1 mask vs V2 concepts, two Molab accounts

Branch: https://github.com/thloc1212/DDVC/tree/exp/v1-v2-contribution-ablation
Base: `exp/event-concept-caption-e8` (V1 architecture plus optional concept extension).
No attention counter, learned duplicate head, H1/H2/H3 or temporal-fusion modules.
Existing V1/V2 model code is unchanged. New files control experiments only.

## Fixed protocol

| Case | Visual context | Auxiliary mask loss | Concept loss | Concept prefix |
|---|---|---:|---:|---|
| C0 | Global | 0 | OFF | OFF |
| C1 | Learned gated proposal mask | 0 | OFF | OFF |
| C2 (V1) | Learned gated proposal mask | 1.0 | OFF | OFF |
| C3 (concept aux) | Learned gated proposal mask | 1.0 | 0.1 | OFF |
| C4 (V2) | Learned gated proposal mask | 1.0 | 0.1 | ON |
| C5 | Learned gated proposal mask | 0 | 0.1 | ON |

All: Gaussian Soft-NMS sigma0.5 ONLY. Both overlap and consecutive merge are OFF.
No hard-local-attention-only run is added here. C1 is NOT old M1: old M1 was hard
local attention. Old M2 corresponds to C2, old M3 to C1 before postprocessing.

Common training: full feature-matched splits; seed777; microbatch1/accumulation1;
Adam LR5e-5, weight decay1e-4; FP32, TF32 OFF; original LR decay/contrastive schedule,
retrieval bank, caption chunking, random two-layer GPT-2 and original counter.
YouCook2: 1237train/436val, 50queries, 15epochs.
ActivityNet: 10009train/4917val, 10queries, 8epochs, both annotation references.
The concept vocabulary is fit from training captions only and shared within dataset.
Same annotation/reference/bank files and hashes across cases. Case changes do not
consume RNG during initialization of the original backbone (existing forked RNG).

Meaning of comparisons:

- C1-C0: learned mask without its auxiliary objective.
- C2-C1: auxiliary mask loss.
- C3-C2: concept supervision without caption prefix.
- C4-C3: predicted concept conditioning in the caption prefix.
- C5-C4: removing mask auxiliary loss when concepts are present.
- C4-C2: total concept extension on V1.

This is not a full factorial: it does not isolate concept effects on a global-only
model or prove all module interactions. One seed provides evidence, not significance.
Evaluate short/long events and nested/overlap subsets after collecting predictions.

## Fresh accounts: common setup on BOTH

Each command below is a terminal/bash command. Run setup/prepare to completion;
queue is backgrounded and runs one case at a time on that account's GPU.

```bash
git clone --single-branch --branch exp/v1-v2-contribution-ablation https://github.com/thloc1212/DDVC.git /marimo/DDVC_Contribution/source
LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh setup
```

Native Molab Python: `/tmp/uv-venv/bin/python` (override `DDVC_PYTHON` if needed).
Setup retains native GPU Torch and existing CUDA13 compatibility, installs packages
into that environment, builds the extension, and runs actual CUDA/CPU tests. No root
or new venv. It inherits the established NaN loss/gradient guards; no batch skipping.

If setup has already compiled CUDA but a later CPU test failed, after pulling a fix:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh verify-setup
```

### Account 1: YouCook2

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh prepare yc2
LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh queue yc2 C0 C1 C2 C3 C4 C5
tail -n 80 -f /marimo/DDVC_Contribution/logs/queue_yc2.log
```

### Account 2: ActivityNet

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh prepare anet
LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh queue anet C0 C1 C2 C3 C4 C5
tail -n 80 -f /marimo/DDVC_Contribution/logs/queue_anet.log
```

Preparation downloads features once (YC2~872MiB, ANET~2.52GiB), annotations and
encodes one train-only CLIP bank per dataset. The six cases share staged files;
they do not download six feature files or encode six banks. Feature caches:

```text
/marimo/DDVC_Contribution/assets/yc2/clipvitl14.pth
/marimo/DDVC_Contribution/assets/anet/clipvitl14.pth
```

These files differ despite the identical filename. Existing correct cached files
are reused. Official Drive quota may require restoring YOUR saved feature file
to the corresponding cache before preparation. The provided script cannot bypass
Google Drive's download quota.

Six cases mean six full training runs per account. You can choose just missing cases:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh queue yc2 C0 C1 C5
```

Queue performs a real model forward/backward and caption generation before EVERY
case, then starts fresh training. Completion skips only already-complete compatible
cases. Existing partial output requires explicit `resume`; no overwrite or automatic
weight transfer from one case to another. Queue stops on the first failed command.
Do not run another training job on the same GPU outside this queue.

## Reuse the CURRENT account runtime/assets (optional)

Keep the new source in its separate directory. If `/marimo/DDVC_Event_Set` already
has a working CUDA extension in this same session:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh reuse-runtime yc2 /marimo/DDVC_Event_Set
```

Use `anet` on account2. This copies the extension and revalidates CUDA; no weights
are copied. It does not reuse event-set counter/duplicate configs as the baseline.

If you have a COMPLETE original V1 or concept-run prepared config with all local
assets still present, reuse its data with:

```bash
DDVC_CONTRIBUTION_SOURCE_CONFIG=/marimo/DDVC_Concept_Aux/prepared/yc2_full_concept_aux_e15_b1/config.yml LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh prepare yc2
```

Second account example:

```bash
DDVC_CONTRIBUTION_SOURCE_CONFIG=/marimo/DDVC_Concept_Aux/prepared/anet_full_concept_aux_e8_b1/config.yml LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh prepare anet
```

Replace examples with the actual config if its directory differs. Source concept
flags are reset only for shared-data staging, then C0-C5 are generated explicitly.
All DDVC weights start fresh. Rejects counter/duplicate-head experiment configs.
If the old session is gone or input files are missing, use fresh preparation instead.

## Progress, results, resume

Queue logs show active case, smoke/train starts, completion/failure. Per-run training:

```bash
tail -n 80 -f /marimo/DDVC_Contribution/logs/yc2_contribution_c0_e15_b1.log
tail -n 80 -f /marimo/DDVC_Contribution/logs/anet_contribution_c0_e8_b1.log
```

Replace `c0` with `c1` ... `c5` for the active case. Ctrl+C stops tail only.
Outputs: `runs/<run>/info.json`, `model-best.pth`, `model-last.pth`,
`prediction/*.json`, `prediction/*.raw.json`. Configs/manifests: `prepared/<run>/`.
Checkpoints preserve optimizer/scheduler/RNG and original save/resume behavior.
Full checkpoints from multiple cases use disk space; no old run is deleted for you.

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh report yc2
LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh report anet
cat /marimo/DDVC_Contribution/analysis/yc2_contribution_summary.tsv
cat /marimo/DDVC_Contribution/analysis/anet_contribution_summary.tsv
```

Summary rows for unrun cases remain blank. Best checkpoint follows METEOR+SODA_c,
as in existing training, not whichever epoch has the highest individual CIDEr.

On interruption in the SAME preserved/restored work directory:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh resume yc2 C0 C1 C2 C3 C4 C5
LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh resume anet C0 C1 C2 C3 C4 C5
```

Run the corresponding dataset only on each account. Completed cases are skipped;
incomplete cases load their own last checkpoint, including a checkpoint-loading
smoke before continuing. Restoring source/config alone is not enough: annotations,
bank, features and runtime must be available. Shared assets can be restaged using:

```bash
cd /marimo/DDVC_Contribution/source
/tmp/uv-venv/bin/python tools/concept_pipeline.py restage --work /marimo/DDVC_Contribution --config /marimo/DDVC_Contribution/prepared/yc2_full_concept_control_e15_b1/config.yml
```

For ANET use `anet_full_concept_control_e8_b1`. Reuse matching source commit and
revalidate runtime after session restoration.

## Rescore existing raw predictions, without retraining

Do not automatically relabel historical runs as C0-C5: match epoch budget, data,
LR/schedule/precision, query count and the three method flags first. Old 15epoch
ANET is not a matched 8epoch ablation. If matching, rescore ALL available raw epochs
with Soft-NMS-only and select best by METEOR+SODA_c. A single best checkpoint chosen
under merges is only a fixed-checkpoint diagnostic, not a complete re-selection.

Example for an existing compatible YC2 run (replace paths if different):

```bash
cd /marimo/DDVC_Contribution/source
PYTHONPATH=/marimo/DDVC_Contribution/source:/marimo/DDVC_Contribution/source/ddvc/ops PATH=/marimo/DDVC_Contribution/tools/java/bin:$PATH /tmp/uv-venv/bin/python -u tools/contribution_ablation.py rescore --config /marimo/DDVC_NoDedup_Molab/prepared/yc2_full_no_dedup_same_merges_e15_b1/config.yml --raw-dir /marimo/DDVC_NoDedup_Molab/runs/yc2_full_no_dedup_same_merges_e15_b1/prediction --output-dir /marimo/DDVC_Contribution/analysis/old_yc2_v1_softnms_only > /marimo/DDVC_Contribution/logs/rescore_old_yc2.log 2>&1
```

No GT enters selection; GT is used by the metric evaluator afterward. This command
needs both reference JSONs/paragraph files for ANET. It validates video IDs, saves
raw/config hashes, selected JSON, metrics, summary.tsv and best_softnms_only.json.
It does not alter old predictions/checkpoints or register old rows as completed runs.

## Google Drive backup and analysis export

On each account connect Drive using Molab's generated connection. Keep this as one
Python cell, not a terminal command; set dataset `yc2` or `anet` once:

```python
import marimo as mo
from pathlib import Path
import sys
from gdrive_fsspec import GoogleDriveFileSystem
contribution_dataset = 'yc2'
contribution_fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
    auth_kwargs={'use_local_webserver': False})
mo.output.clear_console()
sys.path.insert(0, '/marimo/DDVC_Contribution/source/tools')
from drive_backup import DriveBackup
contribution_backup = DriveBackup(contribution_fs, work=Path('/marimo/DDVC_Contribution'),
    remote_root=f'runs/ddvc_contribution_{contribution_dataset}', interval=120)
contribution_backup.start()
```

Do not redefine these globals in another marimo cell. Source stays in GitHub;
backup keeps stable checkpoints/configs/logs/banks/references in two remote slots.
Features must be saved separately. First READY.json is created only after the first
last-checkpoint exists and a sync finishes. Confirm READY before ending a session.

After runs finish, export the analysis bundle (does not include .pth weights):

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh export yc2
LC_ALL=C LANG=C bash /marimo/DDVC_Contribution/source/tools/contribution_molab.sh export anet
```

Run one on its corresponding account, download the printed `CONTRIBUTION_*.zip`,
and provide both for comparison. Includes configs/manifests, predictions/metrics,
logs, TSV, environment records and tracked source code, excludes caches/bank tensors.

## Verification

CPU tests check both dataset recipes x six cases, safe-path script execution,
same backbone initialization, real vendored GPT-2 forward/backward, mask gradients
with auxiliary OFF, concept classifier gradients and enabled/disabled prefix paths.
Existing concept/checkpoint and numerical guard tests are retained. GPU forward
and complete metrics have not been run locally; launcher requires real GPU smoke.
