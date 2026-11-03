# V1 + attention counter + learned duplicate selection

Branch: `exp/v1-attention-counter-duplicate` (local until published).
Base: `exp/v1-mask-ablation`, commit `6790a7a`.
This is V1 learned proposal mask, not the concept-head branch.

## Changes and limits

- Original random two-layer GPT-2 decoder, CLIP features, retrieval, learned proposal
  mask and its auxiliary loss are retained. No language checkpoint is added.
- Counter: query512 + predicted center/duration + sigmoid event confidence ->
  projection128 -> four-head self-attention -> learned attention pooling -> count
  classification. Original count BCE/Gaussian weighting remains coefficient0.5.
  New ordered CDF-distance loss is0.2, final layer only. This is NOT CORAL.
  Count supervision includes overlapping annotated GT events. Attention is applied
  to the counter branch only; queries sent to the caption head are not replaced.
- Duplicate head: symmetric query-feature difference/product plus temporal geometry
  and confidence -> MLP -> pair probability. It does not use caption string similarity
  or an additional pretrained text encoder. Pair loss coefficient0.1, final layer only.
- Many-to-one GT assignment is separate from Hungarian matching: best tIoU>=0.5,
  margin to second GT>=0.1. Same assigned GT=positive, distinct GT=negative.
  Ambiguous/unmatched pairs ignored. Positive/negative BCE averaged separately;
  loss is zero unless both classes exist in the batch. Logs show pair counts:
  `duplicate_supervised_batch_fraction` measures whether both pair classes exist.
  frequent zero positive pairs means insufficient supervision, not successful learning.
- In duplicate mode, learned selection REPLACES old text-based overlap/consecutive
  merges and temporal-only Soft-NMS. It groups only when probability>=0.9 and
  tIoU>=0.5, requiring agreement with every group member. Keep highest-score
  representative and original interval/caption; no interval union or caption rewriting.
  Remaining Soft-NMS penalty is weighted by `max(0,2*p_duplicate-1)`.
  Thus dissimilar events can overlap without a mandatory penalty. This is an
  experimental learned heuristic, not a guarantee of correct event identity.
- Raw JSON includes the probability row indexed by original query ID and selection
  flags; saved predictions remain reproducible without GT or model loading.
- Controls: `control`, `counter`, `duplicate`, `both`. Control/counter retain V1
  Soft-NMS + both old merges. Duplicate/both use learned selection instead.
- New module initialization uses a separate RNG context. No change to V1 LR,
  batch, precision, data IDs or mask coefficients. All model training starts fresh.
- Default: seed777, batch1/accumulation1, YC2 Q50/15epochs, ANET Q10/8epochs.
  Full/30% subset/1% smoke selected via final argument. Sources must match the
  chosen split mode; do not label a reused subset config as full.

## Sources of inspiration (not full reproductions)

- Set Transformer: https://proceedings.mlr.press/v97/lee19d.html
- Learned duplicate removal, Relation Networks CVPR2018:
  https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Relation_Networks_for_CVPR_2018_paper.html
- Original V1 masking inspiration, Masked Transformer CVPR2018:
  https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_End-to-End_Dense_Video_CVPR_2018_paper.html

## Fresh Molab session

Upload the delivered source ZIP to your Drive, download it into Molab using your
existing authenticated Drive connection, and extract to `/marimo/DDVC_Event_Set`.
The archive contains `source/`, with this guide inside `source/docs/`.
Google Drive remote storage is not a mounted local directory. Keep the repository,
features and active checkpoints local; sync stable outputs using DriveBackup.

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Event_Set/source/tools/event_set_molab.sh setup
```

Setup uses Molab's native Python/Torch and the existing CUDA13 compatibility path;
no root or new virtual environment. It logs package/toolchain/build checks.
Preparation auto-downloads features (YC2~872MiB; ANET~2.52GiB) and annotations,
then builds a train-only text bank. Drive download quota may require reusing your
previous feature file. Put features in the correct dataset cache:

```text
/marimo/DDVC_Event_Set/assets/yc2/clipvitl14.pth
/marimo/DDVC_Event_Set/assets/anet/clipvitl14.pth
```

Do not swap these identically named files.

### YouCook2, full, 15 epochs

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Event_Set/source/tools/event_set_molab.sh prepare yc2 both full
LC_ALL=C LANG=C bash /marimo/DDVC_Event_Set/source/tools/event_set_molab.sh smoke yc2 both full
LC_ALL=C LANG=C bash /marimo/DDVC_Event_Set/source/tools/event_set_molab.sh train yc2 both full
tail -n 80 -f /marimo/DDVC_Event_Set/logs/yc2_full_event_set_both_e15_b1.log
```

Smoke performs real-model forward/backward and caption generation, without an
optimizer update, on one training video. Synthetic CPU tests separately exercise
nonzero positive/negative duplicate-pair gradients. It does not evaluate full-val
metrics or prove the new model improves them. Ctrl+C stops tail, not training.

### ActivityNet, full, 8 epochs (second account)

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Event_Set/source/tools/event_set_molab.sh prepare anet both full
LC_ALL=C LANG=C bash /marimo/DDVC_Event_Set/source/tools/event_set_molab.sh smoke anet both full
LC_ALL=C LANG=C bash /marimo/DDVC_Event_Set/source/tools/event_set_molab.sh train anet both full
tail -n 80 -f /marimo/DDVC_Event_Set/logs/anet_full_event_set_both_e8_b1.log
```

### Reuse an existing V1 prepared config, without reusing weights

On the same Molab session, avoid installing over a running source directory.
Use the new extracted source and point preparation to a COMPLETE old V1 config:

```bash
DDVC_EVENT_SET_SOURCE_CONFIG=/marimo/DDVC_NoDedup_Molab/prepared/yc2_full_no_dedup_same_merges_e15_b1/config.yml LC_ALL=C LANG=C bash /marimo/DDVC_Event_Set/source/tools/event_set_molab.sh prepare yc2 both full
```

This path is an example of the prior run: replace it with your actual config.
Referenced local annotations/features/bank must still exist. New session backups
often lack assets, so use fresh preparation if those paths no longer exist.
No weights/optimizer are imported; source config is hashed in the manifest.

## Outputs, comparison, resume and backup

All run artifacts are under:

```text
/marimo/DDVC_Event_Set/runs/yc2_full_event_set_both_e15_b1/
/marimo/DDVC_Event_Set/runs/anet_full_event_set_both_e8_b1/
```

Includes `info.json`, best/last checkpoints, prediction JSON and `.raw.json`.
Logs live in `logs/`; exact YAML/manifest in `prepared/<run>/`; environment/build
records in `provenance/`. Model checkpoints include optimizer/scheduler/RNG state.

For ablations repeat prepare/smoke/train with `control`, `counter`, or `duplicate`
in place of `both` (distinct run names). Do not compare only a new full model
against a historical seed run. Keep epochs, data IDs and evaluator identical.
Report CIDEr, METEOR, BLEU4, SODA_c, P/R; count MAE/under/overcount by GT count;
raw-vs-selected recall and duplicate pair statistics. Also inspect nested-event videos.

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Event_Set/source/tools/event_set_molab.sh report yc2 both full
LC_ALL=C LANG=C bash /marimo/DDVC_Event_Set/source/tools/event_set_molab.sh resume yc2 both full
```

Resume needs the original local config/assets/checkpoint. After session restoration,
restage assets and rerun smoke before resume. New-head flags/coefficients are checked
against checkpoint options; old V1 checkpoints cannot resume as the new architecture.

Use one Molab Python cell after connecting your existing `fs`:

```python
import sys
from pathlib import Path
sys.path.insert(0, '/marimo/DDVC_Event_Set/source/tools')
from drive_backup import DriveBackup
event_set_backup = DriveBackup(
    fs, work=Path('/marimo/DDVC_Event_Set'),
    remote_root='runs/ddvc_event_set_yc2_both', interval=120)
event_set_backup.start()
```

Second account: change remote root to `runs/ddvc_event_set_anet_both`.
Do not reuse shared global names in multiple marimo cells. Confirm a completed
READY.json exists in remote backup before ending the session. Keep the source ZIP
and assets separately; training backups do not promise to include feature caches.

## Verification status

Local CPU: finite gradients, permutation invariance/equivariance, pair symmetry,
ambiguous GT exclusion, nested events retained, actual model hook with final+aux
count losses, unchanged legacy selection when disabled. Molab GPU forward smoke
and complete training/evaluation remain required. No metric improvement claimed.
