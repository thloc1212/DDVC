# PRAWN: joint time–caption Hungarian cost experiment

Release: `prawn-caption-matching-20261117-v1`.
Parent: `91fac5b96d6311f162e36cb228df991377b2f53a`
(`prawn-content-mask-20261117-v2`).

This is a fresh two-stage experiment on the **content-conditioned mask run**
used in the recent selector survey. It is not a continuation of the original
PRAWN checkpoint in the paper tables. Generator, decoder, mask, query counts,
GT sampling, localization/class/caption/counter losses and final Gaussian
Soft-NMS remain the same. Only the final decoder layer's TRAIN matching cost
receives an additional detached reward.

## Method

For each prediction i and sampled TRAIN GT j:

```text
r_ij = temporal_IoU(predicted_interval_i, GT_interval_j)
       * IDF_weighted_caption_unigram_F1(free_caption_i, GT_caption_j)

C_new_ij = C_existing_ij - lambda(epoch) * r_ij

Hungarian(C_new) -> one-to-one matched query/GT pairs -> existing losses
```

The existing cost already contains its enabled classification, box/GIoU and
teacher-forced caption terms; these are retained. The new F1 term uses captions
generated without GT tokens, independently of teacher forcing. Time and caption
must agree with the SAME GT, not independent maxima across different GT events.

- Maximum lambda: **0.5** (not beta).
- Epochs 1–2: lambda=0, exactly the existing matching path.
- Epoch 3: lambda=0.25.
- Epoch 4 onward: lambda=0.5.
- Final decoder layer only; intermediate auxiliary matchers retain their costs.
- Batch 1; decode all Q queries in chunks of **4**, with the same beam count and
  maximum caption length as inference. No per-token vocabulary scores are kept.
- Extra decoding uses no_grad/eval, restores every submodule's training flag and
  the Torch RNG, and never changes model weights.
- TRAIN-only IDF is frozen in prepared/<run>/matching_train_idf.json and hashed.
  This fit uses all generator TRAIN videos. The quality head retains its own
  head-fit-only IDF, as before; no validation reference enters either fit.
- F1 and Hungarian are discrete. Gradients still come from the unchanged
  supervised losses of the selected pairs; F1 itself is not differentiated.
- No training matcher/F1/GT is used at inference.

This may increase training time substantially because it decodes all queries
during active matching. There is no claim of improved localization until the
new raw/final predictions have been evaluated.

## Fixed recipe

| Setting | YouCook2 | ActivityNet Captions |
|---|---:|---:|
| Generator epochs | 15 | 6 |
| Video batch / accumulation | 1 / 1 | 1 / 1 |
| Queries | 50 | 10 |
| Generator seed | 777 | 777 |
| LR / weight decay | 5e-5 / 1e-4 | 5e-5 / 1e-4 |
| Precision | FP32, TF32 off | FP32, TF32 off |
| Auxiliary mask loss | 0 | 0 |
| Fresh frozen-quality head epochs | 20 | 20 |
| Quality batch / seed / fold seed | 8 / 777 / 777 | 8 / 777 / 777 |
| Final beta / Gaussian sigma | 1 / 0.5 | 6 / 0.5 |

Beta retains the earlier PRAWN settings, not a new search on validation. ANET
beta=6 was chosen in earlier validation experiments; this remains exploratory
validation, not an unseen test-set result.

## Start on two NEW Molab sessions

Select a GPU runtime. Upload only `/marimo/KLTN_DRIVE_PRIVATE.env` to each account.
Use the same Drive remote `kltnbackup`, with rclone already available in PATH.
Do not source the private env; the existing runner parses its base64 assignment
without executing shell code or printing tokens.

Account A / YouCook2:

```bash
bash <<'BASH'
set -euo pipefail
export LC_ALL=C LANG=C
test -f /marimo/KLTN_DRIVE_PRIVATE.env
command -v rclone >/dev/null
test ! -e /marimo/PRAWN_Caption_Match/source
git clone --single-branch --branch prawn-caption-matching-20261117-v1 \
  https://github.com/thloc1212/DDVC.git /marimo/PRAWN_Caption_Match/source
bash /marimo/PRAWN_Caption_Match/source/tools/prawn_caption_match.sh start yc2
bash /marimo/PRAWN_Caption_Match/source/tools/prawn_caption_match.sh logs yc2
BASH
```

Account B / ActivityNet:

```bash
bash <<'BASH'
set -euo pipefail
export LC_ALL=C LANG=C
test -f /marimo/KLTN_DRIVE_PRIVATE.env
command -v rclone >/dev/null
test ! -e /marimo/PRAWN_Caption_Match/source
git clone --single-branch --branch prawn-caption-matching-20261117-v1 \
  https://github.com/thloc1212/DDVC.git /marimo/PRAWN_Caption_Match/source
bash /marimo/PRAWN_Caption_Match/source/tools/prawn_caption_match.sh start anet
bash /marimo/PRAWN_Caption_Match/source/tools/prawn_caption_match.sh logs anet
BASH
```

The chain restores Drive auth, reuses the existing verified shared dataset
cache, sets up the native GPU environment, runs the actual full-model GPU
forward/backward + generation smoke **with the new cost active**, then starts a
fresh generator. It backs up the completed generator before extracting fresh
candidates and training a fresh quality head. It does not train another DDVC
baseline first. Summary rows named baseline_topk/baseline_softnms are selectors
on this same new generator, not extra generator training runs.

No existing checkpoint or quality cache is loaded as initialization. Shared
data preparation files are byte-identical to the parent release, so the Drive
data cache identity is unchanged. If that cache is absent, the inherited
preparation may download features (~872 MiB YC2 / 2.52 GiB ANET).

Two accounts are independent: Drive paths separate dataset and unique run ID.
Never use the same local work root for two concurrent datasets in one session.

## Logs and results

Local generator:

```text
/marimo/PRAWN_Caption_Match/runs/yc2_caption_match_e15_b1/
/marimo/PRAWN_Caption_Match/runs/anet_caption_match_e6_b1/
```

Final full-metric quality results:

```bash
cat /marimo/PRAWN_Caption_Match/frozen_quality/runs/yc2/evaluation/summary.tsv
# On the ANET account:
cat /marimo/PRAWN_Caption_Match/frozen_quality/runs/anet/evaluation/summary.tsv
```

Observe these TRAIN log entries:

```text
MATCHING_SCHEDULE epoch=2 weight=0.250000
CAPTION_MATCH ... weight=0.2500 reward_mean=... reward_max=... changed_pairs=k/M decode_s=...
```

Epoch in these entries is 0-based. If changed_pairs remains zero, the tested
coefficient has not changed the assignments; do not claim a matching benefit.
reward_mean/max near zero can indicate still-poor free captions or temporal
alignment. Decode time is reported separately so overhead is visible.

GPU smoke prints `SMOKE_CAPTION_MATCH_STATS`; setup must pass before training.
Local CPU tests do not replace this actual native CUDA smoke.

Drive storage (generator and quality use alternating verified recovery slots):

```text
KLTNRUNS/data/<dataset>/c1-base-assets-v1/                 shared data, unchanged
KLTNRUNS/runs/prawn_caption_match_generator/<dataset>/<run_id>/
KLTNRUNS/runs/prawn_caption_match_quality/<dataset>/<run_id>/
```

## Same-session resume

Use `resume`, never a fresh `start`, for an existing experiment:

```bash
bash /marimo/PRAWN_Caption_Match/source/tools/prawn_caption_match.sh resume yc2
bash /marimo/PRAWN_Caption_Match/source/tools/prawn_caption_match.sh logs yc2
```

Use `anet` on the other account. Resume restores model/optimizer/scheduler/RNG;
matching weight derives from the resumed epoch, not wall time or a new counter.

## Restore after losing a session

Clone the SAME release to the SAME `/marimo/PRAWN_Caption_Match/source` path,
upload the private env and authenticate:

```bash
bash /marimo/PRAWN_Caption_Match/source/tools/prawn_caption_match.sh auth
rclone --config /marimo/secrets/rclone.conf lsf \
  kltnbackup:KLTNRUNS/runs/prawn_caption_match_generator/yc2 --dirs-only
```

Restore the desired generator run ID into that fresh work root:

```bash
bash /marimo/PRAWN_Caption_Match/source/tools/c1_drive.sh restore yc2 --run-id GENERATOR_RUN_ID
```

If a quality run already existed, list its IDs and restore it too **before**
resuming the chain:

```bash
rclone --config /marimo/secrets/rclone.conf lsf \
  kltnbackup:KLTNRUNS/runs/prawn_caption_match_quality/yc2 --dirs-only
bash /marimo/PRAWN_Caption_Match/source/tools/c1_rerank.sh restore yc2 \
  --work /marimo/PRAWN_Caption_Match/frozen_quality \
  --run-id QUALITY_RUN_ID
bash /marimo/PRAWN_Caption_Match/source/tools/prawn_caption_match.sh resume yc2
bash /marimo/PRAWN_Caption_Match/source/tools/prawn_caption_match.sh logs yc2
```

The rerank shell discovers its generator base from the parent of the source
checkout. On ANET substitute `anet` throughout.
Never restore a different matching coefficient into an existing work root.

## Export for analysis

After the chain reports `PRAWN_TWO_STAGE_COMPLETE_BACKED_UP`:

```bash
bash /marimo/PRAWN_Caption_Match/source/tools/prawn_caption_match.sh export yc2
# Other account:
bash /marimo/PRAWN_Caption_Match/source/tools/prawn_caption_match.sh export anet
```

This produces a generator-analysis ZIP under the main work directory and a
quality-analysis ZIP under frozen_quality, printing both ZIP_READY paths.
Generator weights are kept in Drive recovery, not duplicated in analysis ZIPs.
Return both exports from each account for raw/final and end-to-end metric checks.

## Controlled ablation

The direct reference is the parent content-mask run with the same seed/data/
budget, not the paper-table PRAWN checkpoint. A stronger code-matched control is
this release with `DDVC_MATCHING_WEIGHT=0`, started in a separate fresh work root
(clone under `/marimo/PRAWN_Caption_Match_Control/source`). Set that environment
variable **before** start; the saved config freezes the coefficient. Do not edit
the config or coefficient after training has begun.

Compare raw one-to-one recall at tIoU .3/.5/.7/.9, short/nested event coverage,
counter MAE, final P/R and CIDEr/METEOR/BLEU-4/SODA-c. Keep final beta/sigma fixed
for the first comparison. Match-count changes alone are not a localization gain.
