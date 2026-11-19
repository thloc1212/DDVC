# PRAWN: leave-one-out ablations, seed 777

Release: `prawn-loo-20261119-v1`. Existing training releases and checkpoints are preserved.

All experiments retain **caption-aware Hungarian matching** (tIoU × TRAIN-IDF weighted caption unigram F1). Schedule remains unchanged: 2 warmup epochs, 2 ramp epochs, maximum matching weight 0.5. It operates in generator training, not frozen-head fitting/inference. Auxiliary mask loss is **0 in every run**. No concepts, QTS, event-set head, merge or deduplication loss are introduced.

| Part | Matching | Content conditioning | Quality | Optimization |
|---|---|---|---|---|
| Full control, restored | On | On | On | Existing two-stage |
| `no-content` | On | **Off**; retain geometric learned mask + confidence gate | Fresh frozen head | Generator, then head |
| `quality` | On | On | Full vs **Off** | Reuse the same Full generator/candidates |
| `beta` | On | On | Same frozen Full head | Evaluation only |
| `joint` | On | On | On | Train jointly, no head-only fitting |

Generator: batch 1, accumulation 1, seed 777, FP32/TF32 off, Adam 5e-5 / weight decay 1e-4; YC2 15 epochs/Q50, ANET 6 epochs/Q10. Gaussian Soft-NMS only, sigma 0.5, fixed counter budget. Main comparison beta is YC2=1 and ANET=6. The beta sensitivity grid is `[0, .25, .5, 1, 2, 4, 6, 8]`; no beta is chosen separately for each validation metric.

Frozen head: unchanged 2312D features / 280577-parameter head, TRAIN fit/dev split seed777, initialization/training seed777, AdamW .001, batch8 videos, 20 head-only epochs. A no-content generator gets its **own** head; Full's head is not transferred. Joint initializes the same head using seed777 in an isolated RNG context, receives .1 quality-loss weight and a .001 LR parameter group in the original generator Adam optimizer. Joint uses batch1 during generator epochs; its head group follows the original LR scheduler. Quality targets remain detached, generated captions are free-running, and quality labels are from TRAIN head-fit only. Generated token IDs and hard temporal membership are discrete; quality gradients reach existing continuous confidence/duration/lexical-embedding inputs. Caption loss continues to train the content mask.

The Joint vs two-stage comparison is a comparison of **practical training regimes**, not a compute-matched proof about freezing alone: frozen-head training has 20 extra epochs, batch8 and AdamW; Joint uses batch1 and the generator Adam parameter groups. Generator epoch/data/seed/selection budgets are held fixed and the differences above must be reported. Both use the generator's original MetaScore checkpoint selection rule. Quality-aware Joint validation is logged every epoch but does not silently change checkpoint selection.

## 0. New session: same setup on both accounts

Upload `KLTN_DRIVE_PRIVATE.env` to `/marimo`. This script reuses native GPU Torch/Torchvision. No credentials are committed or included in exports.

```bash
bash <<'BASH'
set -euo pipefail
export LC_ALL=C LANG=C
test -f /marimo/KLTN_DRIVE_PRIVATE.env
if [ ! -e /marimo/PRAWN_Ablations/source ]; then
  git -c advice.detachedHead=false clone --single-branch \
    --branch prawn-loo-20261119-v1 \
    https://github.com/thloc1212/DDVC.git /marimo/PRAWN_Ablations/source
fi
test "$(git -C /marimo/PRAWN_Ablations/source rev-parse HEAD)" = \
     "$(git -C /marimo/PRAWN_Ablations/source rev-parse 'prawn-loo-20261119-v1^{commit}')"
bash /marimo/PRAWN_Ablations/source/tools/prawn_ablation.sh auth
BASH
```

Account 1 (YC2):

```bash
printf '%s\n' yc2 > /marimo/PRAWN_Ablations/dataset.txt
```

Account 2 (ANET):

```bash
printf '%s\n' anet > /marimo/PRAWN_Ablations/dataset.txt
```

The remaining blocks are identical on both accounts and read that dataset file. Run one part at a time. Wait for `ABLATION_COMPLETE_BACKED_UP` before starting the next part. `Ctrl+C` stops log viewing only; it does not stop the background experiment.

## 1. Effect of content conditioning

Fresh no-content generator → verified backup → new frozen quality head. Retains geometric learned mask, confidence gate, matching and Soft-NMS.

```bash
DS=$(cat /marimo/PRAWN_Ablations/dataset.txt)
bash /marimo/PRAWN_Ablations/source/tools/prawn_ablation.sh no-content "$DS"
bash /marimo/PRAWN_Ablations/source/tools/prawn_ablation.sh logs "$DS" --part no-content
```

Full is restored, never retrained by this interface. The geometric mask is initialized with the same seed1731 isolation already used by Full; disabling content does not advance the generator RNG differently.

## 2. Effect of quality-aware selection

Restore the completed Full's predictions/metrics and compare Full against its own beta0/quality-off Soft-NMS result. No generator/head training or feature download.

```bash
DS=$(cat /marimo/PRAWN_Ablations/dataset.txt)
bash /marimo/PRAWN_Ablations/source/tools/prawn_ablation.sh quality "$DS"
bash /marimo/PRAWN_Ablations/source/tools/prawn_ablation.sh logs "$DS" --part quality
```

Full control is selected only if both matching and content are on, mask loss is0, and seed/batch/epochs/queries match. The generator and quality are joined by checkpoint/config/raw SHA256 and epoch. Original Global and geometry-only checkpoints are not substituted. A pinned `control_snapshot.json` fixes the control across subsequent parts.

## 3. Sensitivity to the quality coefficient

Only the frozen head, VAL feature shards, annotations and raw predictions are restored. No generator checkpoint, CLIP features or training shards are needed for this part. Official scoring runs for all8 betas; results are printed and verified to Drive after each coefficient. Beta0 and the fixed Full beta must reproduce saved Full selections/metrics.

```bash
DS=$(cat /marimo/PRAWN_Ablations/dataset.txt)
bash /marimo/PRAWN_Ablations/source/tools/prawn_ablation.sh beta "$DS"
bash /marimo/PRAWN_Ablations/source/tools/prawn_ablation.sh logs "$DS" --part beta
```

Sensitivity uses validation data and must be described as such. The existing ANET beta6 was previously validation-tuned; it is not a new independently held-out test choice.

## 4. Joint versus two-stage optimization

Joint trains content mask + matching + generator + quality together, then evaluates the jointly trained head at the same fixed beta as Full. It never starts a second frozen-head optimizer.

```bash
DS=$(cat /marimo/PRAWN_Ablations/dataset.txt)
bash /marimo/PRAWN_Ablations/source/tools/prawn_ablation.sh joint "$DS"
bash /marimo/PRAWN_Ablations/source/tools/prawn_ablation.sh logs "$DS" --part joint
```

Full's two-stage row comes from part2. The Joint log prints `ONLINE_QUALITY_GRADIENT_PASS joint`, `SMOKE_CAPTION_MATCH_STATS`, and `JOINT_VALIDATION_EPOCH` with quality-aware metrics. Real-model CUDA forward/backward and beam-generation smoke must pass before full training. No local CPU test is presented as proof that CUDA compilation or your live Drive authentication works.

## Status, session loss, summaries

```bash
DS=$(cat /marimo/PRAWN_Ablations/dataset.txt)
bash /marimo/PRAWN_Ablations/source/tools/prawn_ablation.sh status "$DS"
```

After session loss, repeat section0 (including the appropriate dataset line), then repeat the exact part command. Trainable parts find their own exact-source recovery namespaces and restore generator/model/optimizer/RNG state plus partial frozen-head work. They never continue from another ablation's weights. Completed generators are restored without changing their config bytes; runtime/data are prepared before frozen-head continuation. Beta output caches are reused only when prediction/reference signatures match. A failure is logged as `FAILED`; do not assume an exit is a completed experiment.

Training needs staged CLIP features/banks from the verified shared data cache; these are necessary for a fresh generator and frozen-head extraction. They are separate from result-only restores in parts2–3.

Local summaries:

```text
/marimo/PRAWN_Ablations/train_no_content/frozen_quality/runs/<DS>/evaluation/summary.tsv
/marimo/PRAWN_Ablations/results/<DS>/quality/summary.tsv
/marimo/PRAWN_Ablations/results/<DS>/beta/summary.tsv
/marimo/PRAWN_Ablations/train_joint/runs/<DS>_loo_joint_e15_b1/joint_evaluation/summary.tsv  (YC2)
/marimo/PRAWN_Ablations/train_joint/runs/<DS>_loo_joint_e6_b1/joint_evaluation/summary.tsv   (ANET)
```

Drive namespaces under `KLTNRUNS`:

```text
runs/prawn_loo_no_content_generator/<DS>/<run_id>/recovery/slot-{0,1}
runs/prawn_loo_no_content_quality/<DS>/<run_id>/recovery/slot-{0,1}
runs/prawn_loo_joint_generator/<DS>/<run_id>/recovery/slot-{0,1}
analysis/prawn_loo/<DS>/...
```

## Export for analysis

```bash
DS=$(cat /marimo/PRAWN_Ablations/dataset.txt)
bash /marimo/PRAWN_Ablations/source/tools/prawn_ablation.sh export "$DS"
```

This writes `/marimo/PRAWN_Ablations/PRAWN_LOO_ANALYSIS_<DS>_<timestamp>.zip` and uploads it to `KLTNRUNS/analysis/prawn_loo/<DS>/exports/`. A combined table is written to `results/<DS>/ablation_summary.tsv`. Source, configs, all saved language metrics, localization P/R/F1, raw/final predictions, training logs and provenance are included; private credentials, checkpoints and feature caches are excluded. F1 is calculated from saved aggregate P/R.

## Validation before release

CPU checks exercise profile isolation, matching enabled in all profiles, mask-loss rejection, existing real tiny GPT2 generation with matching + content + online quality, finite gradient routing, frozen-head beta sweeps without an optimizer, candidate/timestamp/count preservation, scoring-cache invalidation, existing Drive two-slot recovery, and legacy preparation/CLI regressions. Full-control restoration and quality-off extraction were also checked against the user's archived real 436-video YC2 and 4917-video ANET runs. Live Drive and full-model CUDA smoke run on Molab, not on the CPU development workstation.
