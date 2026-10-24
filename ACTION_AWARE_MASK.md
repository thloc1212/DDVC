# Action-aware, GT-supervised proposal masks for DDVC

Branch: [`exp/action-aware-grounded-mask`](https://github.com/thloc1212/DDVC/tree/exp/action-aware-grounded-mask).

This is a new experiment based on `exp/learned-gated-proposal-mask` (`bcdd683`). Existing R15 checkpoints/results remain a historical baseline. No new GPU scores are claimed.

## What changed

1. **Dynamic attention pruning:** feature gating remains; GPT-2 also receives `(weights > threshold OR predicted_context_window) AND valid_frames`. The predicted context window guarantees valid frames. No GT is read in this inference path. The Boolean threshold is not differentiable; soft weights still receive caption gradients. At initialization weights may exceed 0.05 almost everywhere, so pruning is not guaranteed to be strong until learned.
2. **Matched-GT mask supervision:** final-layer Hungarian query/GT pairs select soft logits and GT center/duration in exactly the same order. Only matched queries receive this auxiliary BCE; unmatched candidates do not receive a made-up GT mask. Targets have no context margin. Balanced BCE gives foreground/background equal contribution per event. Optional legacy predicted-window BCE remains available. GT supervision runs in training, not caption inference.
3. **Query conditioning:** project the caption query from hidden dimension 512 to 64; concatenate to predicted/reference boundary encoding (128) and anchor grid (64). The mask MLP input is 256. This supplies query semantics, NOT direct visual-frame features. GPT-2 is still initialized randomly.
4. **Bounded confidence mixing:** `alpha = gate_min + (gate_max - gate_min) * class_confidence`, defaults 0.1–0.85 in the new notebook. This is not learned tIoU calibration; class confidence can remain poorly correlated with boundary accuracy.
5. **Consistent context policy:** hard proposal fallback uses 10% of predicted duration on EACH side and minimum 4 valid positions, matching hard_local control. The attachment's example `duration * (1+0.1)` would add only 5% per side; the implementation uses the existing local_visual convention instead.

These are configurable. Python/CLI defaults retain legacy mask behavior; the **new notebook** explicitly enables the upgraded configuration.

## Additional training fixes

- Optional `corrected_query_contrastive=True`: applies temperature, removes self-negatives from denominator; warmup updates `loss_contrastive` and auxiliary keys as well as the legacy alias. Default notebook enables this in BOTH global and upgraded controls. Historical R15 comparison alone cannot isolate mask gains.
- Logging interval is at least 1 for tiny smoke datasets.
- Removed a machine-specific sys.path entry from train.py.
- Atomic format-2 checkpoints include model, optimizer, scheduler, epoch/next_epoch, iteration, RNG states, config and metric metadata.
- Checkpoints saved before validation have phase `validation_pending`. On resume the loop repeats pending validation and does not retrain that completed train epoch. After validation/scheduler stepping, phase becomes `epoch_complete` and next_epoch advances.
- A consistent completed checkpoint is sufficient even if the separate info.json is absent/stale. Resuming an already completed budget is a no-op.
- Legacy format-1 checkpoints are rejected for resume rather than silently duplicating epochs or scheduler state. Keep using their original notebook, or implement an explicit migration/warm start. Do NOT load R15 into this query-conditioned architecture as if it were an exact resume.
- Model still assumes batch size 1, 2 decoder layers and auxiliary loss in the supported notebook. `matched_gt` requires aux_loss=True; unsafe legacy aux-off path is not advertised.

## Experiment defaults

Notebook: `notebooks/DDVC_Kaggle_YouCook2_Action_Aware_Mask.ipynb`.

```python
RUN_MODE = 'subset'              # smoke=1%, subset=30%, full=all feature-matched
MASK_MODE = 'learned_gated'      # global / hard_local / learned_gated
EPOCHS = 2 if RUN_MODE == 'smoke' else 15
MASK_TARGET = 'matched_gt'
QUERY_CONDITIONING = True
PRUNE_THRESHOLD = 0.05
GATE_MIN, GATE_MAX = 0.1, 0.85
MASK_CONTEXT_RATIO, MASK_MIN_FRAMES = 0.1, 4
BALANCED_MASK_BCE = True
MASK_LOSS_COEF = 1.0
CORRECTED_QUERY_CONTRASTIVE = True
```

Seed 777; queries 50; batch 1; one T4; no pretrained language decoder; data downloads automatically. Full also defaults to 15 epochs.

Output root: `/kaggle/working/DDVC_action_aware_mask/save_action_aware_mask/<actual-run-id>/`. Analysis ZIP has info/config/manifest/data references/logs/predictions, **not model weights**. Preserve checkpoints separately. A hard-killed Kaggle session may prevent the final ZIP cell from executing; inspect the run directory itself.

## Suggested comparisons

1. Corrected global baseline: `MASK_MODE='global'`, corrected contrastive enabled.
2. Upgraded learned_gated, same split/seed/budget/criterion.
3. Same upgraded architecture with mask BCE coefficient 0/0.1/1.
4. Disable one mask feature at a time (query conditioning, pruning threshold=-1, gate bounds 0/1, context 0/min1, predicted vs matched_gt target).

Changing query conditioning changes parameter shapes: use a fresh run ID and train from scratch for each ablation. Resume validation rejects changes to mask settings, dimensions, data signature, schedule or epoch budget.

Do not assume an improvement of a specific magnitude. R14 vs R15 difference is observational; these changes address plausible limitations, not proven causal explanations of CIDEr/SODA_c.

## Diagnostics and tests

Training logs include raw `loss_proposal_mask`, gate mean/min/max, valid attention active fraction, soft-mask target IoU, mask coverage and effective per-epoch objective weights. GT-IoU diagnostic refers to matched-GT targets only when that target mode is used. It is not the detector's tIoU metric.

Tests use CPU, tiny random GPT-2 and AST-extracted actual DDVC methods (no data/model download or CUDA extension):

```bash
python smoke_test_local_visual.py
python smoke_test_proposal_mask.py
python smoke_test_training_state.py
```

Coverage: legacy behavior; matched query-ID/GT ordering; query-projection/caption gradients; bounded gate; padding and min frames; invariance to pruned visual positions during real beam/cached generation; corrected contrastive temperature and self-negative removal; actual train-loop interruption during validation, identical resumed model/scheduler/RNG continuity and no repeated epoch. Tests do not measure real T4 memory or YouCook2 metrics.

## Provenance

- [DDVC paper, ACL 2025](https://aclanthology.org/2025.acl-long.807/), [original repo](https://github.com/siplysagari/DDVC).
- [Masked Transformer, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_End-to-End_Dense_Video_CVPR_2018_paper.html).
- Reference code: [dense-video-cap revision 43914e1](https://github.com/bhubanendra-mishra/dense-video-cap/tree/43914e17769701b9cf98eda203ae4c465b315fab).
- Existing third-party BSD-3-Clause notice remains under `third_party/dense-video-cap/`.
