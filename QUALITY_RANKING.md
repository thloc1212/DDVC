# DDVC localization-quality ranking experiment

## Scope

Based on `f3737c4`: original DDVC with caption-cost chunking, consistent box decoding and separate raw/final exports. Caption attention is **global**, GPT-2 remains randomly initialized. Temporal fusion, bidirectional interaction, semantic matching and narrative memory are disabled. CLIP visual/text encoders remain pretrained as in DDVC.

Default Kaggle experiment: 30% of feature-matched train **and validation**, seed 777, 15 epochs, 50 queries, batch 1, one T4. These metrics must not be compared directly to a full-data paper result.

## New head

`ddvc/localization_quality.py::LocalizationQualityHead` samples 8 interpolated visual features over 1.5 times each predicted interval, clipping samples to the valid timeline. It projects these features and the final localization query into a small MLP, together with normalized center/duration. Padding is excluded. A sigmoid predicts quality in [0,1]. This is a lightweight experimental design, not a reproduction of another paper's quality head.

Supervision is the **maximum temporal IoU with any GT event** for every query, including unmatched queries. Final-layer MSE is weighted by `quality_loss_coef=1`. This target evaluates localization only; it cannot guarantee caption correctness or uniqueness.

Original query, interval and video tensors are detached before this head. The auxiliary objective trains only the quality head; original DDVC localization/caption losses still train the base model. Head initialization preserves the original CPU RNG, and gradient clipping is separate so head gradients do not rescale base gradients. Extra memory is small and does not include an additional caption decoder.

## Ranking

For each candidate:

```
joint_score = alpha * sentence_score / (word_count**2 + 1e-5)
              + proposal_score * quality_score**quality_rank_weight
```

Keep the original predicted event count/top-N selection. With weight 0 this exactly restores the original ranking formula. With weight 1 it uses the predicted quality. The caption-score term and alpha are unchanged; this head is not an NMS or caption rewrite module.

`PostProcess` gathers quality using the same query indices as boxes/captions. Raw JSON retains all candidates and original `proposal_score`, adding `quality_score`. Final JSON is the actual metric input, not raw candidates. Ranking metadata records the formula and weight.

## Controls and diagnostics

- Default: `ENABLE_QUALITY=True`, `QUALITY_RANK_WEIGHT=1.0`.
- No-head training control: `ENABLE_QUALITY=False` (separate run ID).
- Same-candidate ranking control: after training the notebook automatically runs `eval_quality_ablation.py` with weights 0 and 1 on the **same best checkpoint's raw candidates**, without another GPU forward pass.
- Best checkpoint remains selected by validation METEOR + SODA_c. This paired comparison uses that checkpoint selected with quality ranking; it is not two independently chosen best checkpoints and is not a held-out test claim.
- Logs include `loss_quality`, `quality_mean`, `quality_target_mean`, `quality_mae`. Compare selection changes, CIDEr/METEOR/SODA_c/F1 and the full result. Lower head MSE alone does not establish a useful ranker.
- Do not tune many weights on the validation subset and report the best as an unbiased improvement. Use the fixed default first and confirm useful effects on full splits/multiple seeds.

## Files and execution

Use `notebooks/DDVC_Kaggle_YouCook2_Quality_Ranking.ipynb` on Kaggle with GPU and Internet enabled. It downloads annotations/features, selects subsets, builds the training-caption retrieval bank, trains, evaluates and bundles predictions/config/GT/logs plus the paired-ranking report. It does not train feature encoders from videos.

Resume from `model-last.pth` with the same run ID, config, data and head setting. Reusing a prior branch's optimizer checkpoint is unsupported. Kaggle `/kaggle/working` is temporary: save notebook output or download checkpoints to resume in another session. The small analysis ZIP excludes model weights; checkpoints remain in the run folder.

CPU preflight: `python smoke_test_quality.py`. Existing caption preflight: `python smoke_test_local_visual.py`. CUDA extension compilation and a real training/validation epoch still require the Kaggle GPU runtime.
