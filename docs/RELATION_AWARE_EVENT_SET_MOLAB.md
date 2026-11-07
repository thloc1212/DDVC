# DDVC + QTS + Relation-aware Event Set Selection

Branch: https://github.com/thloc1212/DDVC/tree/exp/relation-aware-event-set

Parent: `exp/query-temporal-support`, commit `78f269df0636d482b4a2fdb262992b68c1c4365b`.
This is a fresh experiment, not a promised metric gain. Original QTS and finalized
C2 branches/checkpoints are preserved. Do not switch source underneath an active run.

## Problems and scope

1. Short events missed / long event boundaries incomplete: retain QTS matched-GT
   temporal support BCE and center-duration refinement. No denoising, finer input
   features or new hierarchical temporal attention is added in this branch.
2. Valid overlapping/nested events discarded as duplicates: learn a five-class
   pair relation and penalize only predicted duplicate probability in selection.
3. Count collapses to a common value: preserve the original count head and loss,
   add a small ordered-count CDF loss and use the full count distribution during
   set selection, instead of fixing output size to argmax N.

The counter still uses the original max-pooled query representation. This branch
does not replace it with a new attention/sum-pooling architecture.

## Architecture

```text
Frozen CLIP features [B,T,768] + train-only caption bank
          ↓
Original DDVC visual/retrieval encoders and two-stream decoder
          ↓ h_loc, h_cap [B,Q,512]
          ├─ original confidence / counter heads
          └─ coarse interval → QTS support/refinement → refined interval
                                                     ├─ C1 learned proposal mask
                                                     │       ↓ GPT-2 → Q captions
                                                     └─ EventSetHeads
                                                          ├─ quality logits [B,Q]
                                                          └─ relation logits [B,Q,Q,5]
Q captions + refined intervals + quality + duplicate probabilities + count logits
          ↓
Greedy-prefix search across all feasible output cardinalities
          ↓
Final event set (selected count may differ from counter argmax)
```

The added heads receive query features, local pooled CLIP features, confidence and
predicted geometry. Each head input is detached by default; new head losses train
the heads/projections without sending their noisy early auxiliary gradients into
the decoder/QTS. This is not freezing DDVC: its original caption, classification,
localization, count and QTS objectives still train the original model. The ordinal
count loss does train the original counter and shared representation.
New parameters are initialized inside a separate RNG fork; this is not a claim
that full training is bit-identical to the control (global gradient clipping and
the added count objective can affect updates).
The new heads add **330,886 trainable parameters** at hidden_dim128.

## Supervision

- Quality target: maximum temporal IoU of each predicted interval against training
  GT, detached; smooth-L1 on sigmoid quality, balanced over IoU>=.5 / IoU<.5 groups.
- Relations: `separate`, `overlap`, `contains`, `inside`, `duplicate`.
  Hungarian matched queries retain distinct GT identities. Distinct GT events may
  overlap or even coincide; they are never labeled duplicate solely for that reason.
- Unmatched queries get an identity only when best IoU>=.7 and its margin over the
  next GT is >=.15. Ambiguous unmatched candidates are ignored for relation loss.
  Same-identity pairs are duplicate targets only when both have IoU>=.5 to that GT.
- A deterministic small perturbation of each matched query representation and its
  predicted box supplies training-only duplicate positives. This avoids training
  a duplicate classifier only on negatives when early real duplicates are uncertain.
  It is auxiliary head augmentation, not full DINO decoder denoising.
- Relation cross-entropy is averaged per present class, then across classes.
- Ordered count objective: squared distance between predicted softmax CDF and the
  one-hot GT count CDF, final decoder layer only. Original Gaussian-masked BCE count
  loss and auxiliary decoder count losses are retained.
- No GT boxes, captions, count or identities are passed into the new heads at inference.
  GT is used only after matching for training/validation losses, as before.

New loss coefficients: quality **.5**, relation **.2**, ordinal count **.2**.
Old mask loss remains **0**, support BCE **1**, refinement L1 **1**, count **1**.
These are initial experimental settings, not validation-tuned optimal values.

## Selection

For candidate i, use utility

`u_i = log(quality_i / .3) + .15 log(proposal_confidence_i) + .05 beam_sequence_score_i`.

GPT-2's saved `sequences_scores` already includes generation length penalty. The
new selector does not divide it by decoded word count again; the old Top-N and
Soft-NMS controls intentionally keep their historical scoring formula.

Clamp probabilities away from zero. The quality reference .3 and coefficients are
fixed hyperparameters; they are not learned thresholds. .3 is not a hard filter.

Use a symmetrized duplicate probability from the relation softmax, and optimize
approximately along a deterministic greedy candidate-addition path:

`F(S) = sum_i u_i - 2 sum_{i<j} P_duplicate(i,j) + .2 log P_count(|S|)`.

Counter logits are softened with temperature2, mixed with10% uniform prior to
avoid a near-zero probability for every rare count. Evaluate each prefix size
from1 through min(Q,max_count), take the highest objective. This is a greedy
approximation, not an exact global combinatorial optimum. Min output count1 is
retained for the nonempty-video benchmark protocol. No merge, caption-string
dedup, temporal-IoU suppression or GT oracle is used in the new selection.

Containment alone receives no pair penalty. The relation head must learn low
duplicate probability for distinct nested events; this is not a guarantee that
all such events are recovered. A missing/incorrect raw proposal remains missing.

## Recipe and comparison

| Setting | YouCook2 | ActivityNet |
|---|---:|---:|
| Full feature-matched train / val | 1237 / 436 | 10009 / 4917 |
| Epoch | 15 | 6 |
| Queries / max count | 50 / 20 | 30 / 30 |
| Batch / accumulation / seed | 1 / 1 / 777 | 1 / 1 / 777 |
| Adam LR / WD | 5e-5 / 1e-4 | 5e-5 / 1e-4 |
| Precision / TF32 | FP32 / off | FP32 / off |
| Default variant | `set` | `set` |

Random2-layer GPT-2; frozen pre-extracted CLIP ViT-L/14; no additional DVC
pretraining; no concepts, old mask auxiliary loss, merge or old dedup loss.
LR decay start8/every3/rate.5 is retained; ANET6 does not reach that milestone.
Use the same method/hyperparameters across datasets, apart from the listed budget.

The old `baseline`, `refine`, `support` variants remain available explicitly.
`support` is the control with the same QTS/query/count/data/epoch recipe but no
new heads/ordinal objective and the old Soft-NMS selector. Compare it with `set`
before attributing aggregate gains to the extension. Historical Q10/e8 runs are
not an isolated control for Q30/e6.

## Two fresh Molab accounts: terminal only

On each account select GPU + Internet, upload the private file to
`/marimo/KLTN_DRIVE_PRIVATE.env`. Use the same pinned commit on both accounts.
No new OAuth is needed while that credential remains valid.

```bash
git clone --single-branch --branch exp/relation-aware-event-set https://github.com/thloc1212/DDVC.git /marimo/DDVC_QTS/source
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh auth
```

Acc1:

```bash
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh start yc2
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh logs yc2
```

Acc2:

```bash
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh start anet
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh logs anet
```

The worker handles setup, original shared data cache, actual-model GPU smoke,
fresh training, live logs and verified two-slot rclone backup. It reuses the same
`KLTNRUNS/data/<dataset>/qts-base-assets-v1/` cache because data preparation is
unchanged. Work root remains `/marimo/DDVC_QTS` to preserve absolute input paths.

Fresh run names are `yc2_qts_set_e15_b1` and `anet_qts_set_e6_b1`; old support runs
are not overwritten. Each Drive run also has a unique UTC+UUID ID and variant.
Use fresh sessions; do not edit/checkout source while an old worker is alive.
The private env/config is never included in data/source/recovery uploads.

### GPT-2 tokenizer bootstrap

`tools/prepare_gpt2_tokenizer.py` downloads only vocab.json, merges.txt and
tokenizer_config.json from pinned official GPT-2 revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`; SHA256 is verified per file, then
the vendored tokenizer is loaded locally and checked (50257 IDs, EOS50256,
known hello/world BPE IDs). Total roughly1.43MiB; no model weights are downloaded.
The directory `source/gpt2` satisfies the existing relative from_pretrained('gpt2')
calls, bypassing a failed Hub HEAD/cache resolution. Decoder construction stays
`GPT2LMHeadModel(configuration)` with random weights; model/training recipe unchanged.
Setup, worker retry and direct smoke stage/verify the small assets automatically.
The receipt `provenance/gpt2_tokenizer.json` is included in run backup.

For an existing failed pre-training smoke on the older pinned commit, do not
checkout a different commit or edit the run manifest. Fetch the updated helper
only, run it with `--target /marimo/DDVC_QTS/source/gpt2`, then run
`qts_drive.sh start anet --retry` (or yc2). CUDA/data/config stay in place.
No checkpoint was produced yet, so this is still a fresh training run.

## Logs and saved evidence

```text
/marimo/DDVC_QTS/logs/<dataset>.drive.log
/marimo/DDVC_QTS/logs/<run>.launch.log
/marimo/DDVC_QTS/logs/<run>.heads_smoke.log
/marimo/DDVC_QTS/logs/<run>.log
/marimo/DDVC_QTS/runs/<run>/info.json
/marimo/DDVC_QTS/runs/<run>/prediction/num*_epoch*.raw.json
/marimo/DDVC_QTS/runs/<run>/prediction/num*_epoch*.json
```

Raw rows preserve event_quality, coarse/refined timestamps, original counter N.
Raw top-level metadata stores per-video relation labels and duplicate matrix,
indexed by original query ID (not score-sorted row position). Counter logits are
saved. Final predictions preserve `selected_event_count`; `event_set_selection`
records selected_N, predicted_N, softened count probabilities and cardinality curve.
Full pair matrices stay only in raw JSON to avoid multiplying backup size.
Final JSON is exactly the input evaluated for metrics.

Monitor losses `loss_event_quality`, `loss_event_relation`, `loss_count_ordinal`,
`loss_counter`, `loss_temporal_support`, `loss_temporal_refine`; these are not
benchmark metrics. Count diagnostics: counter_count_error (argmax),
event_counter_expected_error; count selected_N must be evaluated separately.

After at least one completed validation:

```bash
bash /marimo/DDVC_QTS/source/tools/qts_molab.sh report anet set
```

After training, compare three selectors on the identical best-epoch raw candidates
without model inference or retraining:

```bash
bash /marimo/DDVC_QTS/source/tools/qts_molab.sh compare anet set
```

YC2 replaces anet with yc2. Output: `runs/<run>/selection_comparison/epoch*/summary.tsv`.
Rows are Top-N, Soft-NMS .5 and learned event-set selection. Same epoch/references;
official metrics multiplied100. This isolates selection on the new model, not
the contribution of each training loss. Compare reports do not rewrite info.json
or select a new checkpoint separately for each metric.

Backup comparison artifacts after the automatic worker has stopped:

```bash
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh backup anet
bash /marimo/DDVC_QTS/source/tools/qts_molab.sh export anet set
```

Recovery remains epoch/validation-boundary based, with optimizer/scheduler/RNG.
Do not resume checkpoints lacking the new heads or from another variant.
See `docs/QTS_RCLONE_TWO_ACCOUNTS.md` for slot/hash semantics; in this branch,
default run variant is set, not support. `COMPLETE_BACKED_UP` confirms final backup.

## Verification and limits

- `smoke_test_event_set.py`: head/ordinal gradients, detached inputs, nested and
  coincident GT identities, padding, tiny/empty targets, trainability, strict load;
  selector retains parent+child, suppresses duplicate, overrides incorrect count,
  uses query IDs correctly, does not merge equal captions, rejects NaNs.
- `smoke_test_proposal_mask.py --qts --event-set`: actual DDVC matched/full methods,
  actual tiny randomly initialized GPT-2, caption backward and both head gradients.
- `smoke_test_temporal_support.py`: full-split preparation/guards for all4 variants.
- `tools/test_qts_rclone.py`: secret handling, cache, atomic snapshot and fallback.
- On Molab, train requires CUDA head smoke plus actual-model forward/backward,
  generation and postprocess smoke. Offline tests are not a CIDEr/SODA benchmark.

Relation/quality learning and selection may fail to generalize or reduce precision;
measure raw recall, fixed-budget selection recall, short/overlap groups, count MAE
and caption metrics. Do not claim all three original problems are solved yet.

Related ideas, not transplanted implementations:
- IoU-Net (localization confidence): https://arxiv.org/abs/1807.11590
- Relation-DETR (position-relation priors, object detection): https://arxiv.org/abs/2407.11699
- PDVC (parallel caption/localization and event counter): https://arxiv.org/abs/2108.07781
