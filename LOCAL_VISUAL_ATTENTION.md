# Event-local visual attention experiment

Base: original DDVC commit `6cca588`, with the established caption-cost chunking
and final/raw prediction export fixes. This branch does not include temporal
query fusion, gated fusion, temporal matching/order losses or the three-way
interaction experiments.

The caption decoder remains `GPT2LMHeadModel(configuration)`: random language
weights, two layers in the Kaggle recipe. The tokenizer and frozen CLIP video/text
encoders retain the existing DDVC initialization. No external DVC pretraining or
pretrained language decoder is added.

## Change

For each predicted normalized center/duration, construct a binary frame keep-mask
covering `[center - duration/2, center + duration/2]`, expanded by 10% of duration
on each side. At least four nearest valid frames are exposed for short events.
Padding is always excluded. Context ratio and minimum frame count are configurable.

The mask is supplied as GPT-2 `encoder_attention_mask`:

- All query-caption matching pairs use their **query's predicted interval**,
  repeated in exactly the same order as the caption-cost chunks.
- Matched caption training uses the predicted interval of the matched query.
- Inference uses the predicted interval returned by the localization head; no GT
  timestamps are read by the mask builder.

The hard mask and visual features are detached. Existing localization losses
train boundary coordinates; the caption loss trains the caption query/decoder.
This experiment does not claim differentiable caption supervision of boundaries.
Masking does not shrink the cross-attention buffers; the existing batch-size-one
and 16-pair caption-cost chunks are retained for T4 memory.

## Shared correctness fixes and controls

Train and inference now use the same `decode_event_boxes` function. Previously,
at the first decoder layer, inference added a center-only reference to **both**
center and duration logits, while training added it only to center. Duration now
follows the training rule. Final prediction JSON retains the exact reranked metric
input; raw candidates are written separately.

The vendored Transformers generation engine has custom beam code that passes
visual encoder states and masks. It is retained; beam/cached decoding is tested
with the actual vendored GPT-2. No Transformers generation source is changed.

Use the new notebook with `CAPTION_CONTEXT='local'` (default) and `'global'` for
controlled comparisons. Both modes use the same padding, box-decoding and export
fixes, model, losses, data IDs, seed and training schedule. Differences from an
older uncorrected run cannot be attributed solely to local attention.

Recipe: 30% feature-matched train/validation data, 15 epochs, seed 777, 50 queries,
one T4, batch size 1, randomly initialized two-layer caption decoder. Original
`bbox_loss_coef=0`, `set_cost_bbox=0`, GIoU/classification/caption/count/contrastive
settings are retained. Do not resume a gated/temporal run into this branch.

## Files and checks

- `ddvc/CaptioningHead/local_visual.py`: mask building and query interval gathering.
- `ddvc/CaptioningHead/gpt2.py`: train/generation cross-attention mask plumbing.
- `ddvc/ddvc_ret_encdec_clip_capQ.py`: predicted interval calculation and pairing.
- `opts.py`: local attention and caption-cost chunk options.
- `eval_utils_clip.py`: previously verified final/raw export fix.
- `smoke_test_local_visual.py`: small, offline CPU test using the actual caption
  head, vendored GPT-2 and matching-cost function. Tests masking, padding, short
  events, pair ordering, chunk equivalence, gradients and beam/cached generation.

The Kaggle notebook runs this check before training. Full GPU training and metric
improvement require an actual Kaggle run and are not established by the CPU test.
