# DDVC + learned, confidence-gated proposal masks

## Source and scope

Adapted from [bhubanendra-mishra/dense-video-cap](https://github.com/bhubanendra-mishra/dense-video-cap), revision **43914e17769701b9cf98eda203ae4c465b315fab**. This repository implements the CVPR 2018 paper *End-to-End Dense Video Captioning with Masked Transformer*.

Reference components:

- `model/action_prop_dense_cap.py::positional_encodings`
- `ActionPropDenseCap.__init__::mask_model`
- `ActionPropDenseCap.forward` / `inference`: positional boundary encoding + anchor window, confidence-gated fusion of binary proposal mask and learned mask, binary mask objective.
- `model/transformer.py::Encoder.forward`: applies window masks to visual features.

The upstream BSD-3-Clause notice is retained in `third_party/dense-video-cap/LICENSE.txt`. This branch ports the mask components; it does not load the legacy Python 3.6/PyTorch 0.4 training stack or replace DDVC's complete detector/decoder.

DDVC starting point: **f3737c4**, with pair-cost chunking, consistent box decoding and separate raw/final prediction exports. Its released-code retrieval path and legacy losses are retained for comparison with recent global/local runs; this is not the paper-aligned visual-only DDVC baseline.

## Computation

1. For each decoder layer, obtain the predicted center/duration and event-class confidence.
2. Obtain a reference anchor from that layer's decoder reference. The first layer has only a center, so it borrows the predicted duration **detached** as an anchor width.
3. Encode predicted/reference start/end positions with sin-cos encoding. Concatenate the rasterized reference anchor mask.
4. An MLP predicts 64 temporal-grid mask logits, interpolated over the real video feature length. Sigmoid converts them to probabilities.
5. Mix using the event confidence `p`:

```
mask = p * binary_predicted_interval + (1 - p) * sigmoid(learned_mask_logits)
```

6. Multiply the frozen CLIP frame features by these per-event weights, then use DDVC's original randomly initialized GPT-2 cross-attention caption head. Padding is excluded from attention.

This is **feature gating**, not an additive attention-logit bias. With gate 1, frames outside the proposal have zero visual features; attention still sees valid frame positions. This must not be described as numerically identical to the existing hard attention-mask control.

## Important adaptations

- Original convolutional proposal anchors become DDVC decoder references.
- A fixed mask grid + length-aware interpolation replaces the original window-length-specific output.
- LayerNorm replaces BatchNorm to support one matched event / batch 1.
- Learned logits are passed through sigmoid before mixing. The legacy implementation mixes raw logits; this branch intentionally bounds the visual weights in [0,1].
- Original masks were applied inside a separate masked visual encoder. Here they gate CLIP features entering GPT-2; no second legacy visual encoder is added.
- Padded frames are excluded, and sub-frame/zero-width intervals use one nearest valid frame.
- No legacy feature extraction, proposal convolution stack, text vocabulary or language decoder is transplanted.

These are deliberate adaptations, not an exact reproduction of the CVPR 2018 architecture.

## Gradient and supervision

- In matched caption training, caption CE propagates through the learned mask into mask parameters, predicted interval coordinates and event confidence. Frozen CLIP feature tensors are detached **before** gating.
- Hungarian matching still computes every query-caption cost under no-grad, with chunks of 16 pairs and correctly aligned mask/reference/confidence inputs.
- Beam/cached generation uses the same event-conditioned feature gating; no GT timestamps/captions are used to build inference masks.
- A final-layer auxiliary BCE trains learned mask logits against the **detached binary predicted proposal window**, following the upstream mask target. This is a shape-consistency target, **not ground-truth eventness or GT-boundary supervision**. Existing DDVC GIoU/count/class/caption losses provide the original task supervision.
- `loss_proposal_mask` coefficient defaults to 1.0. It is not replicated as a separate auxiliary-layer weight. Caption gradients can still train each caption-head layer's mask path.
- Diagnostics: `proposal_mask_gate` (mean event confidence), `proposal_mask_mean` (mean valid visual weight), `proposal_mask_inside` (binary proposal coverage).

The confidence gate is the existing event confidence, not a new independently learned gate MLP. Improved CIDEr, boundary accuracy or robustness are hypotheses to test.

## Kaggle experiment

Notebook: `notebooks/DDVC_Kaggle_YouCook2_Learned_Proposal_Mask.ipynb`.

- Default: `MASK_MODE='learned_gated'`, subset 30% train/validation, 15 epochs, seed 777, one T4, batch 1, 50 queries.
- Downloads the same annotations/features and builds retrieval memory from selected training captions only.
- `MASK_MODE='global'`: same-branch baseline with the mask module disabled.
- `MASK_MODE='hard_local'`: existing hard visual-attention mask control, with the old 10% context and minimum 4 frames. This control differs in both mask learning and masking placement/context; it is not a pure ablation of just the gate.
- Keep the same data IDs, settings and source for comparisons. The old R12 result is a separate historical run.
- Full mode also uses **15 epochs** by default here; override explicitly if needed. Smoke is 1%, 2 epochs.
- A source/run manifest, actual references/config, logs and final/raw predictions are bundled in a result ZIP. Model checkpoints are left in the run directory, not the small analysis ZIP.
- Resume with the exact run folder ID and matching mode/config. Save Kaggle output/download weights before the temporary session expires. Optimizer checkpoint migration from another branch is not supported.
- CPU preflight runs before training. A real CUDA training/validation run is still needed to assess T4 VRAM and metrics.

## Code map

- `ddvc/CaptioningHead/proposal_mask.py::LearnedGatedProposalMask`: MLP, boundary encoding, gated window, BCE/diagnostics.
- `mask_layer_context`, `proposal_confidence`, `proposal_anchors`: layer/reference/query alignment.
- `GPT2CaptionModel.visual_context`: frozen feature gating and padding attention mask.
- `CM2.parallel_prediction_matched`: all-pair Hungarian costs and matched caption training.
- `CM2.parallel_prediction_full`: generation with per-layer event context.
- `CM2.forward`: final-layer mask objective alongside existing losses.
- `smoke_test_proposal_mask.py`: real tiny random GPT-2 caption gradients, actual two-layer DDVC caption methods, chunking, beam/cached generation and state-dict checks.
