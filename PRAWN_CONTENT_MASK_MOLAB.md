# PRAWN: content-conditioned proposal mask experiment

Fixed release: `prawn-content-mask-20261117-v1`.
Parent: original two-stage PRAWN, commit `08d024eb07e7ef419458049905d9bad03a3597e2`.

## What changes

The original soft mask uses predicted proposal edges, detached reference-anchor
edges and a 64-bin hard anchor mask. It does not inspect visual content.
This variant retains that geometry branch and adds an event-specific, per-time
content residual. The event representation is the caption-stream decoder output
`h_cap`, which already contains localization-stream information under the
original `capQ_addP=after` recipe. It is the same representation used by the
caption prefix MLP, before that projection.

For event i and valid feature step t:

```text
q_i: caption/event query                         [E, 512]
F_t: original frozen CLIP temporal features      [1, T, 768]
b_i = (c_i, d_i), predicted interval             [E, 2]
p_i: event confidence                           [E]

u_it = clamp((t_normalized - c_i) / max(d_i, 1e-4), -4, 4)
r_it = [u_it, d_i]

a_it = tanh(LN(Linear_512_to_128(q_i))
            + LN(Linear_768_to_128(stopgrad(F_t)))
            + Linear_2_to_128(r_it))
z_content_it = Linear_128_to_1(a_it)
S_it = sigmoid(z_geometry_it + z_content_it)       [E, T]
W_it = p_i * H_it + (1 - p_i) * S_it              [E, T]
F_event_it = stopgrad(F_t) * W_it                 [E, T, 768]
```

The last content layer starts at zero. Consequently, the initial mask exactly
matches the original geometry-only mask for the same geometry parameters.
Caption loss first updates that layer, then propagates into its query/visual/
relative-position projections. The new branch adds **164,993 parameters**.

There is no auxiliary mask loss, hard frame threshold, softmax competition across
events, extra boundary-refinement head or change to the quality target. Each
event learns its own sigmoid weights, allowing concurrent/nested events to use
the same feature steps. Padding stays excluded from cross-attention.

In training, E is the number of matched events for caption supervision. During
Hungarian caption-cost calculation, repeated query-caption pairs are aligned by
query IDs and processed in chunks. At inference E = Q. The same content branch
is used in all three paths. The diagnostic mask BCE is still logged with weight
zero and uses the actual content-conditioned logits.

## Fixed experimental recipe

| Setting | YouCook2 | ActivityNet Captions |
|---|---:|---:|
| Generator epochs | 15 | 6 |
| Generator video batch / accumulation | 1 / 1 | 1 / 1 |
| Event queries | 50 | 10 |
| Generator seed | 777 | 777 |
| Precision | FP32, TF32 off | FP32, TF32 off |
| Optimizer / LR / weight decay | Adam / 5e-5 / 1e-4 | Adam / 5e-5 / 1e-4 |
| Auxiliary mask loss coefficient | 0 | 0 |
| Frozen quality head epochs | 20 | 20 |
| Quality video batch | 8 | 8 |
| Quality seed / fit-dev split seed | 777 / 777 | 777 / 777 |
| Fixed final beta | 1 | 6 |
| Gaussian Soft-NMS sigma | 0.5 | 0.5 |

The generator uses the original checkpoint-selection rule, validation METEOR +
SODA_c. Its GT sampling, number of queries, localization loss and counter are
unchanged. Beta is fixed from the earlier PRAWN experiments rather than retuned
on the new validation results. In the earlier experiments ANET beta 6 was
validation-tuned, which should remain explicit when reporting results.

The full chain trains a fresh generator, waits for its verified Drive backup,
extracts candidates from its own best checkpoint, freezes that generator, then
trains a fresh quality head. Old generator weights and old quality caches/heads
are not loaded. Verified shared feature/annotation/bank bytes can be reused.

## Run on two new Molab GPU sessions

Upload the updated private file to `/marimo/KLTN_DRIVE_PRIVATE.env` on each
account. Keep that file outside Git and result archives. These commands run in
the terminal. The launcher preserves the runtime's installed GPU PyTorch,
installs dependencies, restores rclone authentication, stages data, runs CPU
checks and an actual model GPU forward/backward smoke before full training.

Account A, YouCook2:

```bash
bash <<'BASH'
set -euo pipefail
export LC_ALL=C LANG=C
test -f /marimo/KLTN_DRIVE_PRIVATE.env
git clone --single-branch --branch prawn-content-mask-20261117-v1 \
  https://github.com/thloc1212/DDVC.git /marimo/PRAWN_Content_Mask/source
bash /marimo/PRAWN_Content_Mask/source/tools/prawn_content_mask.sh start yc2
bash /marimo/PRAWN_Content_Mask/source/tools/prawn_content_mask.sh logs yc2
BASH
```

Account B, ActivityNet:

```bash
bash <<'BASH'
set -euo pipefail
export LC_ALL=C LANG=C
test -f /marimo/KLTN_DRIVE_PRIVATE.env
git clone --single-branch --branch prawn-content-mask-20261117-v1 \
  https://github.com/thloc1212/DDVC.git /marimo/PRAWN_Content_Mask/source
bash /marimo/PRAWN_Content_Mask/source/tools/prawn_content_mask.sh start anet
bash /marimo/PRAWN_Content_Mask/source/tools/prawn_content_mask.sh logs anet
BASH
```

Data is staged locally for training, about 872 MiB of feature download for YC2
or 2.52 GiB for ANET, plus banks/dependencies/checkpoints. Drive uses separate
namespaces:

```text
KLTNRUNS/data/<dataset>/c1-base-assets-v1/       shared verified data
KLTNRUNS/runs/prawn_content_mask_generator/<dataset>/<run-id>/
KLTNRUNS/runs/prawn_content_mask_quality/<dataset>/<run-id>/
```

Different datasets and unique run IDs avoid two-account output conflicts.
Closing `tail` with Ctrl+C leaves the background worker running. It does not
guarantee that Molab will retain the compute session. Verified recovery slots
are the protection against session loss.

## Monitor, results, export

```bash
bash /marimo/PRAWN_Content_Mask/source/tools/prawn_content_mask.sh status yc2
cat /marimo/PRAWN_Content_Mask/frozen_quality/runs/yc2/evaluation/summary.tsv
cat /marimo/PRAWN_Content_Mask/frozen_quality/runs/yc2/evaluation/temporal_audit.tsv
bash /marimo/PRAWN_Content_Mask/source/tools/prawn_content_mask.sh export yc2
```

Replace `yc2` with `anet` on account B. `summary.tsv` appears after stage 2
evaluation. The method row is `reranker_softnms`. The other rows evaluate the
same generator with different selection policies and do not train extra
generators. Generator validation metrics are printed after each epoch, while
quality-adjusted official metrics are computed after fitting the frozen head.

Generator logs:
`/marimo/PRAWN_Content_Mask/logs/yc2_content_mask_e15_b1.log` or
`/marimo/PRAWN_Content_Mask/logs/anet_content_mask_e6_b1.log`.
Quality logs: `frozen_quality/logs/<dataset>.rerank.log`.
The combined tail includes setup, generator and quality logs.

Export prints two ZIP paths, one under `/marimo/PRAWN_Content_Mask/` and one
under `/marimo/PRAWN_Content_Mask/frozen_quality/`. Both contain code/configs,
logs and predictions needed for comparison. Generator checkpoint weights and
private credentials are excluded. Weights are retained in Drive recovery slots.

Resume an interrupted process in the same session after inspecting its log:

```bash
bash /marimo/PRAWN_Content_Mask/source/tools/prawn_content_mask.sh resume yc2
bash /marimo/PRAWN_Content_Mask/source/tools/prawn_content_mask.sh logs yc2
```

On a new session, clone the exact same tag and authenticate first. Restore the
chosen generator run before setup/training, then restore its quality run too if
one exists. Do not point this experiment at original PRAWN or joint run IDs.

```bash
bash /marimo/PRAWN_Content_Mask/source/tools/prawn_content_mask.sh auth
rclone --config /marimo/secrets/rclone.conf lsf \
  kltnbackup:KLTNRUNS/runs/prawn_content_mask_generator/yc2 --dirs-only
bash /marimo/PRAWN_Content_Mask/source/tools/c1_drive.sh restore yc2 \
  --run-id YOUR_GENERATOR_RUN_ID
# Only if a matching quality run had already started:
# bash /marimo/PRAWN_Content_Mask/source/tools/c1_rerank.sh restore yc2 \
#   --work /marimo/PRAWN_Content_Mask/frozen_quality --run-id YOUR_QUALITY_RUN_ID
bash /marimo/PRAWN_Content_Mask/source/tools/prawn_content_mask.sh resume yc2
bash /marimo/PRAWN_Content_Mask/source/tools/prawn_content_mask.sh logs yc2
```

## Decide whether the hypothesis works

Compare with original PRAWN under the same recipe/seed and evaluation protocol.
`temporal_audit.tsv` reports both ANET references separately, at tIoU 0.3/0.5/
0.7/0.9, before selection and after each policy. Inspect short events at most
5 seconds, at most 5% video length, short overlapping events and long events.
One-to-one recall and independent GT coverage are explicitly different columns/
rows. These diagnostics do not replace official captioning metrics.

A captioning gain alone does not prove better timestamp prediction. If raw short
event recall remains low, this mask change has not solved proposal generation
for short events. It directly changes caption visual support and only affects
localization indirectly through shared decoder/query gradients and caption-cost
matching. It does not add direct boundary supervision or additional candidates.
No result improvement is assumed before the runs finish.

## Source and verification

- `ddvc/CaptioningHead/proposal_mask.py`: geometry + query/visual residual.
- `ddvc/CaptioningHead/gpt2.py`: train/inference visual conditioning.
- `ddvc/ddvc_ret_encdec_clip_capQ.py`: matched/all-query routing and diagnostics.
- `misc/c1_recipe.py`, `tools/c1_pipeline.py`: fixed experimental invariants.
- `tools/prawn_content_mask.sh` / `.py`: two-stage terminal launcher.
- `reranking/`: unchanged quality architecture/target/selection plus temporal audit.
- `tools/test_content_proposal_mask.py`: baseline initial parity, content/query
  sensitivity, padding, repeated matching pairs, actual GPT-2 CE gradients and
  beam generation.
- `smoke_test_c1.py`: full-split preparation and actual tiny GPT-2 checks.
- `tools/test_prawn_content_mask.py`, `tools/test_c1_rclone.py`,
  `tools/test_c1_rerank.py`: recipe, stage ordering, verified shared data, backup
  fallback, quality training and scoring.

Local verification uses CPU. The Molab launcher performs the actual full-model
CUDA smoke test before launching full training.
