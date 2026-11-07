# DDVC + QTS + Relation-aware Event Set Selection

Branch: [`exp/relation-aware-event-set`](https://github.com/thloc1212/DDVC/tree/exp/relation-aware-event-set).

Start here: [fresh two-account terminal/rclone runs, model, losses and comparisons](docs/RELATION_AWARE_EVENT_SET_MOLAB.md).
Default variant is **set**: QTS + localization-quality/pair-relation heads,
an auxiliary ordered-count objective, and GT-free selection over multiple output
cardinalities. Distinct nested events are not duplicate targets. No benchmark
improvement is claimed before full GPU experiments.

New experiment based on C1: learned proposal mask, old mask loss **0.0**.
A query-conditioned visual-support module refines final-layer intervals before
caption generation and Hungarian matching. Matched GT intervals supervise an
event-balanced support BCE and a continuous center/duration refinement L1.
GPT-2 stays randomly initialized; CLIP input features remain frozen.

| Setting | YouCook2 | ActivityNet Captions |
|---|---:|---:|
| Epochs | 15 | **6** |
| Queries | 50 | **30** |
| GT budget | **20** | **30** |
| Counter maximum / logits | 20 / 21 | **30 / 31** |
| Count loss coefficient | **1.0** | **1.0** |

All GT events/captions are retained, with explicit capacity checks. Counter
class frequencies are derived from full training annotations only. The original
`support` control uses counter-guided Gaussian Soft-NMS sigma .5; no merge,
concept heads or deduplication loss. Seed777, batch1, accumulation1, FP32,
TF32 off, Adam LR5e-5. New query/count dimensions require fresh training.

Original QTS: [architecture, ablations, outputs and Drive backup](docs/QUERY_TEMPORAL_SUPPORT_MOLAB.md).
For terminal-only rclone runs with `KLTN_DRIVE_PRIVATE.env`, shared data cache and
verified checkpoint recovery, use the [new event-set runbook](docs/RELATION_AWARE_EVENT_SET_MOLAB.md)
and `tools/qts_drive.sh`.
Default launcher: `tools/qts_drive.sh` (rclone); direct launcher `tools/qts_molab.sh`.
Variants: `baseline`, `refine`, `support`, `set`; the new branch defaults to `set`.
Real-model GPU smoke is required before full train/resume. No new benchmark
scores are claimed from CPU validation.

Previous finalized C2 stays on
[`final/c2-learned-mask`](https://github.com/thloc1212/DDVC/tree/final/c2-learned-mask);
[its historical runbook](docs/C2_FINAL_MOLAB.md) applies to that recipe.

## Original DDVC project

# [ACL 2025] Task-Specific Information Decomposition for End-to-End Dense Video Captioning

This is the official codebase for our ACL 2025 paper:  
**"Task-Specific Information Decomposition for End-to-End Dense Video Captioning"**.

Our framework, **DDVC**, addresses the limitations of query-sharing in end-to-end dense video captioning by introducing decomposed, task-specific query representations for localization and captioning. We further improve task coordination via joint label assignment and contrastive semantic alignment. This repository provides the complete implementation for reproducing our results on the [YouCook2]and [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/) datasets.

---

## 🧠 Highlights

- **Query Decomposition**: Decouples shared event queries into task-specific localization and captioning queries.  
- **Joint Supervision Label Assignment**: Assigns ground-truths based on both localization and caption semantics.  
- **Contrastive Semantic Alignment**: Enhances task-specific representations via visual-textual contrastive learning.  
---

## 🛠️ Setup

**Environment**: Linux, Python ≥ 3.8, PyTorch ≥ 1.7.1

### 1. Create and activate a conda environment

```bash
conda create -n ddvc python=3.8
conda activate ddvc
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install ffmpeg
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### 2. Compile CUDA operators

```bash
cd DDVC/ops
sh make.sh  # Requires GCC >= 5.4
```

---

## 📁 Data Preparation

### 1. Download features

- **ActivityNet (anet) CLIP features**  
  [Download link (Google Drive)](https://drive.google.com/file/d/1v08rs9Hwqh3XIM-8u8rRcFZc3HWUtJo_/view?usp=sharing)  
  Place at:
  ```
  DDVC/data/anet/features/clipvitl14.pth
  ```

- **YouCook2 (yc2) CLIP features**  
  [Download link (Google Drive)](https://drive.google.com/file/d/17H_lxSKFve57kHpkAD7pcYS7ijHIy4-M/view?usp=sharing)  
  Place at:
  ```
  DDVC/data/yc2/features/clipvitl14.pth
  ```

---

## 🚀 Training & Evaluation


You can modify configs in `configs/` to adjust dataset, backbone, loss weights, and optimization settings.

---

## 📊 Results

We report results on both datasets under standard metrics (CIDEr, BLEU-4, METEOR, SODA_c, and F1). For full results and ablation studies, please refer to our paper.

---

## 📄 Citation

---

## 🙏 Acknowledgements

- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) for the transformer backbone  
- [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch) for captioning implementation  
- [PDVC](https://github.com/ttengwang/PDVC) for baseline structure and evaluation tools

We thank the authors for their contributions to the community.
