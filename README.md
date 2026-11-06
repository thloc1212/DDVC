# DDVC — final C2

Branch: [`final/c2-learned-mask`](https://github.com/thloc1212/DDVC/tree/final/c2-learned-mask).

Finalized method: DDVC + confidence-gated learned proposal mask + auxiliary
mask BCE (weight **1.0**), followed by counter-guided Gaussian **Soft-NMS (sigma 0.5)**.
Concept heads/loss/prefix (C3–C5), deduplication loss and merge are removed.
The previous ablation implementation remains on
[`exp/v1-v2-contribution-ablation`](https://github.com/thloc1212/DDVC/tree/exp/v1-v2-contribution-ablation).
This is an experimental fork; the original project attribution is preserved below.

- YouCook2: full feature-matched 1237 train / 436 val, **15 epochs, 50 queries**.
- ActivityNet Captions: 10009 train / 4917 val, **8 epochs, 10 queries**.
- Both: seed 777, batch 1, accumulation 1, FP32, TF32 off, Adam LR 5e-5.
- GPT-2 is randomly initialized; pre-extracted CLIP features remain frozen.
- C2 active-path model math and initialization are retained. No new scores are claimed.

Start here: [Molab commands, outputs, resume and Drive backup](docs/C2_FINAL_MOLAB.md).
Architecture and source attribution: [Learned proposal mask](LEARNED_PROPOSAL_MASK.md).
There is one current launcher: `tools/c2_molab.sh`; no C0–C5 queue or concept modes.

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
