# DDVC C1 — online joint-quality training experiment

Branch: [`exp/c1-online-joint-quality`](https://github.com/thloc1212/DDVC/tree/exp/c1-online-joint-quality).
Fixed tag: `online-quality-joint-20261110`.

Train C1 and its 2312D-input quality scorer together from scratch, then calibrate
beta on TRAIN head-dev and evaluate Top-N/Soft-NMS controls on identical saved
validation candidates. The quality scorer retains the frozen-reranker architecture;
it does not read cached candidates from an older C1 run or ground-truth captions.

- YouCook2: 1237 train /436 validation, **15 epochs, Q50**.
- ActivityNet: 10009 train /4917 validation, **6 epochs, Q10**.
- C1 learned proposal mask, **mask auxiliary loss0**, no concepts/relation/count
  calibration/merge; seed777, batch1, accumulation1, FP32, TF32off.
- C1 Adam LR5e-5; quality LR.001, loss weight.1. GPT-2 starts randomly.
- `joint` mode sends quality gradients through confidence/duration/lexical
  embeddings. `detached` is a separate fresh control for that feedback.
- Generation and hard temporal membership remain discrete; no claim that
  beam search or Soft-NMS is differentiable. No new full-run metric is claimed.

**Start here: [two fresh Molab accounts, Drive cache/backup/resume and results](ONLINE_QUALITY_MOLAB.md).**
Use `tools/online_quality_drive.sh`; older C1/C2/frozen-reranker guides are retained
for provenance, not the launcher/config of this experiment.
This is an experimental fork; original attribution follows.

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
