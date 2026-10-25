# DDVC + normalized GT-aware dedup + Soft-NMS / merge — Molab

Branch: https://github.com/thloc1212/DDVC/tree/exp/gt-aware-dedup-softnms
Base: exp/learned-gated-proposal-mask, commit 71ad96c.

## Những gì thay đổi và không thay đổi

- **Training:** thêm loss_dedup, hệ số 0.1 từ epoch 1 (0-based); epoch 0 hệ số 0.
- **Chuẩn hoá:** lấy tổng background BCE của query dư chia cho số query dư hợp lệ.
  Không có query dư thì finite zero. Không tự L2-normalize feature hoặc gradient.
- **Gradient:** loss này học giảm event confidence của unmatched duplicate queries.
  Matched positive queries không bị phạt. Timestamp/GT/matching/eligibility detach;
  bbox/caption vẫn học bằng các loss DDVC nền, không thêm repulsion boundary loss.
- **Validation/inference:** Soft-NMS chọn output; optional merge gần trùng trước selection.
  Hai bước này không có gradient, nhưng metric mới quyết định checkpoint best.
- Model scratch GPT-2, CLIP features, retrieval bank, mask model, LR/milestones giữ nền.
- Không merge/dedup GT annotations, không dùng GT trong selection lúc inference.
- Sửa logging cardinality_error cho sigmoid; thêm counter_count_error từ pred_count.

## Dedup loss cụ thể

Với one-to-one Hungarian matching cuối decoder, một unmatched query chỉ là hard negative
cho auxiliary loss khi:
1. Confidence >=0.2;
2. Best tIoU với một GT >=0.5;
3. GT đó có matched query đại diện;
4. Query dư overlap matched representative >=0.5;
5. Không bị mơ hồ giữa hai GT: nếu top-2 GT đều IoU>=0.5 và chênh lệch <0.05 thì bỏ qua.

L_dedup = sum(softplus(z_q)) / max(1, số query dư hợp lệ).
L_total = L_DDVC + lambda_dedup * L_dedup.

Tổng loss backward như cũ, sau đó clip_grad_norm_=100 và optimizer.step.
Auxiliary này chỉ áp dụng final decoder layer, tránh nhân đôi theo số layer/pair.
Classification loss gốc vẫn tồn tại: đây là hard-negative reweighting theo redundancy,
không được trình bày như hoàn toàn mới một cơ chế unmatched-background supervision.

## Soft-NMS và merge

Có thêm mode `dedup_softnms_merge_consecutive`: xem [hướng dẫn mới](CONSECUTIVE_MERGE_MOLAB.md).
Mode này nối các selected event liên tiếp cùng caption, KHÁC overlap fusion phía dưới.

Giữ joint score gốc = proposal_score + sentence_score/(word_count^2 + 1e-5).
Gaussian temporal Soft-NMS sigma=0.5, giữ N do event counter dự đoán.
Score âm được xử lý để suppression không làm điểm tăng. Raw candidates giữ riêng.

Merge bật riêng: tIoU>=0.7 và token-set Jaccard caption>=0.9.
Mọi cặp trong nhóm phải thoả (không transitive chain).
Boundary lấy convex weighted mean bằng exp(joint_score-max_score), KHÔNG lấy union.
Caption/confidence/query_id từ representative cao điểm nhất; ghi merged_query_ids.
Không có bảo đảm merge đúng ngữ nghĩa từ token similarity. Hai sự kiện hợp lệ vẫn có
thể overlap và cùng câu; vì vậy default không merge, phải so ablation riêng.
Nếu merge làm còn ít hơn N nhóm, output có thể ít hơn N; không tạo event giả để bù.
raw_box vẫn là box gốc representative, timestamp là box fused; metadata lưu provenance.

## Các mode, cùng recipe

| Mode | Train dedup | Eval Soft-NMS | Merge |
|---|---|---|---|
| baseline | 0 | không | không |
| softnms | 0 | có | không |
| dedup_softnms (default) | 0.1 từ epoch 1 | có | không |
| dedup_softnms_merge | 0.1 từ epoch 1 | có | overlap fusion |
| dedup_softnms_merge_consecutive | 0.1 từ epoch 1 | có | overlap fusion + consecutive union |

Subset 30%: 371 train /131 val, full:1237/436; cả hai 15 epoch.
Adam LR 5e-5, WD 1e-4, LR milestones [8,11], gamma0.5, seed777.
Batch/eval1, accumulation1, FP32, TF32 off, chunk16, workers0, queries50.
Không bật Action-Aware, temporal-prefix/gate, semantic matching hoặc narrative.

## Chạy Molab mới

Giải nén ZIP vào /marimo sẽ tạo /marimo/DDVC_Dedup_SoftNMS.
Hoặc clone branch trực tiếp:

```bash
git clone --depth 1 --single-branch --branch exp/gt-aware-dedup-softnms https://github.com/thloc1212/DDVC.git /marimo/DDVC_Dedup_SoftNMS/source
```

Native môi trường Molab, không tạo venv/không thay native Torch. Chạy từng lệnh:

```bash
LC_ALL=C bash /marimo/DDVC_Dedup_SoftNMS/source/tools/dedup_molab.sh setup full dedup_softnms
bash /marimo/DDVC_Dedup_SoftNMS/source/tools/dedup_molab.sh prepare-smoke full dedup_softnms
bash /marimo/DDVC_Dedup_SoftNMS/source/tools/dedup_molab.sh smoke full dedup_softnms
bash /marimo/DDVC_Dedup_SoftNMS/source/tools/dedup_molab.sh train-smoke full dedup_softnms
```

Cần REAL_MODEL_FORWARD_BACKWARD_PASS, GENERATION_POSTPROCESS_PASS; train-smoke kiểm tra
validation/checkpoint, epoch đầu dedup weight0, epoch tiếp theo0.1. Không train full nếu lỗi.
Data tự download. Có thể chọn DDVC_DATA_ROOT để reuse assets local, không reuse config cũ.

Chuẩn bị full, bật Drive backup bên dưới trước train dài:

```bash
bash /marimo/DDVC_Dedup_SoftNMS/source/tools/dedup_molab.sh prepare-run full dedup_softnms
bash /marimo/DDVC_Dedup_SoftNMS/source/tools/dedup_molab.sh train full dedup_softnms
tail -n 80 -f /marimo/DDVC_Dedup_SoftNMS/logs/yc2_full_dedup_softnms_e15_b1.log
```

Thử merge sau, dùng cùng lệnh nhưng thay argument cuối dedup_softnms thành
**dedup_softnms_merge**; chuẩn bị và smoke cho mode đó trước.
Thay full thành subset để chạy30%. Không chạy nhiều mode đồng thời trên cùng GPU
khi so tốc độ/VRAM. Mode khác có config/run/smoke marker riêng.

## Backup Drive (cell Molab)

Dùng fs hiện có. Nếu chưa có, tạo trong một cell; tránh định nghĩa trùng globals:

```python
import marimo as mo
from gdrive_fsspec import GoogleDriveFileSystem
fs = GoogleDriveFileSystem(root_file_id='1JtVmVYfZYlxbcR96pKMhWqxODi2WkJZI', use_listings_cache=False, skip_instance_cache=True, auth_kwargs={'use_local_webserver':False})
mo.output.clear_console()
```

Cell backup:

```python
import sys as _sys
from pathlib import Path as _Path
_dedup_work = _Path('/marimo/DDVC_Dedup_SoftNMS')
_sys.path.insert(0,str(_dedup_work/'source'/'tools'))
from drive_backup import DriveBackup as _DriveBackup
dedup_backup = _DriveBackup(fs,work=_dedup_work,remote_root='runs/ddvc_dedup_softnms_e15',interval=120)
dedup_backup.start()
```

Giữ notebook hoạt động. Sau train:

```python
dedup_backup.sync_once(force=True)
```

## Kết quả và best

```bash
bash /marimo/DDVC_Dedup_SoftNMS/source/tools/dedup_molab.sh report full dedup_softnms
```

Kết quả: /marimo/DDVC_Dedup_SoftNMS/runs/yc2_full_dedup_softnms_e15_b1/.
info.json best và history.val_result_history dùng metric SAU selected method của run. Best vẫn METEOR+SODA_c.
Trong prediction/: num436_epochN.raw.json chứa mọi candidates; num436_epochN.json chứa
đúng predictions đã chấm và metric tương ứng, có metadata selection. Intermediate rerank
filename chứa method/sigma/merge. Không gắn metric cũ vào file đã hậu xử lý lại.
Không tự tính cả topk và Soft-NMS metric ở mỗi epoch (tránh nhân đôi evaluation time);
raw candidates đủ để tính đối chứng offline.

Log: loss_dedup (chưa nhân hệ số), effective weight dict, diag_dedup_queries,
diag_dedup_confidence, diag_dedup_ambiguous_skipped, counter_count_error.
Dedup có thể không active ở bước không có query đủ điều kiện; không tự tăng loss vì log0.

## Resume

Chỉ resume đúng mode/recipe đã train. Không exact-resume run cũ bằng dedup mới.

```bash
bash /marimo/DDVC_Dedup_SoftNMS/source/tools/dedup_molab.sh resume full dedup_softnms
```

Session mới: setup lại, kết nối fs, tạo backup handle nhưng chưa start;
chạy dedup_backup.restore_latest(), rồi terminal:

```bash
bash /marimo/DDVC_Dedup_SoftNMS/source/tools/dedup_molab.sh restage full dedup_softnms
bash /marimo/DDVC_Dedup_SoftNMS/source/tools/dedup_molab.sh smoke-resume full dedup_softnms
```

Bật backup worker rồi resume. Nohup không giữ process khi Molab thu hồi runtime.

## Chỉ thử selection/merge từ raw file, không train lại

```bash
/tmp/uv-venv/bin/python /marimo/DDVC_Dedup_SoftNMS/source/tools/select_predictions.py --input /duong/dan/num436_epoch11.raw.json --output /duong/dan/epoch11_softnms_merge.json --method softnms --sigma 0.5 --merge
```

Thay hai đường dẫn trên bằng file thực tế. Tool không overwrite input/output, không dùng GT;
metrics phải được evaluator tính lại. Bỏ --merge để thử chính xác Soft-NMS trước đó.

## File code

- ddvc/dedup_loss.py: normalized_dedup_loss.
- ddvc/criterion.py: dùng one-to-one matching cuối để gọi loss, sửa diagnostic count.
- ddvc/ddvc_ret_encdec_clip_capQ.py: đăng ký weight loss_dedup, chỉ final layer.
- train.py: warmup coefficient epoch0=0, epoch>=1=0.1; backward/clip/step nền giữ nguyên.
- misc/proposal_selection.py: select_proposals, merge_proposals.
- eval_utils_clip.py: reranking/evaluate chọn và lưu đúng candidate đã chấm.
- opts.py, tools/molab_pipeline.py, tools/kaggle_recipe.py: flags và drift guards.
- misc/training_state.py: không resume đổi loss/selection/merge âm thầm.

## Validation / giới hạn

CPU tests: gradient vào đúng hard-negative confidence, bảo vệ matched/ambiguous overlap GT;
normalization không tăng loss khi lặp duplicate sample; empty-set zero; actual criterion
optimizer update; actual evaluate/save path; Soft-NMS và topk parity trên 436 video ở
checkpoint11/14 đã audit; merge không union/không gộp câu khác hoặc đoạn xa;
checkpoint/resume guards; actual caption train/backward/beam/cache; config subset/full.
Chưa chạy GPU Molab full cho dedup/merge. CIDEr40.0654 là probe Soft-NMS trên checkpoint
cũ, KHÔNG phải kết quả đã đo của training dedup mới hoặc merge mới. Không hứa tăng metric.
