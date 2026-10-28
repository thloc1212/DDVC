# DDVC + event concepts — Molab, train mới 8 epochs

**Cập nhật:** ActivityNet mặc định 8 epochs; YouCook2 mặc định **15 epochs** theo
yêu cầu mới. Hướng dẫn YouCook2 đầy đủ: [YC2_CONCEPT_E15_MOLAB.md](YC2_CONCEPT_E15_MOLAB.md).
Module/model/loss dùng chung; epoch budget khác theo dataset.

Branch: https://github.com/thloc1212/DDVC/tree/exp/event-concept-caption-e8

Đây là thí nghiệm mới, chưa có metric chứng minh tăng điểm. Dùng cùng module,
loss coefficient và cách conditioning cho YouCook2/ActivityNet. Giữ CLIP features
trích xuất sẵn và GPT-2 khởi tạo mới; không tải pretrained GPT-2 weights.

## Thay đổi chính

1. Từ **caption của tập train**, tạo tối đa 512 word concepts, xuất hiện trong ít
   nhất 5 captions; bỏ stop words. Không đọc validation/test khi fit vocabulary.
   Đây là content words, không phải bộ nhãn action/object có POS annotation.
2. Với mỗi event: nối caption query với CLIP visual feature được mean-pool trong
   **predicted interval**. Có fallback 1 frame cho interval ngắn; bỏ padding.
3. MLP dự đoán nhiều concept. Sau Hungarian matching, lấy concept words của GT
   caption tương ứng làm target weighted BCE; coefficient **0.1**, chỉ final layer.
4. Lấy predicted probabilities để tổng hợp concept embedding và chiếu vào GPT-2
   prefix. Projection khởi tạo zero; GT concept **không** được đưa vào prefix.
5. Áp dụng conditioning trong cả teacher-forced forward, all-pair caption-cost
   chunks và generation. Không đổi thuật toán Hungarian/caption-cost chunk size.

Không đổi mask, retrieval, Soft-NMS/merge, LR, contrastive temperature/weight,
query count hay công thức loss cũ. Không sửa đồng thời các chênh lệch contrastive
đã phát hiện, để cô lập đóng góp concept head.

Ý tưởng liên quan: [MCCL, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32948)
dùng frame-level concept cues và cyclic co-learning. Bản này là event-query
concept supervision/prefix conditioning riêng, **không tái lập toàn bộ MCCL**.

## Cấu hình mặc định

| Setting | Giá trị |
|---|---|
| Training | Fresh model + fresh optimizer; toàn bộ DDVC học |
| Data | Full feature-matched splits của baseline |
| Epochs | **8**, không resume checkpoint 15 epoch cũ |
| Seed / batch / accumulation | 777 / 1 / 1 |
| Precision | FP32, TF32 off như run trước |
| Adam LR / weight decay | 5e-5 / 1e-4 như baseline |
| Queries | ANET 10; YouCook2 50 |
| New loss | weighted concept BCE × 0.1 |
| Concept head | hidden 256; concept embedding 128 |
| Decoder | GPT-2 random, 2 layers; max caption length/beam giữ config cũ |

Giữ lịch LR cũ bắt đầu giảm ở epoch index 8; run 8 epochs (index 0–7) chưa đến
mốc giảm đó. Không tự đưa kết quả decode grid tốt nhất vào config training mới.
Best vẫn chọn bằng METEOR + SODA_c, không mặc định lấy epoch cuối.

## A. Phiên ActivityNet vừa chạy vẫn còn source/data

Chạy từng khối trong Terminal Molab. Không cần thiết lập `$WORK`, `$PY` hay `$RUN`.
Các lệnh dùng folder mới, không switch/ghi đè source của run baseline.

### 1. Clone branch

```bash
git clone --single-branch --branch exp/event-concept-caption-e8 https://github.com/thloc1212/DDVC.git /marimo/DDVC_Concept_E8/source
```

### 2. Tái sử dụng môi trường và extension đang chạy được

```bash
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh reuse-runtime anet full /marimo/DDVC_ActivityNet_Fresh
```

Không cài lại Torch. Sao chép `.so`, áp dụng runtime patches đã có và chạy test
CUDA thật trước khi ghi `cuda_ready`. Các đường private toolkit/cache có thể còn
trỏ vào work cũ, vì vậy không xoá `/marimo/DDVC_ActivityNet_Fresh` trong phiên này.

### 3. Chuẩn bị config 8 epoch; tái sử dụng data/bank

```bash
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh prepare anet full /marimo/DDVC_ActivityNet_Fresh/prepared/anet_full_current_model_q10_e15_b1/config.yml
```

Copy/hardlink input files vào prepared folder của thí nghiệm mới. Không lấy
`model-best.pth` hay optimizer của run cũ. Chờ `CONCEPT_CONFIG_READY`.

### 4. Smoke trên 2 training videos đã chọn, không optimizer update

```bash
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh smoke anet full
```

Phải thấy `CONCEPT_GRADIENT_PASS`, `REAL_MODEL_FORWARD_BACKWARD_PASS` và
`GENERATION_POSTPROCESS_PASS`. Smoke dùng full train-only bank/concept vocabulary,
nhưng chỉ lấy minibatch đầu để forward/backward và generation, không loop full train.

### 5. Train mới 8 epoch

```bash
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh train anet full
```

```bash
tail -n 80 -f /marimo/DDVC_Concept_E8/logs/anet_full_event_concept_e8_b1.log
```

`Ctrl+C` chỉ thoát tail, training nohup vẫn tiếp tục. Không launch lệnh train lần
hai khi PID đang sống. Thấy `FRESH_CONCEPT_TRAINING epochs=8` xác nhận train mới.

### 6. Xem kết quả

```bash
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh report anet full
```

Kết quả tại `/marimo/DDVC_Concept_E8/runs/anet_full_event_concept_e8_b1/`:

- `model-best.pth`, `model-last.pth`: best và last, có optimizer/RNG/config để resume.
- `info.json`: best epoch và lịch sử metric validation.
- `prediction/`: final predictions dùng đo metric và `.raw.json` tách riêng.
- `performance.jsonl`: thời gian/GPU memory của các epoch.

Prepared config/vocabulary nằm trong
`/marimo/DDVC_Concept_E8/prepared/anet_full_event_concept_e8_b1/`.
Log có `loss_event_concept`, `concept_target_coverage`, `concept_mean_probability`,
`concept_prefix_norm`, `concept_prefix_ratio`, `concept_top10_recall`; prefix norm
ban đầu bằng 0 là chủ ý khởi tạo.

## B. Phiên Molab mới hoàn toàn

Clone như bước A1, rồi dùng setup thay vì reuse-runtime:

```bash
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh setup anet full
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh prepare anet full
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh smoke anet full
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh train anet full
```

Prepare tự tải ANET CLIP features (~2.52 GiB), annotations, và tạo train-only bank.
Prepared base-assets config có epoch 15 chỉ là nguồn dữ liệu cũ, **không được train**;
run thật ở `anet_full_event_concept_e8_b1` luôn có 8 epochs.
Setup dùng native Molab environment, giữ GPU Torch; không cần root/venv mới.

## C. Chạy cùng method trên YouCook2

Trong cùng work đã setup, đổi dataset thành `yc2`. Không cần chạy lại setup CUDA:

```bash
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh prepare yc2 full
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh smoke yc2 full
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh train yc2 full
```

YouCook2 features ~872 MiB; bank/vocabulary riêng từ train YouCook2. Run 15 epochs,
50 queries, cùng module/hyperparameters concept. Nếu muốn reuse data YC2 đã có,
thêm đường tuyệt đối config YC2 sau `prepare yc2 full`, như ví dụ ANET ở A3.
Chạy hai dataset tuần tự để dễ theo dõi GPU/log.

```bash
tail -n 80 -f /marimo/DDVC_Concept_E8/logs/yc2_full_event_concept_e15_b1.log
```

## D. Ablation thống nhất trên cả hai dataset

- `full`: concept loss + predicted concept prefix, mặc định.
- `aux`: chỉ concept loss, không cộng concept vào prefix.
- `control`: baseline cùng codebase, không concept head/loss.

Ví dụ đối chứng ANET 8 epochs, reuse data từ run baseline:

```bash
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh prepare anet control /marimo/DDVC_ActivityNet_Fresh/prepared/anet_full_current_model_q10_e15_b1/config.yml
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh smoke anet control
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh train anet control
```

Các variant có run ID/log/checkpoint riêng: ANET 8, YouCook2 15 epochs mặc định.
So sánh method/control trong cùng dataset với cùng budget; báo cáo best theo cùng tiêu chí.

## E. Google Drive backup

Kết nối Drive qua Files > Remote storage. Nếu đã có `fs` được tạo bằng
`GoogleDriveFileSystem`, dùng lại. Không tạo mount path giả. Cell sau dùng tất cả
biến tạm trong cùng cell, chỉ handle `concept_backup` dùng qua cell khác:

```python
import sys as _sys
from pathlib import Path as _Path
_work = _Path('/marimo/DDVC_Concept_E8')
_sys.path.insert(0, str(_work / 'source' / 'tools'))
from drive_backup import DriveBackup as _DriveBackup
concept_backup = _DriveBackup(
    fs,
    work=_work,
    remote_root='runs/ddvc_event_concept_e8',
    interval=120,
)
concept_backup.start()
```

Nếu chưa có kết nối, tạo **một cell riêng** theo mã Molab cung cấp:

```python
from gdrive_fsspec import GoogleDriveFileSystem
fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
    auth_kwargs={"use_local_webserver": False})
mo.output.clear_console()
```

`import marimo as mo` chỉ định nghĩa một lần ở cell riêng nếu notebook chưa có.
Backup gồm prepared config/concept vocabulary/bank, logs, runs và checkpoints;
features `.pth` tải lại được nên không upload. Giữ notebook kernel hoạt động.
Sau train, chạy trong cell:

```python
concept_backup.sync_once(force=True)
```

Chờ `BACKUP_COMPLETE` trong `/marimo/DDVC_Concept_E8/logs/drive_sync.log`.

## F. Tiếp tục chính run mới nếu session bị gián đoạn

Khôi phục đúng source/config/vocabulary/bank/checkpoint, restage feature file tại
đúng đường config nếu cần; setup CUDA trên runtime mới. Sau đó:

```bash
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh restage anet full
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh smoke-resume anet full
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh resume anet full
```

Resume kiểm tra architecture/data/concept vocabulary hash. Không resume từ model
baseline không có concept head, và không đổi variant/vocabulary khi resume.

## Source cần theo dõi

- `misc/event_concepts.py`: vocabulary chỉ từ train, tokenizer content words.
- `ddvc/CaptioningHead/event_concepts.py`: visual pooling, classifier, BCE, prefix delta.
- `ddvc/CaptioningHead/gpt2.py`: cộng predicted concept vào prefix khi train/generate.
- `ddvc/ddvc_ret_encdec_clip_capQ.py`: concept targets sau matching, đăng ký loss weight.
- `tools/concept_pipeline.py`: staging, provenance/config/vocabulary signatures, 8 epochs.
- `tools/concept_molab.sh`: native setup/reuse, smoke, nohup training/log/report/resume.

Không có kết quả GPU training/metric mới trong bundle. Smoke trên Molab là điều
kiện bắt buộc trước full train. Việc concept cues có tăng CIDEr/BLEU/SODA hay không
phải được đo, không suy ra từ việc loss train giảm.

Kiểm tra local đã qua: vocabulary/padding/sub-frame pooling; classifier/query/prefix
gradients; actual vendored GPT-2 forward/generate với tiny offline test vocabulary;
zero-init giữ nguyên common weights/logits của baseline; caption-cost chunk parity;
strict state reload; config preparation cả 2 dataset × 3 variants; legacy source
identity khi bỏ các block opt-in. Các regression NaN caption/pooling cũ cũng pass.
Trên annotations train đã phân tích, vocab 512 có ít nhất một concept cho 99.06%
captions ANET và 99.93% captions YC2; đây là label coverage, không phải model accuracy.
