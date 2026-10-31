# Ablation bản 1: DDVC + learned proposal mask + Soft-NMS + merge

Repo: https://github.com/thloc1212/DDVC

Branch: `exp/v1-mask-ablation`, dựa trên `exp/activitynet-current-model` tại `6ae7bac`.
Đây là kiến trúc bản 1, trước khi thêm concept head. Giữ các sửa lỗi caption target / pooling NaN và PID guard cho ActivityNet. Không có concept head, gated temporal fusion, H1/H2/H3 hoặc dedup loss.

Không reset/xóa nhánh concept; có thể quay lại `exp/event-concept-caption-e8` sau.

## 1. Thiết kế thí nghiệm

Giữ nguyên dữ liệu, train-only retrieval bank, seed 777, decoder GPT-2 khởi tạo từ đầu, LR, scheduler, batch 1, accumulation 1, FP32, caption chunk 16 và số epoch giữa các run. CLIP features trích xuất sẵn. Dùng 15 epoch cho cả hai dataset trong bảng ablation này; YouCook2 50 queries, ActivityNet 10 queries theo cấu hình bản 1.

| ID | Mask cho caption head | Mask auxiliary loss | Cần train lại? |
|---|---|---|---|
| M0 | Global, không proposal mask | 0 | Có |
| M1 | Hard local attention, margin/min-frames giữ từ config gốc | 0 | Có |
| M2 | Learned gated proposal mask, như bản 1 | 1.0 | Có |
| M3 | Cùng learned gated mask, bỏ auxiliary mask loss | 0 | Có |

M0 là đối chứng mask trên cùng codebase, không khẳng định tái lập hoàn toàn mọi chi tiết paper DDVC. M3 vẫn học mask qua caption loss; không đóng băng mask. Gate của bản 1 lấy event confidence, không phải một gate network độc lập có thể tắt bằng cờ.

Các run trên train/eval với top-k và không merge để giữ điều kiện chọn checkpoint nhất quán. Sau đó áp dụng cùng ma trận hậu xử lý lên raw predictions của mỗi checkpoint:

| Ranking | Merge proposal overlap | Merge caption liên tiếp |
|---|---|---|
| topk / Soft-NMS | OFF / ON | OFF / ON |

Tổng cộng 8 cấu hình hậu xử lý, không cần train lại. Trong code hiện tại, overlap merge chạy **trước ranking**, consecutive merge chạy **sau selection**. Không thay thứ tự này khi so sánh.

Ưu tiên M0 và M2 trước. M1 giúp phân biệt lợi ích local context với learned mask; M3 kiểm tra vai trò mask loss. Không cần chạy 8 lần train cho Soft-NMS/merge.

## 2. Tài khoản / session Molab mới

Bật GPU, chạy trong Terminal:

```bash
nvidia-smi
git clone --single-branch --branch exp/v1-mask-ablation https://github.com/thloc1212/DDVC.git /marimo/DDVC_V1_Ablation/source
LC_ALL=C bash /marimo/DDVC_V1_Ablation/source/tools/activitynet_molab.sh setup full
```

Setup dùng native Molab Python/Torch, không tạo venv khác, ghi log vào `/marimo/DDVC_V1_Ablation/logs/`. Chưa train ở bước này. Không chạy đồng thời nhiều training job trên một GPU.

## 3. Chuẩn bị full YouCook2

```bash
cd /marimo/DDVC_V1_Ablation/source
/tmp/uv-venv/bin/python -u tools/molab_pipeline.py prepare --mode full --run-name yc2_v1_assets --epochs 15 --experiment no_dedup_same_merges --data-root /marimo/DDVC_V1_Ablation/assets/yc2 --prepared-root /marimo/DDVC_V1_Ablation/prepared --output-root /marimo/DDVC_V1_Ablation/runs --log-root /marimo/DDVC_V1_Ablation/logs > /marimo/DDVC_V1_Ablation/logs/prepare_yc2.log 2>&1
tail -n 60 /marimo/DDVC_V1_Ablation/logs/prepare_yc2.log
```

Đây là config nguồn, **không train run `yc2_v1_assets`**. Dữ liệu và bank chỉ tải/tạo một lần. Số full feature-matched mong đợi: 1237 train / 436 validation.

Tạo 4 config bằng một khối lệnh tự đủ biến vòng lặp:

```bash
cd /marimo/DDVC_V1_Ablation/source
for mask in global hard learned learned_no_aux; do
  /tmp/uv-venv/bin/python tools/v1_ablation.py prepare --source-config /marimo/DDVC_V1_Ablation/prepared/yc2_v1_assets/config.yml --output "/marimo/DDVC_V1_Ablation/prepared/yc2_${mask}_e15/config.yml" --output-root /marimo/DDVC_V1_Ablation/runs --run-name "yc2_${mask}_e15" --mask "$mask" --epochs 15 || break
done
```

Ví dụ chạy M2 từ đầu:

```bash
LC_ALL=C bash /marimo/DDVC_V1_Ablation/source/tools/v1_ablation.sh smoke /marimo/DDVC_V1_Ablation/prepared/yc2_learned_e15/config.yml
LC_ALL=C bash /marimo/DDVC_V1_Ablation/source/tools/v1_ablation.sh train /marimo/DDVC_V1_Ablation/prepared/yc2_learned_e15/config.yml
tail -n 80 -f /marimo/DDVC_V1_Ablation/logs/yc2_learned_e15.log
```

Smoke chạy forward/backward và generation của model thật, không chạy đủ epoch. Ctrl+C thoát `tail`; job train vẫn chạy nền.

Sau khi M2 xong, thay `yc2_learned_e15` bằng `yc2_global_e15`, `yc2_hard_e15`, `yc2_learned_no_aux_e15` để chạy lần lượt M0, M1, M3. Mỗi run cần smoke riêng. Không resume trọng số từ M2 sang M0/M1/M3.

## 4. ActivityNet: cùng method, dữ liệu riêng

```bash
cd /marimo/DDVC_V1_Ablation/source
/tmp/uv-venv/bin/python -u tools/activitynet_pipeline.py prepare --work /marimo/DDVC_V1_Ablation --mode full --run-name anet_v1_assets --data-root /marimo/DDVC_V1_Ablation/assets/anet > /marimo/DDVC_V1_Ablation/logs/prepare_anet.log 2>&1
tail -n 60 /marimo/DDVC_V1_Ablation/logs/prepare_anet.log
for mask in global hard learned learned_no_aux; do
  /tmp/uv-venv/bin/python tools/v1_ablation.py prepare --source-config /marimo/DDVC_V1_Ablation/prepared/anet_v1_assets/config.yml --output "/marimo/DDVC_V1_Ablation/prepared/anet_${mask}_e15/config.yml" --output-root /marimo/DDVC_V1_Ablation/runs --run-name "anet_${mask}_e15" --mask "$mask" --epochs 15 || break
done
```

Cache `assets/anet` tách khỏi `assets/yc2`: hai file cùng tên `clipvitl14.pth` không được dùng lẫn. ActivityNet mong đợi 10009 train / 4917 validation, 10 queries; giữ hai reference validation.

```bash
LC_ALL=C bash /marimo/DDVC_V1_Ablation/source/tools/v1_ablation.sh smoke /marimo/DDVC_V1_Ablation/prepared/anet_learned_e15/config.yml
LC_ALL=C bash /marimo/DDVC_V1_Ablation/source/tools/v1_ablation.sh train /marimo/DDVC_V1_Ablation/prepared/anet_learned_e15/config.yml
tail -n 80 -f /marimo/DDVC_V1_Ablation/logs/anet_learned_e15.log
```

Tương tự thay tên run để chạy ba đối chứng còn lại.

## 5. Soft-NMS / merge: dùng cùng checkpoint, không train lại

```bash
LC_ALL=C bash /marimo/DDVC_V1_Ablation/source/tools/v1_ablation.sh report /marimo/DDVC_V1_Ablation/prepared/yc2_learned_e15/config.yml
find /marimo/DDVC_V1_Ablation/runs/yc2_learned_e15 -type f -name '*.raw.json'
```

`report` in best epoch và result path. Chọn **raw của chính epoch đó**, không trộn epoch tốt nhất riêng cho mỗi hậu xử lý. Thay `RAW_BEST_EPOCH.raw.json` dưới đây bằng đường dẫn vừa tìm; không chạy nguyên placeholder:

```bash
LC_ALL=C bash /marimo/DDVC_V1_Ablation/source/tools/v1_ablation.sh postprocess /marimo/DDVC_V1_Ablation/prepared/yc2_learned_e15/config.yml /DUONG_DAN_THUC/RAW_BEST_EPOCH.raw.json /marimo/DDVC_V1_Ablation/runs/yc2_learned_e15/postprocess_ablation
LC_ALL=C bash /marimo/DDVC_V1_Ablation/source/tools/v1_ablation.sh score /marimo/DDVC_V1_Ablation/prepared/yc2_learned_e15/config.yml /marimo/DDVC_V1_Ablation/runs/yc2_learned_e15/postprocess_ablation > /marimo/DDVC_V1_Ablation/logs/yc2_learned_e15.selection.log 2>&1
cat /marimo/DDVC_V1_Ablation/runs/yc2_learned_e15/postprocess_ablation/summary.tsv
```

Script tính lại metric chính thức, không lấy metric từ `info.json` rồi gán cho prediction đã đổi. `summary.tsv` dùng thang x100 (ví dụ CIDEr 37, BLEU-4 2); `.metrics.json` giữ thang trả về từ evaluator. Java METEOR được setup ở bước 2. Lặp lại trên ANET với config/reference ANET, không dùng reference YouCook2.

Có thể làm ngay nhóm hậu xử lý trên raw prediction bản 1 cũ đã tải về mà không train mới. Cần config cũ đã restage reference đúng và raw có `pred_event_count`. Không dùng selected JSON thay raw. Kết quả validation dùng để chọn cấu hình phải được ghi rõ là validation tuning.

## 6. Đọc kết quả và backup

- M2 vs M0, cùng hậu xử lý: đóng góp learned mask.
- M2 vs M1: learned gated mask có hơn hard-local attention không.
- M2 vs M3: auxiliary mask loss có ích không. Target mask của bản 1 là cửa sổ **predicted proposal đã detach**, không phải GT segmentation mới.
- Cùng raw M2, topk vs Soft-NMS: đóng góp ranking.
- Cùng ranking, bật/tắt từng merge: đóng góp từng loại merge; theo dõi cả số event, precision/recall, CIDEr, BLEU-4, SODA_c, MetaScore.
- Khác hậu xử lý có thể đổi best epoch nếu áp dụng trong train validation. Bảng này cố định topk khi chọn checkpoint để tránh đổi đồng thời checkpoint và selection. Vì thế epoch best có thể khác run V1 cũ dùng Soft-NMS/merge khi validation.
- Seed 777 là vòng đầu; lặp thêm seed khác trước khi kết luận mức tăng nhỏ là ổn định. Không coi mọi biến thể mask là sẽ cải thiện.

Run output: `/marimo/DDVC_V1_Ablation/runs/<run-id>/`, gồm `model-best.pth`, `model-last.pth`, `info.json`, prediction final/raw, performance logs.

Molab lưu local tạm thời. Dùng hai cell Python cho Drive, không lặp lại biến global ở cell khác:

```python
import marimo as _mo
from gdrive_fsspec import GoogleDriveFileSystem as _DriveFS
v1_drive_fs = _DriveFS(use_listings_cache=False, skip_instance_cache=True,
                      auth_kwargs={"use_local_webserver": False})
_mo.output.clear_console()
```

```python
import sys as _sys
_sys.path.insert(0, '/marimo/DDVC_V1_Ablation/source/tools')
from drive_backup import DriveBackup as _DriveBackup
v1_backup = _DriveBackup(v1_drive_fs, work='/marimo/DDVC_V1_Ablation',
                        remote_root='runs/ddvc_v1_mask_ablation', interval=120)
v1_backup.start()
```

Sau mỗi run / chấm metric xong:

```python
v1_backup.sync_once(force=True)
```

Chờ `BACKUP_COMPLETE` trong `/marimo/DDVC_V1_Ablation/logs/drive_sync.log`. Backup chứa prepared/config/log/checkpoints/results, không thay thế bản source clone và không chứa toàn bộ feature cache. Source và branch ở đầu tài liệu cần được giữ cùng commit để khôi phục.
