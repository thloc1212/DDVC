# Concept loss only trên hai tài khoản Molab mới

Thử nghiệm này giữ Bản 1 và concept auxiliary loss, **không đưa concept embedding vào GPT-2 prefix**. Đây không phải train lại Bản 2 đầy đủ. Mục đích: tách tác động của semantic supervision khỏi prefix conditioning. Chưa có metric chứng minh biến thể này tốt hơn.

| Tài khoản | Dataset | Epoch | Run ID |
|---|---|---:|---|
| A | YouCook2 full | 15 | yc2_full_concept_aux_e15_b1 |
| B | ActivityNet full | 8 | anet_full_concept_aux_e8_b1 |

Mode `aux` là biến thể concept, không phải subset. Cùng model/loss trên hai dataset, concept vocabulary được fit riêng từ train. Giữ learned proposal mask và mask loss của Bản 1; dedup OFF; Soft-NMS/cả hai merge ON. Không kết hợp thêm thay đổi M3 (tắt mask loss) vào lần thử này. Decoder GPT-2 khởi tạo từ đầu, pre-extracted CLIP features; batch 1, accumulation 1, FP32, seed 777, concept BCE coefficient 0.1. Queries: YouCook2 50, ActivityNet 10.

## Setup chung trên mỗi tài khoản

Bật GPU và Internet. Chạy trong Terminal trên **cả hai acc**, không cần runtime hoặc data từ session cũ:

```bash
nvidia-smi
git clone --single-branch --branch exp/event-concept-caption-e8 https://github.com/thloc1212/DDVC.git /marimo/DDVC_Concept_Aux/source
git -C /marimo/DDVC_Concept_Aux/source log -1 --oneline
LC_ALL=C bash /marimo/DDVC_Concept_Aux/source/tools/concept_molab.sh setup yc2 aux
```

Setup sử dụng môi trường Molab native, không tạo venv mới hoặc thay GPU Torch. Argument `yc2` ở bước setup không tải data; bước này dùng được cho cả hai acc. Có log setup trong `/marimo/DDVC_Concept_Aux/logs/`. Nhánh đã có sửa caption-target rỗng, temporal pooling rỗng và chốt gradient NaN/Inf trước optimizer update. Lỗi không bị biến thành loss 0 hoặc tự skip video.

## Acc A chạy YouCook2

```bash
DDVC_CONCEPT_EPOCHS=15 LC_ALL=C bash /marimo/DDVC_Concept_Aux/source/tools/concept_molab.sh prepare yc2 aux
DDVC_CONCEPT_EPOCHS=15 LC_ALL=C bash /marimo/DDVC_Concept_Aux/source/tools/concept_molab.sh smoke yc2 aux
```

Prepare tự tải features khoảng 872 MiB và annotations, tạo train-only bank và concept vocabulary. Mong đợi 1237 train / 436 validation; `variant=aux`; 15 epochs. Đợi prepare xong và smoke pass (`CONCEPT_GRADIENT_PASS`, `REAL_MODEL_FORWARD_BACKWARD_PASS`, `GENERATION_POSTPROCESS_PASS`) rồi mới train:

```bash
DDVC_CONCEPT_EPOCHS=15 LC_ALL=C bash /marimo/DDVC_Concept_Aux/source/tools/concept_molab.sh train yc2 aux
tail -n 80 -f /marimo/DDVC_Concept_Aux/logs/yc2_full_concept_aux_e15_b1.log
```

Kết quả: `/marimo/DDVC_Concept_Aux/runs/yc2_full_concept_aux_e15_b1/`.

```bash
DDVC_CONCEPT_EPOCHS=15 LC_ALL=C bash /marimo/DDVC_Concept_Aux/source/tools/concept_molab.sh report yc2 aux
```

## Acc B chạy ActivityNet

Sau setup chung:

```bash
DDVC_CONCEPT_EPOCHS=8 LC_ALL=C bash /marimo/DDVC_Concept_Aux/source/tools/concept_molab.sh prepare anet aux
DDVC_CONCEPT_EPOCHS=8 LC_ALL=C bash /marimo/DDVC_Concept_Aux/source/tools/concept_molab.sh smoke anet aux
```

Features ActivityNet khoảng 2.52 GiB, cần chờ tải/tạo bank train. Mong đợi 10009 train / 4917 validation; 10 queries; 8 epochs. Chạy khi smoke pass:

```bash
DDVC_CONCEPT_EPOCHS=8 LC_ALL=C bash /marimo/DDVC_Concept_Aux/source/tools/concept_molab.sh train anet aux
tail -n 80 -f /marimo/DDVC_Concept_Aux/logs/anet_full_concept_aux_e8_b1.log
```

Kết quả: `/marimo/DDVC_Concept_Aux/runs/anet_full_concept_aux_e8_b1/`.

```bash
DDVC_CONCEPT_EPOCHS=8 LC_ALL=C bash /marimo/DDVC_Concept_Aux/source/tools/concept_molab.sh report anet aux
```

Train chạy nền với nohup. Ctrl+C thoát xem tail không dừng training. Nohup không giữ compute sống khi session Molab bị hủy. Không nhấn train nhiều lần; launcher có PID guard.

## Backup Google Drive trước khi train

Hai cell Python notebook, không dán trực tiếp vào Bash. Dùng remote root khác nhau cho hai tài khoản. Kernel notebook cần còn sống để worker backup hoạt động.

Cell kết nối trên mỗi acc:

```python
import marimo as _mo
from gdrive_fsspec import GoogleDriveFileSystem as _DriveFS
concept_aux_drive_fs = _DriveFS(use_listings_cache=False, skip_instance_cache=True,
                              auth_kwargs={"use_local_webserver": False})
_mo.output.clear_console()
```

Cell backup Acc A:

```python
import sys as _sys
_sys.path.insert(0, '/marimo/DDVC_Concept_Aux/source/tools')
from drive_backup import DriveBackup as _DriveBackup
yc2_aux_backup = _DriveBackup(concept_aux_drive_fs, work='/marimo/DDVC_Concept_Aux',
                            remote_root='runs/ddvc_concept_aux_yc2_e15', interval=120)
yc2_aux_backup.start()
```

Cell backup Acc B:

```python
import sys as _sys
_sys.path.insert(0, '/marimo/DDVC_Concept_Aux/source/tools')
from drive_backup import DriveBackup as _DriveBackup
anet_aux_backup = _DriveBackup(concept_aux_drive_fs, work='/marimo/DDVC_Concept_Aux',
                             remote_root='runs/ddvc_concept_aux_anet_e8', interval=120)
anet_aux_backup.start()
```

Sau run, gọi `yc2_aux_backup.sync_once(force=True)` ở acc A hoặc `anet_aux_backup.sync_once(force=True)` ở acc B. Chờ `BACKUP_COMPLETE` trong `/marimo/DDVC_Concept_Aux/logs/drive_sync.log` trước khi kết thúc session.

Backup giữ checkpoint/config/annotation/log/results, không giữ toàn bộ feature cache và source. Khôi phục session khác cần clone đúng commit, setup runtime và restage data. Ghi lại `git rev-parse HEAD` cùng kết quả.

## Diễn giải kết quả

So theo từng dataset với Bản 1 và Bản 2, cùng decoding/hậu xử lý. ANET 8 epoch chưa là đối chứng cùng budget với Bản 1 ANET 15 epoch; báo cáo rõ khác biệt hoặc bổ sung control 8 epoch sau. Không mặc định biến thể mới sẽ giữ nguyên YouCook2 hoặc tăng ActivityNet.

Nhánh source: https://github.com/thloc1212/DDVC/tree/exp/event-concept-caption-e8
