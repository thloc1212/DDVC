# Train mới từ đầu: learned proposal mask, tắt dedup loss

Đây là recipe chính của fork https://github.com/thloc1212/DDVC, nhánh `master`.
Không phải thay đổi trong repo chính thức của tác giả.

## Cấu hình

- Full feature-matched YouCook2: 1.237 train / 436 validation; seed 777.
- 15 epoch, batch 1, accumulation 1, FP32, TF32 off; Adam LR 5e-5,
  weight decay 1e-4, giảm LR tại epoch 8/11 với gamma 0.5.
- 50 event queries; GPT-2 hai layer khởi tạo ngẫu nhiên; learned proposal mask giữ bật.
- **dedup_loss_coef=0**, không đóng băng localization/caption head.
- Soft-NMS sigma 0.5; overlap merge và consecutive merge bật ở hậu xử lý.
- CLIP ViT-L/14 video features trích xuất sẵn; CLIP text encoder dùng tạo bank,
  không fine-tune backbone CLIP. Không load checkpoint DDVC cũ.
- Dùng thư mục mới `/marimo/DDVC_NoDedup_Molab`, không đè run cũ.

Không tăng epoch/batch/LR đồng thời với thí nghiệm này. Recipe giữ giống run
no-dedup trước để so sánh. GPU/CUDA khác có thể cho số khác dù cùng seed.

## 1. Lấy source — terminal Molab

```bash
git clone --single-branch --branch master https://github.com/thloc1212/DDVC.git /marimo/DDVC_NoDedup_Molab/source
```

Nếu clone báo thư mục tồn tại thì kiểm tra dữ liệu trong đó trước; không xoá hoặc reset.
Nếu dùng ZIP source đi kèm thay cho Git: upload `DDVC_NoDedup_Molab_Source.zip`
vào `/marimo`, rồi giải nén tạo cùng đường dẫn. Không dùng đồng thời cả hai cách.

```bash
/tmp/uv-venv/bin/python -m zipfile -e /marimo/DDVC_NoDedup_Molab_Source.zip /marimo
```

## 2. Cài/build trên môi trường native Molab

```bash
LC_ALL=C bash /marimo/DDVC_NoDedup_Molab/source/tools/dedup_molab.sh setup full no_dedup_same_merges
```

Chờ thành công. Native Torch được giữ nguyên; CUDA extension phải build/test
trên GPU session mới. Không chép file .so hoặc cuda_ready từ session cũ.

## 3. Smoke ít dữ liệu: kiểm tra forward/backward thực tế

```bash
LC_ALL=C bash /marimo/DDVC_NoDedup_Molab/source/tools/dedup_molab.sh prepare-smoke full no_dedup_same_merges
LC_ALL=C bash /marimo/DDVC_NoDedup_Molab/source/tools/dedup_molab.sh smoke full no_dedup_same_merges
```

Chờ REAL_MODEL_FORWARD_BACKWARD_PASS và GENERATION_POSTPROCESS_PASS. Không cần
train thêm smoke hai epoch. Bước smoke tạo dữ liệu/CLIP bank nhỏ riêng; bước full
bên dưới tạo đủ dữ liệu và bank mới từ annotations, không cần restore prepared cũ.

## 4. Chuẩn bị full data, kiểm tra recipe rồi train

```bash
LC_ALL=C bash /marimo/DDVC_NoDedup_Molab/source/tools/dedup_molab.sh prepare-run full no_dedup_same_merges
```

Chờ `CONFIG_READY` và `KAGGLE_RECIPE_OK`. Kiểm tra quyết định train từ đầu:

```bash
/tmp/uv-venv/bin/python - <<'PY'
from pathlib import Path
import yaml
p=Path('/marimo/DDVC_NoDedup_Molab/prepared/yc2_full_no_dedup_same_merges_e15_b1/config.yml')
c=yaml.safe_load(p.read_text())
assert c['dedup_loss_coef']==0
assert not c.get('start_from') and not c.get('pretrain')
assert not c.get('only_ft_captioner') and not c.get('only_ft_class_head')
print('FRESH_TRAIN_ALL_DDVC_PASS',c['epoch'],'epochs',c['lr'],'LR')
PY
```

Bật backup theo mục 5 trước hoặc ngay sau khi khởi động training:

```bash
LC_ALL=C bash /marimo/DDVC_NoDedup_Molab/source/tools/dedup_molab.sh train full no_dedup_same_merges
```

Launcher chạy nền và từ chối `train` nếu thư mục run đã có checkpoint. Không tự
resume hoặc ghi đè run cũ. Nếu session bị ngắt, restore và dùng `resume` có chủ đích.

## 5. Backup Google Drive — đúng một cell Python

Biến tạm dấu `_` chỉ dùng trong cell này; `no_dedup_backup` là handle dùng cho
cell sync sau. Nếu đã có cell định nghĩa no_dedup_backup thì sửa cell đó.
Kết nối dưới dùng gốc Drive mặc định, giống nơi đã tìm được READY của run trước.
Không thêm root_file_id ở phiên restore trừ khi cả backup/restore cùng dùng nó.

```python
import sys as _fresh_sys
from pathlib import Path as _FreshPath
from gdrive_fsspec import GoogleDriveFileSystem as _FreshFS

_fresh_fs = _FreshFS(
    use_listings_cache=False, skip_instance_cache=True,
    auth_kwargs={"use_local_webserver": False},
)
_fresh_work = _FreshPath('/marimo/DDVC_NoDedup_Molab')
_fresh_sys.path.insert(0, str(_fresh_work / 'source' / 'tools'))
from drive_backup import DriveBackup as _FreshBackup

no_dedup_backup = _FreshBackup(
    _fresh_fs, work=_fresh_work,
    remote_root='runs/ddvc_no_dedup_fresh_e15', interval=120,
)
no_dedup_backup.start()
```

Giữ notebook/kernel hoạt động để worker chạy. Worker chỉ bắt đầu upload khi
đã có model-last.pth. Poll 120 giây không có nghĩa upload xong trong 120 giây.

## 6. Xem tiến độ và kết quả

```bash
tail -n 80 -f /marimo/DDVC_NoDedup_Molab/logs/yc2_full_no_dedup_same_merges_e15_b1.log
```

Ctrl+C thoát tail, không dừng train chạy nền.

```bash
LC_ALL=C bash /marimo/DDVC_NoDedup_Molab/source/tools/dedup_molab.sh report full no_dedup_same_merges
tail -n 30 /marimo/DDVC_NoDedup_Molab/logs/drive_sync.log
```

Kết quả: `/marimo/DDVC_NoDedup_Molab/runs/yc2_full_no_dedup_same_merges_e15_b1/`.
Có model-best.pth, model-last.pth, info.json, performance.jsonl và prediction.
Best chọn bằng METEOR + SODA_c; max CIDEr có thể ở epoch khác. Saved prediction
là output đã selection/merge thực sự dùng tính metric; `.raw.json` chứa các candidates.

Khi train xong, cell mới:

```python
no_dedup_backup.sync_once(force=True)
```

Chờ BACKUP_COMPLETE trong drive_sync.log và xác nhận READY trên Drive. Backup
bao gồm prepared (annotations/vocabulary/bank/config), runs và logs; video features
lớn có thể tải lại. Source được lấy riêng qua Git hoặc ZIP, không nằm trong worker backup.
Không chỉ tải mỗi thư mục runs nếu muốn restore đủ dữ liệu cho session mới.

## Khôi phục sau session loss

Lấy lại cùng commit source, chạy setup. Dùng cùng cell tạo kết nối và handle backup
nhưng **thay `.start()` bằng `.restore_latest()`** ngay trong chính cell đó.
Chưa start worker khi restore. Tải lại features:

```bash
LC_ALL=C bash /marimo/DDVC_NoDedup_Molab/source/tools/dedup_molab.sh restage full no_dedup_same_merges
LC_ALL=C bash /marimo/DDVC_NoDedup_Molab/source/tools/dedup_molab.sh smoke-resume full no_dedup_same_merges
```

Sau smoke thành công, bật worker `.start()`, rồi:

```bash
LC_ALL=C bash /marimo/DDVC_NoDedup_Molab/source/tools/dedup_molab.sh resume full no_dedup_same_merges
```

Không đổi mode/LR/loss khi resume. Đây là khôi phục optimizer/scheduler/RNG từ
checkpoint; không chỉ load weights rồi reset lịch train.
