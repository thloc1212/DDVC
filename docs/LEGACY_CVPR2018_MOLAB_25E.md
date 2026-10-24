# DDVC + CVPR 2018 masking cũ — Molab, full data, 25 epoch

## 1. Đúng bản nào?

Bản này dành cho `DDVC_CVPR2018_Masking_Source_Notebook.zip` ban đầu.

- [Branch cũ](https://github.com/thloc1212/DDVC/tree/exp/learned-gated-proposal-mask), archive commit `bcdd683565c825369a4f03d6a25bfb18234b388c`.
- Model/notebook training source: `4918a0cb89bac0966f74c39e7320d44f0a780430`.
- [Branch Molab mới](https://github.com/thloc1212/DDVC/tree/exp/legacy-cvpr2018-mask-molab).
- [DDVC, ACL 2025](https://aclanthology.org/2025.acl-long.807/).
- [Masked Transformer, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_End-to-End_Dense_Video_CVPR_2018_paper.html).
- [Repo masking tham khảo](https://github.com/bhubanendra-mishra/dense-video-cap/tree/43914e17769701b9cf98eda203ae4c465b315fab).

Đây là bản thích nghi module masking vào DDVC, không phải toàn bộ kiến trúc CVPR 2018.
Commit chính xác của gói mới nằm trong `BUNDLE_MANIFEST.json`.

### Recipe giữ lại / thay đổi

| Thiết lập | ZIP Kaggle cũ | Molab này |
|---|---|---|
| Learned mask model + GPT-2 + DDVC core | Legacy learned gated mask | Giữ nguyên source model |
| Mask target | Predicted interval | Giữ nguyên; không matched-GT |
| Alpha | Event confidence trực tiếp | Giữ nguyên [0,1]; không bounded gate |
| Query conditioning / pruning / balanced BCE | Không | Không |
| Query contrastive | Legacy | Legacy; không bật bản corrected |
| Data mặc định | Subset 30% | Full feature-matched 1237/436 |
| Epoch full | 15 | **25 theo yêu cầu** |
| Batch thật / accumulation | 1 / 1 | **1 / 1** |
| Precision | Không autocast | **FP32, TF32 tắt** |
| Caption chunk / bank encoding batch | 16 / 64 | 16 / 64 |
| Workers | 0 | 0 |
| Seed / queries | 777 / 50 | 777 / 50 |
| GPT-2 | 2 layer, train từ đầu | Giữ nguyên |
| LR / optimizer / weight decay | 5e-5 / Adam / 1e-4 | Giữ nguyên |
| LR decay | Công thức MultiStepLR của source | Giữ nguyên công thức với budget 25 epoch |

Với full data và 25 epoch có 30.925 optimizer updates. Tăng epoch từ bản ZIP 15 lên 25 là thay đổi có chủ ý. Không hứa metric trùng R15 vì khác budget, GPU, Python/Torch và CUDA; không gọi đây là tái lập bitwise.

## 2. Những sửa Molab đã giữ lại

- Dùng native interpreter Molab, không tạo `.venv`, không cài đè native Torch.
- CUDA compiler/header 13.0 riêng khi native Torch là cu130; không trộn nvcc 13.3 với runtime headers 13.0. Script có header/compiler compile probe rồi CUDA forward/backward test.
- Tự tìm CCCL, lỗi import extension được báo rõ; không `except: pass` che lỗi gốc.
- Smoke nạp parser defaults như train, không đọc YAML thành namespace thiếu thuộc tính.
- Backup source không bắt buộc `cfgs`; lưu `effective_config.json`, bỏ build/cache artifacts.
- Atomic format-2 checkpoint có optimizer/scheduler/RNG, metadata chuyển khỏi lambda-defaultdict để tránh lỗi pickle.
- Resume phân biệt epoch_complete và validation_pending, giữ data signature và kiểm tra recipe.
- Log unbuffered, per-epoch performance, prediction raw/final tách rõ, bỏ ActivityNet grounding ở YouCook2 như notebook cũ.
- Tokenizer public GPT-2 và compatibility patches giống notebook; CUDA `.scalar_type()`/`.data_ptr()`.

Các sửa này phục vụ khả năng chạy/lưu/resume. Không thêm kiến trúc, loss hay supervision từ nhánh Action-Aware.

## 3. Link cố định của bạn

- [Code ZIP hiện tại](https://drive.google.com/file/d/1JmJL2hVEfnqkIhUak-ol5EMM9XNBBXjE/view?usp=drive_link).
- [Folder Drive chứa code lâu dài](https://drive.google.com/drive/folders/1JtVmVYfZYlxbcR96pKMhWqxODi2WkJZI).
- File ID: `1JmJL2hVEfnqkIhUak-ol5EMM9XNBBXjE`.
- Folder ID: `1JtVmVYfZYlxbcR96pKMhWqxODi2WkJZI`.

ZIP hiện tại không cần tạo lại: phần hướng dẫn mới chỉ đổi nguồn tải và nơi backup, không đổi code/model hay recipe trong ZIP.
Local work riêng: `/marimo/DDVC_CVPR2018_Molab`. Không ghi đè run RTX/Action-Aware cũ.

## 4. Tải code — terminal Molab

Chọn GPU. Các lệnh dùng native interpreter đã thấy trong log Molab `/tmp/uv-venv/bin/python`; không tạo venv mới. Nếu runtime đổi đường dẫn interpreter thì lấy `sys.executable` của notebook và đặt `DDVC_PYTHON` theo đường dẫn thật khi gọi launcher.

```bash
uv pip install --python /tmp/uv-venv/bin/python gdown
```

```bash
/tmp/uv-venv/bin/python -m gdown '1JmJL2hVEfnqkIhUak-ol5EMM9XNBBXjE' -O /marimo/DDVC_CVPR2018_Masking_Molab_25E.zip
```

Không dùng `--fuzzy`. Nếu file private và gdown bị từ chối quyền, dùng kết nối `legacy_fs` ở bước 8 rồi `legacy_fs.ls('')` để xác nhận tên file; tải bằng fs.get theo tên thực tế trong folder, không đổi quyền folder backup thành public.

```bash
/tmp/uv-venv/bin/python -m zipfile -t /marimo/DDVC_CVPR2018_Masking_Molab_25E.zip
```

Giải nén lần đầu; nếu source đã có thì giữ nguyên:

```bash
if [ -d /marimo/DDVC_CVPR2018_Molab/source ]; then
  echo 'Source đã tồn tại; không giải nén đè.'
else
  /tmp/uv-venv/bin/python -m zipfile -e /marimo/DDVC_CVPR2018_Masking_Molab_25E.zip /marimo
fi
```

```bash
/tmp/uv-venv/bin/python -c 'import json; from pathlib import Path; m=json.loads(Path("/marimo/DDVC_CVPR2018_Molab/BUNDLE_MANIFEST.json").read_text()); assert m["source_commit"]=="c385e6641c5ead3f0371f79141a0d773762aa71f"; print("LEGACY_PACKAGE_OK",m["source_commit"])'
```

Phải có `LEGACY_PACKAGE_OK`. Mỗi bản ZIP tương lai nên có tên phiên bản/commit riêng trong folder Drive; không giải nén đè source đang train.

## 5. Setup thư viện và CUDA — terminal

Nếu dùng ZIP source c385e66 đã upload, chuẩn hoá launcher từ CRLF về LF trước (không đổi code hay recipe):

```bash
/tmp/uv-venv/bin/python -c 'from pathlib import Path; p=Path("/marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh"); p.write_bytes(p.read_bytes().replace(b"\r\n",b"\n")); print("LAUNCHER_LF_OK")'
```

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh setup
```

Lệnh tự nạp đường dẫn, không cần khai báo WORK/RUN hoặc source env cũ. Nó cài native dependencies, giữ Torch/torchvision, patch tokenizer/CUDA API, xử lý CCCL/compiler/header cu130, kiểm tra CUDA extension. Log vừa hiện trên terminal vừa ghi vào `/marimo/DDVC_CVPR2018_Molab/logs/`.

Với native Torch cu130, toolkit ~105 MB ở `toolchains/cuda130-pinned` là compiler assets, không phải venv. Nếu setup lỗi, script dừng; không chạy bước sau. Cài một lần cho runtime hiện tại, không bảo đảm tồn tại sau Molab reset.

## 6. Smoke model legacy — terminal

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh prepare-smoke
```

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh smoke
```

Data tự tải annotation + CLIP features ~0.9 GB + CLIP weights tạo train-only bank. Smoke dùng 12 train/4 validation. Cần `REAL_MODEL_FORWARD_BACKWARD_PASS` và `GENERATION_POSTPROCESS_PASS`.

Có thể kiểm tra thêm cả validation/Java/save với 2 epoch nhỏ (chạy một lần):

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh train-smoke
```

Smoke của model Action-Aware cũ không thay thế smoke của model legacy này. Chưa dùng metric smoke làm kết quả full.

## 7. Chuẩn bị full — terminal

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh prepare-full
```

Cần `LEGACY_FULL_CONFIG_OK`: 1237 train/436 validation, 25 epoch, batch thật 1, accumulation 1, FP32, TF32 tắt, chunk 16, seed 777, 50 queries. Giữ legacy mask target predicted; không bật các cải tiến Action-Aware.

## 8. Kết nối đúng folder Drive — cell Molab

Folder của bạn được dùng làm filesystem root bằng `root_file_id`, theo [API gdrive-fsspec](https://github.com/fsspec/gdrive-fsspec/blob/master/src/gdrive_fsspec/core.py). Vì vậy không cần đoán tên folder hoặc đường dẫn My Drive.

Tạo handle riêng `legacy_fs` trong MỘT cell. Không định nghĩa lại tên này ở cell khác. Handle `fs` đang dùng cho run cũ được giữ nguyên.

```python
import marimo as _mo
from gdrive_fsspec import GoogleDriveFileSystem as _GoogleDriveFileSystem

legacy_fs = _GoogleDriveFileSystem(
    root_file_id='1JtVmVYfZYlxbcR96pKMhWqxODi2WkJZI',
    use_listings_cache=False,
    skip_instance_cache=True,
    auth_kwargs={"use_local_webserver": False},
)
_mo.output.clear_console()
print(legacy_fs.ls(''))
```

Tài khoản đăng nhập cần quyền ghi vào folder để backup. Không cần đặt toàn bộ folder thành public. Không copy token vào terminal/source. Nếu thiếu package gdrive_fsspec, thêm qua Molab Packages panel.

Nếu gdown ở bước 4 không có quyền tải và file ở ngay root folder với tên như ZIP cung cấp, sau khi xác nhận listing có đúng tên, dùng cell:

```python
legacy_fs.get('DDVC_CVPR2018_Masking_Molab_25E.zip', '/marimo/DDVC_CVPR2018_Masking_Molab_25E.zip')
```

Nếu listing cho tên khác thì dùng đúng tên thực tế. Không di chuyển/xoá ZIP đã upload.

## 9. Bật backup — cell Molab

```python
import sys as _sys
from pathlib import Path as _Path

_legacy_work = _Path('/marimo/DDVC_CVPR2018_Molab')
_sys.path.insert(0, str(_legacy_work / 'source' / 'tools'))
from drive_backup import DriveBackup as _DriveBackup

legacy_backup = _DriveBackup(
    legacy_fs,
    work=_legacy_work,
    remote_root='runs/ddvc_cvpr2018_e25_b1',
    interval=120,
)
legacy_backup.start()
```

Chỉ tạo một worker cho remote root này. Tất cả đường dẫn remote ở trên đều tương đối với folder ID của bạn. Backup nằm trong:

```text
Folder Drive 1JtVmVYfZYlxbcR96pKMhWqxODi2WkJZI/
  ZIP code do bạn upload
  runs/
    ddvc_cvpr2018_e25_b1/
      slot-0/
      slot-1/
```

Hai slot lưu model-last/best, metric, config, refs, bank và logs. Một slot chỉ hoàn tất khi có READY.json; slot còn lại được giữ nếu upload lỗi. Không backup raw video features/HF cache/compiler vì tải lại được. Không thao tác với ZIP code ở root.

```bash
tail -n 40 -f /marimo/DDVC_CVPR2018_Molab/logs/drive_sync.log
```

Sau khi checkpoint upload xong phải có BACKUP_COMPLETE. Chưa có model-last thì worker chưa có checkpoint để upload. Cần đủ dung lượng cho 2 slot; upload có thể mất thời gian. Epoch đang dở chưa có mid-epoch checkpoint.

## 10. Train full — terminal

Chạy sau khi setup/smoke/prepare-full thành công và bật backup. Nên đợi run cũ kết thúc để tránh tranh GPU; launcher không tự dừng run khác.

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh train
```

Lệnh khởi động background và lưu PID. Không chạy lại để tạo process thứ hai. Nohup không giữ process sống khi cả runtime bị thu hồi.

## 11. Xem tiến độ và metric — đường dẫn đầy đủ

```bash
tail -n 80 -f /marimo/DDVC_CVPR2018_Molab/logs/yc2_full_cvpr2018_e25_b1.log
```

Ctrl+C chỉ thoát tail, không dừng train.

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh status
```

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh report
```

Kết quả local:

```text
/marimo/DDVC_CVPR2018_Molab/runs/yc2_full_cvpr2018_e25_b1/
  model-last.pth
  model-best.pth
  info.json
  performance.jsonl
  train.log
  prediction/
```

Best chọn theo METEOR + SODA_c. `info.json` chứa metric history; validation tính CIDEr, METEOR, BLEU-4, SODA_c và localization scores theo evaluator cũ. Prediction final là sau rerank/top-N; raw để debug.

Sau khi train xong, cell:

```python
legacy_backup.sync_once(force=True)
```

Đợi BACKUP_COMPLETE trước khi đóng runtime.

## 12. Resume

Cùng runtime: chỉ khi process cũ đã dừng:

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh resume
```

Session mới:

1. Tải/giải nén và setup lại khi cần.
2. Kết nối `legacy_fs` bằng đúng folder ID ở bước 8.
3. Tạo `legacy_backup` như bước 9 nhưng CHƯA gọi start.
4. Cell restore:

```python
legacy_backup.restore_latest()
```

5. Terminal:

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh restage
```

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh smoke-resume
```

6. Khi CHECKPOINT_LOAD_PASS và smoke thành công, bật `legacy_backup.start()` trong cell rồi chạy lệnh resume.

Không prepare đè config restore. Checkpoint Action-Aware/accumulation 8 không dùng để resume bản này. Checkpoint Kaggle legacy cũ không đủ format-2 scheduler/RNG cũng không được gọi là exact resume; đây là fresh run recipe legacy 25 epoch.

## 13. Kiểm tra và giới hạn

Đã chạy CPU: actual legacy model matched forward/backward, mask gradients, generation/beam/cache, chunk equivalence, training checkpoint interruption/resume, config defaults, backup transport giả lập. Đối chiếu core source với archive gốc. Không tuyên bố đã train model này trên Molab GPU từ local; dùng setup/smoke/train-smoke trên máy thật để xác nhận.

Folder Drive được cấu hình đúng theo ID đã cung cấp nhưng hướng dẫn này không tự đăng nhập hay ghi vào Drive thay bạn. ZIP trên Drive vẫn là source c385e66; tài liệu external này cập nhật link mà không yêu cầu thay ZIP. Không hứa metric trùng Kaggle vì khác epoch budget, GPU và môi trường số học.
