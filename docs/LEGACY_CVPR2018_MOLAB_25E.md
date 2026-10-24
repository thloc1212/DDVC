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

## 3. Giải nén trên Molab

Chọn RTX Pro 6000 và native GPU runtime. Không dừng/xoá run Action-Aware hiện có.
Gói mới dùng thư mục riêng `/marimo/DDVC_CVPR2018_Molab`.

Cách A: upload `DDVC_CVPR2018_Masking_Molab_25E.zip` vào Files của Molab tại `/marimo/`, rồi chạy:

```bash
/tmp/uv-venv/bin/python -m zipfile -t /marimo/DDVC_CVPR2018_Masking_Molab_25E.zip
```

Chỉ giải nén khi chưa có thư mục source/run cùng tên:

```bash
test ! -e /marimo/DDVC_CVPR2018_Molab && /tmp/uv-venv/bin/python -m zipfile -e /marimo/DDVC_CVPR2018_Masking_Molab_25E.zip /marimo
```

Cách B: nếu ZIP đặt trong Google Drive tại `DDVC_Molab/DDVC_CVPR2018_Masking_Molab_25E.zip`, dùng `fs` đã kết nối trong một cell:

```python
fs.get('DDVC_Molab/DDVC_CVPR2018_Masking_Molab_25E.zip', '/marimo/DDVC_CVPR2018_Masking_Molab_25E.zip')
```

Rồi dùng hai lệnh kiểm tra/giải nén bên trên. Drive là remote storage qua fs, không có đường dẫn mount giả.

Nếu chưa có `fs`, tạo một lần bằng cell dưới. Nếu đã có thì dùng lại để tránh redefines variables:

```python
import marimo as mo
from gdrive_fsspec import GoogleDriveFileSystem
fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, auth_kwargs={"use_local_webserver": False})
mo.output.clear_console()
```

## 4. Setup môi trường + CUDA

Mỗi lệnh tự lấy WORK/REPO/RUN và CUDA env. Không phải `source ...` hay khai báo biến trước, không lệ thuộc thư mục terminal hiện tại.

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh setup
```

Output hiện trên terminal và được ghi vào `logs/`. Nếu lỗi, script dừng, không tự chạy tiếp training.
Với native Torch cu130 script tạo compiler/header ở `toolchains/cuda130-pinned` (~105 MB download, uv có thể tái dùng cache). Đây là assets compiler, không phải `.venv`. Native Torch giữ nguyên; không sửa source/runtime của run đang chạy ở thư mục khác.

Setup cài một lần cho runtime hiện tại. Molab reset có thể phải cài/build lại. Không thể đảm bảo package tồn tại vĩnh viễn. Không dùng sudo.
Nếu interpreter Molab không phải `/tmp/uv-venv/bin/python`, chỉ định đường dẫn thật qua `DDVC_PYTHON` khi gọi script. Không đổi Torch tuỳ tiện.

## 5. Smoke data và model

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh prepare-smoke
```

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh smoke
```

Tải annotation + CLIP features ~0.9 GB và CLIP ViT-L/14 weights để encode train-only memory bank; không cần raw video hay pretrained caption checkpoint. Smoke dùng 1% (12 train, 4 validation). Cần thấy `REAL_MODEL_FORWARD_BACKWARD_PASS` và `GENERATION_POSTPROCESS_PASS`.

Kiểm tra thêm cả validation/metrics/Java/save bằng 2 epoch nhỏ (chạy một lần):

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh train-smoke
```

Đây là metric smoke, không báo thành metric full. Chỉ train full sau khi đã qua smoke. Smoke cuả run Action-Aware không thay thế smoke model legacy này.

## 6. Full data, 25 epoch

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh prepare-full
```

Cần `LEGACY_FULL_CONFIG_OK 1237 436 ...`. Script từ chối các profile làm đổi batch/precision/recipe. Không cần Kaggle Input dataset.

Bật backup ở bước 8 trước khi train dài. Sau đó khởi động background:

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh train
```

Không chạy song song với run cũ nếu muốn đo tốc độ/VRAM riêng và tránh tranh tài nguyên. Script không tự dừng run khác.
`nohup` không giữ process sống khi cả runtime bị thu hồi.

## 7. Log và kết quả — đường dẫn đầy đủ

```bash
tail -n 80 -f /marimo/DDVC_CVPR2018_Molab/logs/yc2_full_cvpr2018_e25_b1.log
```

Ctrl+C chỉ thoát tail, không dừng train background.

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh status
```

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh report
```

Run directory:

```text
/marimo/DDVC_CVPR2018_Molab/runs/yc2_full_cvpr2018_e25_b1/
  model-last.pth
  model-best.pth
  info.json
  performance.jsonl
  train.log
  prediction/
```

`info.json` lưu metric và history; best chọn theo METEOR + SODA_c. Evaluator YouCook2 trên validation tính CIDEr, METEOR, BLEU-4, SODA_c và localization scores theo code cũ. Prediction final khớp rerank/top-N; raw để debug.

## 8. Backup Google Drive

Dùng lại `fs` đã xác thực. Tạo handle riêng trong một cell (khác tên handle của run cũ):

```python
import sys as _sys
from pathlib import Path as _Path
_legacy_work = _Path('/marimo/DDVC_CVPR2018_Molab')
_sys.path.insert(0, str(_legacy_work / 'source' / 'tools'))
from drive_backup import DriveBackup as _DriveBackup
legacy_backup = _DriveBackup(fs, work=_legacy_work, remote_root='DDVC_Molab/legacy_cvpr2018_e25_b1', interval=120)
legacy_backup.start()
```

Chỉ tạo một worker cho remote root này. Backup 2 slot: checkpoint best/last, config, references, bank, metadata và logs. Một slot chỉ hoàn tất khi có READY.json; không đụng slot hoàn tất kia nếu upload lỗi. Không backup video features/HF cache/compiler vì tải lại được.

```bash
tail -n 40 -f /marimo/DDVC_CVPR2018_Molab/logs/drive_sync.log
```

Sau khi train xong, cell:

```python
legacy_backup.sync_once(force=True)
```

Đợi BACKUP_COMPLETE. Session có thể mất upload/epoch đang dở; checkpoint hiện ở boundary epoch, không mid-epoch.

## 9. Resume

Cùng runtime, sau khi xác nhận process cũ dừng:

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh resume
```

Run bị lỗi trước checkpoint sẽ được giữ lại với suffix startup_failed khi chạy lại `train`; không xoá. Nếu đã có checkpoint thì `train` từ chối, phải dùng `resume`.

Session mới: giải nén/setup lại, kết nối fs và tạo legacy_backup như trên nhưng CHƯA gọi start. Sau đó cell:

```python
legacy_backup.restore_latest()
```

Terminal:

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh restage
```

```bash
bash /marimo/DDVC_CVPR2018_Molab/source/tools/legacy_molab.sh smoke-resume
```

Phải có CHECKPOINT_LOAD_PASS và model smoke pass; bật lại `legacy_backup.start()` trong cell, rồi chạy `resume`.
Không prepare đè config restore, không đổi epoch/batch/loss giữa run.

Checkpoint Action-Aware/accumulation 8 không dùng để resume bản này. Checkpoint legacy cũ từ Kaggle không có đầy đủ format-2 scheduler/RNG cũng không được gọi là exact resume; bản này khởi tạo fresh để tái chạy recipe cũ 25 epoch.

## 10. Kiểm tra và giới hạn

Đã chạy CPU: actual legacy model matched forward/backward, mask gradients, generation/beam/cache, chunk equivalence, training checkpoint interruption/resume, config defaults, backup transport giả lập. Đối chiếu core source với archive gốc. Không tuyên bố đã train model này trên Molab GPU từ local; dùng các bước setup/smoke/train-smoke trên GPU để xác nhận.

GPU/Python/Torch mới có thể làm kết quả số khác Kaggle. Bản này giữ phương pháp legacy để so sánh, không tự bật các cải tiến mới hay hứa CIDEr tăng.
