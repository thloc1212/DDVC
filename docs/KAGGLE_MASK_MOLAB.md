# DDVC + CVPR2018 masking: Kaggle recipe trên Molab

> Tài liệu nền cho branch cũ. Trên branch exp/gt-aware-dedup-softnms, dùng
> docs/DEDUP_SOFTNMS_MOLAB.md và tools/dedup_molab.sh.

Branch: https://github.com/thloc1212/DDVC/tree/exp/learned-gated-proposal-mask

Giữ model learned proposal mask/GPT-2/DDVC core của ZIP gốc. Không dùng time prefix,
Action-Aware, corrected contrastive, query-conditioned mask hoặc pruning.
Đây là Kaggle recipe port; không bảo đảm bitwise reproduction giữa GPU/Torch khác nhau.

## Config

- Mặc định subset 30%: 371 train / 131 validation; full tùy chọn: 1237 / 436.
- Cả subset/full: 15 epoch; smoke: 1%, 2 epoch.
- Seed 777; batch train/eval 1; accumulation 1; 50 queries; GPT-2 scratch 2 layers.
- Adam, LR 5e-5, weight decay 1e-4, grad clip 100.
- MultiStepLR: mốc 8 và 11, gamma 0.5 theo công thức code Kaggle.
- Mask predicted target, confidence gate [0,1], bins 64, encoding 128, BCE coefficient 1.
- BBox L1 loss/cost 0; GIoU 4; caption loss 2; counter loss 0.5.
- soft_k 20, window_size 80, frame_embedding_num 200, max_caption_len 30, beam 3.
- Caption cost chunk 16; workers 0; bank encoding batch 64; CLIP frozen.
- FP32, TF32 off explicit trên Molab; không BF16. Kaggle không đặt autocast;
  TF32 mặc định Kaggle chưa được ghi lại nên không khẳng định giống bitwise.
- tools/kaggle_mask_recipe.json pin 234 config values; validator dừng nếu drift.
  Đường dẫn, run IDs, vocab/data phụ thuộc tập chọn được loại khỏi so sánh số học.

## Chạy mới, không resume checkpoint 25 epoch

ZIP giải nén vào /marimo sẽ tạo /marimo/DDVC_CVPR2018_KaggleRecipe.
Hoặc clone source:

```bash
git clone --depth 1 --single-branch --branch exp/learned-gated-proposal-mask https://github.com/thloc1212/DDVC.git /marimo/DDVC_CVPR2018_KaggleRecipe/source
```

Terminal, chạy lần lượt:

```bash
LC_ALL=C bash /marimo/DDVC_CVPR2018_KaggleRecipe/source/tools/kaggle_molab.sh setup
bash /marimo/DDVC_CVPR2018_KaggleRecipe/source/tools/kaggle_molab.sh prepare-smoke
bash /marimo/DDVC_CVPR2018_KaggleRecipe/source/tools/kaggle_molab.sh smoke
bash /marimo/DDVC_CVPR2018_KaggleRecipe/source/tools/kaggle_molab.sh train-smoke
```

Cần REAL_MODEL_FORWARD_BACKWARD_PASS và GENERATION_POSTPROCESS_PASS, rồi train-smoke
hoàn tất validation/checkpoint. Data tự tải, không cần raw videos. Setup giữ native Torch,
không tạo venv mới. Giữ các sửa CUDA/build/header, public tokenizer, checkpoint/resume,
backup/log cần thiết. Không gọi đây là môi trường giống Kaggle hoàn toàn.

Subset giống default notebook:

```bash
bash /marimo/DDVC_CVPR2018_KaggleRecipe/source/tools/kaggle_molab.sh prepare-run subset
bash /marimo/DDVC_CVPR2018_KaggleRecipe/source/tools/kaggle_molab.sh train subset
tail -n 80 -f /marimo/DDVC_CVPR2018_KaggleRecipe/logs/yc2_subset_cvpr2018_kaggle_e15_b1.log
```

Hoặc full để so run R15 full, vẫn 15 epoch:

```bash
bash /marimo/DDVC_CVPR2018_KaggleRecipe/source/tools/kaggle_molab.sh prepare-run full
bash /marimo/DDVC_CVPR2018_KaggleRecipe/source/tools/kaggle_molab.sh train full
tail -n 80 -f /marimo/DDVC_CVPR2018_KaggleRecipe/logs/yc2_full_cvpr2018_kaggle_e15_b1.log
```

Không chạy hai run cùng GPU khi so hiệu năng. Ctrl+C ở tail chỉ thoát xem log.
Nohup không bảo vệ khỏi runtime bị thu hồi. Mỗi mode có config/run/log riêng.

## Backup Drive: bật trước train dài

Dùng lại fs hiện có, hoặc sửa cell kết nối hiện tại (không định nghĩa trùng fs):

```python
import marimo as mo
from gdrive_fsspec import GoogleDriveFileSystem
fs = GoogleDriveFileSystem(root_file_id='1JtVmVYfZYlxbcR96pKMhWqxODi2WkJZI', use_listings_cache=False, skip_instance_cache=True, auth_kwargs={'use_local_webserver':False})
mo.output.clear_console()
```

Cell riêng:

```python
import sys as _sys
from pathlib import Path as _Path
_kaggle_work = _Path('/marimo/DDVC_CVPR2018_KaggleRecipe')
_sys.path.insert(0, str(_kaggle_work/'source'/'tools'))
from drive_backup import DriveBackup as _DriveBackup
kaggle_recipe_backup = _DriveBackup(fs, work=_kaggle_work, remote_root='runs/ddvc_cvpr2018_kaggle_recipe_e15', interval=120)
kaggle_recipe_backup.start()
```

Giữ notebook chạy để backup hoạt động. Sau train:

```python
kaggle_recipe_backup.sync_once(force=True)
```

Kết quả full ở /marimo/DDVC_CVPR2018_KaggleRecipe/runs/yc2_full_cvpr2018_kaggle_e15_b1/:
info.json, model-best.pth, model-last.pth, prediction/, train.log.
Subset dùng yc2_subset_cvpr2018_kaggle_e15_b1. Best chọn METEOR + SODA_c như Kaggle.

```bash
bash /marimo/DDVC_CVPR2018_KaggleRecipe/source/tools/kaggle_molab.sh report full
```

## Resume đúng recipe

Nếu runtime còn file và process đã dừng:

```bash
bash /marimo/DDVC_CVPR2018_KaggleRecipe/source/tools/kaggle_molab.sh resume full
```

Session mới: tải source/setup, kết nối fs, tạo backup handle như trên nhưng CHƯA start;
chạy kaggle_recipe_backup.restore_latest(). Tiếp theo:

```bash
bash /marimo/DDVC_CVPR2018_KaggleRecipe/source/tools/kaggle_molab.sh restage full
bash /marimo/DDVC_CVPR2018_KaggleRecipe/source/tools/kaggle_molab.sh smoke-resume full
```

Bật worker backup rồi resume. Đổi full thành subset nếu run subset.
Không reuse config/checkpoint của 25-epoch/Action-Aware/temporal-caption experiment.

## Validation

CPU: actual config expression subset/full so với notebook; reject LR/epoch/batch/mask drift;
model source hash identity; checkpoint/scheduler/RNG continuity; CLI patch idempotence;
backup transport fixtures. GPU/CUDA/full metrics phải smoke và đo trên Molab thực tế.
