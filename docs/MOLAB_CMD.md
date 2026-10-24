# DDVC Action-Aware Mask trên Molab — terminal, 25 epoch, effective batch 4

## Thiết lập của gói này

- Source: branch [`exp/action-aware-grounded-mask`](https://github.com/thloc1212/DDVC/tree/exp/action-aware-grounded-mask). Commit chính xác nằm trong `BUNDLE_MANIFEST.json` của ZIP.
- Không tạo `.venv`; cài vào **đúng interpreter của môi trường Molab đang mở**. Không thay GPU PyTorch/torchvision có sẵn.
- Full feature-matched: dự kiến 1.237 train / 436 validation; 25 epochs, seed 777, 50 queries.
- **Micro-batch thật = 1 video; gradient accumulation = 4; effective batch tối đa = 4 video/update.** Nhóm cuối epoch có thể nhỏ hơn 4. Không phải batch thật 4, không tạo contrastive negatives liên video, không tăng throughput tương đương multi-GPU. Nhánh caption/mask hiện không hỗ trợ batch thật >1.
- Giữ LR `5e-5`, không tự scale LR theo batch. Số optimizer updates mỗi epoch ít hơn batch 1; 25 epoch là recipe bạn yêu cầu, không gọi là tái lập chính xác recipe paper.
- Toàn bộ setup/data/smoke/train/report dùng **Linux terminal/Bash của Molab**. Chỉ dùng vài cell ngắn cho kết nối Drive vì Google Drive không phải ổ mount POSIX.
- Model dùng CLIP features và train-only caption memory; GPT-2 decoder khởi tạo mới. Không cần notebook Kaggle hay tải pretrained caption model.

**Về “cài một lần”:** cài một lần cho runtime hiện tại, không cài lại mỗi epoch hoặc lần chạy command. Nếu Molab tạo runtime mới/reset môi trường, có thể vẫn cần chạy lại bước cài dependency và build CUDA. Dùng native environment không bảo đảm package tồn tại vĩnh viễn. Checkpoint/log phải backup Drive; xem [chính sách storage Molab](https://marimo.io/pages/molab/storage) và [remote-storage API](https://docs.marimo.io/guides/working_with_data/remote_storage/).

## 0. Upload gói source lên Drive

Tải file `DDVC_Action_Aware_Molab.zip` được cung cấp cùng tài liệu này. Upload vào Google Drive, ví dụ:

```text
DDVC_Molab/
└── DDVC_Action_Aware_Molab.zip
```

ZIP chỉ có source, script CLI, license và tài liệu; không có `.venv`, dataset hoặc checkpoint đã train. Các evaluator JAR/data của repo được giữ để tính metric. Khoảng trống cần cho data/cache/model/optimizer có thể nhiều GB; kiểm tra `df -h` trước.

## 1. Chọn GPU và kết nối Drive — cell nhỏ duy nhất cho authentication

Chọn GPU trong Molab. Nếu đã có `fs` từ Remote Storage thì **dùng lại**, không định nghĩa `fs` hoặc `mo` ở cell khác. Nếu chưa có, dùng đúng mẫu sau trong một cell:

```python
import marimo as mo
from gdrive_fsspec import GoogleDriveFileSystem

fs = GoogleDriveFileSystem(
    use_listings_cache=False,
    skip_instance_cache=True,
    auth_kwargs={"use_local_webserver": False},
)
mo.output.clear_console()
```

Nếu thiếu `gdrive_fsspec`, thêm bằng Packages panel Molab trước. Không copy OAuth token vào source hay ZIP. Drive được truy cập bằng `fs.get/fs.put`, không có đường dẫn giả kiểu `/content/drive`.

## 2. Tải source ZIP xuống, unzip và tạo environment file

Chạy cell nhỏ này, sửa **chỉ** `_remote_zip` nếu tên thư mục Drive khác. Nếu không chắc remote path, trước hết xem `fs.ls('')`/Files → Remote Storage.

```python
import os as _os
import sys as _sys
import shlex as _shlex
import zipfile as _zipfile
from pathlib import Path as _Path

_remote_zip = 'DDVC_Molab/DDVC_Action_Aware_Molab.zip'
_base = _Path.cwd()
_work = _base / 'DDVC_Action_Aware_Molab'
_archive = _base / 'DDVC_Action_Aware_Molab.zip'
if _work.exists():
    raise FileExistsError(f'{_work} đã tồn tại. Dùng source hiện có; không unzip đè run đang train.')
fs.get(_remote_zip, str(_archive))
with _zipfile.ZipFile(_archive) as _zip:
    for _name in _zip.namelist():
        _target = (_base / _name).resolve()
        if not _target.is_relative_to(_work.resolve()):
            raise ValueError(f'Unsafe ZIP path: {_name}')
    _zip.extractall(_base)
_env = _base / 'ddvc_molab_env.sh'
_env.write_text(
    f'export PY={_shlex.quote(_sys.executable)}\n'
    f'export WORK={_shlex.quote(str(_work))}\n'
    'export REPO="$WORK/source"\n'
    'export PYTHONPATH="$REPO:$REPO/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"\n'
    'export PATH="$WORK/tools/java/bin:$PATH"\n'
    'export HF_HOME="$WORK/hf-cache"\n'
    'export TOKENIZERS_PARALLELISM=false\n'
    'export PYTHONUNBUFFERED=1\n'
    'export CUDA_VISIBLE_DEVICES=0\n'
    'export RUN=yc2_full_a4_e25\n'
    'export CFG="$WORK/prepared/$RUN/config.yml"\n'
)
print('Mở terminal và chạy:')
print('source ' + _shlex.quote(str(_env)))
```

Mở terminal Molab, copy **đúng dòng `source ...` được cell in ra**. Không tự đoán đường dẫn máy khác.

Từ đây các block `bash` đều chạy trong terminal, mỗi block theo thứ tự:

```bash
cd "$REPO"
mkdir -p "$WORK/logs" "$WORK/provenance" "$WORK/runs" "$WORK/prepared"
printf 'Python: %s\nSource: %s\n' "$PY" "$REPO"
"$PY" tools/molab_pipeline.py env
df -h "$WORK"
```

Lệnh `env` phải in `cuda: true`. Nếu không, chọn lại GPU/đúng interpreter trước. Khi mở terminal mới, chạy lại dòng `source ...` đó; không cần cài lại trong cùng runtime.

## 3. Cài dependency một lần vào native Molab environment

Ghi nhận Torch/torchvision trước khi cài:

```bash
"$PY" - <<'PY'
import os,json,torch,torchvision
from pathlib import Path
assert torch.cuda.is_available(), 'Molab GPU chưa khả dụng'
record={'torch':torch.__version__, 'torchvision':torchvision.__version__}
(Path(os.environ['WORK'])/'provenance/native_torch.json').write_text(json.dumps(record))
print(record)
PY
```

Nếu import torchvision lỗi, dừng và gửi lỗi; không cài đè torch phiên bản khác một cách tự động.

```bash
uv pip install --python "$PY" -r tools/molab-requirements.txt
uv pip install --python "$PY" --no-build-isolation --no-deps \
  'git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1'

"$PY" - <<'PY'
import os,json,torch,torchvision,clip
from pathlib import Path
old=json.loads((Path(os.environ['WORK'])/'provenance/native_torch.json').read_text())
assert old=={'torch':torch.__version__,'torchvision':torchvision.__version__}, 'GPU Torch stack đã thay đổi'
assert callable(clip.load), 'Đang import nhầm package clip'
print('Native GPU Torch retained; OpenAI CLIP OK')
PY
uv pip freeze --python "$PY" > "$WORK/provenance/resolved_packages.txt"
"$PY" tools/molab_pipeline.py env > "$WORK/provenance/runtime.json"
```

Dependency tương thích vendored Transformers được khai báo rõ: tokenizers `>=0.19,<0.24`, hub `<1.0`; OpenAI CLIP revision cố định. Đây là runtime compatibility của nhánh đã chạy, không tải package Transformers mới thay cho source vendored. Không sử dụng `requirement.txt` cũ để pin lại Torch/CUDA legacy.

Nếu môi trường chưa có `uv`, cài package qua Packages panel theo danh sách trong `tools/molab-requirements.txt`; không tạo `.venv` phụ hoặc dùng `sudo`.

## 4. Áp dụng compatibility patches, Java và build CUDA

```bash
"$PY" tools/molab_pipeline.py patch
"$PY" tools/molab_pipeline.py java --target "$WORK/tools/java"
java -version

"$PY" -u tools/molab_pipeline.py build-cuda --jobs 2 \
  > "$WORK/logs/build_cuda.log" 2>&1
```

Trong terminal thứ hai (đã `source ...`), theo dõi build:

```bash
tail -n 60 -f "$WORK/logs/build_cuda.log"
```

`Ctrl+C` chỉ thoát `tail`, không dừng lệnh build đang chạy ở terminal khác. Build dùng đúng CUDA compute capability của GPU được cấp, không hard-code T4. Không cần root; extension nằm trong source và được tìm qua `PYTHONPATH`.

Nếu báo thiếu `nvcc`/CUDA toolkit thì **không train full**: runtime đó thiếu công cụ compile extension, cần môi trường có CUDA development toolkit. Không dùng CPU fallback để giả vờ GPU pipeline chạy đúng.

Kiểm tra extension + tests CPU:

```bash
"$PY" -c "import MultiScaleDeformableAttention; print('CUDA extension import OK')"
"$PY" smoke_test_local_visual.py
"$PY" smoke_test_proposal_mask.py
"$PY" smoke_test_training_state.py
"$PY" smoke_test_drive_backup.py
"$PY" smoke_test_molab_cli.py
```

## 5. Smoke trên ít data trước

Data setup tải annotation, CLIP video features khoảng 0.9 GB, và CLIP ViT-L/14 weights để encode train-caption bank. Không cần video gốc hoặc ASR cho cấu hình này. Script in nguồn và số video thực dùng.

```bash
"$PY" -u tools/molab_pipeline.py prepare \
  --mode smoke --run-name smoke_a4_e2 --epochs 2 --accumulation 4 \
  --data-root "$WORK/assets" --prepared-root "$WORK/prepared" \
  --output-root "$WORK/runs" --log-root "$WORK/logs" \
  > "$WORK/logs/prepare_smoke.log" 2>&1

"$PY" -u tools/molab_pipeline.py smoke \
  --config "$WORK/prepared/smoke_a4_e2/config.yml" \
  > "$WORK/logs/smoke_forward.log" 2>&1
tail -n 30 "$WORK/logs/smoke_forward.log"
```

Phải có `REAL_MODEL_FORWARD_BACKWARD_PASS`; đây là model thật + CUDA + bank + data, không chỉ import. Fresh run không có pretrained caption checkpoint để load; khi restore dùng thêm `--checkpoint ...` ở bước 10 để xác nhận checkpoint.

Có thể chạy thêm 2 epoch smoke để kiểm tra cả validation/Java/save:

```bash
"$PY" -u train.py --cfg_path "$WORK/prepared/smoke_a4_e2/config.yml" \
  > "$WORK/logs/smoke_train.log" 2>&1
tail -n 50 "$WORK/logs/smoke_train.log"
```

Chỉ chuẩn bị mỗi `run-name` một lần. Script từ chối ghi đè config đã có để không làm sai data/schedule của run cũ. Nếu smoke lỗi, xem log trước, không bỏ qua rồi train full.

## 6. Chuẩn bị full data, 25 epoch, effective batch 4

```bash
"$PY" -u tools/molab_pipeline.py prepare \
  --mode full --run-name "$RUN" --epochs 25 --accumulation 4 --log-every 10 \
  --data-root "$WORK/assets" --prepared-root "$WORK/prepared" \
  --output-root "$WORK/runs" --log-root "$WORK/logs" \
  > "$WORK/logs/prepare_full.log" 2>&1
tail -n 35 "$WORK/logs/prepare_full.log"
```

Nếu cần effective batch 8, chọn `--accumulation 8` và run-name mới trước khi train; **không đổi `batch_size` thành 4/8**. Đừng đổi accumulation hoặc epoch budget của checkpoint rồi gọi là exact resume.

Config sẽ bật matched-GT balanced mask BCE, query conditioning, pruning threshold 0.05, bounded gate 0.1–0.85 và context margin 0.1/min 4. Source/loss sửa giống nhánh action-aware; không thêm pretrained language decoder.

## 7. Bật backup Drive (khuyến nghị trước khi train dài)

Chạy một cell Molab, dùng lại `fs` đã authentication. Chỉ định nghĩa `ddvc_backup` trong **một** cell; khi rerun phải dừng worker cũ trước. Không dùng cell này thay lệnh train.

```python
import sys as _sys
from pathlib import Path as _Path
_work = _Path.cwd() / 'DDVC_Action_Aware_Molab'
_sys.path.insert(0, str(_work / 'source' / 'tools'))
from drive_backup import DriveBackup as _DriveBackup

ddvc_backup = _DriveBackup(
    fs,
    work=_work,
    remote_root='DDVC_Molab/action_aware_a4_e25',
    interval=120,
)
ddvc_backup.start()
```

Worker kiểm tra checkpoint mới mỗi 120 giây; chỉ khi `model-last.pth` đổi mới upload snapshot, không upload model liên tục theo mỗi dòng log. Hai slot luân phiên giữ checkpoint best/last, metadata, references, bank, logs và environment manifests. Không backup video-feature `.pth` hoặc HF caches vì có thể tải lại. Một slot chỉ được coi là hoàn tất khi có `READY.json`; không xoá slot hoàn tất còn lại khi upload lỗi. Đích remote ở trên phải dành riêng cho run của bạn; không cho hai session cùng ghi vào nó.

Theo dõi backup từ terminal:

```bash
tail -n 40 -f "$WORK/logs/drive_sync.log"
```

Kiểm tra có `BACKUP_COMPLETE`. Checkpoint có thể nhiều GB, upload mất thời gian và cần đủ dung lượng Drive cho hai slot. Một lần upload chưa xong khi session bị kill có thể mất; slot đã hoàn tất trước đó dùng để restore. Không có checkpoint giữa epoch: phần epoch đang train dở chưa được backup.

## 8. Train background và xem `.log`

Trong terminal đã source environment:

```bash
cd "$REPO"
nohup "$PY" -u train.py --cfg_path "$CFG" \
  > "$WORK/logs/$RUN.log" 2>&1 < /dev/null &
echo $! > "$WORK/logs/$RUN.pid"
printf 'PID=%s\nLOG=%s\n' "$(cat "$WORK/logs/$RUN.pid")" "$WORK/logs/$RUN.log"
```

Không chạy lệnh này hai lần khi process cũ còn chạy. `nohup` giúp không phụ thuộc terminal, **không giữ process sống khi Molab thu hồi cả runtime**.

Theo dõi:

```bash
tail -n 80 -f "$WORK/logs/$RUN.log"
```

Hoặc chỉ xem tiến độ:

```bash
grep -E 'EPOCH_START|PROGRESS|VALIDATION_START|EPOCH_COMPLETE|Traceback|Error' \
  "$WORK/logs/$RUN.log" | tail -n 35
```

Ví dụ dòng log (minh hoạ):

```text
EPOCH_START epoch=3/25 videos=1237 micro_batch=1 accumulation=4 effective_batch=4
PROGRESS epoch=3/25 micro_batch=120/1237 optimizer_steps=650 loss=... lr=... elapsed_s=... eta_epoch_s=...
VALIDATION_START epoch=3/25 videos=436
VALIDATION_PROGRESS batches=120/436
EPOCH_COMPLETE epoch=3/25 optimizer_steps=... checkpoint=.../model-last.pth
```

Progress `epoch=3/25` là cách đếm 1-based để đọc; best epoch trong `info.json` vẫn 0-based. ETA chỉ ước lượng phần train của epoch, không gồm validation. Dòng PROGRESS xuất hiện batch đầu, mỗi 10 micro-batches và batch cuối. Với 1.237 video/epoch và accumulation 4 có 310 optimizer updates/epoch.

Kiểm tra process/GPU:

```bash
ps -p "$(cat "$WORK/logs/$RUN.pid")" -o pid,etime,%cpu,%mem,args
nvidia-smi
```

Lưu log trong run cũng có ở `$WORK/runs/$RUN/train.log`. File `.log` bên ngoài còn chứa stdout/stderr và traceback.

## 9. Xem kết quả và checkpoint

```bash
"$PY" tools/molab_pipeline.py report --config "$CFG"
ls -lh "$WORK/runs/$RUN/model-last.pth" "$WORK/runs/$RUN/model-best.pth"
ls -lh "$WORK/runs/$RUN/prediction/"
```

`info.json` lưu metric history; best chọn theo METEOR + SODA_c, không theo CIDEr. Final prediction JSON là output sau rerank/top-N, raw JSON để debug. Sau train kết thúc, chạy trong cell đọc handle sẵn có để backup lần cuối:

```python
ddvc_backup.sync_once(force=True)
```

Không cần notebook cell nào để training hoặc đo metric: train.py đã chạy validation/evaluator theo epoch.

## 10. Resume trong cùng runtime

Sau khi chắc chắn process cũ đã dừng:

```bash
"$PY" tools/molab_pipeline.py resume-config --config "$CFG"
nohup "$PY" -u train.py --cfg_path "$CFG" \
  >> "$WORK/logs/$RUN.log" 2>&1 < /dev/null &
echo $! > "$WORK/logs/$RUN.pid"
tail -n 80 -f "$WORK/logs/$RUN.log"
```

Format-2 checkpoint chứa model/optimizer/scheduler/RNG/config/metric metadata. `validation_pending` nghĩa là train epoch đã xong: resume chỉ chạy lại validation rồi tiến tiếp; `epoch_complete` bắt đầu ở epoch kế tiếp. Bản fix serialization lambda-defaultdict đã có trong ZIP.

**Checkpoint Kaggle cũ batch 1 / 15 epoch không phải exact resume của run mới accumulation 4 / 25 epoch.** Hãy tạo run mới như trên, không sửa checkpoint để né kiểm tra recipe.

## 11. Resume sau khi Molab reset / session mới

1. Kết nối Drive và tải/unzip source như bước 1–2. Source ZIP vẫn nằm trên Drive.
2. Source env; kiểm tra/cài dependencies nếu môi trường mới; patch và build extension như bước 3–4.
3. Tạo backup handle ở cell bước 7 nhưng **chưa gọi `.start()`**. Nếu đã start, gọi `ddvc_backup.stop()` và đợi worker upload kết thúc.
4. Restore slot đã hoàn tất:

```python
ddvc_backup.restore_latest()
```

5. Trong terminal, khôi phục video features bị bỏ khỏi backup và kiểm tra checkpoint:

```bash
"$PY" -u tools/molab_pipeline.py restage --config "$CFG" --data-root "$WORK/assets" \
  > "$WORK/logs/restage.log" 2>&1

"$PY" -u tools/molab_pipeline.py smoke --config "$CFG" \
  --checkpoint "$WORK/runs/$RUN/model-last.pth" \
  > "$WORK/logs/resume_smoke.log" 2>&1
tail -n 30 "$WORK/logs/resume_smoke.log"
```

Phải thấy `CHECKPOINT_LOAD_PASS` và `REAL_MODEL_FORWARD_BACKWARD_PASS`. Smoke không cập nhật file checkpoint; process train sẽ load lại trạng thái gốc.

6. Bật lại `ddvc_backup.start()` trong cell rồi chạy lệnh resume ở bước 10.

Không chạy `prepare` đè config/data đã restore. Restore có remap local work path trong config; train/val IDs và bank được giữ. Muốn chạy cấu hình mới phải chọn run-name mới.

## 12. Giới hạn đã biết

- Chưa kiểm thử trên GPU Molab của bạn. Có test CPU actual training loop, gradient accumulation với nhóm cuối thiếu batch, checkpoint interruption/resume và backup qua transport giả lập. Cần smoke CUDA ở bước 5 trước full.
- Mô hình vẫn là batch thật 1. Muốn batch thật 2–4 cần sửa multi-video retrieval/matching/mask alignment, không chỉ sửa config.
- Ngưỡng pruning không bảo đảm nhiều frame bị loại lúc soft mask mới khởi tạo; đọc active-fraction log.
- Epoch dài vẫn có rủi ro mất phần train dở nếu thu hồi compute; checkpoint hiện là boundary epoch, không phải mid-epoch sampler resume.
- Gói này không hứa tăng CIDEr; thay effective batch và epoch budget cần ghi rõ khi so với R14/R15.

## File chính

- `source/tools/molab_pipeline.py`: patch/build/data/bank/config/smoke/restore-features/report CLI.
- `source/tools/drive_backup.py`: backup 2 slot qua `fs` Molab.
- `source/tools/molab-requirements.txt`: native-environment dependencies, không pin/thay Torch.
- `source/train.py`, `source/misc/training_state.py`: accumulation/progress log/atomic checkpoint.
- `source/ACTION_AWARE_MASK.md`: kiến trúc và các flag ablation của nhánh.
