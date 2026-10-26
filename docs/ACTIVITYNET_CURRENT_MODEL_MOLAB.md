# Model hiện tại trên ActivityNet Captions — 10 queries — Molab + Drive

Branch: https://github.com/thloc1212/DDVC/tree/exp/activitynet-current-model

## Phạm vi

Giữ kiến trúc/model của run YouCook2 no-dedup: learned gated proposal mask bật,
dedup loss bằng 0, retrieval bật, Soft-NMS sigma 0.5 và overlap/consecutive merge
bật. GPT-2 khởi tạo ngẫu nhiên, localization và caption cùng train từ đầu. Không
load checkpoint YouCook2. Dùng **10 queries**, batch/accumulation 1, FP32 TF32 off,
15 epoch, seed 777, Adam LR 5e-5, WD 1e-4, milestones 8/11 gamma 0.5, chunk 16.

Đây là **model hiện tại chuyển dataset và dùng 10 queries theo ActivityNet**, không
phải tái hiện toàn bộ DDVC gốc vì vẫn giữ learned mask/retrieval/selection mở rộng.
Giữ caption/mask, matcher, loss weights, optimizer và selection. Có một sửa lỗi
số học được ghi rõ: GT interval có thể thành [i,i] sau làm tròn frame index dù
thời lượng dương. Loss visual alignment nay lấy ít nhất một frame hợp lệ thay
vì mean(empty)=NaN; mọi slice hợp lệ không rỗng giữ nguyên phép tính cũ. Thêm
adapter dữ liệu và đọc archive features qua mmap để hạn chế nhân đôi RAM.

Soft-NMS/merge vẫn giữ theo yêu cầu, chưa có bằng chứng chúng tốt trên ActivityNet.
Đừng áp trực tiếp metric YouCook2 để kết luận chất lượng trên dataset này.

## Nguồn và protocol dữ liệu

- Features: link DDVC chính thức https://drive.google.com/file/d/1v08rs9Hwqh3XIM-8u8rRcFZc3HWUtJo_/view
- Annotations: processed ActivityNet Captions từ PDVC, được DDVC dẫn nguồn:
  https://github.com/ttengwang/PDVC/tree/0b025c84f42fe27da51c312e8871c4b19628a04c/data/anet/captiondata
- Pin PDVC commit `0b025c84f42fe27da51c312e8871c4b19628a04c`.
  Nguồn có 10.009 train videos, val_1 4.917, val_2 4.885 (hợp 4.917 video).
  Đây là số trong processed annotations, không tự coi bằng số chính thức trong paper.
- Lọc theo video features thực sự có, không zero-padding video thiếu. Chưa tải
  file features thật trên local nên chưa khẳng định số video dùng cuối cùng.
- Loader validation lấy val_1, fallback val_2; evaluator giữ **hai reference tách biệt**.
  Không gộp hai bộ timestamps để train hoặc coi chúng thành hai video riêng.
- Training annotations có hai event zero-duration (v_N7ppHQNikv8/event 2 và
  v_4rKTw99bM8g/event 1, chỉ số 0-based). Adapter bỏ hai event này, ghi manifest.
  Không sửa timestamps/reference validation. Giữ giới hạn sample tối đa 10 GT
  events/video của loader hiện tại; đây không phải giới hạn số annotation evaluator.
- Vocabulary và text memory bank chỉ từ train ActivityNet được chọn. Không trộn
  caption validation vào bank. CLIP encoder chỉ encode, không fine-tune.
- Manifest lưu annotation hashes, train/val IDs, feature coverage/missing IDs,
  sự kiện bị bỏ và chữ ký train/val/vocab. Subset lấy 30% feature-matched IDs;
  full lấy toàn bộ feature-matched IDs. Smoke 1%, chỉ test một video forward/backward.

## 1. Source — terminal Molab

```bash
git clone --single-branch --branch exp/activitynet-current-model https://github.com/thloc1212/DDVC.git /marimo/DDVC_ActivityNet/source
```

Không clone/giải nén đè thư mục có source/checkpoint cũ. ZIP đi kèm có cùng layout
`DDVC_ActivityNet/source/`; nếu dùng ZIP thay Git, upload bản Q10 vào /marimo rồi:

```bash
/tmp/uv-venv/bin/python -m zipfile -e /marimo/DDVC_ActivityNet_Current_Model_Q10.zip /marimo
```

Chỉ dùng một trong hai cách lấy source.

### Nếu đã clone bản 50 queries trong cùng session

Chờ/dừng tiến trình training 50-query trước; không chạy hai run cùng GPU. Không
kill tiến trình Python khác bằng pkill. Nếu muốn dừng đúng run cũ, dùng PID đã lưu:

```bash
if [ -f /marimo/DDVC_ActivityNet/logs/anet_full_current_model_e15_b1.pid ]; then
  ps -p "$(cat /marimo/DDVC_ActivityNet/logs/anet_full_current_model_e15_b1.pid)" -o pid,args
fi
```

Chỉ khi dòng args đúng train.py/config run 50-query cũ, dùng `kill -INT PID` với
PID đó. Kiểm tra tiến trình đã dừng trước khi chạy Q10. Checkpoint cũ được giữ,
nhưng không thể exact-resume nó với số query khác.

```bash
git -C /marimo/DDVC_ActivityNet/source pull --ff-only
```

Không cần chạy lại setup/build CUDA nếu cùng session đã setup thành công. Sau đó
chạy lại prepare-smoke, smoke, prepare-run, train ở các mục dưới. Tên run mới thêm
`q10`, tên smoke marker cũng riêng. Launcher tự dùng lại annotations/features/vocab/
bank của prepared 50-query khi có config cũ; **không load model hoặc optimizer cũ**.
Nếu prepared cũ không có, tạo dữ liệu như bình thường. Cache feature gốc vẫn dùng lại.
Nếu dùng ZIP trước đó, tải bản Q10 rồi chép đúng hai file tools/activitynet_pipeline.py
và tools/activitynet_molab.sh từ bản Q10 vào source cũ; không giải nén đè cả thư mục run.

## 2. Setup native Molab

```bash
LC_ALL=C bash /marimo/DDVC_ActivityNet/source/tools/activitynet_molab.sh setup full
```

Giữ native GPU Torch/torchvision, không tạo venv. Setup cài dependency như workflow
đã dùng, patch đường tokenizer/CUDA tương thích, build và kiểm tra extension.
Phiên bản Torch mới khác major CUDA13.0 chưa được kiểm tra; không bỏ qua lỗi build.

## 3. Features, annotations, smoke

File features ActivityNet lớn hơn YouCook2, có thể tốn nhiều dung lượng/RAM khi
chuẩn bị. Kiểm tra trước:

```bash
df -h /marimo
free -h
```

Nếu đã tải features theo lệnh trước vào `/marimo/DDVC_ActivityNet/assets/clipvitl14.pth`,
adapter tái sử dụng. Nếu chưa có, prepare-smoke tự tải từ đúng Drive ID ở trên.
Muốn dùng file sẵn ở nơi khác, đặt `DDVC_ANET_FEATURES` bằng đường dẫn tuyệt đối;
không trỏ vào file YouCook2. Feature cache chung dùng lại giữa smoke và full.

```bash
LC_ALL=C bash /marimo/DDVC_ActivityNet/source/tools/activitynet_molab.sh prepare-smoke full
```

```bash
LC_ALL=C bash /marimo/DDVC_ActivityNet/source/tools/activitynet_molab.sh smoke full
```

Chờ REAL_MODEL_FORWARD_BACKWARD_PASS và GENERATION_POSTPROCESS_PASS. Lần tải
features ban đầu vẫn là cả file vì Drive cung cấp một archive, không thể tải riêng
1% chỉ bằng mode smoke. Text bank smoke chỉ encode caption của train smoke.

## 4. Chuẩn bị full hoặc subset

Full:

```bash
LC_ALL=C bash /marimo/DDVC_ActivityNet/source/tools/activitynet_molab.sh prepare-run full
```

Hoặc 30% data, vẫn 15 epoch:

```bash
LC_ALL=C bash /marimo/DDVC_ActivityNet/source/tools/activitynet_molab.sh prepare-run subset
```

Không chạy cả hai nếu không định thực nghiệm cả hai. Phần chuẩn bị encode bank
ActivityNet riêng, không chỉ thay tên yc2 thành anet. Chờ ACTIVITYNET_CURRENT_MODEL_OK
và CONFIG_READY. Nếu chuẩn bị gãy, chạy lại cùng run-name; config chỉ viết sau khi
đã xong bank/refs. Nếu config đã tồn tại, launcher giữ và kiểm tra nó.

## 5. Bật Drive backup — một cell Python Molab

Giữ toàn bộ biến `_` trong **chính cell này**. Không dùng chúng ở cell khác.

```python
import marimo as mo
import sys as _anet_sys
from pathlib import Path as _AnetPath
from gdrive_fsspec import GoogleDriveFileSystem as _AnetFS

_anet_fs = _AnetFS(
    use_listings_cache=False, skip_instance_cache=True,
    auth_kwargs={"use_local_webserver": False},
)
mo.output.clear_console()
_anet_work = _AnetPath('/marimo/DDVC_ActivityNet')
_anet_sys.path.insert(0, str(_anet_work / 'source' / 'tools'))
from drive_backup import DriveBackup as _AnetBackup

anet_backup = _AnetBackup(
    _anet_fs, work=_anet_work,
    remote_root='runs/ddvc_activitynet_current_model_q10_e15', interval=120,
)
anet_backup.start()
```

Nếu notebook đã định nghĩa `mo` ở cell khác, bỏ dòng `import marimo as mo` trong
cell trên và dùng biến mo đã có. Nếu có cell anet_backup cũ, sửa cell đó thay vì
tạo tên trùng. Kết nối gốc Drive mặc định, giống nơi READY của lần trước được tìm thấy.

## 6. Train full, log, report

```bash
LC_ALL=C bash /marimo/DDVC_ActivityNet/source/tools/activitynet_molab.sh train full
```

```bash
tail -n 80 -f /marimo/DDVC_ActivityNet/logs/anet_full_current_model_q10_e15_b1.log
```

```bash
LC_ALL=C bash /marimo/DDVC_ActivityNet/source/tools/activitynet_molab.sh report full
```

Nếu chọn subset, thay `full` bằng `subset` ở lệnh train/report và log tên
`anet_subset_current_model_q10_e15_b1.log`. Launcher không tự resume hoặc đè checkpoint.
Nếu chỉ là run khởi động thất bại chưa có .pth, launcher giữ nguyên folder cũ với
đuôi `_failed_start_<timestamp>` rồi cho train lại từ đầu. Không xoá evidence.
Ctrl+C chỉ thoát tail. Log không xuất liên tục lúc evaluator/Java chạy không đủ
để kết luận treo; xem tiến trình trước khi dừng.

Output full: `/marimo/DDVC_ActivityNet/runs/anet_full_current_model_q10_e15_b1/`.
Output subset: `/marimo/DDVC_ActivityNet/runs/anet_subset_current_model_q10_e15_b1/`.
Metric CIDEr, BLEU-4, METEOR, SODA_c, Precision/Recall và MetaScore nằm trong
info.json/prediction. F1 có thể tính từ aggregate Precision/Recall, ghi rõ công thức.
Model-best chọn METEOR+SODA_c, không nhất thiết max CIDEr. Prediction đã selection
và raw predictions được giữ riêng. Đây là điểm validation, không phải test.

Khi xong, cell mới:

```python
anet_backup.sync_once(force=True)
```

Chờ BACKUP_COMPLETE. Worker chỉ bắt đầu upload sau khi có model-last.pth và cần
kernel còn hoạt động. Backup cả runs/prepared/logs/provenance; prepared chứa refs,
vocab, text bank và config; file video features lớn được tải lại khi restore.

## Session mới: restore, restage, smoke, resume

Lấy cùng source commit, chạy setup. Sửa cell backup trên, thay `anet_backup.start()`
bằng `anet_backup.restore_latest()` **trong cùng cell**, chưa start worker.

```bash
LC_ALL=C bash /marimo/DDVC_ActivityNet/source/tools/activitynet_molab.sh restage full
LC_ALL=C bash /marimo/DDVC_ActivityNet/source/tools/activitynet_molab.sh smoke-resume full
```

Sau smoke checkpoint thành công, bật worker `anet_backup.start()`, rồi:

```bash
LC_ALL=C bash /marimo/DDVC_ActivityNet/source/tools/activitynet_molab.sh resume full
```

## Validation đã thực hiện trước khi bàn giao

- Kiểm tra annotation thật ở commit PDVC đã pin: split/ref counts, hai event lỗi.
- CPU tests cho aliases video IDs, features sai shape/ID trùng, leakage,
  giữ reference riêng, vocabulary chỉ từ train, config drift và mmap giữ tensor values.
- Legacy model identity test: caption/mask giữ nguyên; backbone chỉ thêm đúng
  sửa lỗi pooling đoạn GT rỗng đã mô tả, không thay cấu trúc mạng/loss weights.
- CPU regression recipe YouCook2 và shell syntax/LF checks.
- **Chưa tải full ActivityNet features hoặc chạy GPU full train trên local**.
  Molab smoke là điều kiện bắt buộc trước khi train; không có metric ActivityNet mới
  hoặc cam kết tái hiện điểm paper trong gói này.

## Lỗi non-finite ngay epoch đầu

Không tắt finite guard và không giảm LR khi chưa biết loss nào lỗi. Cập nhật source:

```bash
git -C /marimo/DDVC_ActivityNet/source pull --ff-only
```

Quét đúng frame length trong file features đã chuẩn bị, không load model/GPU:

```bash
/tmp/uv-venv/bin/python /marimo/DDVC_ActivityNet/source/tools/check_temporal_spans.py --config /marimo/DDVC_ActivityNet/prepared/anet_full_current_model_q10_e15_b1/config.yml
/tmp/uv-venv/bin/python /marimo/DDVC_ActivityNet/source/smoke_test_temporal_pooling.py
```

`EMPTY_GT_FRAME_SLICES` đếm event bị collapse sau quantization. Các event này không
bị bỏ khỏi data; pool lấy frame gần nhất. Không cần download features hay build
CUDA lại vì chỉ đổi Python. Chạy lại smoke rồi `train full` nếu chưa có checkpoint;
nếu đã có model-last.pth thì smoke-resume/resume. Guard vẫn dừng trước update nếu
NaN/Inf; khi đó file `runs/<run>/nonfinite_diagnostics.json` ghi video ID, bad loss
terms, raw captions, quantized GT spans và tình trạng finite của video tensor.
Gửi file đó để xác định nguyên nhân thực sự, không khẳng định mọi NaN đều do pooling.
