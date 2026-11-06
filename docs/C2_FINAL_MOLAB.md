# Final C2 — hướng dẫn Molab

Branch: [`final/c2-learned-mask`](https://github.com/thloc1212/DDVC/tree/final/c2-learned-mask).
Nhánh ablation cũ: [`exp/v1-v2-contribution-ablation`](https://github.com/thloc1212/DDVC/tree/exp/v1-v2-contribution-ablation).

## Phạm vi chốt

Đây là **C2 trong bộ contribution C0–C5**: DDVC + confidence-gated learned proposal mask + auxiliary mask loss **1.0**, sau đó chọn output bằng Gaussian **Soft-NMS σ=0.5** và số event do counter dự đoán. Không có concept head, concept loss, concept prefix, dedup loss hoặc merge. C3/C4/C5 và các launcher tương ứng đã được gỡ khỏi nhánh này; lịch sử còn nguyên trên nhánh ablation.

Không đổi công thức mask, khởi tạo tham số, decoder, matching hoặc loss chính của C2. GPT-2 vẫn khởi tạo ngẫu nhiên; CLIP features đã trích xuất sẵn và không được fine-tune. Các biện pháp chống NaN, kiểm tra gradient hữu hạn và checkpoint đầy đủ vẫn giữ nguyên. Đây không phải cấu hình historical V1 có merge từng đạt CIDEr 40.0358; không gán kết quả đó cho C2.

| Thiết lập | YouCook2 | ActivityNet Captions |
|---|---:|---:|
| Train / validation feature-matched | 1237 / 436 | 10009 / 4917 |
| Epoch | 15 | 8 |
| Queries | 50 | 10 |
| Batch / gradient accumulation | 1 / 1 | 1 / 1 |
| Seed | 777 | 777 |
| Precision / TF32 | FP32 / tắt | FP32 / tắt |
| Adam LR / weight decay | 5e-5 / 1e-4 | 5e-5 / 1e-4 |
| LR decay start / every / rate | 8 / 3 / 0.5 | 8 / 3 / 0.5 |
| Caption-cost chunk | 16 | 16 |
| Auxiliary mask loss | 1.0 | 1.0 |

Dataset, features, train-only bank và annotation được kiểm tra trước khi chạy. Config cuối và SHA-256 input được ghi trong `prepared/<run>/c2_manifest.json`. Không thay LR, batch hay precision tự động theo GPU. CUDA operators có thể không quyết định hoàn toàn dù giữ cùng seed.

## 1. Session / tài khoản mới

Chạy trong **terminal Molab**. Cài vào Python có sẵn; không tạo môi trường mới, không thay GPU Torch/Torchvision.

```bash
git clone --single-branch --branch final/c2-learned-mask https://github.com/thloc1212/DDVC.git /marimo/DDVC_C2/source
LC_ALL=C LANG=C bash /marimo/DDVC_C2/source/tools/c2_molab.sh setup
```

`setup` cài dependency, kiểm tra C2 trên CPU, chuẩn bị Java, build CUDA và chạy test CUDA. Chỉ tiếp tục khi lệnh kết thúc thành công. Không tự tạo marker `cuda_ready` để bỏ qua test. Interpreter mặc định là `/tmp/uv-venv/bin/python`; nếu nền tảng đổi đường dẫn, đặt `DDVC_PYTHON` trỏ tới Python GPU hiện có.

## 2A. Tài khoản chạy YouCook2

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C2/source/tools/c2_molab.sh prepare yc2
LC_ALL=C LANG=C bash /marimo/DDVC_C2/source/tools/c2_molab.sh train yc2
bash /marimo/DDVC_C2/source/tools/c2_molab.sh logs yc2
```

Run: `yc2_c2_e15_b1`. Tải khoảng 872 MiB CLIP features, annotation và model/tokenizer phục vụ pipeline. Không cần raw video.

## 2B. Tài khoản khác chạy ActivityNet

Chạy bước 1 trên tài khoản thứ hai rồi:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C2/source/tools/c2_molab.sh prepare anet
LC_ALL=C LANG=C bash /marimo/DDVC_C2/source/tools/c2_molab.sh train anet
bash /marimo/DDVC_C2/source/tools/c2_molab.sh logs anet
```

Run: `anet_c2_e8_b1`. CLIP features riêng khoảng 2.52 GiB. Không dùng features YouCook2 cho ANET. Hai tài khoản độc lập có thể dùng cùng đường dẫn local; remote backup phải khác dataset.

Lệnh `train` chạy ngầm và **tự chạy real-model GPU forward/backward + generation smoke trước khi train**. Log smoke/startup nằm ở `logs/<run>.launch.log`; log train nằm ở `logs/<run>.log`. Dùng Ctrl+C để thoát xem log; việc đó không dừng job chạy ngầm. `status yc2` hoặc `status anet` xem PID. Launcher từ chối chạy trùng hoặc ghi đè run cũ.

## 3. Tận dụng dữ liệu đã tải

Nếu Drive nguồn bị quota, tải đúng CLIP features của dataset từ bản sao của bạn, rồi truyền đường dẫn file `.pth`:

```bash
DDVC_FEATURES=/marimo/clipvitl14_yc2.pth bash /marimo/DDVC_C2/source/tools/c2_molab.sh prepare yc2
```

Với ANET đổi dataset thành `anet` và chọn features ANET. Không đổi tên file YouCook2 để giả làm ANET.

Có thể tái sử dụng config **C2 cũ** cùng dataset và các file annotation/features/bank còn tồn tại bằng `DDVC_C2_SOURCE_CONFIG=/đường/dẫn/config.yml` trước lệnh `prepare`. Chỉ dùng config có mask loss 1, Soft-NMS .5, concepts/dedup/merge tắt. Input được stage byte-for-byte; không lấy checkpoint để train fresh.

## 4. Kết quả, báo cáo và ZIP phân tích

YouCook2:

```bash
bash /marimo/DDVC_C2/source/tools/c2_molab.sh report yc2
bash /marimo/DDVC_C2/source/tools/c2_molab.sh export yc2
```

ActivityNet dùng hai lệnh tương tự với `anet`. Output chính:

- `/marimo/DDVC_C2/runs/yc2_c2_e15_b1/` hoặc `/marimo/DDVC_C2/runs/anet_c2_e8_b1/`.
- `model-best.pth`, `model-last.pth`, `info.json`: checkpoint và lịch sử metric. Best checkpoint vẫn theo tiêu chí DDVC đã có, không tự đổi sang CIDEr.
- Các prediction `num*_epoch*.json`: **final được chấm metric**; `num*_epoch*.raw.json`: toàn bộ candidates trước selection để debug. Tìm trong run bằng `find /marimo/DDVC_C2/runs -name 'num*epoch*.json'`.
- ZIP `C2_ANALYSIS_<dataset>_<timestamp>.zip` dưới `/marimo/DDVC_C2`: config, annotations/references, logs, metadata, prediction và source để phân tích; **không chứa checkpoint weights hay video features**. Lưu checkpoints qua backup bên dưới.

Không có `info.json` sau khi vừa khởi động là bình thường; cần hoàn tất validation/checkpoint. Nếu log có exception, kiểm tra exception trước khi báo metric.

## 5. Backup Google Drive

Chạy trong **cell Python của Molab**, không dán Python vào Bash. `fs` phải là Google Drive filesystem đã xác thực của bạn. Không dùng Drive mount giả hoặc public link chứa token.

```python
from gdrive_fsspec import GoogleDriveFileSystem
fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
    auth_kwargs={"use_local_webserver": False})
```

Sau khi xác thực, cell khác:

```python
import sys
sys.path.insert(0, "/marimo/DDVC_C2/source/tools")
from drive_backup import DriveBackup

c2_backup = DriveBackup(fs, work="/marimo/DDVC_C2",
    remote_root="runs/ddvc_final_c2/yc2", interval=120)
c2_backup.start()
```

Tài khoản chạy ANET dùng remote root `runs/ddvc_final_c2/anet`. Backup bắt đầu khi có `model-last.pth`; giữ hai slot và ghi `READY.json` cuối cùng. Nó lưu runs (best/last checkpoint), prepared configs/annotation/bank, logs và provenance; không lưu source hoặc full video features. Source lấy từ branch/commit đã ghi, features tải lại bằng `restage`. Xem tiến độ ở `/marimo/DDVC_C2/logs/drive_sync.log`.

Trước khi kết thúc session, chạy cell và chờ hoàn tất:

```python
c2_backup.stop()
print("BACKUP_DONE:", c2_backup.sync_once(force=True))
```

Trên session mới: clone đúng commit và setup, xác thực lại `fs`, tạo `c2_backup` cùng remote root nhưng **chưa gọi start**, rồi:

```python
print(c2_backup.restore_latest())
```

Giữ nguyên work root `/marimo/DDVC_C2` để các manifest/input paths không đổi. Tiếp đó chạy terminal:

```bash
bash /marimo/DDVC_C2/source/tools/c2_molab.sh restage yc2
bash /marimo/DDVC_C2/source/tools/c2_molab.sh resume yc2
```

ANET dùng `anet`. Resume khôi phục optimizer, scheduler, RNG và phase validation; không phải fresh training hoặc nạp pretrained caption decoder. Đừng chạy `prepare` để ghi đè config đã restore.

## Kiểm tra trước khi phát hành branch

CPU smoke so sánh C2 với commit ablation `ed5494c9dce7371ea30683a3e8a5320ccc19b2a7`: cùng initialization/checkpoint keys, logits/loss, gradient, optimizer step và generation trên tiny GPT-2 thật. Kiểm tra thêm matching-caption chunks, raw/final selection, resume và chống NaN. CPU parity không thay thế full GPU experiment; launcher buộc smoke trên GPU Molab trước mỗi train/resume.

Đã đối chiếu các export C2: 436 video YouCook2 (epoch 11) và 4.917 video ANET (epoch 6). Query IDs, caption, timestamp và thứ tự selection khớp final đã lưu; selection score sai số tối đa cho phép 1e-12 giữa nền tảng. Tính lại bằng hàm selection cũ và mới trên cùng máy cho kết quả giống hệt. Các hyperparameter active từ `best.opt` giữ nguyên; chỉ đổi run ID, đường dẫn output và nhãn recipe. Test resume dùng vòng `train()` thật với fixture CPU: optimizer/scheduler/RNG liên tục, không lặp epoch đã xong; Drive backup và các guard chống NaN cũng đã qua test.
