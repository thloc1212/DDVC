# QTS: hai tài khoản Molab, toàn bộ thao tác bằng terminal + rclone

**Historical support-variant runbook.** On `exp/relation-aware-event-set`, use
`docs/RELATION_AWARE_EVENT_SET_MOLAB.md`; defaults/run names now use variant `set`.

Nhánh: https://github.com/thloc1212/DDVC/tree/exp/query-temporal-support

Model gốc của experiment này ở commit `89a3f89f0c92153150abdbdcd1108213e2b64111`.
Launcher rclone chỉ bổ sung cách khởi động/lưu/khôi phục; không thay đổi model,
loss, seed, data split, precision, batch hoặc learning rate.

| Recipe | YouCook2 | ActivityNet Captions |
|---|---:|---:|
| Full feature-matched train / validation | 1237 / 436 | 10009 / 4917 |
| Epoch | 15 | 6 |
| Event queries | 50 | 30 |
| Max count | 20 | 30 |
| Batch / accumulation / seed | 1 / 1 / 777 | 1 / 1 / 777 |
| Adam LR / weight decay | 5e-5 / 1e-4 | 5e-5 / 1e-4 |
| Precision / TF32 | FP32 / off | FP32 / off |
| Old mask loss | 0 | 0 |
| Support BCE / refine L1 / count loss | 1 / 1 / 1 | 1 / 1 / 1 |
| Post-processing | Gaussian Soft-NMS .5 only | Gaussian Soft-NMS .5 only |

Module: C1 learned proposal mask + query-conditioned temporal support/refinement.
Không concepts, merge hay dedup. GPT-2 random initialization 2 layers; không nạp
checkpoint cũ khi `start` fresh. CLIP features trích xuất sẵn/frozen.

## Một lần trên mỗi tài khoản mới

Chọn Molab GPU + Internet trước. Đưa file riêng `KLTN_DRIVE_PRIVATE.env` vào
`/marimo/KLTN_DRIVE_PRIVATE.env`. Không đặt file vào source hoặc KLTNRUNS.
File này chứa quyền OAuth; base64 không phải mã hóa bảo mật.

```bash
git clone --single-branch --branch exp/query-temporal-support https://github.com/thloc1212/DDVC.git /marimo/DDVC_QTS/source
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh auth
```

Trong một đợt so sánh, cả hai tài khoản phải checkout **cùng full commit hash**
được cung cấp trong hướng dẫn run. `git rev-parse HEAD` ghi trong provenance.

`auth` chỉ đọc đúng assignment RCLONE_CONFIG_B64 bằng parser, không `source .env`,
không in token. Nó phục hồi remote `kltnbackup` vào `/marimo/secrets/rclone.conf`
với mode600 và kiểm tra truy cập Drive. Nếu đã có config hợp lệ cùng client,
giữ token hiện tại để không ghi đè token đã refresh.

Nếu không tìm thấy rclone, cài binary từ https://rclone.org/install/ trước.
Không thay GPU Torch. Không upload khóa riêng vào GitHub/ZIP công khai.

## Acc 1: YouCook2 fresh

```bash
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh start yc2
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh logs yc2
```

## Acc 2: ActivityNet fresh

```bash
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh start anet
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh logs anet
```

`start` kiểm tra GPU và chạy worker tách khỏi terminal. Worker setup dependencies,
Java/CUDA, operator tests, chuẩn bị full data, GPU model smoke rồi mới train.
Setup hiện giữ nguyên native PyTorch/Torchvision; compiler CUDA13.0 riêng nếu cần.
Không reuse `.venv`, không restore compiled CUDA extension từ máy khác.

Xem log dùng `tail -F` với cả drive/setup log, startup/smoke log và train log.
Ctrl+C chỉ thoát xem log. Nó không đảm bảo Molab giữ session khi runtime hết hạn.

```bash
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh status yc2
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh status anet
```

Nếu lỗi setup hoặc prepare trước khi tạo run directory, sửa nguyên nhân rồi:

```bash
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh start yc2 --retry
```

Không dùng `--retry` để ghi đè một run đã train. Nếu có model-last, dùng `--resume`.

## Data dùng chung: lần đầu và các lần sau

Data cache được lưu tại:

```text
KLTNRUNS/data/yc2/qts-base-assets-v1/
KLTNRUNS/data/anet/qts-base-assets-v1/
```

Mỗi cache có prepared base annotations/vocabulary/train-only CLIP bank, filtered
CLIP feature map, config và hashes; không chứa model weights hay validation bank.
Một generation riêng được upload và `rclone check --one-way`; READY và LATEST
chỉ xuất hiện sau khi upload hoàn tất. Download kiểm tra SHA256 từng file trước
khi đưa data vào prepared folder. Giữ cùng work root `/marimo/DDVC_QTS` vì config
và QTS manifests chứa absolute paths. Hai dataset không dùng lẫn features/bank.

Lần đầu, nếu cache trống, launcher dùng downloader chính thức như recipe trước.
YC2 khoảng872MiB và ANET khoảng2.52GiB. Public Google Drive có thể bị quota;
rclone của bạn không loại bỏ quota download của file nguồn công khai.

Nếu đã có features đúng dataset, đặt một bản tại một trong hai vị trí Drive:

```text
KLTNRUNS/data/yc2/clipvitl14.pth
KLTNRUNS/data/anet/clipvitl14.pth
```

Launcher ưu tiên file đó trước downloader. Có thể đặt trực tiếp local tại
`/marimo/DDVC_QTS/assets/<dataset>/clipvitl14.pth` trước `start`.
Không đổi tên features YC2 thành ANET hoặc ngược lại.

Sau khi cache đã hoàn tất, session mới tải data cache riêng của bạn, không tải
lại CLIP features từ public link và không phải encode lại caption bank.
Không mount Drive để đọc tensor trực tiếp trong từng training iteration.

## Backup và vị trí kết quả

Mỗi acc tự tạo một run ID UTC+UUID, in ngay khi start. Không ghi đè run của acc khác:

```text
KLTNRUNS/
  code/<full_commit>/source.zip
  data/<dataset>/qts-base-assets-v1/...
  runs/ddvc_qts/<dataset>/<unique_run_id>/
    RUN_MANIFEST.json
    live_logs/
    recovery/
      slot-0/READY.json + run artifacts
      slot-1/READY.json + run artifacts
      LATEST.json
```

Log local:

```text
/marimo/DDVC_QTS/logs/yc2.drive.log
/marimo/DDVC_QTS/logs/yc2_qts_support_e15_b1.log
/marimo/DDVC_QTS/logs/anet.drive.log
/marimo/DDVC_QTS/logs/anet_qts_support_e6_b1.log
```

Kết quả local:

```text
/marimo/DDVC_QTS/runs/yc2_qts_support_e15_b1/
/marimo/DDVC_QTS/runs/anet_qts_support_e6_b1/
```

Mỗi120giây, worker copy log và kiểm tra checkpoint mới. Chỉ checkpoint thay đổi
mới tạo snapshot mới. Checkpoint được hard-link inode ổn định trước network copy
(train ghi bằng atomic replace). Text/log được copy riêng; secrets, venv,
toolchain, CUDA binaries, full feature maps và run của dataset kia bị loại khỏi
recovery. Video features nằm trong data cache; source ZIP riêng trong code/.

Hai slot luân phiên, kiểm tra file bằng `rclone check --one-way`, lưu SHA256,
READY ghi cuối và LATEST cập nhật sau READY. Upload dở không đánh dấu thành công;
restore kiểm tra toàn bộ hashes trước khi đưa file ra work root, thử slot cũ nếu
slot mới bị lỗi. Bản sao có cả model-last/model-best, optimizer/scheduler/RNG
trong checkpoint, config, bank, annotations, predictions, log, provenance.

Checkpoint vẫn theo boundary epoch hoặc trước/sau validation như train.py;
không hứa resume đúng micro-batch giữa epoch. Snapshot text/log có thể mới hơn
checkpoint, nhưng train resume lấy info/phase/RNG từ chính model-last format2.
Mất session có thể mất phần epoch chưa checkpoint hoặc chưa upload xong.

Chỉ `COMPLETE_BACKED_UP` xác nhận train xong và final snapshot đã kiểm tra.
`TRAIN_COMPLETE_BACKUP_PENDING`/`FINAL_BACKUP_PENDING` cần retry backup.

Nếu worker đã dừng, có thể backup thủ công:

```bash
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh backup yc2
```

Không chạy backup thủ công đồng thời worker; launcher chặn hai writer cùng run.
Không dùng rclone sync/purge để backup. Hai slot không giữ tất cả checkpoint
epoch lịch sử; giữ model-best và model-last theo state tại snapshot gần nhất.

## Khôi phục session gãy: dùng run ID đã lưu

Trên session mới, clone/checkout đúng commit của run, đưa file riêng vào /marimo,
chạy `auth` như trên. Restore trước setup/start để không ghi đè local artifacts:

```bash
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh restore anet --run-id RUN_ID_DA_LUU
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh start anet --resume
bash /marimo/DDVC_QTS/source/tools/qts_drive.sh logs anet
```

YC2 thay `anet` bằng `yc2`. Worker dựng lại runtime, lấy features từ cache đã
kiểm tra, dùng checkpoint smoke rồi mới resume. Không đổi epoch/query/count/
precision/accumulation giữa fresh và resume. Token Testing có thể hết hiệu lực;
nếu OAuth bị thu hồi thì cần xác thực lại, không phải lỗi checkpoint.

## Export để phân tích

```bash
bash /marimo/DDVC_QTS/source/tools/qts_molab.sh report yc2
bash /marimo/DDVC_QTS/source/tools/qts_molab.sh export yc2
```

YC2/ANET có thể export riêng. ZIP phân tích không chứa model weights/features.
Weights đã nằm trong verified recovery slots trên Drive.

## Kiểm tra đã có và giới hạn

`python tools/test_qts_rclone.py`: offline fake-transport tests cho private auth,
stable checkpoint inode, loại secrets/features/other dataset, network failure,
hash verification và fallback, shared data cache. Không có Drive thật trong test.
GPU train/smoke và OAuth/rclone upload thật phải được kiểm tra trên Molab.

Tài liệu gốc:
- https://rclone.org/remote_setup/
- https://rclone.org/commands/rclone_copy/
- https://rclone.org/commands/rclone_check/
