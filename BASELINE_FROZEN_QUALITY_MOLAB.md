# DDVC global baseline + frozen quality: hai tài khoản Molab

Branch: [`exp/baseline-frozen-quality`](https://github.com/thloc1212/DDVC/tree/exp/baseline-frozen-quality).
Release cố định: [`baseline-frozen-quality-20261111-importfix`](https://github.com/thloc1212/DDVC/tree/baseline-frozen-quality-20261111-importfix).

Đây là đối chứng C0 **global visual attention** theo cùng recipe với thí nghiệm C1/frozen quality. Giữ DDVC, retrieval bank và GPT-2 khởi tạo ngẫu nhiên; **tắt learned proposal mask**, mask loss, concept, QTS, event-set, dedup loss và mọi merge. Đây là đối chứng trong code hiện tại, không khẳng định tái lập chính xác con số DDVC công bố.

## Hai giai đoạn

1. Train toàn bộ phần DDVC học được; dùng CLIP features trích xuất sẵn. Validation mỗi epoch với Soft-NMS như recipe C1. Lưu best checkpoint theo MetaScore.
2. Khi đủ số epoch và đã xác minh backup baseline, tự đóng băng best checkpoint. Trích xuất candidates TRAIN, train riêng cùng quality head 2312D → scalar như bản frozen C1; giữ nguyên timestamps, captions và counter. Đánh giá bốn cách chọn trên cùng candidates VAL.

| Thông số | Account 1: YouCook2 | Account 2: ActivityNet Captions |
|---|---:|---:|
| Train / validation videos | 1237 / 436 | 10009 / 4917 |
| Epoch DDVC | 15 | 6 |
| Queries | 50 | 10 |
| Batch DDVC / accumulation | 1 / 1 | 1 / 1 |
| Precision / TF32 | FP32 / OFF | FP32 / OFF |
| Optimizer DDVC | Adam, lr 5e-5, WD 1e-4 | Adam, lr 5e-5, WD 1e-4 |
| Seed | 777 | 777 |
| Caption visual context | Global; padding mask only | Global; padding mask only |
| Quality head | 280577 tham số; 20 epochs | 280577 tham số; 20 epochs |
| Optimizer head | AdamW, lr .001, WD 1e-4; 8 videos/batch | Giống YouCook2 |
| Head fit / head dev | 90% / 10% của TRAIN, seed 777 | Giống YouCook2 |
| Beta grid | 0, .25, .5, 1, 2 | Giống YouCook2 |
| Gaussian Soft-NMS | sigma=.5 | sigma=.5 |

Head và beta được chọn từ head-dev của TRAIN; VAL chỉ dùng chấm metric ở bước cuối. Tuy nhiên DDVC ở giai đoạn 1 đã học toàn bộ TRAIN, gồm head-dev. Đây không phải một holdout chưa từng được backbone nhìn thấy. Best DDVC checkpoint cũng được chọn theo validation như run C1 cũ; kết quả báo cáo là validation, không phải test độc lập.

**Không lấy lại quality head C1 đã train để gắn thẳng vào baseline**: candidate distribution và GPT-2 embedding khác, phải fit head mới. Kiến trúc/loss quality giữ nguyên; beta được chọn lại cho từng baseline/dataset. Quality + Top-N dùng cùng beta đã chọn cho Quality + Soft-NMS, không dò beta trên VAL.

## Sửa lỗi setup/import ở các release đầu (chưa bắt đầu train)

Nếu bị AssertionError tại `check_global_caption` trong setup: release đầu dùng phép so sánh toàn bộ logits FP32 với `atol=1e-6` khi đổi padding thành 1000. Log cũ không lưu mức lệch nên chưa xác định được nguyên nhân số học trên runtime đó. Bản sửa kiểm tra trực tiếp attention vào padding bằng 0, in mức lệch FP32 để chẩn đoán, và dùng FP64 chỉ cho phép thử padding invariance. Full training vẫn FP32; không thay mô hình/loss/lr/epoch.

Nếu CUDA đã báo `EXTENSION_LOAD_PASS`, `PASS: cuda FP32 ...` rồi dừng với `No module named 'baseline_assets'`: CUDA đã hoạt động, lỗi là worker chưa thêm `source/tools` vào import path khi Python bật safe-path. Bản importfix bootstrap đường dẫn ngay trong worker, kiểm tra lại CPU smoke khi dùng CUDA đã build, và cho phép retry sau CUDA readiness khi chưa có run train/checkpoint. Không thay đổi model, LR, batch hay epoch.

Trên account đã lỗi, thoát tail bằng Ctrl+C, rồi cập nhật:

```bash
git -C /marimo/DDVC_Baseline_Quality/source fetch origin tag baseline-frozen-quality-20261111-importfix
git -C /marimo/DDVC_Baseline_Quality/source checkout --detach baseline-frozen-quality-20261111-importfix
```

Account YC2:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh retry yc2
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh logs yc2
```

Account ANET nếu cũng bị cùng lỗi:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh retry anet
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh logs anet
```

`retry` chỉ nhận worker setup/data đã thất bại và chưa có folder train/checkpoint/head; CUDA có thể đã sẵn sàng và được giữ lại. Nó lưu lại state lỗi, ghi lịch sử source fix rồi chạy tiếp cả hai giai đoạn. Không xóa data, log hay credentials. Nếu account kia chưa khởi động run lần nào thì dùng `run anet` sau khi cập nhật, không dùng `retry`.

Các mốc cần thấy: `PADDING_FP32_DIAGNOSTIC`, `PADDING_FP64_INVARIANCE_PASS`, rồi smoke mô hình thật và `EPOCH_START`. Nếu xuất hiện `Padding received cross-attention` hoặc `FP64 padding invariance failed`, giữ nguyên lỗi và gửi log; không bỏ qua kiểm tra.

## 1. Chuẩn bị cho cả hai tài khoản

Chọn GPU trên Molab. Upload riêng file `KLTN_DRIVE_PRIVATE.env` vào `/marimo` trên **mỗi tài khoản**. File phải có remote đã xác thực `kltnbackup`; Drive API đã bật.

Mỗi account dùng một dataset. Cùng đường dẫn local trên hai máy khác nhau không xung đột. Các run YC2/ANET được lưu ở hai nhánh Drive riêng, có run ID riêng. Source và shared data có thể dùng chung. Không upload `.env` vào Git hoặc ZIP phân tích.

Trong terminal **của cả hai account**, chạy:

```bash
test -f /marimo/KLTN_DRIVE_PRIVATE.env
command -v rclone
git clone --single-branch --branch baseline-frozen-quality-20261111-importfix https://github.com/thloc1212/DDVC.git /marimo/DDVC_Baseline_Quality/source
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh auth
```

Clone từ tag cố định nên Git có thể báo detached HEAD; đó là chủ đích. Lệnh clone không ghi đè folder có sẵn. Không checkout/pull vào folder đang train.

## 2. Account 1: YouCook2

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh run yc2
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh logs yc2
```

## 3. Account 2: ActivityNet Captions

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh run anet
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh logs anet
```

`run` khởi động coordinator chạy nền. Nó tự setup môi trường, dựng CUDA extension cho GPU hiện tại, smoke forward/backward qua mô hình thực, train DDVC, xác minh backup, rồi chuyển sang frozen quality. Giữ nguyên GPU torch/torchvision của Molab; không tạo thêm venv.

Data được lấy từ shared cache Drive đã xác minh hash và đổi đường dẫn local. **Chỉ tái sử dụng data, không tái sử dụng trọng số DDVC.** Nếu cache chưa có, downloader gốc dùng khoảng 872 MiB cho YC2 và 2.52 GiB cho ANET CLIP features, thêm annotations/train-caption bank. Data được stage local để train; không đọc feature trực tiếp qua Drive từng batch. Public Drive có thể quota nếu phải tải mới.

Lúc chưa đến giai đoạn quality, `tail -F` có thể báo file `frozen_quality/...rerank.log` chưa tồn tại; nó sẽ tự theo dõi khi file xuất hiện. `Ctrl+C` chỉ thoát xem log, không dừng coordinator/train.

## 4. Nhận biết tiến độ và xem kết quả

Các mốc log:

- `BASELINE_GLOBAL_OK ... learned_mask False`: đúng baseline global.
- `EPOCH_START` / `PROGRESS` / `EPOCH_COMPLETE`: train DDVC; 15 epoch YC2, 6 epoch ANET.
- `BASELINE_COMPLETE_STARTING_FROZEN_QUALITY`: xong DDVC và backup, bắt đầu giai đoạn 2.
- `FROZEN_BASELINE_READY`, cache extraction: đang tạo đặc trưng reranker từ checkpoint tốt nhất.
- `HEAD_EPOCH`: head-only training, 20 epochs.
- `HEAD_FROZEN_BEFORE_VALIDATION ... beta ...`: đã chọn head/beta từ TRAIN head-dev.
- `RESULT`: kết quả chấm lần lượt từng selection mode.
- `TWO_STAGE_COMPLETE_BACKED_UP`: cả hai giai đoạn đã xong và backup được xác minh.

YC2:

```bash
cat /marimo/DDVC_Baseline_Quality/frozen_quality/runs/yc2/evaluation/summary.tsv
```

ANET:

```bash
cat /marimo/DDVC_Baseline_Quality/frozen_quality/runs/anet/evaluation/summary.tsv
```

| Hàng kết quả | Ý nghĩa |
|---|---|
| `baseline_topk` | DDVC global, chọn Top-N bằng score gốc |
| `baseline_softnms` | Cùng baseline + Gaussian Soft-NMS |
| `reranker_topk` | Baseline + quality, không Soft-NMS |
| `reranker_softnms` | Baseline + quality + Soft-NMS |

Score trước suppression là `base_score + beta * quality`. Counter N giữ nguyên ở cả bốn hàng; không sửa/gộp caption hay timestamp. Các metric trong TSV đã nhân 100. So hai hàng đầu để đo ảnh hưởng Soft-NMS; so hàng 2 và 4 để đo quality khi cùng dùng Soft-NMS. Khi so với C1 + quality, phải ghi rõ checkpoint, budget, seed và beta của mỗi run; một seed chưa đủ kết luận ổn định thống kê.

Baseline checkpoints và predictions:

```text
/marimo/DDVC_Baseline_Quality/runs/yc2_baseline_e15_b1/
/marimo/DDVC_Baseline_Quality/runs/anet_baseline_e6_b1/
```

Quality head, manifest, cache, beta, kết quả:

```text
/marimo/DDVC_Baseline_Quality/frozen_quality/runs/yc2/
/marimo/DDVC_Baseline_Quality/frozen_quality/runs/anet/
```

## 5. Backup và gửi kết quả

Worker tự backup mỗi 120 giây; baseline snapshot mới sau khi có checkpoint mới. Hai slot xác minh checksum, READY viết sau cùng. Baseline lưu model/optimizer/scheduler/RNG/config/log; frozen quality lưu cache đã commit, head, optimizer, folds, beta và metrics. Source commit cũng được lưu riêng. Có thể mất phần train của epoch hiện tại nếu session mất trước checkpoint tiếp theo.

Drive:

```text
kltnbackup:KLTNRUNS/runs/ddvc_baseline/yc2/<run_id>/
kltnbackup:KLTNRUNS/runs/ddvc_baseline/anet/<run_id>/
kltnbackup:KLTNRUNS/runs/ddvc_baseline_rerank/yc2/<run_id>/
kltnbackup:KLTNRUNS/runs/ddvc_baseline_rerank/anet/<run_id>/
```

Sau khi hoàn tất, account YC2 chạy:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh export yc2
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh quality-export yc2
```

Account ANET chạy:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh export anet
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh quality-export anet
```

Mỗi account in hai đường dẫn `ZIP_READY`: ZIP baseline ở work root và ZIP quality ở `frozen_quality`. Tải cả hai ZIP của mỗi dataset để đối chiếu. ZIP phân tích không chứa checkpoint DDVC lớn; checkpoint đầy đủ vẫn ở recovery trên Drive. ZIP quality gồm small head và VAL feature cache, không gồm TRAIN feature cache.

## 6. Nếu gãy hoặc cần chạy thủ công từng giai đoạn

Xem trạng thái ở terminal khác:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh status yc2
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh quality-status yc2
```

Account ANET thay `yc2` bằng `anet`. Chưa đến giai đoạn 2 thì chưa có quality state.

Không gọi `run` lần nữa trên output đã tồn tại. Nếu baseline đã hoàn tất trên cùng session nhưng quality chưa bắt đầu, chạy:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh quality yc2
```

Nếu quality đã tạo state nhưng gãy, dùng `quality yc2 --resume`. Nếu baseline gãy và còn local checkpoint, dùng `start yc2 --resume`; nếu gãy trước checkpoint đầu, kiểm tra log rồi dùng `start yc2 --retry`. Sau khi baseline chạy lại xong, gọi quality thủ công. Chỉ một worker cho một work root.

Session mới: clone **cùng tag vào cùng đường dẫn**, upload private env và auth như bước 1. Liệt kê run ID baseline và quality:

```bash
rclone --config /marimo/secrets/rclone.conf lsf kltnbackup:KLTNRUNS/runs/ddvc_baseline/yc2 --dirs-only
rclone --config /marimo/secrets/rclone.conf lsf kltnbackup:KLTNRUNS/runs/ddvc_baseline_rerank/yc2 --dirs-only
```

Dùng đúng ID đã lưu; thay placeholder ở các lệnh dưới bằng tên folder thực:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh restore yc2 --run-id BASELINE_RUN_ID
```

Nếu baseline chưa xong: `start yc2 --resume` sẽ setup lại môi trường/CUDA và tải lại feature từ cache, rồi resume. Nếu baseline đã xong và cần khôi phục quality, **không train thêm epoch**; cần dựng lại môi trường và restage features:

```bash
DDVC_RUN_VARIANT=baseline_quality LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/c1_molab.sh setup
```

Sau setup, lấy feature từ cache Drive mà **không sửa config baseline**, rồi restore head:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh restore-data yc2
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh quality-restore yc2 --run-id QUALITY_RUN_ID
LC_ALL=C LANG=C bash /marimo/DDVC_Baseline_Quality/source/tools/baseline_quality.sh quality yc2 --resume
```

Không gọi baseline resume sau khi đã fit quality chỉ để lấy data: nó có thể sửa trường `start_from`, làm thay đổi config hash mà frozen head đang gắn với. `restore-data` chỉ khôi phục feature bytes và xác minh config hash không thay đổi.

Nếu trước đây chưa chạy quality thì bỏ `quality-restore` và gọi `quality yc2` không có `--resume`. Account ANET thay dataset bằng `anet`. Giữ nguyên config/source/paths khi resume; không dùng checkpoint C1 mask để resume baseline global.

## 7. Mã và kiểm tra

- Global context, không feature gating: `ddvc/CaptioningHead/gpt2.py::visual_context`.
- Recipe/guard: `misc/baseline_recipe.py`, `tools/c1_pipeline.py`.
- Quality input/target/scoring: `reranking/core.py`; network/loss giữ nguyên trong `reranking/head.py`.
- Fit/dev/beta và bốn cách evaluate: `reranking/pipeline.py`.
- Coordinator: `tools/baseline_quality_chain.py`.
- Private auth, shared cache, model backup: `tools/c1_rclone.py`, `tools/baseline_assets.py`.
- Frozen quality backup/restore: `tools/c1_rerank.py`.

Kiểm tra local gồm full-split config/staging, GPT-2 forward/backward thật ở chế độ global, parity khởi tạo phần chung với C1, frozen head fit/resume, bốn mode evaluation trên fixture, selector parity, backup slot/checksum và loại secret khỏi backup. Chưa train full trên GPU Molab tại máy local; launcher bắt buộc compile/import CUDA và real-model smoke trước full training.
