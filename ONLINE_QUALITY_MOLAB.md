# C1 + Quality: học online cùng lúc trên hai tài khoản Molab

Ngày10/11/2026. Branch `exp/c1-online-joint-quality`.
Tag cố định: `online-quality-joint-20261110`.
Repo: https://github.com/thloc1212/DDVC/tree/exp/c1-online-joint-quality

## 1. Thí nghiệm này đo gì?

Hai giai đoạn trước: train C1, đóng băng C1, sinh cache candidates, học quality
head20epoch. Bản mới: **khởi tạo C1 và quality head từ đầu**, học trong cùng một
run15epochYC2/6epochANET. Không nạp C1 weights hoặc quality head cũ.

Giữ kiến trúc quality head280.577parameters và2312Dinput:
local CLIP768 + outside-context CLIP768 + generated-caption lexical embedding768
+8auxiliary scalars. Không thay thành query-state head, không dùng GT caption
làm đầu vào quality. Không bật mask loss/concept/relation/count-calibrator/merge.

```text
Video features → C1 đang học → timestamps/confidence + generated captions
                         ↓
                 live candidate features2312D
                         ↓
                    Quality head
                         ↓
       L_total = L_C1 + 0.1 × L_quality
```

Toàn bộ DDVC có optimizer và nhận các loss C1 gốc. Quality loss truyền gradient
qua confidence, duration và token embedding GPT-2 của các từ đã sinh; không
backprop qua lựa chọn token beam search hoặc membership của hard temporal pool.
Do đó không gọi đây là differentiable caption generation/ranking end-to-end.
Soft-NMS cũng là selection lúc inference, không phải loss differentiable.

## 2. Recipe và đối chứng

| Thông số | YouCook2 | ActivityNet |
|---|---:|---:|
| Train / validation videos |1237/436|10009/4917|
| DDVC + quality online epochs |15|6|
| Queries |50|10|
| Batch / accumulation |1/1|1/1|
| Seed |777|777|
| DDVC optimizer / LR / WD |Adam /5e-5 /1e-4|Adam /5e-5 /1e-4|
| Quality parameter group LR |.001|.001|
| Quality loss coefficient |.1|.1|
| Runtime |FP32, TF32off|FP32, TF32off|
| Mask auxiliary loss |0|0|
| Counter / candidate budget |C1 cũ|C1 cũ|
| Soft-NMS |Gaussian sigma.5|Gaussian sigma.5|

Hai parameter groups trong cùng Adam; LR cùng đi theo scheduler C1. C1 giữ
nguyên phần loss/schedule cũ. Clip gradient norm100 **riêng cho DDVC và quality
group** để head norm không làm rescale DDVC gradient trong detached control.
Head mới được init trong RNG fork; generation và scorer-dropout không tiến RNG
của C1. Checkpoint chứa cả C1, quality head, optimizer groups, scheduler và RNG.

Quality head dùng cùng objective như frozen: balanced Smooth-L1 trên sigmoid
quality, cộng0.2within-video preference loss. GT quality target vẫn là max theo
**cùng GT event** của tIoU × fit-IDF-weighted unigram F1; target stop-gradient.

TRAIN vẫn chia90/10 theo seed777. Quality loss chỉ học trên90% head-fit;10%
head-dev dùng chọn beta. DDVC vẫn học trên toàn TRAIN. Chưa phải out-of-fold
stacking, head-dev không phải dữ liệu chưa được C1 nhìn thấy.

So sánh với frozen có khác lịch huấn luyện/head optimizer/budget: frozen head
AdamW20epoch trên cache cố định; online head Adam parameter-group cùng DDVC
15/6epoch trên candidates thay đổi. Vì vậy **online vs frozen đo cả training
regime**, không được quy mọi chênh lệch chỉ cho gradient từ quality về DDVC.

Đã thêm chế độ `detached` có cùng kiến trúc, generation, optimizer, loss, RNG
isolation và thời lượng; chỉ ngắt gradient quality feature về C1. Joint-vs-detached
là đối chứng trực tiếp hơn cho tác dụng của quality feedback. Có thể chạy sau,
không phải mặc định phát sinh thêm hai full runs ngay bây giờ.

## 3. Trước khi chạy

- Chọn **GPU + Internet** trên mỗi tài khoản Molab.
- Upload `KLTN_DRIVE_PRIVATE.env` vào `/marimo/` của **cả hai tài khoản**.
- Dán vào terminal. Không dùng notebook Python cell, không `source` file key.
- Môi trường `/tmp/uv-venv/bin/python` có sẵn; giữ GPU Torch/torchvision gốc.
- Work mới `/marimo/DDVC_Joint_Quality`; không clone đè `/marimo/DDVC_C1` cũ.

Data: tự kiểm hash và tái sử dụng `KLTNRUNS/data/<dataset>/c1-base-assets-v1`
hoặc QTS data cache cũ, chỉ lấy data, **không lấy checkpoint/model cũ**. Config
được chuyển đường dẫn sang work mới sau khi kiểm bytes. Không nhân bản features
lên Drive khi đã có bundle. Nếu chưa có cache, downloader gốc được dùng;
YC2~872MiB, ANET~2.52GiB, cộng bank/annotation nhỏ hơn.

## 4. Tài khoản1 — YouCook2,15epoch

```bash
test -f /marimo/KLTN_DRIVE_PRIVATE.env
command -v rclone
git clone --single-branch --branch exp/c1-online-joint-quality https://github.com/thloc1212/DDVC.git /marimo/DDVC_Joint_Quality/source
git -C /marimo/DDVC_Joint_Quality/source checkout --detach online-quality-joint-20261110
LC_ALL=C LANG=C bash /marimo/DDVC_Joint_Quality/source/tools/online_quality_drive.sh auth
LC_ALL=C LANG=C bash /marimo/DDVC_Joint_Quality/source/tools/online_quality_drive.sh start yc2
LC_ALL=C LANG=C bash /marimo/DDVC_Joint_Quality/source/tools/online_quality_drive.sh logs yc2
```

## 5. Tài khoản2 — ActivityNet,6epoch

```bash
test -f /marimo/KLTN_DRIVE_PRIVATE.env
command -v rclone
git clone --single-branch --branch exp/c1-online-joint-quality https://github.com/thloc1212/DDVC.git /marimo/DDVC_Joint_Quality/source
git -C /marimo/DDVC_Joint_Quality/source checkout --detach online-quality-joint-20261110
LC_ALL=C LANG=C bash /marimo/DDVC_Joint_Quality/source/tools/online_quality_drive.sh auth
LC_ALL=C LANG=C bash /marimo/DDVC_Joint_Quality/source/tools/online_quality_drive.sh start anet
LC_ALL=C LANG=C bash /marimo/DDVC_Joint_Quality/source/tools/online_quality_drive.sh logs anet
```

`start` chạy nền: setup native environment/CUDA → kiểm data → real-model GPU
smoke forward/backward + generation → train mới → calibrate beta → đánh giá
4đối chứng → verified Drive backup. Auth/setup không tự train. Nếu cần setup
riêng trước khi start: `bash .../tools/online_quality_molab.sh setup yc2` hoặc anet.

Hai tài khoản cùng Drive không ghi đè nhau vì namespace có dataset/runID riêng.
`Ctrl+C` chỉ thoát tail. Không chạy đồng thời cả hai datasets trên một GPU.

## 6. Log cần thấy

```text
ONLINE_REUSED_VERIFIED_C1_ASSETS_NO_WEIGHTS
Fresh training: random decoder; no pretrained caption checkpoint requested.
ONLINE_QUALITY_GRADIENT_PASS joint {"confidence": true, "boxes": true, "quality_head": true}
REAL_MODEL_FORWARD_BACKWARD_PASS
GENERATION_POSTPROCESS_PASS
FULL_OFFICIAL_SCORER_SMOKE_PASS_SYNTHETIC
ONLINE_QUALITY_OPTIMIZER ...
EPOCH_START ...
PROGRESS ...
EPOCH_COMPLETE ...
ONLINE_CALIBRATION_TRAIN_DEV ...
ONLINE_POLICY_FROZEN_BEFORE_VALIDATION ...
ONLINE_RESULT ...
ONLINE_QUALITY_EVALUATION_COMPLETE ...
COMPLETE_BACKED_UP ...
```

Phần đầu có thể là `DATA_CACHE_VERIFIED` hoặc `VERIFIED_QTS_DATA_REUSED_FOR_C1`
nếu cache ở namespace khác. Đừng tạo marker smoke bằng tay. Nếu gradient route,
NaN, CUDA hoặc feature hash sai, workflow dừng trước khi tiếp tục full train.

Generation online làm mỗi step nặng hơn C1 cũ: mỗi video phải sinh caption cho
Qqueries bằng beam decoder. Không hứa số giờ trước khi đo thực tế. Generation
được chia8queries/chunk để hạn chế VRAM; không giảm Q, max caption length hoặc beam.

## 7. Chọn checkpoint và beta

- Trong training, chọn best checkpoint theo **METEOR+SODA_c của base Soft-NMS**,
  giữ cùng criterion C1. Validation mỗi epoch lưu raw candidates kèm live quality.
- Khi đủ15/6epoch, load best **cả C1+head**; không train thêm một optimizer khác.
- Generate candidates trên TRAIN head-dev, chọn beta từ {0,.25,.5,1,2} theo
  same-GT one-to-one joint reward, tie chọn beta nhỏ hơn.
- Policy ghi trước khi chấm4mode trên saved validation raw của best checkpoint.
- Nếu beta0 được chọn, quality inference chưa chứng minh contribution; không ép
  beta1/2 bằng cách nhìn validation.

Không sinh validation caption lại cho từng mode. Timestamp/caption/N giữ nguyên
giữa4mode của cùng online checkpoint. Các mode là:

| Method | Score | Selection |
|---|---|---|
| online_base_topk | DDVC joint score | Top-N |
| online_base_softnms | DDVC joint score | Soft-NMS |
| online_quality_topk | DDVC score + beta×quality | Top-N |
| online_quality_softnms | DDVC score + beta×quality | Soft-NMS |

`online_base_*` vẫn là DDVC đã được train cùng quality loss trong modejoint;
không được gọi là DDVC gốc hoặc C1 frozen checkpoint. Bốn dòng này chỉ tách lợi
ích dùng quality **lúc inference**, không tự tách contribution của joint training.

## 8. Xem kết quả

YC2:

```bash
cat /marimo/DDVC_Joint_Quality/runs/yc2_online_quality_joint_e15_b1/quality_evaluation/summary.tsv
cat /marimo/DDVC_Joint_Quality/runs/yc2_online_quality_joint_e15_b1/quality_evaluation/policy.json
```

ANET:

```bash
cat /marimo/DDVC_Joint_Quality/runs/anet_online_quality_joint_e6_b1/quality_evaluation/summary.tsv
cat /marimo/DDVC_Joint_Quality/runs/anet_online_quality_joint_e6_b1/quality_evaluation/policy.json
```

Chỉ đọc summary cuối khi có `ONLINE_QUALITY_EVALUATION_COMPLETE`; best scores
trong info.json là base Soft-NMS criterion, không phải bảng quality đã calibrate.

Khi calibrate/evaluate đang chạy mà lệnh `logs` chỉ hiển thị log train đã dừng,
xem log riêng:

```bash
tail -n 80 -F /marimo/DDVC_Joint_Quality/logs/yc2_online_quality_joint_e15_b1.evaluation.log
```

ANET:

```bash
tail -n 80 -F /marimo/DDVC_Joint_Quality/logs/anet_online_quality_joint_e6_b1.evaluation.log
```

## 9. Backup, session gãy và resume

Drive tự backup2slot có hash/READY, khoảng120giây/lần và lúc kết thúc. Đích:

```text
KLTNRUNS/runs/ddvc_online_quality/joint/yc2/<RUN_ID>/
KLTNRUNS/runs/ddvc_online_quality/joint/anet/<RUN_ID>/
  RUN_MANIFEST.json
  recovery/LATEST.json
  recovery/slot-0/READY.json
  recovery/slot-1/READY.json
```

Không ghi vào `ddvc_c1`, `ddvc_c1_rerank` hoặc `ddvc_c1_relation_count` cũ.
Model/optimizer/scheduler/RNG/head đều nằm chung checkpoint. Restore tiếp tục
từ epoch hoàn tất gần nhất; không đảm bảo đúng minibatch giữa epoch.

Còn local và worker đã dừng:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Joint_Quality/source/tools/online_quality_drive.sh start yc2 --resume
LC_ALL=C LANG=C bash /marimo/DDVC_Joint_Quality/source/tools/online_quality_drive.sh logs yc2
```

ANET thay yc2 bằng anet. Nếu chỉ setup/prepare lỗi và chưa có checkpoint, dùng
`start anet --retry` hoặc `start yc2 --retry`; không gọi resume khi chưa có model-last.

Session mới: clone đúng tag, auth như trên, liệt kê ID:

```bash
rclone --config /marimo/secrets/rclone.conf lsf kltnbackup:KLTNRUNS/runs/ddvc_online_quality/joint/anet --dirs-only
```

Thay `ID_DA_LUU` bằng tên thư mục thật:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Joint_Quality/source/tools/online_quality_drive.sh restore anet --run-id ID_DA_LUU
LC_ALL=C LANG=C bash /marimo/DDVC_Joint_Quality/source/tools/online_quality_drive.sh start anet --resume
LC_ALL=C LANG=C bash /marimo/DDVC_Joint_Quality/source/tools/online_quality_drive.sh logs anet
```

YC2 thay đường dẫn/dataset bằng yc2. Source/mode/work/recipe khác thì restore bị
từ chối. Beta đã chốt không bị chọn lại khi chỉ chấm tiếp evaluation; đổi checkpoint
hoặc raw/config semantics sẽ bị chặn. Calibration shards cũng được đưa vào backup.

## 10. Xuất ZIP phân tích

Sau `COMPLETE_BACKED_UP`, tài khoảnYC2:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Joint_Quality/source/tools/online_quality_drive.sh export yc2
```

Tài khoảnANET:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_Joint_Quality/source/tools/online_quality_drive.sh export anet
```

ZIP `ONLINE_QUALITY_ANALYSIS_<dataset>_joint_<time>.zip` trong work. Có source,
config,logs,loss,policy,raw/final predictions và4mode metrics; không chứa private.env,
token, CLIP archive hoặc checkpoint DDVC lớn. Weights nằm trong recovery trên Drive.

## 11. Detached control nếu cần kiểm chứng riêng gradient

Chỉ chạy sau khi joint run trên tài khoản đó đã kết thúc. Không cần clone mới:

```bash
DDVC_QUALITY_MODE=detached LC_ALL=C LANG=C bash /marimo/DDVC_Joint_Quality/source/tools/online_quality_drive.sh start yc2
DDVC_QUALITY_MODE=detached LC_ALL=C LANG=C bash /marimo/DDVC_Joint_Quality/source/tools/online_quality_drive.sh logs yc2
```

ANET thay yc2 bằng anet. Đây là **một run train từ đầu khác**, không fine-tune
joint checkpoint. Runname có `_detached_`, Drive namespace là
`KLTNRUNS/runs/ddvc_online_quality/detached/<dataset>/<RUN_ID>/`.
Đọc/export/resume/restore control cũng phải thêm `DDVC_QUALITY_MODE=detached`.

## 12. Phạm vi kiểm chứng

CPU tests: forward-feature parity trong sai số cacheFP16, actual tiny GPT-2
generation + quality backward, joint/detached gradient routes, RNG preservation,
head-dev label isolation, separate clipping, same-GT target, score/budget parity,
beta freeze/resume/model-identity guards, CLI safe imports, hash-verified C1 data
relocation, online backup namespace và slot corruption fallback. Có regression
tests C1 cũ để kiểm phần không bật online vẫn giữ hành vi cũ.

Local không có CUDA. Workflow trên Molab bắt buộc chạy smoke **mô hình DDVC thật**,
gồm forward/backward/generated captions và kiểm quality gradient route, trước
khi full train; scorer Java thật được kiểm bằng dữ liệu synthetic, không dùng
GT validation để làm smoke hay chọn beta. Chưa có metric full-run mới và không
bảo đảm joint sẽ tốt hơn frozen.
