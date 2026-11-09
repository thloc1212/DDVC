# Trở về C1 và kiểm tra nguyên nhân — hai tài khoản Molab

## 1. Bản chạy này là gì?

Nhánh GitHub: https://github.com/thloc1212/DDVC/tree/exp/c1-grounding-diagnostics

Mốc mã cố định cho run mới: `c1-diagnostics-20261109-importfix`.
Mốc ban đầu `c1-diagnostics-20261109` giữ nguyên để restore các run đã chạy.

Đây là **C1 đã có trong ablation**: DDVC + confidence-gated learned proposal mask,
không auxiliary mask loss, dùng Gaussian Soft-NMS. Không có QTS, quality/relation
heads, ordinal-count loss, concept conditioning, merge hay dedup loss.

Model được lấy từ nhánh đã dọn gọn ở commit `4ce76e9`; cấu hình C1 được đóng băng
từ `info.json` của hai run `yc2_contribution_c1_e15_b1` và
`anet_contribution_c1_e8_b1`, lưu trong `tools/c1_frozen_options.json`.
ANET đổi thành **6 epoch theo yêu cầu**, nên không gọi đây là tái lập đầy đủ run 8 epoch.

| Thông số | YouCook2 | ActivityNet |
|---|---:|---:|
| Feature-matched train / validation | 1237 / 436 | 10009 / 4917 |
| Epoch | 15 | 6 |
| Queries | 50 | 10 |
| GT proposal sample budget | 10 | 10 |
| Counter logits | 21 (0–20) | 21 (0–20) |
| Count loss coefficient | 0.5 | 0.5 |
| Seed / batch / accumulation | 777 / 1 / 1 | 777 / 1 / 1 |
| Adam LR / weight decay | 5e-5 / 1e-4 | 5e-5 / 1e-4 |
| Precision / TF32 | FP32 / OFF | FP32 / OFF |
| Mask loss | 0 | 0 |
| Selection | Soft-NMS, sigma=0.5 | Soft-NMS, sigma=0.5 |
| Caption decoder | GPT-2 ngẫu nhiên, 2 layer | GPT-2 ngẫu nhiên, 2 layer |

Giữ Q10/GT budget10 để có đối chứng C1; đây không phải khẳng định Q10 tối ưu cho
ANET. GT vượt Q sẽ được ghi riêng trong báo cáo chẩn đoán. Chưa thêm module học mới.

Runtime giữ các sửa tương thích CUDA/NaN đã có; tokenizer GPT-2 được tải riêng có
checksum, không tải GPT-2 pretrained weights. Giữ nguyên GPU Torch của Molab.

## 2. Chuẩn bị mỗi tài khoản

- Chọn session GPU.
- Upload file riêng `KLTN_DRIVE_PRIVATE.env` vào `/marimo/`.
- Dùng terminal. Không `source` nội dung env; công cụ chỉ đọc giá trị base64, không
  thực thi lệnh hay in token.
- Dùng **session mới** hoặc đường dẫn `/marimo/DDVC_C1` chưa tồn tại. Giữ các run
  QTS cũ để đối chiếu; không giải nén/clone đè chúng.

Chạy trên **cả hai tài khoản**:

```bash
test -f /marimo/KLTN_DRIVE_PRIVATE.env
command -v rclone
git clone --single-branch --branch exp/c1-grounding-diagnostics https://github.com/thloc1212/DDVC.git /marimo/DDVC_C1/source
git -C /marimo/DDVC_C1/source checkout --detach c1-diagnostics-20261109-importfix
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh auth
```

Chỉ tiếp tục khi thấy `DRIVE_CONNECTED`. Nếu có lỗi OAuth/API, giữ nguyên source
và sửa kết nối trước. Không gửi nội dung env/token lên chat.

## 3. Tài khoản 1 — YouCook2

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh start yc2
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh logs yc2
```

## 4. Tài khoản 2 — ActivityNet

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh start anet
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh logs anet
```

`start` chạy ngầm: kiểm tra GPU → setup → tải/kiểm tra dữ liệu → smoke thật của
model (forward/backward + generation) → train từ đầu → backup. Nếu smoke lỗi thì
full training không được chạy. Lần đầu build CUDA có thể mất thời gian; tiến độ
nằm trong log Drive/setup. Không chạy đồng thời hai dataset trong cùng session.

Hai tài khoản dùng chung Drive vẫn độc lập: dataset và ID run duy nhất khác nhau.
Không cần shell variables `WORK`, `PY` hay `RUN`.

## 5. Dữ liệu, kết quả và log ở đâu?

Nếu có cache dữ liệu QTS trong `KLTNRUNS/data/<dataset>/qts-base-assets-v1`, công cụ
kiểm tra SHA256 rồi tái sử dụng annotations/features/bank, chỉ chuyển các đường
dẫn trong config. Không lấy weights hoặc các module QTS và **không upload thêm
một bản features giống nhau lên Drive**. ANET annotations/vocabulary trong ZIP
QTS đã được so với C1 cũ và khớp byte-for-byte.

`auth` không tải data; `start yc2` chỉ lấy YC2, `start anet` chỉ lấy ANET. Data được
stage về ổ local Molab để train. Nếu chưa có cache, trình tải gốc được dùng;
quota Google Drive công khai vẫn có thể xuất hiện.

Local:

```text
/marimo/DDVC_C1/logs/yc2.drive.log
/marimo/DDVC_C1/logs/yc2_c1_e15_b1.launch.log
/marimo/DDVC_C1/logs/yc2_c1_e15_b1.log
/marimo/DDVC_C1/runs/yc2_c1_e15_b1/

/marimo/DDVC_C1/logs/anet.drive.log
/marimo/DDVC_C1/logs/anet_c1_e6_b1.launch.log
/marimo/DDVC_C1/logs/anet_c1_e6_b1.log
/marimo/DDVC_C1/runs/anet_c1_e6_b1/
```

Trong run: `info.json`, `model-best.pth`, `model-last.pth`,
`prediction/num*_epoch*.json` (final đã chấm), và `.raw.json` (toàn bộ candidates).

Drive:

```text
KLTNRUNS/runs/ddvc_c1/yc2/<ID_DUY_NHAT>/
KLTNRUNS/runs/ddvc_c1/anet/<ID_DUY_NHAT>/
```

Log được đồng bộ khoảng 120 giây/lần. Checkpoint mới được lưu qua hai slot,
kiểm tra hash và ghi READY cuối cùng. Chờ `COMPLETE_BACKED_UP` trước khi coi run
đã lưu xong. `Ctrl+C` ở lệnh xem log chỉ thoát tail, không dừng worker train.

Xem trạng thái:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh status yc2
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh status anet
```

Chạy dòng ứng với dataset của tài khoản đó.

## 6. Sau train — kiểm tra count và ranking, không train lại

Nếu run đã train ở tag ban đầu và báo `No module named c1_pipeline` hoặc
`c1_diagnostics`: giữ nguyên checkout để không đổi source commit của backup.
Đợi lệnh backup hiện tại hoàn tất, rồi thêm đường dẫn tools trong terminal:

```bash
export PYTHONPATH="/marimo/DDVC_C1/source/tools:/marimo/DDVC_C1/source:/marimo/DDVC_C1/source/ddvc/ops${PYTHONPATH:+:$PYTHONPATH}"
```

Sau đó chạy lại các lệnh dưới đây. Không cần train lại; không đổi weights/config.
Run mới ở tag `importfix` đã tự khai báo các đường dẫn này. Có regression test
chạy CLI thật với Python safe-path và cwd khác thư mục repo.

YC2:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh diagnose yc2
cat /marimo/DDVC_C1/runs/yc2_c1_e15_b1/diagnostics/summary.tsv
```

ANET:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh diagnose anet
cat /marimo/DDVC_C1/runs/anet_c1_e6_b1/diagnostics/summary.tsv
```

Chấm bốn trường hợp trên **cùng raw của best checkpoint**:

| Case | N | Chọn proposal |
|---|---|---|
| `C1_softnms` | Counter | Soft-NMS gốc C1 |
| `oracle_N_ref0` | Số GT reference_0, giới hạn bởi Q | Soft-NMS |
| `oracle_rank_fixed_N` | Counter | GT tIoU + matching một-một |
| `oracle_N_and_rank_ref0` | GT reference_0, giới hạn bởi Q | GT tIoU + matching một-một |

Ba case oracle có GT hỗ trợ, chỉ dùng để chẩn đoán, không được đưa vào bảng kết
quả của phương pháp inference. Caption giữ nguyên. Oracle ranking tối ưu định vị
ở tIoU0.5, không tối ưu trực tiếp CIDEr; CIDEr không bắt buộc tăng.

`C1_softnms` phải tái lập metric lưu trong final, nếu khác công cụ dừng để kiểm
tra input/evaluator. Có thêm `strata.tsv` (ngắn/overlap/nested),
`count_histograms.json` và `per_video.tsv`.

Diễn giải: oracle N tăng nhiều → nút thắt count; oracle rank tăng nhiều → nút
thắt ranking; cả hai vẫn thấp → cần xem raw boundary/caption. Dùng cả official
metric và one-to-one recall chẩn đoán, không trộn hai định nghĩa.

## 7. Kiểm tra grounding bằng checkpoint — không update weights

Chỉ chạy sau train, khi worker không còn dùng GPU.

Thử 2 video trước trên mỗi tài khoản:

```bash
DDVC_PROBE_VIDEOS=2 LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh probe yc2
```

Hoặc trên tài khoản ANET:

```bash
DDVC_PROBE_VIDEOS=2 LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh probe anet
```

Nếu thành công, chạy mặc định 200 video:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh probe yc2
```

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh probe anet
```

Công cụ load `model-best.pth`, strict state dict, dùng `eval()` + no-grad, không
tạo optimizer. Cùng video chạy hai lần: nguyên bản và dịch vòng 25% chuỗi feature.
GT được dịch tương ứng; event đi qua điểm nối bị loại khỏi **cả hai** tập GT.
Nội dung ở điểm nối bị thay đổi cấu trúc nên đây là can thiệp chẩn đoán, không
phải benchmark thông thường hay bằng chứng nhân quả duy nhất.

Không buộc query index cố định: so sánh tập prediction bằng matching. Cả raw
caption, confidence, timestamp và counter của hai lần đều được lưu để xem.

Kết quả ở `runs/<run>/diagnostics/visual_probe/summary.json`, `pairs.json`,
`per_video.tsv`. Hash checkpoint được kiểm tra trước/sau. Nếu prediction vẫn gần
interval cũ hơn interval được dịch, đó là thêm bằng chứng về prior thời gian.

## 8. Backup thêm chẩn đoán và xuất ZIP gửi phân tích

Sau `diagnose` và `probe`, trên tài khoản YC2:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh backup yc2
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh export yc2
rclone --config /marimo/secrets/rclone.conf copy /marimo/DDVC_C1 kltnbackup:KLTNRUNS/analysis/c1/yc2 --max-depth 1 --include '/C1_ANALYSIS_yc2_*.zip' --progress
```

Tài khoản ANET:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh backup anet
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh export anet
rclone --config /marimo/secrets/rclone.conf copy /marimo/DDVC_C1 kltnbackup:KLTNRUNS/analysis/c1/anet --max-depth 1 --include '/C1_ANALYSIS_anet_*.zip' --progress
```

`export` in tên ZIP trong `/marimo/DDVC_C1/`. ZIP chứa source, config, log, raw/final,
metric và chẩn đoán; không chứa weights, features hay env/token. Weights nằm trong
recovery backup riêng để resume/probe. Không chạy backup thủ công khi worker tự
backup còn đang chạy; chờ `COMPLETE_BACKED_UP`.

## 9. Nếu session gãy

Tài khoản mới: upload env, clone và checkout đúng tag như bước 2; auth. Lấy ID từ
log `DRIVE_RUN` hoặc liệt kê:

```bash
rclone --config /marimo/secrets/rclone.conf lsf kltnbackup:KLTNRUNS/runs/ddvc_c1/anet --dirs-only
```

Thay `ID_RUN_DA_LUU` bằng đúng thư mục của run C1:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh restore anet --run-id ID_RUN_DA_LUU
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh start anet --resume
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh logs anet
```

YC2 thay `anet` bằng `yc2`. Restore giữ model/optimizer/scheduler/RNG/epoch từ
checkpoint đã xác minh; không đảm bảo tiếp tục chính xác minibatch giữa epoch.
Không restore QTS/event-set vào C1.

Nếu chỉ setup/prepare lỗi và chưa có run/checkpoint:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1/source/tools/c1_drive.sh start anet --retry
```

## 10. Phạm vi xác minh

Đã kiểm tra CPU: cấu hình hai full split, staging byte-for-byte, chống config
drift, caption/mask gradient khi auxiliary loss=0, parity với caption head ở
commit ablation gốc, kiểm tra oracle/matching/shift, hai-slot backup/restore và
nhập cache QTS có xác minh hash. Chưa chạy full training/GPU probe trên máy local;
launcher bắt buộc smoke GPU thật trên Molab trước full train. Không hứa metric
sẽ khớp tuyệt đối giữa môi trường hoặc tăng nhờ các công cụ chẩn đoán.

Giữ C1 làm mốc. Chỉ sau khi đọc các đối chứng trên mới chọn một thay đổi học mới
và thử riêng, không thêm đồng thời QTS/count/relation/concept lần nữa.
