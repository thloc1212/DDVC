# C1 + Query-conditioned Temporal Support Learning

Branch: [`exp/query-temporal-support`](https://github.com/thloc1212/DDVC/tree/exp/query-temporal-support).
Nhánh C2 cuối trước đây vẫn ở [`final/c2-learned-mask`](https://github.com/thloc1212/DDVC/tree/final/c2-learned-mask).

## Những gì đã thay đổi

| Thiết lập | YouCook2 | ActivityNet Captions |
|---|---:|---:|
| Full feature-matched train / val | 1237 / 436 | 10009 / 4917 |
| Epoch | **15** | **6** |
| Queries | **50** | **30** |
| `gt_proposal_sample_num` | **20** | **30** |
| `max_eseq_length` | **20** | **30** |
| Count logits | 21 | **31** |
| Count loss coefficient | **1.0** | **1.0** |
| Old proposal-mask loss coefficient | **0.0** | **0.0** |
| New temporal-support loss coefficient | **1.0** | **1.0** |
| New temporal-refine L1 coefficient | **1.0** | **1.0** |
| Seed / batch / accumulation | 777 / 1 / 1 | 777 / 1 / 1 |
| Precision / TF32 | FP32 / off | FP32 / off |
| Adam LR / weight decay | 5e-5 / 1e-4 | 5e-5 / 1e-4 |
| Selection | Gaussian Soft-NMS .5 | Gaussian Soft-NMS .5 |

Đây là nhánh mới dựa trên **C1: learned proposal mask, không old mask loss**. GPT-2 vẫn random initialization (2 layers), dùng CLIP features trích xuất sẵn. Không nạp checkpoint C1/C2 để train fresh. Concepts, merge và dedup tắt.

Gaussian count-loss formulation giữ nguyên; tăng hệ số từ .5 lên 1.0. Bảng tần suất lớp count được tính lại **chỉ từ full training annotations** theo 21/31 lớp. Không padding bảng hard-coded bằng số tuỳ ý và không dùng validation để tính prior.

Loader giữ **toàn bộ event và caption** trong cả train/validation. Các giới hạn 20/30 trở thành kiểm tra capacity: nếu annotation vượt query/count/GT budget, báo lỗi thay vì âm thầm cắt GT. Trong annotations hiện tại, train max là 16 (YC2) và 27 (ANET). Matching, class labels, caption labels và count target cùng dùng một tập GT đầy đủ.

Giữ LR decay start=8/every=3/rate=.5 như recipe trước. ANET dừng ở epoch6 nên chưa tới LR milestone; không tự rescale schedule. Việc đổi epoch, query/count head và GT protocol yêu cầu **train mới**. Resume chỉ dùng checkpoint của cùng nhánh, cùng variant/config.

## Module mới hoạt động thế nào?

```text
DDVC final decoder layer
  ├─ localization query h_loc [B,Q,512]
  ├─ coarse center-duration [B,Q,2]
  └─ frozen CLIP features F [B,T,768]
                     ↓
     query projection + frame projection + relative-time features
                     ↓
       support logits / sigmoid weights [B,Q,T]
       (independent per query, padding excluded)
                     ↓
       visual pooling + support temporal moments
                     ↓
       bounded correction to center/duration logits
                     ↓
       refined center-duration [B,Q,2]
          ├─ Hungarian matching + localization loss
          ├─ C1 proposal mask → GPT-2 caption generation
          └─ raw candidates → counter-guided Soft-NMS → final
```

Forward chỉ nhận query, visual features, predicted boxes và padding mask. **GT không đi vào forward/inference mask.** Sau matching, objective lấy GT của từng matched query để tạo fractional occupancy theo các temporal feature bins. Event ngắn hơn một bin vẫn có target dương. Mỗi query có map riêng nên hai GT lồng nhau có thể đồng thời là positive ở cùng temporal step.

- `loss_temporal_support`: BCE cân bằng foreground/background theo từng matched event. Đây là GT supervision, khác với mask loss cũ bắt chước detached predicted hard window.
- `loss_temporal_refine`: L1 giữa refined center/duration và matched GT. GIoU loss gốc vẫn áp dụng lên refined boxes. `bbox_loss_coef=0` của loss cũ vẫn được giữ; loss L1 mới có key/trọng số riêng.
- Chỉ thêm support/refinement ở **final decoder layer**; intermediate DDVC auxiliary layers giữ nguyên.
- Refiner zero-initialization để bắt đầu gần identity. Module khởi tạo trong RNG fork riêng để không thay RNG của các thành phần cũ.
- Với hidden support dim128, nhánh mới có 198.435 tham số học được; tensor support là Q×T, không tạo thêm event queries hoặc decoder GPT-2.
- Không nhân support map mới trực tiếp vào CLIP trước GPT-2. Nó tinh chỉnh interval; C1 mask hiện có sử dụng interval đó để điều tiết caption context.

Kỳ vọng là cải thiện raw proposal recall và boundary, chưa có kết quả metric mới để khẳng định hiệu quả. Feature temporal resolution vẫn là giới hạn nếu evidence của event ngắn đã mất trước model.

Hướng boundary refinement có liên quan tới [BMN, ICCV2019](https://arxiv.org/abs/1907.09702) và [AFSD, CVPR2021](https://arxiv.org/abs/2103.13137). Đây là thiết kế thích nghi vào DDVC với query-conditioned GT support; không transplant code/weights hoặc tuyên bố tái lập toàn bộ hai phương pháp đó. Tính mới và hiệu quả cần được đánh giá qua related work/ablation.

## Chạy tài khoản mới — terminal Molab

Mỗi tài khoản chạy một dataset. Môi trường Python GPU hiện có được giữ nguyên; không tạo venv mới hoặc thay GPU Torch/Torchvision. Bật GPU và Internet.

### Bước chung trên cả hai tài khoản

```bash
git clone --single-branch --branch exp/query-temporal-support https://github.com/thloc1212/DDVC.git /marimo/DDVC_QTS/source
LC_ALL=C LANG=C bash /marimo/DDVC_QTS/source/tools/qts_molab.sh setup
```

Setup cài dependency, test CPU, Java, CUDA extension và CUDA operator test. Chỉ tiếp tục khi thành công; không tự tạo marker `cuda_ready`.

### Tài khoản 1 — YouCook2, 15 epoch

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_QTS/source/tools/qts_molab.sh prepare yc2
LC_ALL=C LANG=C bash /marimo/DDVC_QTS/source/tools/qts_molab.sh train yc2
bash /marimo/DDVC_QTS/source/tools/qts_molab.sh logs yc2
```

### Tài khoản 2 — ActivityNet, 6 epoch

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_QTS/source/tools/qts_molab.sh prepare anet
LC_ALL=C LANG=C bash /marimo/DDVC_QTS/source/tools/qts_molab.sh train anet
bash /marimo/DDVC_QTS/source/tools/qts_molab.sh logs anet
```

Default variant là `support` (bản đầy đủ). Train chạy ngầm. Ctrl+C chỉ thoát xem log. `status yc2` / `status anet` xem PID. Mỗi lần train/resume đều chạy real-model GPU smoke trước, gồm forward/backward, generation/postprocess, finite gradients và kiểm tra nhánh QTS. Smoke lấy video nhiều GT nhất trong split train để kiểm tra áp lực caption/matching sau tăng budget.

Features được tải riêng: YC2 khoảng 872 MiB, ANET khoảng 2.52 GiB. Không dùng raw video. Nếu Drive bị quota, dùng bản features đúng dataset đã tải:

```bash
DDVC_FEATURES=/marimo/anet_clipvitl14.pth bash /marimo/DDVC_QTS/source/tools/qts_molab.sh prepare anet
```

Nếu assets đã có, có thể dùng `DDVC_QTS_SOURCE_CONFIG=/đường/dẫn/config.yml` trước lệnh prepare để tái sử dụng C1/C2 data config cùng dataset. Source annotation/vocabulary/references/bank được stage byte-for-byte; không reuse weights. Các thay đổi budget/epoch/count prior được ghi rõ trong config/manifest mới.

## Logs, kết quả và export

Run IDs:

- YC2: `yc2_qts_support_e15_b1`.
- ANET: `anet_qts_support_e6_b1`.

Log startup/smoke: `/marimo/DDVC_QTS/logs/<run>.launch.log` và `<run>.smoke.log`.
Log train: `/marimo/DDVC_QTS/logs/<run>.log`.
Config và `qts_manifest.json`: `/marimo/DDVC_QTS/prepared/<run>/`.
Kết quả: `/marimo/DDVC_QTS/runs/<run>/` với `model-best.pth`, `model-last.pth`, `info.json` và `prediction/`.

Theo dõi thêm `loss_counter`, `counter_count_error`, `loss_temporal_support`, `loss_temporal_refine`, `support_inside`, `support_outside`, `support_boundary_delta`. Các chỉ số support là diagnostic, không phải metric benchmark.

Prediction `.raw.json` giữ all Q candidates; `.json` là final được chấm metric. Khi QTS bật, từng row có thêm `coarse_timestamp` và `support_mean` để đối chiếu trước/sau refinement. Cả raw/final lưu `counter_predictions` theo video gồm logits và predicted N, giúp kiểm tra collapse count. Không gọi sigmoid/softmax của các logits đó là xác suất đã calibration.

```bash
bash /marimo/DDVC_QTS/source/tools/qts_molab.sh report yc2
bash /marimo/DDVC_QTS/source/tools/qts_molab.sh export yc2
```

ANET đổi `yc2` thành `anet`. ZIP phân tích dưới work root có source, config, annotations/references, logs và prediction; không có checkpoint weights/video features. Lưu checkpoints qua Drive backup trước khi session hết.

## Ablation đã chuẩn bị sẵn

| Variant | Support/refinement module | GT support BCE | Refine L1 |
|---|---:|---:|---:|
| `baseline` | Tắt | 0 | 0 |
| `refine` | Bật | 0 | 1 |
| `support` (default) | Bật | 1 | 1 |

Cả ba dùng C1 mask loss0, full GT, count loss1, cùng query/count budget, epoch, data IDs và recipe. `baseline` là **đối chứng C1 theo protocol mới**, không phải run C1 cũ. So sánh với các run cũ đồng thời thay đổi nhiều yếu tố nên không đủ để kết luận riêng contribution của QTS.

Ví dụ chạy control ANET:

```bash
bash /marimo/DDVC_QTS/source/tools/qts_molab.sh prepare anet baseline
bash /marimo/DDVC_QTS/source/tools/qts_molab.sh train anet baseline
bash /marimo/DDVC_QTS/source/tools/qts_molab.sh logs anet baseline
```

`refine` tương tự. Một work/GPU chỉ cho một train supervisor chạy; không xếp chồng các variant. Đánh giá raw recall theo duration, GT overlap và boundary coverage trước khi đọc caption metrics. Giữ counter/selection protocol giống nhau trong bảng ablation.

## Backup Google Drive và resume

Cell Python Molab xác thực theo API hiện có:

```python
from gdrive_fsspec import GoogleDriveFileSystem
fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
    auth_kwargs={"use_local_webserver": False})
```

Sau khi xác thực, cell khác:

```python
import sys
sys.path.insert(0, "/marimo/DDVC_QTS/source/tools")
from drive_backup import DriveBackup
qts_backup = DriveBackup(fs, work="/marimo/DDVC_QTS",
    remote_root="runs/ddvc_qts/yc2", interval=120)
qts_backup.start()
```

Tài khoản ANET dùng remote root `runs/ddvc_qts/anet`. Hai tài khoản không ghi cùng remote root. Theo dõi `/marimo/DDVC_QTS/logs/drive_sync.log`. Trước khi kết thúc, chờ cell sau chạy xong:

```python
qts_backup.stop()
print("BACKUP_DONE:", qts_backup.sync_once(force=True))
```

Backup hai slot, `READY.json` ghi cuối; lưu checkpoint/config/annotation/bank/logs/provenance. Không lưu full video features hoặc source. Session mới clone đúng commit và setup, xác thực `fs`, tạo `qts_backup` nhưng chưa start, rồi gọi `qts_backup.restore_latest()`. Giữ nguyên `/marimo/DDVC_QTS` để các manifest absolute paths khớp. Tiếp theo terminal:

```bash
bash /marimo/DDVC_QTS/source/tools/qts_molab.sh restage anet
bash /marimo/DDVC_QTS/source/tools/qts_molab.sh resume anet
```

YC2 dùng `yc2`; nếu variant khác thì thêm tên variant ở cuối. Resume kiểm tra recipe, budget, train-derived prior và module/loss; khôi phục optimizer/scheduler/RNG/validation phase, không nhận checkpoint của nhánh Q10/count21 cũ.

## Code map và validation

- `ddvc/temporal_support.py`: query/frame projections, dense support, bounded final refinement, fractional GT occupancy, matched objectives.
- `ddvc/ddvc_ret_encdec_clip_capQ.py`: final-layer integration trước caption pair costs, generation và matching; loss weights.
- `ddvc/criterion.py`: expanded count prior và matched support/refine objective.
- `data/video_dataset.py`: all-GT supervision và capacity guard.
- `misc/qts_recipe.py`: protocol/ablation invariants và dataset budgets.
- `tools/qts_pipeline.py`, `tools/qts_molab.sh`: prepare, smoke, train, resume, logs, report, export.
- `smoke_test_temporal_support.py`: actual loader/collate, actual criterion 31 logits/count27, support gradients, short/nested targets, synthetic optimization, full-data config preparation.
- `smoke_test_proposal_mask.py --qts`: actual DDVC matched/full prediction methods với tiny GPT-2, final/aux paths, cost chunks và generation.

Kiểm tra CPU không thay thế benchmark GPU. Chưa có kết quả CIDEr/SODA_c mới; GPU smoke và full training phải chạy trên Molab. New support module, Q30 và full GT có thể tăng thời gian/VRAM; launcher dừng nếu smoke thất bại, không tự giảm query hoặc bỏ GT.
