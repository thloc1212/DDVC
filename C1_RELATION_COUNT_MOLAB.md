# C1: learned duplicate relation + candidate-conditioned count calibration

Ngày 10/11/2026. Branch: `exp/c1-relation-count`.

Repo: https://github.com/thloc1212/DDVC/tree/exp/c1-relation-count

Tag cố định: `c1-relation-count-20261110-jsonfix`.

## Phạm vi

Đây là hai phần **3 + 4** trong phân tích vấn đề: phân biệt proposal trùng với
overlap hợp lệ và hiệu chỉnh số event. Không phải boundary refinement.

Hai head mới học từ feature cache của C1 và head quality đã đóng băng. Không chạy
`train.py`, không load/đổi trọng số DDVC, không sinh lại caption/timestamp, không
đổi Q, không bật mask loss/concept/QTS/event-set/merge. Số N chỉ thay ở ablation count.

Mốc nguồn là **C1 rerun 09/11: CIDEr37.8991** và **quality reranker39.5025**.
Không gọi đây là checkpoint C1 ablation39.8365 hoặc Ours lịch sử40.0358.
Mọi kết quả mới phải so cùng mốc đã khôi phục.

## A. Chạy trên Molab mới — YouCook2

Chọn **GPU + Internet**. Đặt file riêng tư `KLTN_DRIVE_PRIVATE.env` vào `/marimo`.
File này phải chứa remote đã xác thực tên `kltnbackup`. Không paste nội dung key
vào terminal/chat. Dùng cùng Drive chứa backup C1/reranker cũ.

Dán từng khối vào **terminal**, không phải cell Python. Không cần khai báo PY/WORK/RUN.

### 1. Tải source

```bash
git clone --single-branch --branch exp/c1-relation-count https://github.com/thloc1212/DDVC.git /marimo/DDVC_C1_Relation_Count/source
git -C /marimo/DDVC_C1_Relation_Count/source checkout --detach c1-relation-count-20261110-jsonfix
```

### 2. Chuẩn bị môi trường native

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1_Relation_Count/source/tools/relation_count.sh setup
```

Giữ nguyên GPU Torch có sẵn của Molab, dùng `/tmp/uv-venv/bin/python`; không tạo
venv mới. Chỉ cài numpy/scipy/tqdm/PyYAML và dùng Java hệ thống hoặc JRE ở work folder.
Kiểm tra phiên bản Torch trước/sau cài. Workflow không compile CUDA extension.
Molab cần có `rclone` trên PATH; nếu thiếu, setup dừng trước khi truy cập Drive.
[Rclone có bản binary portable chính thức](https://rclone.org/install/).

### 3. Kết nối và khôi phục đúng cache đã phân tích

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1_Relation_Count/source/tools/relation_count.sh prepare yc2 --rank-run 20261109T092734Z_05a32edc0bd5
```

Nó tự khôi phục auth từ file riêng tư, rồi tải:

- Train/val candidate feature shards, head quality và manifest từ backup reranker.
- Annotation/reference nhỏ từ backup C1 được ghi trong provenance reranker.
- Kiểm tra model-best SHA trong metadata C1 bằng đúng SHA của nguồn candidate.
  Không tải file model-best.pth đó xuống.

Tổng cache khoảng vài trăm MiB; log in dung lượng thực khôi phục. Không tải CLIP
feature archive lớn, checkpoint DDVC/optimizer nhiều GiB hoặc pretrained decoder.
Mỗi file được kiểm hash từ READY snapshot. Slot mới hỏng thì thử slot cũ.
Không ghi đè input đã có nếu hash/identity khác.

### 4. Chạy hai head mới và đánh giá

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1_Relation_Count/source/tools/relation_count.sh start yc2
tail -n 80 -F /marimo/DDVC_C1_Relation_Count/logs/yc2.relation_count.log
```

Chạy nền. `Ctrl+C` chỉ thoát tail, không dừng worker.
Hai head train **20 epoch**, batch8video, FP32/TF32off, seed777, AdamW lr0.001,
weight decay0.0001; đây không phải 20epoch huấn luyện DDVC.

Trước khi train: kiểm hash toàn bộ input; forward/backward **RelationHead + CountHead
thật** trên restored feature; chạy evaluator thật với dữ liệu synthetic để kiểm
Java và đầy đủ CIDEr/METEOR/BLEU/SODA. Smoke scorer không dùng GT validation DVC.

Log:

```text
FROZEN_CACHE_READY
REAL_RELATION_COUNT_FORWARD_BACKWARD_PASS cuda
FULL_OFFICIAL_SCORER_SMOKE_PASS_SYNTHETIC
RELATION_COUNT_EPOCH ...
POLICY_FROZEN_BEFORE_VALIDATION ...
OFFICIAL_SCORING ...
RESULT ...
RELATION_COUNT_COMPLETE ...
COMPLETE_BACKED_UP ...
```

### 5. Xem kết quả và xuất ZIP

```bash
cat /marimo/DDVC_C1_Relation_Count/runs/yc2/evaluation/summary.tsv
cat /marimo/DDVC_C1_Relation_Count/runs/yc2/policy.json
cat /marimo/DDVC_C1_Relation_Count/runs/yc2/evaluation/count_diagnostics.json
LC_ALL=C LANG=C bash /marimo/DDVC_C1_Relation_Count/source/tools/relation_count.sh export yc2
```

Lệnh cuối in `ZIP_READY: /marimo/DDVC_C1_Relation_Count/RELATION_COUNT_ANALYSIS_yc2_....zip`.
ZIP có log, source, hai head/optimizer, policy, prediction/metric/count/strata;
không chứa DDVC weights hoặc CLIP archive. Gửi ZIP để phân tích.

## B. Sáu ablation tự động

| Tên | Selection | N |
|---|---|---|
| C1_softnms | Joint score C1 + Soft-NMS | Counter C1 |
| quality_topk | Quality score, không suppression | Counter C1 |
| quality_softnms | Quality reranker + Soft-NMS cũ | Counter C1 |
| quality_relation | Quality + learned relation policy | Counter C1 |
| quality_softnms_count | Quality + Soft-NMS cũ | Count calibrator |
| quality_relation_count | Quality + learned relation policy | Count calibrator |

`quality_relation - quality_softnms` đo selection cùng N.
`quality_softnms_count - quality_softnms` đo count riêng.
`quality_relation_count - quality_relation` đo count trên relation selector.
`quality_topk` giúp phân biệt lợi ích học quan hệ với việc đơn thuần bỏ Soft-NMS.

Code kiểm tra C1_softnms và quality_softnms tái hiện metric nguồn trong sai số1e-4;
không khớp thì dừng. Mọi selected caption/timestamp phải lấy nguyên từ cùng raw.
Output ghi cả `base_pred_event_count` và `pred_event_count` mới.

`strata.tsv` là diagnostic one-to-one tIoU>=0.5 theo duration/overlap/count trên
val annotation, khác official Recall nhiều threshold/reference. Không trộn hai loại.

## C. Relation head học gì?

Mỗi candidate dùng frozen feature2312chiều, predicted quality và joint score.
Encoder nhỏ64chiều; pair MLP nhìn biểu diễn đối xứng và9feature quan hệ:
tIoU, containment, duration ratio, khoảng cách tương đối, cosine visual/text,
chênh lệch quality. Không tạo caption mới hoặc merge timestamps.

Nhãn TRAIN: gán candidate vào GT có joint reward cao nhất, chỉ nhận nếu tIoU>=.5,
reward>=.2 và khoảng cách với GT hạng2>.03. Hai candidate đáng tin cùng GT là
positive duplicate; khác GT là negative, kể cả khi temporal overlap/nested.
Các cặp mơ hồ bị bỏ qua. Hard negatives là khác GT nhưng pair-IoU>=.3.

Loss cân bằng positive và negative; nhóm negative cân bằng hard/easy. Không
gán duplicate chỉ vì hai câu giống nhau hoặc vì hai interval overlap.

Score greedy: `score_j -= lambda * P_duplicate(i,j)` sau khi chọn i.
Hệ sốlambda chọn trên head-dev thuộc TRAIN. Cùng train/dev split với quality head
cũ; không dùng DVC validation để chọn policy. Frozen C1 đã được train trên toàn
bộ TRAIN nên không gọi head-dev là dữ liệu chưa từng được C1 nhìn thấy.

## D. Count calibrator học gì?

Set encoder nhìn toàn bộ candidate feature + quality; mean/max/std pooling làm
head bất biến với hoán vị query. Có thêm N cũ và thống kê quality làm context.
Đầu ra là phân bố count0..Q, không suy N bằng tổng quality (quality không phải
xác suất event tồn tại độc lập).

Nhãn count lấy **số GT đầy đủ trước subsampling10**, chỉ cap theo số candidate
thực có. Loss categorical CE với class weights tính từ TRAIN head-fit, cộng0.1
ordinal CDF loss. Model selection dùng head-dev full-count MAE rồi loss tie-break.
Inference lấy median count, sau đó blend với N cũ; blend chọn trên head-dev MAE.

Q vẫn giữ50YC2/10ANET. Do đó head này không thể xuất >10event trên ANET Q10;
capacity thiếu cần experiment khác có base retraining. Không tuyên bố sửa hết
counter hoặc event ngắn chỉ bằng hai head này.

Fallback được phép: nếu head-dev không tốt hơn, policy giữ Soft-NMS hoặc blend0.
Khi đó các hàng tương ứng có thể giống nhau, phải báo contribution bằng0 thay
vì ép chọn hệ số để có khác biệt. Hyperparameter được đóng băng trước validation.

## E. Drive backup và chạy tiếp

Tự backup mỗi120giây và khi hoàn tất vào:

```text
kltnbackup:KLTNRUNS/runs/ddvc_c1_relation_count/yc2/<RUN_ID>/
  RUN_MANIFEST.json
  recovery/LATEST.json
  recovery/slot-0/READY.json
  recovery/slot-1/READY.json
```

Hai slot kiểm hash, READY ghi sau cùng. Chỉ backup head/optimizer/policy/manifest/log;
feature cache vẫn giữ ở namespace reranker gốc, không nhân bản vào từng run mới.
Source commit được archive trong `KLTNRUNS/code/<commit>/source.zip`.

Session còn local nhưng worker đã dừng:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1_Relation_Count/source/tools/relation_count.sh start yc2 --resume
tail -n 80 -F /marimo/DDVC_C1_Relation_Count/logs/yc2.relation_count.log
```

Session mới để tiếp tục **run hai head đã bắt đầu**: clone đúng tag, chạy setup,
liệt kê run ID (đã khôi phục auth bằng `auth yc2`), rồi restore:

```bash
bash /marimo/DDVC_C1_Relation_Count/source/tools/relation_count.sh auth yc2
rclone --config /marimo/secrets/rclone.conf lsf kltnbackup:KLTNRUNS/runs/ddvc_c1_relation_count/yc2 --dirs-only
bash /marimo/DDVC_C1_Relation_Count/source/tools/relation_count.sh restore yc2 --run-id RUN_ID_DA_LUU
bash /marimo/DDVC_C1_Relation_Count/source/tools/relation_count.sh start yc2 --resume
```

Thay duy nhất `RUN_ID_DA_LUU` bằng ID thật. Restore tự lấy lại cache nguồn đúng ID,
kiểm input receipt giống cũ, rồi khôi phục model/optimizer đến epoch hoàn tất gần nhất.
Giữ runtime evidence của session mới. Không ghi đè run local đang có.

## F. ActivityNet

Thuật toán hỗ trợ cả hai dataset, cùng recipe head. Nhưng cần backup frozen C1
reranker **của ActivityNet**: train/val cache và head đã hoàn tất. Không dùng cache
YC2 cho ANET và không tự train lại base khi thiếu.

Nếu có cache ANET, chạy trên tài khoản kia:

```bash
bash /marimo/DDVC_C1_Relation_Count/source/tools/relation_count.sh prepare anet
```

Lệnh liệt kê run cache có sẵn và yêu cầu chỉ rõ ID:

```bash
bash /marimo/DDVC_C1_Relation_Count/source/tools/relation_count.sh prepare anet --rank-run ID_CACHE_ANET_THAT
bash /marimo/DDVC_C1_Relation_Count/source/tools/relation_count.sh start anet
tail -n 80 -F /marimo/DDVC_C1_Relation_Count/logs/anet.relation_count.log
```

Mốc C1 base ANET ở pipeline này là6epoch; head riêng20epoch. Nếu không có ANET cache,
phải phục hồi đúng C1 và tạo cache frozen trước, không lấy checkpoint khác rồi
gọi là cùng mốc. C1 ablation ANET8epoch thuộc run khác.

## G. Kiểm chứng

- 13 kiểm thử: same/different GT overlap; repeated caption ở event khác; ambiguous
  pairs; score âm; symmetry/permutation; full count14 vs cap10; input restore
  fallback/corruption/model-hash mismatch; train/dev isolation; fit/resume6ablation;
  head backup/restore; native scorer thực với Java và paragraph protocol.
- Nhãn relation từ TRAIN YC2 thật đã được kiểm coverage: head-fit57.646pair,
  8.112positive và319hard-negative; head-dev6.306pair,898positive và20hard-negative.
- Local chỉ có CPU. Molab sẽ kiểm GPU forward/backward trên hai head thật trước
  full training. Chưa có kết quả tăng metric mới để khẳng định.

Ý tưởng học suppression có liên quan [Learning Non-Maximum Suppression](https://openaccess.thecvf.com/content_cvpr_2017/html/Hosang_Learning_Non-Maximum_Suppression_CVPR_2017_paper.html).
Đây là adaptation sang temporal-caption candidates, không tuyên bố learned NMS là ý tưởng mới.

## H. Sửa lỗi ghi prediction `float32 is not JSON serializable`

Một score ở selector Soft-NMS có thể là NumPy scalar, đặc biệt khi scalar promotion
giữa các phiên bản NumPy khác nhau. Bản sửa chuyển NumPy scalar sang kiểu Python
chỉ khi ghi JSON; không đổi phép tính, thứ tự ranking, loss, checkpoint hoặc policy.
NumPy arrays/feature tensors vẫn bị từ chối thay vì tự dump vào JSON lớn.

Nếu run đã train xong hai head và báo FAILED_BACKED_UP ở `experiment.evaluate()`,
giữ nguyên local work và chạy:

```bash
git -C /marimo/DDVC_C1_Relation_Count/source fetch origin tag c1-relation-count-20261110-jsonfix
git -C /marimo/DDVC_C1_Relation_Count/source checkout --detach c1-relation-count-20261110-jsonfix
LC_ALL=C LANG=C bash /marimo/DDVC_C1_Relation_Count/source/tools/relation_count.sh start yc2 --resume
tail -n 80 -F /marimo/DDVC_C1_Relation_Count/logs/yc2.relation_count.log
```

Migration chỉ được chấp nhận từ source đầu `46255740675ecf0869abb334e2cacddd53798245`,
khi `heads-last.pt` đã đủ20epoch, input/manifest signature khớp và hash hai best-head
khớp policy đã chốt. Source history được ghi lại. Không train lại hoặc chọn policy
bằng validation. Log có:

```text
JSON_FIX_REUSE_COMPLETE_HEADS_AND_POLICY_NO_TRAINING
RELATION_COUNT_ALREADY_FROZEN_NO_TRAINING
OFFICIAL_SCORING ...
```

Backup cũ từ source đầu cũng khôi phục được bằng tag sửa này; kiểm điều kiện hoàn
tất khi start --resume. Không cho phép resume tùy ý từ source không liên quan.
