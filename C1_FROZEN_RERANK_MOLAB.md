# Thử reranker trên C1 hiện có — không train lại DDVC

Repo: https://github.com/thloc1212/DDVC/tree/exp/c1-frozen-reranker

Tag cố định: `c1-frozen-rerank-20261109`

Session đích: `/marimo/DDVC_C1` đã train xong, còn checkpoint, data và môi trường GPU.

## 1. Mục tiêu và phạm vi

Phân tích C1 cho thấy có candidate khớp GT nhưng không được chọn vào final.
Thử nghiệm này học **một head chất lượng riêng, 280.577 tham số**, rồi cộng điểm
của nó vào joint score trước Gaussian Soft-NMS. Đây là giả thuyết thực nghiệm;
không thể suy ra mức tăng metric từ kết quả oracle ranking.

- Giữ nguyên checkpoint C1, caption, timestamp, Q và N do counter dự đoán.
- Không train lại DDVC; không thay mask, counter, caption decoder hay data recipe.
- Không thêm concept/QTS/event-set, merge, dedup loss hoặc pretrained model mới.
- Dùng đúng candidates validation đã lưu tại best epoch; không sinh lại để thay kết quả.
- C1 đã chọn best checkpoint bằng validation theo protocol cũ. **Chỉ head mới**
  dùng train split riêng; vì vậy không gọi phần head-dev là dữ liệu chưa từng được C1 thấy.
- Thử nghiệm này không khắc phục được event không có trong raw candidates,
  caption đã sai hoặc ngân sách N quá nhỏ. Soft-NMS vẫn có thể phạt overlap hợp lệ.

**Không pull/checkout/reset trong `/marimo/DDVC_C1/source`.** Backup C1 đang gắn
với commit cũ. Source mới và đầu ra head được đặt riêng trong `/marimo/DDVC_C1_Rerank`.

## 2. Chạy trên session C1 hiện tại: YouCook2

Dán lần lượt vào **terminal Molab**. Không dán vào cell Python.

```bash
git clone --single-branch --branch exp/c1-frozen-reranker https://github.com/thloc1212/DDVC.git /marimo/DDVC_C1_Rerank/source
git -C /marimo/DDVC_C1_Rerank/source checkout --detach c1-frozen-rerank-20261109
```

Nếu thư mục source reranker đã tồn tại, thay hai lệnh trên bằng:

```bash
git -C /marimo/DDVC_C1_Rerank/source fetch origin tag c1-frozen-rerank-20261109
git -C /marimo/DDVC_C1_Rerank/source checkout --detach c1-frozen-rerank-20261109
```

Khởi chạy nền, rồi theo dõi log:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1_Rerank/source/tools/c1_rerank.sh start yc2
tail -n 80 -F /marimo/DDVC_C1_Rerank/logs/yc2.rerank.log
```

`Ctrl+C` thoát tail; không dừng worker. Không chạy đồng thời train/probe C1 khác
trên GPU này. Workflow tự kiểm tra khóa GPU, checkpoint, config và data.

Không cần cài lại Torch, compile CUDA, tải CLIP features hay tokenizer. Dùng lại
`/tmp/uv-venv/bin/python`, CUDA env, Java và Hugging Face cache từ C1.
Rclone dùng `/marimo/secrets/rclone.conf`; nếu thiếu sẽ khôi phục từ
`/marimo/KLTN_DRIVE_PRIVATE.env`, không in key/token.

## 3. Nó sẽ chạy những gì?

1. Kiểm tra C1 đã hoàn tất, best checkpoint khớp info, train/val tách biệt.
2. **Smoke trên mô hình thật**: load C1 đóng băng, chạy hai video validation,
   so query ID/caption với raw đã lưu và kiểm tra sai lệch timestamp chuẩn hóa ≤ 1e-5.
   Không khớp thì dừng trước khi học head.
3. Chạy C1 inference trên **train** để tạo candidate; không cập nhật tham số C1.
   Mỗi 32 video ghi một shard feature kèm hash và ready marker.
4. Validation dùng raw candidates cũ. Chỉ trích feature, không tạo nhãn chất lượng GT.
5. Học head nhỏ **20 epoch**, batch 8 video, AdamW lr 0.001, weight decay 0.0001.
   Đây là epoch của head, không phải 20 epoch DDVC.
6. Chọn checkpoint head và hệ số kết hợp từ head-dev thuộc **TRAIN**.
7. Đóng băng head + hệ số, chấm validation theo ba chế độ cùng candidates, cùng N:
   baseline Top-N; baseline Soft-NMS; reranker + cùng Soft-NMS sigma 0.5.
8. Kiểm tra baseline Soft-NMS tái hiện final C1 cũ, số event không đổi, caption và
   timestamp đều lấy nguyên từ raw; hash checkpoint C1 không thay đổi.

Log lần lượt có `REAL_FROZEN_C1_GPU_SMOKE_PASS`, `CANDIDATE`, `SHARD_COMMITTED`,
`HEAD_EPOCH`, `HEAD_FROZEN_BEFORE_VALIDATION`, `RESULT`, `RERANK_EVALUATION_COMPLETE`.
Giai đoạn trích train candidates cần thời gian; mỗi candidate phải sinh caption.
Không ước lượng thời gian từ tốc độ huấn luyện head.

## 4. Head học và xếp hạng bằng gì?

Đầu vào mỗi candidate gồm:

- Mean CLIP feature trong predicted interval: 768 chiều.
- Mean CLIP feature ở vùng lân cận ngoài interval: 768 chiều.
- Mean token embedding của generated caption lấy từ **GPT-2 C1 đang đóng băng**: 768 chiều.
  Đây là embedding đã được C1 huấn luyện từ khởi tạo ngẫu nhiên, không tải GPT-2 pretrained.
- 8 số phụ: confidence, sentence score, joint score, độ dài caption, thời lượng tương đối,
  độ biến thiên visual trong đoạn, cosine local/context và tỷ lệ feature steps được chọn.

Tổng: `[Q, 2312]`, lưu cache FP16; head tính FP32. Không đưa query ID,
start/end/center tuyệt đối hay decoder query vector vào head. Điều này hạn chế
shortcut vị trí nhưng không đảm bảo head sẽ học grounding đúng.

Nhãn train cho candidate i:

```text
y_i = max_j { tIoU(pred_i, GT_j) × IDF-weighted unigram F1(caption_i, caption_GT_j) }
```

Hai thành phần phải thuộc **cùng GT j**. Đây là mục tiêu gần đúng cho chất lượng
timestamp-caption, không phải CIDEr differentiable và không đánh giá hoàn hảo paraphrase.

Train video được chia cố định seed 777: 90% head-fit, 10% head-dev.
IDF chỉ tính từ caption head-fit. Head học Smooth-L1 trên sigmoid quality cộng
0.2 pairwise preference loss trong cùng video; chọn epoch theo head-dev loss.

```text
score_i = proposal_score_i + sentence_score_i / (word_count_i² + 1e-5)
rerank_score_i = score_i + beta × predicted_quality_i
```

Beta chọn trong `{0, 0.25, 0.5, 1, 2}` theo one-to-one joint reward trên head-dev;
tie chọn beta nhỏ hơn. Không dùng GT validation để chọn proposal/checkpoint/beta.
Nếu beta=0 được chọn, workflow giữ lựa chọn Soft-NMS C1 cũ. Beta khác 0 cũng
không đảm bảo tăng metric validation.

## 5. Xem và gửi kết quả

```bash
cat /marimo/DDVC_C1_Rerank/runs/yc2/evaluation/summary.tsv
LC_ALL=C LANG=C bash /marimo/DDVC_C1_Rerank/source/tools/c1_rerank.sh status yc2
LC_ALL=C LANG=C bash /marimo/DDVC_C1_Rerank/source/tools/c1_rerank.sh export yc2
```

Lệnh cuối in `ZIP_READY` và đường dẫn ZIP trong `/marimo/DDVC_C1_Rerank`.
ZIP gồm source, manifest, log, head nhỏ, features validation, prediction của ba
chế độ và metrics. Không chứa checkpoint DDVC lớn hoặc CLIP feature archive.
Tải ZIP đó về và gửi để phân tích. Có thể lưu ZIP lên Drive bằng:

```bash
rclone --config /marimo/secrets/rclone.conf copy /marimo/DDVC_C1_Rerank kltnbackup:KLTNRUNS/analysis/c1_rerank/yc2 --max-depth 1 --include 'FROZEN_RERANK_ANALYSIS_yc2_*.zip' --progress
```

## 6. Backup, gãy session và chạy tiếp

Worker tự backup mỗi 120 giây và khi kết thúc vào:

```text
kltnbackup:KLTNRUNS/runs/ddvc_c1_rerank/yc2/<RUN_ID>/
  RUN_MANIFEST.json
  recovery/LATEST.json
  recovery/slot-0/READY.json
  recovery/slot-1/READY.json
```

Chỉ slot upload và kiểm tra hash xong mới có READY mới. Khôi phục thử slot mới
trước, hỏng thì thử slot cũ. Cache/nhãn train và head optimizer được backup;
CLIP archive và DDVC checkpoint tiếp tục nằm trong backup C1 riêng. Source mới
được archive tại `KLTNRUNS/code/<commit>/source.zip`. Không upload `.env`/rclone config.

Nếu worker dừng nhưng local còn nguyên:

```bash
LC_ALL=C LANG=C bash /marimo/DDVC_C1_Rerank/source/tools/c1_rerank.sh start yc2 --resume
tail -n 80 -F /marimo/DDVC_C1_Rerank/logs/yc2.rerank.log
```

Tiếp tục từ shard 32 video hoàn tất gần nhất và epoch head gần nhất.
Head/beta đã chốt không được chọn lại khi chỉ chạy tiếp evaluation.

Nếu session mất hoàn toàn: khôi phục **C1 checkpoint/data/môi trường đúng run cũ**
trước theo hướng dẫn C1, không train một C1 mới. Clone source reranker đúng tag
trong mục 2. Liệt kê run reranker rồi restore bằng ID thực được in ra:

```bash
rclone --config /marimo/secrets/rclone.conf lsf kltnbackup:KLTNRUNS/runs/ddvc_c1_rerank/yc2 --dirs-only
bash /marimo/DDVC_C1_Rerank/source/tools/c1_rerank.sh restore yc2 --run-id RUN_ID_TU_DANH_SACH
bash /marimo/DDVC_C1_Rerank/source/tools/c1_rerank.sh start yc2 --resume
```

Restore chỉ vào work reranker chưa có state/run tương ứng, giữ nguyên đường dẫn
base/work và commit. Không ghi đè run đang có. `RUN_ID_TU_DANH_SACH` là chỗ duy
nhất cần thay bằng ID thật; không dùng nguyên placeholder.

## 7. ActivityNet trên session C1 tương ứng

Chỉ dùng khi C1 ANET 6 epoch đã hoàn tất trong `/marimo/DDVC_C1` trên tài khoản kia:

```bash
bash /marimo/DDVC_C1_Rerank/source/tools/c1_rerank.sh start anet
tail -n 80 -F /marimo/DDVC_C1_Rerank/logs/anet.rerank.log
cat /marimo/DDVC_C1_Rerank/runs/anet/evaluation/summary.tsv
bash /marimo/DDVC_C1_Rerank/source/tools/c1_rerank.sh export anet
```

Cùng phương pháp, cùng hyperparameter head; giữ Q=10 và N của C1 ANET.
Trích train 10.009 video sẽ nặng hơn YouCook2. Hai tài khoản dùng namespace
dataset/run ID riêng, không ghi đè nhau.

## 8. Kiểm chứng trước khi phát hành

- 11 kiểm thử CPU: selector beta=0, target cùng GT, giới hạn fit/dev, feature pooling,
  gradient head, cache hash, head fit/resume, validation inference, import safe-path,
  backup hai slot và fallback khi corrupt.
- Đối chiếu raw/final YC2 epoch 11 có sẵn: beta=0 chọn đúng toàn bộ 436 video,
  3.163 event, query ID/timestamp/caption giống hệt final. Sai số score lớn nhất
  do số thực giữa môi trường là 5.6e-17.
- Kiểm thử scorer trong integration test là giả lập; chưa chạy GPU/official scorer
  tại máy local. Molab sẽ chạy real-model GPU smoke và scorer chính thức trước khi
  công bố summary. Không có số cải thiện reranker nào được khẳng định trước lần chạy đó.
