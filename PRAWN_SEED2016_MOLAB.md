# PRAWN-DVC seed 2016 trên hai tài khoản Molab

Mỗi tài khoản dùng một session GPU mới. Upload file riêng tư `KLTN_DRIVE_PRIVATE.env` vào `/marimo`. Tài khoản thứ nhất chạy YouCook2, tài khoản thứ hai chạy ActivityNet Captions. Tất cả lệnh bên dưới chạy trong terminal.

## Cấu hình đã chốt

| Thành phần | YouCook2 | ActivityNet Captions |
| --- | --- | --- |
| Train / validation reference 1 | 1237 / 436 | 10009 / 4917 |
| Event queries | 50 | 10 |
| Epoch giai đoạn tạo sự kiện | 15 | 6 |
| Batch / accumulation | 1 / 1 | 1 / 1 |
| Generator training seed | 2016 | 2016 |
| Quality training seed | 2016 | 2016 |
| TRAIN head-fit/head-dev split seed | 777 | 777 |
| Quality epochs / batch videos | 20 / 8 | 20 / 8 |
| Beta cố định | 1 | 6 |
| Soft-NMS sigma | 0.5 | 0.5 |

Giữ Adam, learning rate `5e-5`, weight decay `1e-4`, FP32 và TF32 tắt trong giai đoạn một. Giữ AdamW, learning rate `1e-3`, weight decay `1e-4` trong giai đoạn hai. Giữ lịch giảm learning rate của generator từ epoch 8, mỗi 3 epoch nhân 0.5. Giữ GPT-2 hai lớp khởi tạo ngẫu nhiên, learned proposal mask và hệ thống retrieval của phiên bản đã chốt. Auxiliary mask loss bằng 0. Soft-NMS thực hiện ở bước chọn kết quả.

Cả generator và quality head được huấn luyện mới với seed 2016. Tập head-fit/head-dev vẫn chia bằng seed 777 để giữ nguyên tập dữ liệu dùng so sánh. Dữ liệu CLIP và TRAIN caption bank được tái sử dụng qua cache Drive sau khi kiểm tra hash. Các checkpoint seed 777 giữ nguyên trên Drive.

Beta được cố định từ thực nghiệm seed 777. Giá trị 6 của ActivityNet đã được chọn qua khảo sát validation trước đó. Run seed 2016 giữ giá trị này trong suốt thực nghiệm. Grid trên TRAIN head-dev vẫn được ghi lại để kiểm tra, giá trị beta đánh giá cuối cùng lấy từ bảng trên.

## Tài khoản 1 chạy YouCook2

```bash
test -f /marimo/KLTN_DRIVE_PRIVATE.env
rclone version
git clone --single-branch --branch prawn-seed2016-20261113 https://github.com/thloc1212/DDVC.git /marimo/PRAWN_Seed2016/source
LC_ALL=C LANG=C bash /marimo/PRAWN_Seed2016/source/tools/prawn_seed2016.sh start yc2
LC_ALL=C LANG=C bash /marimo/PRAWN_Seed2016/source/tools/prawn_seed2016.sh logs yc2
```

## Tài khoản 2 chạy ActivityNet Captions

```bash
test -f /marimo/KLTN_DRIVE_PRIVATE.env
rclone version
git clone --single-branch --branch prawn-seed2016-20261113 https://github.com/thloc1212/DDVC.git /marimo/PRAWN_Seed2016/source
LC_ALL=C LANG=C bash /marimo/PRAWN_Seed2016/source/tools/prawn_seed2016.sh start anet
LC_ALL=C LANG=C bash /marimo/PRAWN_Seed2016/source/tools/prawn_seed2016.sh logs anet
```

Lệnh `start` kết nối Drive trước khi khởi chạy worker nền. Worker cài dependencies vào Python native của Molab, build CUDA operator, chạy smoke test, tải dữ liệu của dataset đang chạy, train generator, backup checkpoint, đóng băng generator, tạo candidate cache, train quality head rồi đánh giá. Python native GPU PyTorch được kiểm tra trước và sau cài đặt.

Ở lần tải dữ liệu đầu tiên trên session mới, cần khoảng 872 MiB cho YouCook2 hoặc 2.52 GiB cho ActivityNet CLIP features cùng annotation và caption bank. Cache có sẵn được kiểm tra hash và chuyển đường dẫn sang thư mục mới. Việc tải dữ liệu chỉ áp dụng cho dataset của tài khoản đó.

Trong terminal xem log, `Ctrl+C` dừng theo dõi log. Worker đã được khởi chạy thành process riêng và tiếp tục chạy trong session Molab.

## Xem trạng thái và kết quả

YouCook2

```bash
LC_ALL=C LANG=C bash /marimo/PRAWN_Seed2016/source/tools/prawn_seed2016.sh status yc2
cat /marimo/PRAWN_Seed2016/frozen_quality/runs/yc2/evaluation/summary.tsv
```

ActivityNet

```bash
LC_ALL=C LANG=C bash /marimo/PRAWN_Seed2016/source/tools/prawn_seed2016.sh status anet
cat /marimo/PRAWN_Seed2016/frozen_quality/runs/anet/evaluation/summary.tsv
```

File summary xuất hiện sau khi đánh giá xong. Dòng `reranker_softnms` là kết quả PRAWN cuối cùng. Hai dòng `baseline_topk` và `baseline_softnms` trong file này dùng cùng generator learned-mask, nhằm đối chiếu riêng tác động của quality ranking.

Worker báo `PRAWN_TWO_STAGE_COMPLETE_BACKED_UP` sau khi hoàn tất hai giai đoạn và xác minh backup.

## Lưu trữ và khôi phục

Kết quả generator được lưu trong `KLTNRUNS/runs/prawn_seed2016_generator/<dataset>/<run_id>/`. Kết quả quality head được lưu trong `KLTNRUNS/runs/prawn_seed2016_quality/<dataset>/<run_id>/`. Mỗi giai đoạn có hai recovery slot, hash manifest và trạng thái riêng. Worker đồng bộ định kỳ 120 giây, tùy dữ liệu mới và thời gian truyền. Các checkpoint đã ghi xong được đưa vào snapshot.

Nếu worker dừng trong session hiện tại, xem log để xử lý nguyên nhân rồi chạy lại chain bằng lệnh phù hợp.

```bash
LC_ALL=C LANG=C bash /marimo/PRAWN_Seed2016/source/tools/prawn_seed2016.sh resume yc2
```

```bash
LC_ALL=C LANG=C bash /marimo/PRAWN_Seed2016/source/tools/prawn_seed2016.sh resume anet
```

Khi chuyển sang session mới, clone lại cùng tag, upload file `.env`, chạy `auth` rồi dùng các lệnh `c1_drive.sh restore` và `c1_rerank.sh restore` với run ID thực tế của seed 2016. Sau khi khôi phục, dùng `resume` để tiếp tục từ checkpoint. Nếu mất session trước checkpoint đầu tiên, cần khởi tạo lại giai đoạn tương ứng từ đầu.

## Xuất ZIP gửi phân tích

Sau khi hoàn tất YouCook2

```bash
LC_ALL=C LANG=C bash /marimo/PRAWN_Seed2016/source/tools/prawn_seed2016.sh export yc2
find /marimo/PRAWN_Seed2016 -maxdepth 2 -type f -name '*ANALYSIS*.zip'
```

Sau khi hoàn tất ActivityNet

```bash
LC_ALL=C LANG=C bash /marimo/PRAWN_Seed2016/source/tools/prawn_seed2016.sh export anet
find /marimo/PRAWN_Seed2016 -maxdepth 2 -type f -name '*ANALYSIS*.zip'
```

Mỗi tài khoản xuất hai ZIP gồm generator analysis và frozen quality analysis. Gửi cả hai file để đối chiếu configuration, training history, prediction và ranking. Checkpoint đầy đủ được lưu qua recovery trên Drive.

## Kiểm chứng trước phát hành

Đã kiểm tra local bằng CPU cho việc tạo cấu hình hai dataset, gradient của GPT-2 và mask, train/resume quality head, beta cố định, thứ tự hai giai đoạn, chuyển cache giữa đường dẫn mới và backup/restore hai slot. GPU forward/backward thực tế được kiểm tra bằng smoke test trong session Molab trước full training.
