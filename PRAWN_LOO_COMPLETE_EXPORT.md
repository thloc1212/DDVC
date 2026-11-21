# Gom ZIP ablation đầy đủ từ Drive sau khi mất session

Collector chạy riêng bên ngoài source training. Không thay checkout, không chạy model, không train, không tính lại benchmark và không tải checkpoint hoặc feature cache. Cần thư mục source hiện có `/marimo/PRAWN_Ablations/source` và private env `/marimo/KLTN_DRIVE_PRIVATE.env`. Python standard library là đủ; không cần GPU.

```bash
bash <<'BASH'
set -euo pipefail
export LC_ALL=C LANG=C PYTHONUTF8=1
test -f /marimo/KLTN_DRIVE_PRIVATE.env
test -f /marimo/PRAWN_Ablations/source/tools/c1_rclone.py
curl -fL --retry 3 \
  https://raw.githubusercontent.com/thloc1212/DDVC/prawn-loo-export-20261121-v1/tools/collect_prawn_loo.py \
  -o /tmp/collect_prawn_loo.py
/tmp/uv-venv/bin/python -u /tmp/collect_prawn_loo.py anet --upload \
  2>&1 | tee /marimo/collect_prawn_loo_anet.log
BASH
```

Đổi `anet` thành `yc2` để dùng cho YouCook2. Dataset là đối số trực tiếp, không phụ thuộc `$DS` trong terminal cũ.

## Các nguồn kết quả được gom

- Full + w/o Quality: backup generator `runs/prawn_caption_match_generator/<dataset>` và frozen quality `runs/prawn_caption_match_quality/<dataset>`.
- Quét beta: `analysis/prawn_loo/<dataset>/full_<generator_run_id>/beta`. Collector yêu cầu đủ 8 beta: 0, .25, .5, 1, 2, 4, 6, 8. Identity trong kết quả beta quyết định đúng Full generator/head phải đi kèm.
- w/o Content: `runs/prawn_loo_no_content_generator/<dataset>` và `runs/prawn_loo_no_content_quality/<dataset>`.
- Joint: `runs/prawn_loo_joint_generator/<dataset>`, lấy quality-aware evaluation ở đúng best-generator epoch. Không phụ thuộc alias `joint_evaluation` còn local hay đã mất.

Collector chỉ dùng generator đã hoàn tất 15 epoch YC2 hoặc 6 epoch ANET, seed777, batch1, query50/10, caption-aware matching bật với weight.5, auxiliary mask loss0. Content chỉ tắt ở w/o Content; online quality chỉ bật ở Joint. Các nhánh phải có cùng data signature và cùng GT reference hashes.

## Kiểm tra trước khi xuất

- Metadata và artifact native phải khớp size/SHA256 trong READY.
- Artifact analysis được kiểm tra size/MD5 khi Drive cung cấp MD5.
- Ghép frozen head với generator bằng checkpoint/config/raw SHA256 và epoch.
- Metric JSON phải có đủ metric chính, hữu hạn và khớp fingerprint của prediction + references + scoring protocol.
- Các kết quả beta dùng cùng frozen head; beta0 tái hiện w/o Quality và beta cố định tái hiện Full.
- Final giữ nguyên query/caption/timestamp từ raw và cùng counter budget.
- Validation pool phải đủ 436 video YC2 hoặc 4917 ANET. Các reference ANET vẫn được giữ riêng.

## Đầu ra

Khi đủ bốn phần, collector in:

```text
ALL_FOUR_ABLATIONS_COMPLETE anet
ZIP_READY /marimo/PRAWN_LOO_COMPLETE/PRAWN_LOO_COMPLETE_anet_<timestamp>.zip
ZIP_ON_DRIVE kltnbackup:KLTNRUNS/analysis/prawn_loo/anet/exports/...
```

ZIP có bảng tổng hợp 12 dòng (Full, w/o Quality, w/o Content, Joint và 8 beta), raw/final của các checkpoint cần đối chiếu, GT references, manifests, head history khi có, quality scores, metric JSON và source liên quan. Không có private env/rclone.conf, weights, train/val feature cache hoặc toolchain.

Nếu chưa đủ, collector in `INCOMPLETE_COLLECTION` và ghi:

```text
/marimo/PRAWN_LOO_COMPLETE/anet/COLLECTION_REPORT.json
```

Không có ZIP gắn nhãn đầy đủ trong trường hợp đó. Thiếu artifact không tự động có nghĩa là phải train lại: cần xem namespace/run được liệt kê và kiểm tra những kết quả đã lưu ở account hoặc thư mục Drive khác. Collector không tự chạy bù thí nghiệm.

Source training đang chạy không bị thay đổi. Snapshot gốc và mọi file trên Drive được đọc, ngoại trừ thao tác upload ZIP hoàn tất khi có `--upload`.
