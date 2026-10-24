# DDVC Temporal Caption Prefix — branch và cách chạy Molab

## 1. Tóm tắt thay đổi

Branch: [exp/temporal-caption-prefix](https://github.com/thloc1212/DDVC/tree/exp/temporal-caption-prefix).
Nền: branch Molab legacy DDVC; detector, retrieval, Hungarian matching, localization/caption losses và GPT-2 scratch decoder được giữ. Đây là thử nghiệm thay learned proposal feature-mask bằng conditioning thời gian ở caption head.

1. **Time-conditioned caption prefix:** từ predicted normalized center/duration `(c,d)`, mã hoá bằng raw coordinates + multi-frequency sin/cos, MLP 34→128→768. Cộng residual vào prefix GPT-2 **sau fc2**, trước word tokens.
2. **Frame temporal embedding:** mỗi valid frame có vị trí tương đối `(j+0.5)/L`, raw+sin/cos 17 chiều, projection 17→768 rồi cộng vào CLIP frame feature.
3. **Không feature gating / pruning:** caption head nhận tất cả valid CLIP frames; chỉ padding bị attention mask. Không nhân soft/hard mask vào feature và không thêm mask BCE.
4. **Khởi tạo an toàn:** projection cuối của time prefix và frame projection đều zero-init. Trước optimizer update đầu, logits/cross-attention context giống global baseline. RNG khởi tạo backbone giữ giống nhau giữa các ablation.
5. **Predicted boxes detach mặc định:** caption CE học time MLP, nhưng không tạo thêm đường gradient trực tiếp qua tọa độ; localization vẫn học bằng loss gốc. Không dùng GT timestamp để tạo prefix lúc train hay infer.
6. **Nhất quán:** forward có teacher forcing, all-pair caption cost, matched caption loss, beam generation/cache đều dùng cùng helper. Query/caption expansion giữ đúng ID của predicted box.
7. **Semantic CLIP alignment giữ nguyên:** `input_project()` vẫn semantic-only; time embedding chỉ cộng ở `caption_prefix()`, không buộc timestamp embedding phải giống CLIP text embedding.
8. **Diagnostics:** prefix/frame temporal norm và ratio so với semantic prefix/raw CLIP context được log, không cộng vào loss.

Đây là continuous temporal-conditioned prefix, **không phải triển khai lại Time Tokens/Vid2Seq**. Không thay GPT-2 vocabulary, không thêm T5/JAX hay pretrained language weights, không cần raw video.

## 2. Đã tham khảo code nào?

### Vid2Seq — repo chính thức Google Research Scenic

- [Project chính thức](https://antoyang.github.io/vid2seq.html).
- [Paper CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_CVPR_2023_paper.html).
- [models.py, pin 693290c7](https://github.com/google-research/scenic/blob/693290c7b7927b27c9539561fca4ae8c8a84f98e/scenic/projects/vid2seq/models.py): visual encoder có temporal positional embeddings; text model mở rộng vocabulary để có discrete time tokens.
- [data_utils.py cùng revision](https://github.com/google-research/scenic/blob/693290c7b7927b27c9539561fca4ae8c8a84f98e/scenic/projects/vid2seq/data_utils.py): xử lý timestamp theo duration/temporal bins.

**Áp dụng ở đây:** cung cấp time representation cho visual/context và decoder. **Khác:** DDVC vẫn predict boxes bằng detection head; GPT-2 chỉ sinh caption, không sinh time tokens. Code mới viết bằng PyTorch, không copy mô hình Scenic/JAX hay dùng checkpoint Vid2Seq.

### DETR — positional encoding có xét valid/padding

- [models/position_encoding.py, pin a5cd934f](https://github.com/facebookresearch/detr/blob/a5cd934faa64a4977d10ab86010d4ad9094c0a8f/models/position_encoding.py).
- Tham khảo việc lấy vị trí từ valid-mask cumsum, chuẩn hoá và dùng sin/cos.
- Bản này là 1D temporal, dùng frame-center và các band `2^k`; không tái lập công thức 2D temperature-based của DETR.
- Giữ thêm raw `(c,d)` / frame position để tránh Fourier periodicity làm 0 và 1 có mã giống nhau.

## 3. Các ablation — cùng recipe, khác đúng flags

| Mode | Time prefix | Frame time embedding | Caption visual context |
|---|---|---|---|
| global | Không | Không | Global, không feature mask |
| time_only | Có | Không | Global, không feature mask |
| frame_only | Không | Có | Global, không feature mask |
| **time_frame (mặc định)** | Có | Có | Global, không feature mask |

Default **full feature-matched 1237 train / 436 validation, 25 epoch, batch thật 1, accumulation 1, FP32, TF32 off, seed 777, 50 queries, caption chunk 16, workers 0, Adam LR 5e-5**. Thêm khoảng 116.608 tham số khi bật cả hai nhánh; không thay backbone width hay số layer GPT-2 (2 layer, khởi tạo mới).

Không bật Action-Aware/query-conditioned mask, matched-GT mask supervision, corrected contrastive hay Gated Temporal Fusion. Giữ legacy contrastive objective để không đổi thêm loss khi so sánh. Những sửa Molab về CUDA/compiler/header, native dependencies, parser defaults, backup thiếu cfgs, atomic checkpoint/resume, raw/final predictions và log được kế thừa. Best-config metadata được chụp riêng để LR của best không bị ghi thành LR cuối.

## 4. File/function cần xem

| File | Function/class | Vai trò |
|---|---|---|
| ddvc/CaptioningHead/temporal_caption.py | TemporalCaptionConditioning.encode / prefix_delta / frame_delta | Encoding liên tục, zero-init projections, detach timestamp |
| ddvc/CaptioningHead/gpt2.py | GPT2CaptionModel.caption_prefix | Cộng time embedding sau semantic projection |
| ddvc/CaptioningHead/gpt2.py | visual_context | Cộng frame time embedding, giữ valid frames |
| ddvc/CaptioningHead/gpt2.py | forward / generate | Dùng cùng prefix helper trong train/infer |
| ddvc/ddvc_ret_encdec_clip_capQ.py | caption_prediction / caption_prediction_no_grad / caption_prediction_eval | Giữ predicted-box alignment trong matching, matched loss, generation |
| ddvc/ddvc_ret_encdec_clip_capQ.py | forward | Gắn diagnostics vào log loss dict; không thêm weighted loss |
| tools/molab_pipeline.py | conditioning_options / prepare / smoke | Data/bank/config theo từng ablation + actual model smoke |
| tools/temporal_recipe.py | validate | Kiểm tra batch/precision/mode/loss/data trước full |
| tools/temporal_molab.sh | launcher | Setup, smoke, train, logs, report, resume không phụ thuộc biến terminal |
| misc/training_state.py | validate_resume | Không cho đổi architecture/mode/recipe giữa resume |

Training: video/CLIP features → DDVC event queries → predicted intervals → caption semantic prefix + time residual; CLIP frames + frame time residual → GPT-2 cross-attention → caption CE. Hungarian/localization/event-count losses vẫn như nền. GT captions dùng teacher forcing/loss; GT intervals dùng matching/localization, không làm input cho time module.

Inference: video + train-only memory bank → predicted intervals/queries → cùng prefix/frame helpers → beam generation → event-count/rerank → selected captions + timestamps. Không dùng GT của video suy luận.

## 5. Chuẩn bị code trên Molab

Chọn GPU và Internet. Gói mới dùng thư mục riêng `/marimo/DDVC_Temporal_Caption`; không ghi đè source/checkpoint của run cũ. Nên đợi run GPU cũ kết thúc để đo riêng tài nguyên/thời gian.

### Cách A: clone branch để chạy ngay, không cần upload ZIP

```bash
git clone --depth 1 --single-branch --branch exp/temporal-caption-prefix https://github.com/thloc1212/DDVC.git /marimo/DDVC_Temporal_Caption/source
```

### Cách B: dùng ZIP mới

Upload `DDVC_Temporal_Caption_Molab.zip` vào folder code Drive lâu dài của bạn:
[folder 1JtVmVYfZYlxbcR96pKMhWqxODi2WkJZI](https://drive.google.com/drive/folders/1JtVmVYfZYlxbcR96pKMhWqxODi2WkJZI).
Không dùng link ZIP legacy cũ vì chưa có module mới.
Nếu `fs` đã kết nối và root là folder này, cell:

```python
assert fs.info('')['id'] == '1JtVmVYfZYlxbcR96pKMhWqxODi2WkJZI'
fs.get('DDVC_Temporal_Caption_Molab.zip', '/marimo/DDVC_Temporal_Caption_Molab.zip')
```

Terminal, chỉ giải nén khi chưa có thư mục cùng tên:

```bash
/tmp/uv-venv/bin/python -m zipfile -t /marimo/DDVC_Temporal_Caption_Molab.zip
```

```bash
if [ -d /marimo/DDVC_Temporal_Caption ]; then
  echo 'Folder đã tồn tại; không giải nén đè source/run.'
else
  /tmp/uv-venv/bin/python -m zipfile -e /marimo/DDVC_Temporal_Caption_Molab.zip /marimo
fi
```

Shell scripts trong gói mới dùng LF; không cần sửa CRLF. Native interpreter dùng đường dẫn đã thấy trong Molab; có thể cấu hình `DDVC_PYTHON` nếu runtime thực tế đổi. Không tạo venv mới, không yêu cầu root và không tự thay Torch.

## 6. Setup và smoke — terminal

Mặc định mode time_frame. Mỗi lệnh tự nạp workdir, native interpreter, CUDA env và run ID, không cần khai báo WORK/RUN.

```bash
LC_ALL=C bash /marimo/DDVC_Temporal_Caption/source/tools/temporal_molab.sh setup
```

```bash
bash /marimo/DDVC_Temporal_Caption/source/tools/temporal_molab.sh prepare-smoke time_frame
```

```bash
bash /marimo/DDVC_Temporal_Caption/source/tools/temporal_molab.sh smoke time_frame
```

Cần REAL_MODEL_FORWARD_BACKWARD_PASS và GENERATION_POSTPROCESS_PASS. Setup có CUDA extension test; model smoke dùng 1% data, check actual model/loss/backward/generation/postprocess. Data tự tải annotation, CLIP features ~0.9 GB, CLIP weights tạo train-only caption bank. Không cần Kaggle Input/raw videos/pretrained caption checkpoint.

Kiểm tra thêm validation/Java/checkpoint bằng 2 epoch nhỏ (chạy một lần):

```bash
bash /marimo/DDVC_Temporal_Caption/source/tools/temporal_molab.sh train-smoke time_frame
```

Full chưa chạy khi smoke/config fail. Nếu Molab reset có thể cần setup lại native packages và compiler. CUDA 13.0 private toolchain (~105 MB) là compiler assets, không phải venv. Giữ checkpoint/log trên Drive.

## 7. Full 25 epoch

```bash
bash /marimo/DDVC_Temporal_Caption/source/tools/temporal_molab.sh prepare-full time_frame
```

Phải thấy TEMPORAL_CAPTION_CONFIG_OK time_frame. Trước train dài, bật backup ở bước 9.

```bash
bash /marimo/DDVC_Temporal_Caption/source/tools/temporal_molab.sh train time_frame
```

```bash
tail -n 80 -f /marimo/DDVC_Temporal_Caption/logs/yc2_full_time_frame_e25_b1.log
```

Ctrl+C chỉ thoát xem log. Nohup không giữ process khi cả runtime bị thu hồi.

```bash
bash /marimo/DDVC_Temporal_Caption/source/tools/temporal_molab.sh status time_frame
```

```bash
bash /marimo/DDVC_Temporal_Caption/source/tools/temporal_molab.sh report time_frame
```

Kết quả: `/marimo/DDVC_Temporal_Caption/runs/yc2_full_time_frame_e25_b1/` gồm model-last/best, info.json, prediction/, train.log, performance.jsonl. Best vẫn chọn theo METEOR + SODA_c. Diagnostics `diag_caption_time_prefix_norm/ratio` và `diag_caption_frame_time_norm/ratio` nằm trong loss history/train log; không là metric chất lượng và không đóng góp vào total loss.

## 8. Chạy đối chứng

Thay mode cuối lệnh bằng global, time_only hoặc frame_only. Mỗi mode có run/config/log riêng, cùng seed/data/budget/optimizer. Ví dụ global:

```bash
bash /marimo/DDVC_Temporal_Caption/source/tools/temporal_molab.sh prepare-smoke global
```

```bash
bash /marimo/DDVC_Temporal_Caption/source/tools/temporal_molab.sh smoke global
```

```bash
bash /marimo/DDVC_Temporal_Caption/source/tools/temporal_molab.sh prepare-full global
```

```bash
bash /marimo/DDVC_Temporal_Caption/source/tools/temporal_molab.sh train global
```

Nên chạy tuần tự. So time_frame với global trước; thêm time_only/frame_only khi cần tách đóng góp, không kết luận từ việc đổi nhiều yếu tố cùng lúc. Run legacy có learned mask không thay thế global control của branch này.

## 9. Backup Drive — cell Molab

Nếu chưa có fs, tạo trong một cell. Nếu đã có thì dùng lại hoặc sửa ngay cell kết nối hiện tại, không định nghĩa fs ở cell khác:

```python
import marimo as mo
from gdrive_fsspec import GoogleDriveFileSystem
fs = GoogleDriveFileSystem(root_file_id='1JtVmVYfZYlxbcR96pKMhWqxODi2WkJZI', use_listings_cache=False, skip_instance_cache=True, auth_kwargs={"use_local_webserver": False})
mo.output.clear_console()
```

Cell backup:

```python
import sys as _sys
from pathlib import Path as _Path
assert fs.info('')['id'] == '1JtVmVYfZYlxbcR96pKMhWqxODi2WkJZI'
_temporal_work = _Path('/marimo/DDVC_Temporal_Caption')
_sys.path.insert(0, str(_temporal_work / 'source' / 'tools'))
from drive_backup import DriveBackup as _DriveBackup
temporal_backup = _DriveBackup(fs, work=_temporal_work, remote_root='runs/ddvc_temporal_caption_25e_b1', interval=120)
temporal_backup.start()
```

Chỉ một worker cho remote root này. Folder backup riêng bên trong Drive do bạn chọn, không đụng ZIP code hay backup legacy/Action-Aware. Hai slot giữ model/optimizer/RNG/config/bank/refs/metrics/logs của các run trong workdir; không backup video features/HF cache/toolchain tải lại được. Đợi BACKUP_COMPLETE; slot chỉ hoàn tất khi có READY.json. Chưa có checkpoint thì chưa có gì để upload.

```bash
tail -n 40 -f /marimo/DDVC_Temporal_Caption/logs/drive_sync.log
```

Sau train, cell:

```python
temporal_backup.sync_once(force=True)
```

## 10. Resume

Sau khi process cũ dừng:

```bash
bash /marimo/DDVC_Temporal_Caption/source/tools/temporal_molab.sh resume time_frame
```

Session mới: tải source/setup lại nếu cần, kết nối fs, tạo temporal_backup nhưng CHƯA start, rồi cell:

```python
temporal_backup.restore_latest()
```

Terminal:

```bash
bash /marimo/DDVC_Temporal_Caption/source/tools/temporal_molab.sh restage time_frame
```

```bash
bash /marimo/DDVC_Temporal_Caption/source/tools/temporal_molab.sh smoke-resume time_frame
```

Khi checkpoint load + smoke pass, bật temporal_backup.start() rồi resume. Không đổi mode hoặc batch/loss/precision giữa resume; không dùng checkpoint Action-Aware/legacy làm exact resume. Giữ model-best/last tương ứng run ID, không prepare đè config restore.

## 11. Kiểm chứng và giới hạn

CPU tests chạy actual scratch GPT-2 CE/backward, actual DDVC pair expansion/matched loss/full generation, beam/cache, padding invariance, predicted-box detach, zero-init identity, shared-backbone RNG identity ở cả 4 mode, chunk equivalence, checkpoint/resume và backup transport giả lập. Không gọi đây là Molab GPU/full training benchmark đã làm ở local; cần setup/smoke/train-smoke trên máy thật.

Thời gian là tương đối theo duration video, không phải phút/giây tuyệt đối. Giữ toàn bộ frame không bảo đảm hết nhiễu. Time prefix không bảo đảm model tự biết frame nào liên quan nếu chưa học alignment. Không hứa CIDEr tăng và không gọi là idea Time Tokens hoàn toàn mới. Cần so cùng full data, 25 epoch, batch/accumulation 1, FP32 và cùng evaluator với global control.
