# YouCook2 — cùng concept model với ActivityNet, train mới 15 epochs

Branch: https://github.com/thloc1212/DDVC/tree/exp/event-concept-caption-e8

Thư mục `/marimo/DDVC_Concept_E8` và tên branch giữ nguyên để dùng source/runtime
đã cài. Tên folder không quyết định số epoch. Run YouCook2 mới có suffix **e15**.

Model/concept head/loss giữ nguyên bản ActivityNet đã chạy. Không đổi temperature,
contrastive coefficient, mask, retrieval, matching hay ranking. Không dùng weights
ActivityNet cho YouCook2: train toàn bộ DDVC mới từ đầu, GPT-2 random.

| Setting | YouCook2 mới |
|---|---|
| Epochs | **15** |
| Full feature-matched data | 1.237 train / 436 validation videos |
| Queries | 50, theo recipe YouCook2 |
| Batch / accumulation / seed | 1 / 1 / 777 |
| LR / weight decay | 5e-5 / 1e-4; lịch giảm LR cũ |
| Concept | cùng module, 512 training word concepts, BCE coefficient 0.1 |
| Runtime | FP32, TF32 off, native Molab Torch |
| Input | tự tải CLIP ViT-L/14 features ~872 MiB và annotations |
| Bank/vocabulary | riêng từ caption train YouCook2 |

## 1. Cập nhật source đã clone

Trong Terminal Molab:

```bash
git -C /marimo/DDVC_Concept_E8/source pull --ff-only
git -C /marimo/DDVC_Concept_E8/source log -1 --oneline
```

Commit cập nhật có message `Separate dataset feature caches and run YouCook2 concepts for 15 epochs`.
Nếu đã setup branch này trong cùng session để chạy ANET, bỏ qua setup, sang bước 2.
Chạy tuần tự sau ANET nếu đang dùng cùng GPU.

Nếu là session mới hoàn toàn và chưa có source:

```bash
git clone --single-branch --branch exp/event-concept-caption-e8 https://github.com/thloc1212/DDVC.git /marimo/DDVC_Concept_E8/source
```

Nếu chưa setup môi trường/GPU trong session mới:

```bash
LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh setup yc2 full
```

## 2. Tải/chuẩn bị data và config 15 epoch

```bash
DDVC_CONCEPT_EPOCHS=15 LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh prepare yc2 full
```

Tải feature vào **`assets/yc2/clipvitl14.pth`**. ActivityNet dùng cache `assets/anet/`.
Không dùng file `assets/clipvitl14.pth` chung cũ vì không thể suy ra dataset từ tên file.
Không xoá cache/run ActivityNet. Nếu đã có prepared base-assets đúng của YouCook2,
script tái sử dụng các file đó. Có thể thấy epoch 15 ở config base-assets nhưng đây
chỉ là staging, chưa train model.

Chờ `CONCEPT_CONFIG_READY` và kiểm tra:

```bash
/tmp/uv-venv/bin/python - <<'PY'
from pathlib import Path
import yaml
p=Path('/marimo/DDVC_Concept_E8/prepared/yc2_full_event_concept_e15_b1/config.yml')
c=yaml.safe_load(p.read_text())
assert c['epoch']==15 and c['num_queries']==50
assert c['dataset']=='yc2' and c['bank_type']==['yc2']
assert c['event_concepts'] and c['concept_condition_prefix']
assert not c.get('start_from') and not c.get('pretrain')
print('YC2_CONCEPT_15_EPOCHS_READY', c['id'])
PY
```

## 3. Smoke thật trước full train

```bash
DDVC_CONCEPT_EPOCHS=15 LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh smoke yc2 full
```

Phải thấy `CONCEPT_GRADIENT_PASS`, `REAL_MODEL_FORWARD_BACKWARD_PASS` và
`GENERATION_POSTPROCESS_PASS`. Smoke chỉ forward/backward minibatch đầu từ 2
training video được chọn và kiểm tra generation; không optimizer update/train full.

## 4. Train mới

Bật backup theo mục 7 trước khi train nếu muốn lưu qua session loss.

```bash
DDVC_CONCEPT_EPOCHS=15 LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh train yc2 full
```

```bash
tail -n 80 -f /marimo/DDVC_Concept_E8/logs/yc2_full_event_concept_e15_b1.log
```

`Ctrl+C` chỉ dừng tail; nohup training tiếp tục. Không gọi train hai lần cùng run.

## 5. Kết quả

```bash
DDVC_CONCEPT_EPOCHS=15 LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh report yc2 full
```

Folder kết quả:

`/marimo/DDVC_Concept_E8/runs/yc2_full_event_concept_e15_b1/`

Gồm `model-best.pth`, `model-last.pth`, `info.json`, `prediction/*.json`,
`prediction/*.raw.json`, `performance.jsonl`. Best chọn theo METEOR + SODA_c,
không mặc định là epoch 14 cuối cùng.

## 6. Nếu cần resume chính run 15 epoch này

```bash
DDVC_CONCEPT_EPOCHS=15 LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh restage yc2 full
DDVC_CONCEPT_EPOCHS=15 LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh smoke-resume yc2 full
DDVC_CONCEPT_EPOCHS=15 LC_ALL=C bash /marimo/DDVC_Concept_E8/source/tools/concept_molab.sh resume yc2 full
```

Nếu session mới: restore đúng prepared metadata/bank/checkpoint rồi setup CUDA
trước các lệnh trên. Không resume checkpoint e8 thành e15; khác training budget,
script yêu cầu run mới. Không dùng checkpoint ANET để resume YouCook2.

## 7. Google Drive backup

Nếu handle `concept_backup` đã chạy với `work=/marimo/DDVC_Concept_E8`, nó bao gồm
cả run ANET và YouCook2 trong work này. Không tạo thêm worker cho cùng remote root.
Kết nối bằng `GoogleDriveFileSystem` như mã Molab cung cấp; không tạo mount Drive.

Nếu notebook đã có `fs` nhưng chưa có backup handle, chạy trong một cell Python:

```python
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, '/marimo/DDVC_Concept_E8/source/tools')
from drive_backup import DriveBackup as _DriveBackup
concept_yc2_backup = _DriveBackup(
    fs, work=_Path('/marimo/DDVC_Concept_E8'),
    remote_root='runs/ddvc_concepts_yc2_e15', interval=120,
)
concept_yc2_backup.start()
```

Sau train, dùng đúng handle đã tạo:

```python
concept_yc2_backup.sync_once(force=True)
```

Hoặc `concept_backup.sync_once(force=True)` nếu dùng handle cũ. Chờ
`BACKUP_COMPLETE` trong `/marimo/DDVC_Concept_E8/logs/drive_sync.log`.
Notebook kernel cần tiếp tục hoạt động. Features `.pth` tải lại được nên không
upload; config/concept vocabulary/bank/checkpoints/logs/predictions được lưu.

## Thay đổi so với commit concept đầu tiên

- Default epochs theo dataset: ANET 8, YouCook2 15. Các run ID/log/config tách riêng.
- Vẫn có thể chọn e8 bằng `DDVC_CONCEPT_EPOCHS=8` khi cần ablation; không đổi config
  e8 đã có. Bản ghi config/manifest cũ tiếp tục được kiểm tra như trước.
- Download/restage cache tách `assets/anet` và `assets/yc2`, tránh đọc nhầm hai
  feature files cùng tên. Không thay đổi tensors/model/loss của run ANET đã chạy.
