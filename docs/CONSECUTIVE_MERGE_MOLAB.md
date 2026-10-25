# DDVC: thêm merge event liên tiếp cùng caption

Branch: https://github.com/thloc1212/DDVC/tree/exp/gt-aware-dedup-softnms
Đây là bản cập nhật của nhánh dedup/Soft-NMS. Bản ZIP mới có folder riêng:
`DDVC_Dedup_SoftNMS_Consecutive/`.

## Quy tắc mới

Sau overlap fusion (nếu bật), Soft-NMS chọn N event, sắp xếp timestamp, rồi xét
**các selected event kề nhau** để merge liên tiếp.

- Caption phải giống hệt sau lowercase, bỏ dấu câu và chuẩn hoá khoảng trắng.
  Thứ tự từ được giữ: token-set giống nhưng thứ tự khác vẫn không đủ điều kiện.
- Khoảng gap tối đa **2 giây**; cho phép overlap nhỏ tối đa 2 giây.
- Đoạn sau phải kéo dài về phía trước; không merge một interval chứa/lồng interval khác.
- Tối đa **3 selected events/nhóm**, tổng span tối đa **60 giây**.
- Không bỏ qua một event caption khác ở giữa để nối hai event cùng câu.
- Timestamp mới từ đầu đoạn đầu tới cuối đoạn cuối, bao gồm các gap nhỏ được cho phép.
- Caption/confidence lấy từ representative có joint_score gốc cao nhất.
- Ghi `consecutive_merge_segments`, `consecutive_merge_size`, `merged_query_ids` để truy vết.
  `raw_box` vẫn là box gốc representative. `pred_event_count` vẫn là dự đoán counter;
  số event thực tế sau merge là len(results[video]), có thể nhỏ hơn N.

Ví dụ:

```text
0:30–0:40  stir the soup
0:40–0:50  stir the soup
0:50–1:00  stir the soup
→ 0:30–1:00  stir the soup
```

Đây là heuristic hậu xử lý, không chứng minh 3 caption giống nhau thật sự cùng hành động.
Có thể kéo background/gap vào event hoặc mất event thật, nên phải so metric và visualization.
Dedup loss training vẫn như bản trước: normalized background BCE cho query dư,
hệ số0.1 từ epoch1, matched positives/GT overlap mơ hồ được bảo vệ.
Consecutive merge không dùng GT và không tạo thêm gradient.

## Mode

- baseline: train nền, topk.
- softnms: train nền, Soft-NMS.
- dedup_softnms: dedup training + Soft-NMS, không merge.
- dedup_softnms_merge: thêm fusion các interval gần trùng (tIoU>=0.7, caption Jaccard>=0.9).
- **dedup_softnms_merge_consecutive**: overlap fusion + Soft-NMS + merge liên tiếp.

Mode mới bật cả `eval_merge_duplicates` và `eval_merge_consecutive`.
Các mode cũ giữ nguyên hành vi. Mặc định script vẫn dedup_softnms để không đổi âm thầm.
LR5e-5, Adam, WD1e-4, milestones8/11 gamma0.5; batch/accumulation1,
FP32 TF32off, seed777,50queries,15epoch full hoặc subset đều giữ.

## Chạy Molab với ZIP mới

Giải nén ZIP tại /marimo, hoặc clone vào folder mới:

```bash
git clone --depth 1 --single-branch --branch exp/gt-aware-dedup-softnms https://github.com/thloc1212/DDVC.git /marimo/DDVC_Dedup_SoftNMS_Consecutive/source
```

Chạy lần lượt, cần GPU+Internet; native môi trường Molab, không tạo venv/thay Torch:

```bash
LC_ALL=C bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh setup full dedup_softnms_merge_consecutive
bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh prepare-smoke full dedup_softnms_merge_consecutive
bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh smoke full dedup_softnms_merge_consecutive
bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh train-smoke full dedup_softnms_merge_consecutive
```

Đợi REAL_MODEL_FORWARD_BACKWARD_PASS, GENERATION_POSTPROCESS_PASS và smoke validation/save pass.
Data tự download. Nếu muốn dùng lại assets local, có thể đặt DDVC_DATA_ROOT;
không dùng config/run đang train của nhánh khác.

Bật backup Drive trước train dài (cell phía dưới), rồi:

```bash
bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh prepare-run full dedup_softnms_merge_consecutive
bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh train full dedup_softnms_merge_consecutive
tail -n 80 -f /marimo/DDVC_Dedup_SoftNMS_Consecutive/logs/yc2_full_dedup_softnms_merge_consecutive_e15_b1.log
```

Thay full bằng subset để dùng30%. Run/config riêng theo mode.
Muốn đối chứng không merge: thay argument cuối bằng dedup_softnms và chạy smoke riêng.
Nohup không ngăn Molab thu hồi runtime. Ctrl+C ở tail chỉ thoát xem log.

## Drive backup

Dùng lại fs hiện có. Nếu chưa có, tạo trong một cell (không định nghĩa lại ở cell khác):

```python
import marimo as mo
from gdrive_fsspec import GoogleDriveFileSystem
fs = GoogleDriveFileSystem(root_file_id='1JtVmVYfZYlxbcR96pKMhWqxODi2WkJZI', use_listings_cache=False, skip_instance_cache=True, auth_kwargs={'use_local_webserver':False})
mo.output.clear_console()
```

Cell backup riêng:

```python
import sys as _sys
from pathlib import Path as _Path
_consecutive_work = _Path('/marimo/DDVC_Dedup_SoftNMS_Consecutive')
_sys.path.insert(0,str(_consecutive_work/'source'/'tools'))
from drive_backup import DriveBackup as _DriveBackup
consecutive_backup = _DriveBackup(fs,work=_consecutive_work,remote_root='runs/ddvc_dedup_softnms_consecutive_e15',interval=120)
consecutive_backup.start()
```

Giữ notebook chạy. Sau train:

```python
consecutive_backup.sync_once(force=True)
```

## Output / resume

```bash
bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh report full dedup_softnms_merge_consecutive
```

Kết quả tại runs/yc2_full_dedup_softnms_merge_consecutive_e15_b1/ trong workdir mới:
info.json best/history dùng metric sau merge, model-best/last.pth,
prediction/num436_epochN.raw.json (candidates), num436_epochN.json (đã chọn/merge và chấm).
Metadata selection lưu flags/gap/span/maxevents. Average proposal count tính theo output sau merge.

Chỉ resume cùng loss và selection recipe:

```bash
bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh resume full dedup_softnms_merge_consecutive
```

Session mới: setup/source+fs, tạo backup handle nhưng chưa start, restore_latest(),
restage và smoke-resume với đúng mode, rồi bật backup và resume.
Không resume run trước bằng flag merge mới để giữ lịch sử best nhất quán.

## Thử trên raw prediction có sẵn, không cần train lại

```bash
/tmp/uv-venv/bin/python /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/select_predictions.py --input /duong/dan/num436_epoch11.raw.json --output /duong/dan/epoch11_softnms_consecutive.json --method softnms --merge --merge-consecutive --consecutive-gap 2 --consecutive-max-span 60 --consecutive-max-events 3
```

Đổi input/output thành đường dẫn thật. Tool giữ input, không overwrite output đã tồn tại,
không giữ metric cũ; cần evaluator chấm lại. Bỏ --merge nếu chỉ muốn consecutive merge sau Soft-NMS.

## Kiểm chứng

CPU tests đã kiểm tra: chuỗi3event đúng ví dụ; gap2/2.01giây; overlapnhỏ/lớn;
giới hạn count/span; caption khác chặn chuỗi; không mutate input; raw provenance;
actual evaluate/save roundtrip, config5modes, resume guards; dedup backward/normalization.
Các mode không bật consecutive vẫn khớp selection đã audit trên checkpoint11/14.
Chưa có GPU/full-training metric cho mode consecutive mới, không hứa CIDEr tăng.
