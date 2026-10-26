"""Keep the fail-before-update guard and persist evidence for its cause."""
import json
import math
from pathlib import Path
import torch


def report_nonfinite_loss(loss,weights,dt,save_folder,epoch,batch_index,iteration):
    values={}
    for name,value in loss.items():
        tensor=torch.as_tensor(value).detach()
        if tensor.numel()==1:
            scalar=float(tensor.cpu())
            values[name]=scalar if math.isfinite(scalar) else str(scalar)
        else:values[name]=dict(shape=list(tensor.shape),all_finite=bool(torch.isfinite(tensor).all()))
    bad=[name for name,value in loss.items() if not torch.isfinite(torch.as_tensor(value)).all()]
    stats={}
    for name in ('video_tensor','video_length','gt_boxes'):
        value=dt.get(name)
        if isinstance(value,torch.Tensor):
            value=value.detach();finite=torch.isfinite(value)
            stats[name]=dict(shape=list(value.shape),all_finite=bool(finite.all()),nonfinite=int((~finite).sum()))
    spans=dt.get('gt_featstamps',[])
    if isinstance(spans,torch.Tensor):spans=spans.detach().cpu().tolist()
    spans=[[int(s),int(e)] for s,e in spans]
    report=dict(epoch=epoch,batch_index=batch_index,iteration=iteration,
        video_ids=dt.get('video_key'),captions=dt.get('cap_raw'),
        loss_values=values,loss_weights=weights,bad_loss_terms=bad,
        quantized_gt_spans=spans,empty_gt_spans=[i for i,(s,e) in enumerate(spans) if e<=s],
        tensor_stats=stats)
    path=Path(save_folder)/'nonfinite_diagnostics.json'
    path.write_text(json.dumps(report,indent=2,default=str),encoding='utf8')
    return path,bad
