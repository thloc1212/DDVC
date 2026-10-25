"""Extra hard-negative confidence loss, not a replacement for Hungarian matching."""
import torch
from torch.nn import functional as F


def interval_iou(boxes, other):
    a=torch.stack((boxes[:,0]-boxes[:,1]/2,boxes[:,0]+boxes[:,1]/2),-1).clamp(0,1)
    b=torch.stack((other[:,0]-other[:,1]/2,other[:,0]+other[:,1]/2),-1).clamp(0,1)
    inter=(torch.minimum(a[:,None,1],b[None,:,1])-torch.maximum(a[:,None,0],b[None,:,0])).clamp_min(0)
    return inter/((a[:,1]-a[:,0])[:,None]+(b[:,1]-b[:,0])[None,:]-inter).clamp_min(1e-8)


def normalized_dedup_loss(outputs, targets, matching, iou_threshold=.5, confidence_threshold=.2):
    logits=outputs['pred_logits']
    if logits.shape[-1]!=1:raise ValueError('GT-aware dedup currently supports one-class sigmoid DDVC')
    if not 0<iou_threshold<=1 or not 0<=confidence_threshold<1:raise ValueError('Invalid dedup thresholds')
    # Matcher returns (one-to-one list, many-to-one list); protect one-to-one positives.
    indices=matching[0]
    total=logits.sum()*0.; count=0; ambiguous_count=0
    confidences=[]
    for b,(src,tgt) in enumerate(indices):
        gt=targets[b]['boxes'];boxes=outputs['pred_boxes'][b].detach()
        if not len(gt) or not len(src):continue
        src=src.to(logits.device);tgt=tgt.to(logits.device)
        with torch.no_grad():
            overlap=interval_iou(boxes,gt.detach())
            best,owner=overlap.max(-1)
            inverse=torch.full((len(gt),),-1,device=logits.device,dtype=torch.long);inverse[tgt]=src
            paired=inverse[owner]
            eligible=(paired>=0)&(best>=iou_threshold)&(logits[b,:,0].detach().sigmoid()>=confidence_threshold)
            eligible[src]=False
            pair_overlap=interval_iou(boxes,boxes)[torch.arange(len(boxes),device=logits.device),paired.clamp_min(0)]
            eligible &= pair_overlap>=iou_threshold
            if len(gt)>1:
                top=overlap.topk(2,dim=-1).values
                ambiguous=(top[:,1]>=iou_threshold)&((top[:,0]-top[:,1])<.05)
                ambiguous_count+=int((eligible&ambiguous).sum())
                eligible &= ~ambiguous
        if eligible.any():
            selected=logits[b,eligible,0]
            total=total+F.softplus(selected).sum() # BCE-with-logits(target=background)
            count+=selected.numel();confidences.append(selected.detach().sigmoid())
    # Mean per eligible duplicate query, not a sum over Q^2 pairs. Empty set is finite zero.
    loss=total/max(1,count)
    scalar=logits.new_tensor
    return dict(loss_dedup=loss,diag_dedup_queries=scalar(float(count)),
                diag_dedup_ambiguous_skipped=scalar(float(ambiguous_count)),
                diag_dedup_confidence=torch.cat(confidences).mean() if confidences else scalar(0.))
