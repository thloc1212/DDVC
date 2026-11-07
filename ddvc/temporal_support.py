"""Query-conditioned visual support and final-layer boundary refinement.

No GT is accepted by forward(). Independent query maps permit nested events.
The auxiliary objective is computed only after Hungarian matching.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class QueryTemporalSupport(nn.Module):
    def __init__(self, query_dim=512, feature_dim=768, hidden_dim=128, max_delta=2.):
        super().__init__()
        self.query_proj=nn.Sequential(nn.Linear(query_dim,hidden_dim),nn.LayerNorm(hidden_dim))
        self.frame_proj=nn.Sequential(nn.Linear(feature_dim,hidden_dim),nn.LayerNorm(hidden_dim))
        self.position=nn.Sequential(nn.Linear(3,32),nn.GELU(),nn.Linear(32,1))
        self.refiner=nn.Sequential(nn.Linear(2*hidden_dim+4,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,2))
        nn.init.zeros_(self.refiner[-1].weight);nn.init.zeros_(self.refiner[-1].bias)
        self.max_delta=float(max_delta)

    def forward(self, queries, features, boxes, valid_frames):
        if queries.ndim!=3 or features.ndim!=3 or boxes.shape!=(*queries.shape[:2],2):
            raise ValueError('Expected queries [B,Q,D], features [B,T,F], boxes [B,Q,2]')
        if valid_frames.shape!=features.shape[:2] or queries.shape[0]!=features.shape[0]:
            raise ValueError('Temporal support batch/mask mismatch')
        valid=valid_frames.to(device=features.device,dtype=torch.bool)
        if not bool(valid.any(-1).all()):raise ValueError('Temporal support needs valid visual steps')
        if not bool(torch.isfinite(boxes).all()):raise ValueError('Nonfinite coarse boxes')
        # Fixed CLIP inputs, trainable projections; padding cannot affect any query.
        clean=features.detach().masked_fill(~valid[...,None],0.)
        q=self.query_proj(queries.float());v=self.frame_proj(clean.float())
        lengths=valid.sum(-1,keepdim=True).float()
        positions=(valid.long().cumsum(-1).float()-.5)/lengths
        center=boxes[...,0:1].float();duration=boxes[...,1:2].float().clamp_min(1e-5)
        distance=positions[:,None,:]-center
        geometry=torch.stack((distance,(distance/duration).clamp(-10,10),
                              duration.expand_as(distance)),dim=-1)
        logits=torch.einsum('bqd,btd->bqt',q,v)/math.sqrt(q.shape[-1])
        logits=logits+self.position(geometry).squeeze(-1)
        weights=logits.sigmoid()*valid[:,None,:]
        attention=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        pooled=torch.einsum('bqt,btd->bqd',attention,v)
        mean=(attention*positions[:,None,:]).sum(-1,keepdim=True)
        variance=(attention*(positions[:,None,:]-mean).square()).sum(-1,keepdim=True)
        summary=torch.cat((q,pooled,boxes.float(),mean,(variance+1e-8).sqrt()),dim=-1)
        delta=self.max_delta*self.refiner(summary).tanh()
        coarse=boxes.float().clamp(1e-5,1-1e-5)
        refined=(torch.logit(coarse)+delta).sigmoid()
        return dict(logits=logits,weights=weights,valid_frames=valid,
                    coarse_boxes=boxes,pred_boxes=refined)


def fractional_interval_targets(boxes, valid):
    """Fractional GT occupancy per valid bin, preserving sub-step short events."""
    lengths=valid.sum(-1,keepdim=True).float()
    if not bool((lengths>0).all()):raise ValueError('Empty support target video')
    left=(valid.long().cumsum(-1).float()-1)/lengths;right=left+1/lengths
    start=(boxes[:,:1]-boxes[:,1:]/2).clamp(0,1)
    end=(boxes[:,:1]+boxes[:,1:]/2).clamp(0,1)
    occupancy=(torch.minimum(right,end)-torch.maximum(left,start)).clamp_min(0)*lengths
    return occupancy.clamp(0,1)*valid


def temporal_support_objective(context, targets, indices):
    """Event-balanced foreground/background BCE + continuous center/duration L1."""
    logits=context['logits'];device=logits.device
    batch_ids=[];query_ids=[];matched=[]
    for b,((query,target),gt) in enumerate(zip(indices,targets)):
        if len(query):
            batch_ids.append(torch.full_like(query,b,device=device))
            query_ids.append(query.to(device))
            matched.append(gt['boxes'].to(device).index_select(0,target.to(device)))
    if not batch_ids:
        zero=logits.sum()*0+context['pred_boxes'].sum()*0
        return {'loss_temporal_support':zero,'loss_temporal_refine':zero}
    b=torch.cat(batch_ids);q=torch.cat(query_ids);gt=torch.cat(matched).detach().float()
    valid=context['valid_frames'].index_select(0,b)
    target=fractional_interval_targets(gt,valid)
    z=logits[b,q].float();positive=target.sum(-1);negative=((1-target)*valid).sum(-1)
    pos=(F.softplus(-z)*target).sum(-1)/positive.clamp_min(1e-8)
    neg=(F.softplus(z)*(1-target)*valid).sum(-1)/negative.clamp_min(1e-8)
    terms=(positive>0).float()+(negative>0).float()
    balanced=((pos*(positive>0)+neg*(negative>0))/terms.clamp_min(1)).mean()
    refined=context['pred_boxes'][b,q]
    loss=F.l1_loss(refined,gt,reduction='mean')
    with torch.no_grad():
        probability=z.sigmoid()
        inside=((probability*target).sum(-1)/positive.clamp_min(1e-8)).mean()
        outside=((probability*(1-target)*valid).sum(-1)/negative.clamp_min(1e-8)).mean()
        correction=(refined-context['coarse_boxes'][b,q]).abs().mean()
    return dict(loss_temporal_support=balanced,loss_temporal_refine=loss,
                support_inside=inside,support_outside=outside,support_boundary_delta=correction)
