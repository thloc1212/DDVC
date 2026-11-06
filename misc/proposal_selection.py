"""Counter-guided Gaussian Soft-NMS for finalized C2; no merge or dedup loss."""
import copy
import math


def temporal_iou(a, b):
    intersection = max(0., min(a[1], b[1]) - max(a[0], b[0]))
    union = max(0., a[1]-a[0]) + max(0., b[1]-b[0]) - intersection
    return intersection / max(union, 1e-12)


def joint_score(row, alpha=1., temperature=2.):
    return float(row['proposal_score']) + alpha * float(row['sentence_score']) / (
        float(len(row['sentence'].split())) ** temperature + 1e-5)


def select_proposals(rows, top_n, method='softnms', sigma=.5, alpha=1., temperature=2.):
    if method not in ('topk','softnms') or not math.isfinite(sigma) or sigma<=0:
        raise ValueError('Invalid proposal selection method/sigma')
    if top_n<0: raise ValueError('Event count cannot be negative')
    pending=copy.deepcopy(rows)
    for row in pending:
        interval=row['timestamp']
        if len(interval)!=2 or not all(math.isfinite(v) for v in interval) or interval[1]<interval[0]:
            raise ValueError('Invalid predicted temporal interval')
        row['joint_score']=joint_score(row,alpha,temperature)
        if not math.isfinite(row['joint_score']):raise ValueError('Non-finite joint score')
    if method=='topk':
        chosen=sorted(pending,key=lambda p:p['joint_score'],reverse=True)[:int(top_n)]
    else:
        pending=[(p,p['joint_score']) for p in pending];chosen=[]
        while pending and len(chosen)<int(top_n):
            index=max(range(len(pending)),key=lambda i:pending[i][1])
            row,score=pending.pop(index);row['selection_score']=score;chosen.append(row)
            updated=[]
            for p,s in pending:
                factor=math.exp(-temporal_iou(row['timestamp'],p['timestamp'])**2/sigma)
                # Negative joint scores must decrease, never improve toward zero.
                updated.append((p,s*factor if s>0 else s-abs(s)*(1-factor)))
            pending=updated
    chosen=sorted(chosen,key=lambda p:p['timestamp'])
    return chosen
