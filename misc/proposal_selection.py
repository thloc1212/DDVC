"""GT-free temporal selection. Merge is optional, conservative and not a loss."""
import copy
import math
import re


def temporal_iou(a, b):
    intersection = max(0., min(a[1], b[1]) - max(a[0], b[0]))
    union = max(0., a[1]-a[0]) + max(0., b[1]-b[0]) - intersection
    return intersection / max(union, 1e-12)


def caption_tokens(text):
    return set(re.findall(r'\w+', text.lower()))


def joint_score(row, alpha=1., temperature=2.):
    return float(row['proposal_score']) + alpha * float(row['sentence_score']) / (
        float(len(row['sentence'].split())) ** temperature + 1e-5)


def merge_proposals(rows, iou_threshold=.7, text_threshold=.9):
    """Fuse near-identical candidates; never merge just because captions repeat.

    Every member must agree with every other member (no transitive chaining).
    Boundaries are convex score-weighted averages, not interval unions. Caption,
    confidence and query_id remain those of the highest-scoring representative.
    """
    if not 0 < iou_threshold <= 1 or not 0 < text_threshold <= 1:
        raise ValueError('Merge thresholds must be in (0,1]')
    groups=[]
    for row in sorted(rows, key=lambda p:p['joint_score'], reverse=True):
        tokens=caption_tokens(row['sentence'])
        group=next((g for g in groups if tokens and all(
            temporal_iou(row['timestamp'], other['timestamp']) >= iou_threshold and
            len(tokens & caption_tokens(other['sentence'])) / max(1,len(tokens | caption_tokens(other['sentence']))) >= text_threshold
            for other in g)),None)
        if group is None: groups.append([row])
        else: group.append(row)
    merged=[]
    for group in groups:
        representative=copy.deepcopy(group[0])
        if len(group)>1:
            high=max(p['joint_score'] for p in group)
            weights=[math.exp(p['joint_score']-high) for p in group]
            representative['timestamp']=[sum(w*p['timestamp'][j] for w,p in zip(weights,group))/sum(weights) for j in [0,1]]
            # raw_box is kept as the original representative box for provenance.
            representative['merged_query_ids']=[p.get('query_id') for p in group]
            representative['merge_size']=len(group)
        merged.append(representative)
    return merged


def select_proposals(rows, top_n, method='topk', sigma=.5, alpha=1., temperature=2.,
                     merge=False, merge_iou=.7, merge_text=.9):
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
    if merge:pending=merge_proposals(pending,merge_iou,merge_text)
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
    return sorted(chosen,key=lambda p:p['timestamp'])
