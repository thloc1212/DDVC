import copy
import numpy as np
from reranking.core import iou_matrix,joint_score,dev_reward
from scipy.optimize import linear_sum_assignment

SCHEMA='c1-cached-relation-count-v1'
RECIPE=dict(seed=777,epochs=20,lr=.001,weight_decay=.0001,hidden=64,
            pair_iou=.5,pair_quality=.2,pair_margin=.03,
            relation_lambdas=[0.,.25,.5,1.,2.],count_blends=[0.,.25,.5,1.])


def pair_targets(rows,annotation,reward):
    """GT identity, not overlap/text equality, defines a duplicate TRAIN pair."""
    overlap=iou_matrix([r['timestamp'] for r in rows],annotation['timestamps'])
    reward=np.asarray(reward)
    if reward.shape!=overlap.shape:raise ValueError('Training reward/GT identity mismatch')
    assignment=reward.argmax(-1);best=reward[np.arange(len(rows)),assignment]
    margin=best-np.partition(reward,-2,axis=1)[:,-2] if reward.shape[1]>1 else best
    reliable=((overlap[np.arange(len(rows)),assignment]>=RECIPE['pair_iou']) &
              (best>=RECIPE['pair_quality']) & (margin>RECIPE['pair_margin']))
    valid=np.triu(reliable[:,None]&reliable[None,:],k=1)
    labels=(assignment[:,None]==assignment[None,:]).astype(np.float32)
    hard=(iou_matrix([r['timestamp'] for r in rows],[r['timestamp'] for r in rows])>=.3)
    return labels,valid,hard


def geometry(rows,duration,features,quality):
    if duration<=0:raise ValueError('Invalid video duration')
    boxes=np.asarray([r['timestamp'] for r in rows],np.float32)/duration
    if np.any(boxes[:,1]<=boxes[:,0]) or not np.isfinite(boxes).all():raise ValueError('Invalid cached intervals')
    s,e=boxes.T;d=e-s;c=(s+e)/2
    inter=np.maximum(0,np.minimum(e[:,None],e[None,:])-np.maximum(s[:,None],s[None,:]))
    short=np.minimum(d[:,None],d[None,:]);long=np.maximum(d[:,None],d[None,:])
    f=np.asarray(features,np.float32);q=np.asarray(quality,np.float32)
    unit=lambda x:x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
    local=unit(f[:,:768]);text=unit(f[:,1536:2304])
    g=np.stack([iou_matrix(boxes,boxes),inter/np.maximum(short,1e-8),short/np.maximum(long,1e-8),
                np.minimum(4,np.abs(c[:,None]-c[None,:])/np.maximum(long,1e-8)),
                np.abs(s[:,None]-s[None,:]),np.abs(e[:,None]-e[None,:]),
                local@local.T,text@text.T,np.abs(q[:,None]-q[None,:])],axis=-1)
    if not np.isfinite(g).all():raise ValueError('Nonfinite relation features')
    return g.astype(np.float32)


def relation_select(rows,n,quality,beta,duplicate,penalty):
    """Keep original caption/interval; suppress learned duplication, not temporal overlap."""
    q=np.asarray(quality,float);dup=np.asarray(duplicate,float)
    if dup.shape!=(len(rows),len(rows)) or not np.isfinite(dup).all() or np.any((dup<0)|(dup>1)):
        raise ValueError('Invalid duplicate probabilities')
    scores=np.array([joint_score(r) for r in rows])+beta*q
    if not np.isfinite(scores).all() or penalty<0:raise ValueError('Invalid relation scores')
    pending=list(range(len(rows)));chosen=[]
    while pending and len(chosen)<n:
        index=max(pending,key=lambda i:scores[i]);pending.remove(index)
        row=copy.deepcopy(rows[index]);row.update(joint_score=joint_score(row),selection_score=float(scores[index]),
                  reranker_quality=float(q[index]),reranker_beta=float(beta),relation_penalty=float(penalty))
        chosen.append(row)
        # Scores may be negative: subtracting a nonnegative penalty never improves them.
        for j in pending:scores[j]-=penalty*dup[index,j]
    return sorted(chosen,key=lambda r:r['timestamp'])


def calibrated_count(base_n,head_n,blend,q):
    # These benchmark videos contain events. Q remains the hard candidate capacity.
    if blend==0:return int(np.clip(base_n,0,q))
    return int(np.clip(np.floor((1-blend)*base_n+blend*head_n+.5),1,q))


def median_count(probabilities):
    probabilities=np.asarray(probabilities)
    return int(np.searchsorted(probabilities.cumsum(),.5,side='left').clip(0,len(probabilities)-1))


def normalize_features(features,rows,quality):
    f=np.asarray(features,np.float32);q=np.asarray(quality,np.float32)
    base=np.clip([joint_score(r) for r in rows],-2,2).astype(np.float32)/2
    return np.concatenate([f,q[:,None],base[:,None]],axis=-1)


def temporal_matches(rows,gt):
    matrix=iou_matrix([r['timestamp'] for r in rows],gt)
    if not matrix.size:return set()
    a,b=linear_sum_assignment(-((matrix>=.5)*(min(matrix.shape)+1)+matrix))
    return {int(j) for i,j in zip(a,b) if matrix[i,j]>=.5}
