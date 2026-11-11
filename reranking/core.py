"""Train-only quality targets and GT-free fixed-budget proposal selection."""
import collections
import copy
import math
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

SCHEMA='c1-frozen-joint-quality-v1'
AUX=8


def words(text):return re.findall(r'[a-z0-9]+',text.lower())


def fit_idf(annotations,fit_ids):
    """Video-document IDF, fitted ONLY on the head's fit partition of TRAIN."""
    frequency=collections.Counter()
    for vid in fit_ids:
        frequency.update(set(w for s in annotations[vid]['sentences'] for w in words(s)))
    n=len(fit_ids)
    if not n:raise ValueError('Empty fit partition')
    return dict(documents=n,values={w:1+math.log((n+1)/(c+1)) for w,c in frequency.items()})


def caption_f1(a,b,idf):
    a=collections.Counter(words(a));b=collections.Counter(words(b))
    if not a or not b:return 0.
    # Unknown words receive the same fixed smoothing weight on fit/dev/inference.
    weight=lambda w:idf['values'].get(w,1+math.log(idf['documents']+1))
    hit=sum(min(a[w],b[w])*weight(w) for w in a.keys()&b.keys())
    return 2*hit/max(1e-12,sum(n*weight(w) for w,n in a.items())+sum(n*weight(w) for w,n in b.items()))


def iou_matrix(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1,2);b=np.asarray(b,dtype=np.float64).reshape(-1,2)
    inter=np.maximum(0,np.minimum(a[:,None,1],b[None,:,1])-np.maximum(a[:,None,0],b[None,:,0]))
    union=(a[:,1]-a[:,0])[:,None]+(b[:,1]-b[:,0])[None,:]-inter
    return inter/np.maximum(union,1e-12)


def targets(rows,annotation,idf):
    """One joint target: temporal IoU x weighted lexical F1 for the SAME GT event."""
    gt=annotation['timestamps'];captions=annotation['sentences']
    if len(gt)!=len(captions) or not gt:raise ValueError('Invalid training annotation')
    lexical=np.array([[caption_f1(r['sentence'],s,idf) for s in captions] for r in rows])
    reward=iou_matrix([r['timestamp'] for r in rows],gt)*lexical
    return reward.max(-1).astype(np.float32),reward.astype(np.float32)


def joint_score(row):
    return float(row['proposal_score'])+float(row['sentence_score'])/(len(row['sentence'].split())**2+1e-5)


def temporal_iou(a,b):
    x=max(0.,min(a[1],b[1])-max(a[0],b[0]))
    return x/max(1e-12,(a[1]-a[0])+(b[1]-b[0])-x)


def select(rows,n,quality,beta,method='softnms'):
    """Same Gaussian Soft-NMS as C1; only initial score receives beta*quality.

    No GT argument. Counter N, timestamps and generated sentences are retained.
    beta=0 reproduces the original selector, including negative-score handling.
    """
    if method not in ('topk','softnms'):raise ValueError('Unknown selection method')
    if len(quality)!=len(rows) or beta<0 or not math.isfinite(beta):raise ValueError('Invalid reranking input')
    pending=[]
    for index,(row,q) in enumerate(zip(rows,quality)):
        if not math.isfinite(q) or not 0<=q<=1:raise ValueError('Invalid quality')
        p=copy.deepcopy(row);score=joint_score(p)
        p.update(joint_score=score,reranker_quality=float(q),reranker_beta=float(beta))
        pending.append((p,score+beta*q,index))
    chosen=[]
    while pending and len(chosen)<n:
        index=max(range(len(pending)),key=lambda i:pending[i][1])
        p,score,_=pending.pop(index);p['selection_score']=score;chosen.append(p)
        if method=='topk':continue
        updated=[]
        for row,value,original in pending:
            factor=math.exp(-temporal_iou(p['timestamp'],row['timestamp'])**2/.5)
            updated.append((row,value*factor if value>0 else value-abs(value)*(1-factor),original))
        pending=updated
    return sorted(chosen,key=lambda p:p['timestamp'])


def dev_reward(rows,chosen,reward):
    """Video-balanced one-to-one reward, used on TRAIN holdout only."""
    byid={int(r['query_id']):i for i,r in enumerate(rows)}
    m=reward[[byid[int(r['query_id'])] for r in chosen]]
    if not m.size:return 0.
    a,b=linear_sum_assignment(-m)
    return float(m[a,b].sum()/max(m.shape[1],len(chosen),1))


def feature_pool(features,rows,duration,wte,tokenizer):
    """Local/context CLIP + frozen, trained-from-scratch GPT2 lexical embedding.

    No GT, query ID, absolute center/start/end or query-state vector is an input.
    wte is the existing C1 token embedding, not a new pretrained language model.
    """
    import torch
    from torch.nn import functional as F
    f=features.detach().float();length=len(f)
    if not length or f.ndim!=2 or f.shape[-1]!=768 or duration<=0:raise ValueError('Invalid features')
    if not bool(torch.isfinite(f).all()):raise ValueError('Nonfinite features')
    position=(torch.arange(length,device=f.device)+.5)/length
    local,context,text,aux=[],[],[],[]
    with torch.no_grad():
        for row in rows:
            s,e=[float(v)/duration for v in row['timestamp']]
            if not 0<=s<e<=1.00001:raise ValueError('Invalid candidate interval')
            inside=(position>=s)&(position<=e)
            if not bool(inside.any()):inside[(position-(s+e)/2).abs().argmin()]=True
            margin=max((e-s)/2,1/length)
            outside=((position>=s-margin)&(position<=e+margin))&~inside
            if not bool(outside.any()):outside=~inside
            if not bool(outside.any()):outside=torch.ones_like(inside)
            a=f[inside].mean(0);b=f[outside].mean(0)
            an=F.normalize(a,dim=0);bn=F.normalize(b,dim=0)
            tokens=tokenizer.encode(row['sentence'],add_special_tokens=False)
            tokens=tokens[:64] or [tokenizer.eos_token_id]
            ids=torch.tensor(tokens,device=wte.weight.device,dtype=torch.long)
            embedding=F.normalize(wte(ids).float().mean(0),dim=0)
            variance=float((f[inside]-a).square().mean()/(f[inside].square().mean()+1e-8))
            local.append(an);context.append(bn);text.append(embedding.to(f.device))
            aux.append([float(row['proposal_score']),max(-10.,min(0.,float(row['sentence_score'])))/10,
                        max(-2.,min(2.,joint_score(row)))/2,math.log1p(len(words(row['sentence'])))/4,
                        min(1.,e-s),min(2.,variance),float((an*bn).sum()),float(inside.float().mean())])
    x=torch.cat((torch.stack(local),torch.stack(context),torch.stack(text),f.new_tensor(aux)),dim=-1)
    if not bool(torch.isfinite(x).all()):raise ValueError('Nonfinite cached features')
    return x.cpu().numpy().astype(np.float16)
