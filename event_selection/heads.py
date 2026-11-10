import torch
from torch import nn
from torch.nn import functional as F


class RelationHead(nn.Module):
    def __init__(self,hidden=64):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(2314,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.pair=nn.Sequential(nn.Linear(3*hidden+9,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1))

    def forward(self,x,geometry):
        h=self.encoder(x);a=h[:,None,:].expand(-1,len(h),-1);b=a.transpose(0,1)
        return self.pair(torch.cat([a+b,(a-b).abs(),a*b,geometry],dim=-1)).squeeze(-1)


def relation_loss(logits,labels,valid,hard):
    losses=F.binary_cross_entropy_with_logits(logits,labels,reduction='none')
    positive=valid&(labels>.5);negative=valid&~positive
    terms=[]
    if bool(positive.any()):terms.append(losses[positive].mean())
    groups=[losses[m].mean() for m in (negative&hard,negative&~hard) if bool(m.any())]
    if groups:terms.append(torch.stack(groups).mean())
    return torch.stack(terms).mean() if terms else logits.sum()*0


class CountHead(nn.Module):
    def __init__(self,queries,hidden=64):
        super().__init__();self.queries=queries
        self.encoder=nn.Sequential(nn.Linear(2314,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.score=nn.Sequential(nn.Linear(3*hidden+5,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,queries+1))

    def forward(self,x,base_n):
        h=self.encoder(x);q=x[:,-2]
        aux=torch.stack([x.new_tensor(base_n/self.queries),q.mean(),q.std(unbiased=False),
                         (q>.25).float().mean(),(q>.5).float().mean()])
        pooled=torch.cat([h.mean(0),h.amax(0),h.std(0,unbiased=False),aux])
        return self.score(pooled)


def count_loss(logits,target,weights):
    # Full annotated TRAIN count, capped only by candidate Q; never sampled caption count.
    target=torch.as_tensor(target,device=logits.device,dtype=torch.long)
    ce=F.cross_entropy(logits[None],target[None],reduction='sum')*weights[target]
    cdf=logits.softmax(-1).cumsum(-1)
    desired=(torch.arange(len(logits),device=logits.device)>=target).float()
    return ce+.1*(cdf-desired).square().mean()
