"""A small independent head. DDVC is never its child or optimizer parameter."""
import torch
from torch import nn
from torch.nn import functional as F
from .core import AUX


class JointQualityHead(nn.Module):
    def __init__(self,hidden=128):
        super().__init__()
        self.visual=nn.Sequential(nn.Linear(768,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.text=nn.Sequential(nn.Linear(768,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.score=nn.Sequential(nn.Linear(5*hidden+AUX,hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,1))

    def forward(self,features):
        a,b,t,extra=features.split((768,768,768,AUX),dim=-1)
        a=self.visual(a);b=self.visual(b);t=self.text(t)
        fused=torch.cat((a,t,a*t,(a-t).abs(),a-b,extra),dim=-1)
        return self.score(fused).squeeze(-1)


def objective(logits,quality,valid):
    """Video-balanced soft-target regression plus within-video preference pairs."""
    if logits.shape!=quality.shape or valid.shape!=quality.shape:raise ValueError('Loss shape mismatch')
    losses=[]
    for p,y,mask in zip(logits,quality,valid):
        p=p[mask];y=y[mask]
        if not len(p):raise ValueError('Empty candidate video')
        error=F.smooth_l1_loss(p.sigmoid(),y,reduction='none')
        groups=[error[m].mean() for m in (y>=.3,y<.3) if bool(m.any())]
        regression=torch.stack(groups).mean()
        preferences=(y[:,None]-y[None,:])>.1
        ranking=F.softplus(p[None,:]-p[:,None])[preferences].mean() if bool(preferences.any()) else p.sum()*0
        losses.append(regression+.2*ranking)
    return torch.stack(losses).mean()
