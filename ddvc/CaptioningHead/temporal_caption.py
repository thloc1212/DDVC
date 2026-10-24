"""Continuous temporal conditioning of a scratch caption decoder.

Inspired by temporal representations in Vid2Seq (Scenic), not its discrete
timestamp vocabulary. Original PyTorch code; no external checkpoint/dependency.
"""
import math
import torch
from torch import nn


class TemporalCaptionConditioning(nn.Module):
    def __init__(self, dimension=768, bands=8, hidden=128, time_prefix=True,
                 frame_positions=True, prefix_scale=1., frame_scale=1., detach_boxes=True):
        super().__init__()
        if bands < 1 or bands > 12 or dimension < 1 or hidden < 1:
            raise ValueError('Invalid temporal caption embedding dimensions')
        if not math.isfinite(prefix_scale) or not math.isfinite(frame_scale) or min(prefix_scale,frame_scale)<0:
            raise ValueError('Temporal scales must be finite and nonnegative')
        self.detach_boxes=detach_boxes
        self.prefix_scale,self.frame_scale=prefix_scale,frame_scale
        self.register_buffer('frequencies',2.**torch.arange(bands,dtype=torch.float32),persistent=False)
        self.prefix_mlp=None
        self.frame_projection=None
        if time_prefix:
            self.prefix_mlp=nn.Sequential(nn.Linear(2+4*bands,hidden),nn.GELU(),nn.Linear(hidden,dimension))
            nn.init.zeros_(self.prefix_mlp[-1].weight)
            nn.init.zeros_(self.prefix_mlp[-1].bias)
        if frame_positions:
            self.frame_projection=nn.Linear(1+2*bands,dimension,bias=False)
            nn.init.zeros_(self.frame_projection.weight)

    def encode(self,coordinates):
        phase=2*math.pi*coordinates.float().unsqueeze(-1)*self.frequencies
        fourier=torch.cat((phase.sin(),phase.cos()),-1).flatten(-2)
        # Raw coordinates disambiguate 0 and 1, identical for integer-period Fourier bands.
        return torch.cat((coordinates.float(),fourier),-1)

    def prefix_delta(self,boxes):
        if self.prefix_mlp is None:
            raise RuntimeError('Time prefix conditioning is disabled')
        if boxes is None or boxes.ndim!=2 or boxes.shape[-1]!=2 or not len(boxes):
            raise ValueError('Time prefix requires aligned predicted [events, center/duration] boxes')
        if not torch.isfinite(boxes).all():
            raise ValueError('Predicted caption intervals contain NaN/Inf')
        coordinates=(boxes.detach() if self.detach_boxes else boxes).clamp(0,1)
        return self.prefix_scale*self.prefix_mlp(self.encode(coordinates))[:,None,:]

    def frame_delta(self,valid_frames):
        if self.frame_projection is None:
            raise RuntimeError('Frame temporal positions are disabled')
        if valid_frames.ndim!=2 or valid_frames.shape[0]!=1:
            raise ValueError('Caption frame positions currently require batch size 1')
        valid=valid_frames.bool()
        lengths=valid.sum(-1,keepdim=True)
        if (lengths==0).any():raise ValueError('Caption video contains no valid frames')
        # Relative positions use only valid frames; adding padding never changes them.
        positions=((valid.long().cumsum(-1).float()-.5)/lengths).masked_fill(~valid,0)
        encoded=self.encode(positions.unsqueeze(-1))
        return self.frame_scale*self.frame_projection(encoded)*valid.unsqueeze(-1)
