"""Adapt learned/gated proposal masks from dense-video-cap to DDVC.

Based on ActionPropDenseCap.mask_model and positional_encodings, revision
43914e17769701b9cf98eda203ae4c465b315fab, Copyright (c) 2018 salesforce.com,
inc. SPDX-License-Identifier: BSD-3-Clause. See third_party/dense-video-cap/LICENSE.txt.

Adaptations: normalized DDVC boxes/reference anchors, fixed mask grid with
length-aware interpolation, LayerNorm for tiny batches, sigmoid probabilities,
and gating frozen CLIP features before GPT-2 cross-attention.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .local_visual import event_visual_mask


def proposal_confidence(others, query_ids=None):
    logits = others.get('mask_event_logits')
    if logits is None:
        return None
    if logits.shape[0] != 1:
        raise ValueError('Proposal masks currently require DDVC batch size 1')
    confidence = logits[0].sigmoid().amax(-1)
    return confidence if query_ids is None else confidence.index_select(0, query_ids.to(confidence.device))


def proposal_anchors(others, query_ids=None):
    anchors = others.get('mask_anchor_boxes')
    if anchors is None:
        return None
    anchors = anchors[0]
    return anchors if query_ids is None else anchors.index_select(0, query_ids.to(anchors.device))


def mask_layer_context(others, logits, reference, boxes):
    if reference.shape[-1] == 1:
        reference = torch.cat((reference, boxes[..., 1:].detach()), -1)
    return dict(others, mask_event_logits=logits, mask_anchor_boxes=reference.detach())


def interval_edges(boxes):
    center, duration = boxes.unbind(-1)
    return torch.stack((center - duration / 2, center + duration / 2), -1).clamp(0, 1)


@torch.no_grad()
def hard_window(boxes, positions):
    edges = interval_edges(boxes.detach())
    keep = (positions >= edges[:, :1]) & (positions <= edges[:, 1:])
    # A sub-frame proposal still needs one frame; no forced full-timeline cover.
    closest = (positions - boxes.detach()[:, :1]).abs().argmin(-1, keepdim=True)
    fallback = torch.zeros_like(keep).scatter(1, closest, True)
    return keep | (fallback & ~keep.any(-1, keepdim=True))


class LearnedGatedProposalMask(nn.Module):
    def __init__(self, bins=64, encoding_dim=128, query_dim=0,
                 gate_min=0.0, gate_max=1.0, context_ratio=0.0, min_frames=1):
        super().__init__()
        if bins < 2 or encoding_dim < 8 or encoding_dim % 8:
            raise ValueError('bins >= 2 and encoding_dim divisible by 8 are required')
        self.bins, self.encoding_dim = bins, encoding_dim
        if not 0 <= gate_min <= gate_max <= 1 or context_ratio < 0 or min_frames < 1:
            raise ValueError('Invalid gate bounds or context settings')
        self.gate_min, self.gate_max = gate_min, gate_max
        self.context_ratio, self.min_frames = context_ratio, min_frames
        self.query_proj = nn.Linear(query_dim, 64) if query_dim else None
        # Original uses BatchNorm1d; LayerNorm supports a single matched event.
        self.mask_model = nn.Sequential(
            nn.Linear(encoding_dim + bins + (64 if query_dim else 0), encoding_dim, bias=False),
            nn.LayerNorm(encoding_dim), nn.ReLU(), nn.Linear(encoding_dim, bins))

    def position_encoding(self, coordinates):
        channels = self.encoding_dim // 4
        index = torch.arange(channels, device=coordinates.device)
        frequency = 10000.0 ** (2 * torch.div(index, 2, rounding_mode='floor') / channels)
        phase = coordinates.float().unsqueeze(-1) * self.bins / frequency
        return torch.where(index % 2 == 0, phase.sin(), phase.cos()).flatten(1)

    def forward(self, boxes, valid_frames, event_scores, anchor_boxes=None, query=None):
        if boxes.ndim != 2 or boxes.shape[-1] != 2 or len(boxes) == 0:
            raise ValueError('Expected nonempty [events, 2] predicted center/duration')
        if valid_frames.ndim != 2 or valid_frames.shape[0] != 1:
            raise ValueError('Expected one video valid mask [1, frames]')
        if event_scores is None or event_scores.shape != boxes.shape[:1]:
            raise ValueError('Each proposal requires an aligned event confidence')
        if not torch.isfinite(boxes).all() or not torch.isfinite(event_scores).all():
            raise ValueError('Non-finite proposal mask inputs')
        valid = valid_frames.to(boxes.device).bool()[0]
        length = int(valid.sum())
        if length == 0:
            raise ValueError('Proposal mask requires at least one valid video frame')
        anchor_boxes = boxes.detach() if anchor_boxes is None else anchor_boxes.detach()
        if anchor_boxes.shape != boxes.shape:
            raise ValueError('Anchors and proposals must have the same shape')
        grid = (torch.arange(self.bins, device=boxes.device).float() + .5) / self.bins
        anchor_mask = hard_window(anchor_boxes, grid[None]).float()
        coordinates = torch.cat((interval_edges(boxes), interval_edges(anchor_boxes)), -1)
        encoded = self.position_encoding(coordinates)
        inputs = [encoded, anchor_mask]
        if self.query_proj is not None:
            if query is None or query.shape != (len(boxes), self.query_proj.in_features):
                raise ValueError('Query-conditioned masks need aligned [events, query_dim] states')
            inputs.append(self.query_proj(query))
        grid_logits = self.mask_model(torch.cat(inputs, -1))
        # Interpolate only along real frame positions; padding never changes a mask.
        local_logits = F.interpolate(grid_logits[:, None], size=length, mode='linear', align_corners=False)[:, 0]
        logits = local_logits.new_zeros((len(boxes), valid.numel()))
        logits[:, valid] = local_logits
        positions = (torch.arange(length, device=boxes.device).float() + .5)[None] / length
        hard = torch.zeros_like(logits)
        if self.context_ratio or self.min_frames > 1:
            hard = event_visual_mask(boxes, valid_frames, self.context_ratio, self.min_frames).to(logits.dtype)
        else:
            hard[:, valid] = hard_window(boxes, positions).to(logits.dtype)
        soft = logits.sigmoid() * valid
        gate = (self.gate_min + (self.gate_max - self.gate_min) * event_scores.clamp(0, 1)).unsqueeze(-1)
        weights = (gate * hard + (1 - gate) * soft) * valid
        return {'weights': weights, 'logits': logits, 'hard': hard, 'gate': gate, 'soft': soft}

    def objective(self, boxes, valid_frames, event_scores, anchor_boxes=None,
                  query=None, target_boxes=None, balanced=False, prune_threshold=-1.):
        details = self(boxes, valid_frames, event_scores, anchor_boxes, query)
        valid = valid_frames.to(boxes.device).bool().expand_as(details['logits'])
        target = details['hard']
        if target_boxes is not None:
            if target_boxes.shape != boxes.shape:
                raise ValueError('GT boxes must follow matched query order')
            target = torch.zeros_like(details['logits'])
            length = int(valid_frames.bool().sum())
            positions = (torch.arange(length, device=boxes.device).float() + .5)[None] / length
            target[:, valid[0]] = hard_window(target_boxes.detach(), positions).to(target.dtype)
        element_loss = F.binary_cross_entropy_with_logits(details['logits'], target, reduction='none')
        if balanced:
            # Equal foreground/background contribution per matched event; no dataset-dependent pos_weight.
            pos, neg = valid & target.bool(), valid & ~target.bool()
            pos_loss = (element_loss * pos).sum(-1) / pos.sum(-1).clamp_min(1)
            neg_loss = (element_loss * neg).sum(-1) / neg.sum(-1).clamp_min(1)
            groups = pos.any(-1).float() + neg.any(-1).float()
            loss = ((pos_loss + neg_loss) / groups.clamp_min(1)).mean()
        else:
            loss = element_loss[valid].mean()
        with torch.no_grad():
            active = valid if prune_threshold < 0 else valid & (
                (details['weights'] > prune_threshold) | details['hard'].bool())
            soft_binary = details['soft'] >= .5
            intersection = (soft_binary & target.bool() & valid).sum(-1).float()
            union = ((soft_binary | target.bool()) & valid).sum(-1).clamp_min(1)
        return {'loss_proposal_mask': loss,
                'proposal_mask_gate': details['gate'].detach().mean(),
                'proposal_mask_gate_min': details['gate'].detach().min(),
                'proposal_mask_gate_max': details['gate'].detach().max(),
                'proposal_mask_active_fraction': active.sum().float() / valid.sum(),
                'proposal_mask_target_iou': (intersection / union).mean(),
                'proposal_mask_mean': details['weights'].detach()[valid].mean(),
                'proposal_mask_inside': details['hard'].detach()[valid].mean()}
