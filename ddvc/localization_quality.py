"""Detached temporal RoI quality estimation; no annotation input at inference."""
import torch
from torch import nn
from torch.nn import functional as F


def interval_edges(boxes):
    center, duration = boxes.unbind(-1)
    return torch.stack((center - duration / 2, center + duration / 2), -1).clamp(0, 1)


@torch.no_grad()
def quality_targets(boxes, targets):
    """Best tIoU against any annotated event, including unmatched queries."""
    edges = interval_edges(boxes)
    result = boxes.new_zeros(boxes.shape[:2])
    for b, target in enumerate(targets):
        gt = interval_edges(target['boxes'].to(boxes))
        if not len(gt):
            continue
        pred = edges[b]
        intersection = (torch.minimum(pred[:, None, 1], gt[None, :, 1]) -
                        torch.maximum(pred[:, None, 0], gt[None, :, 0])).clamp(min=0)
        union = (pred[:, 1] - pred[:, 0])[:, None] + (gt[:, 1] - gt[:, 0])[None] - intersection
        result[b] = (intersection / union.clamp(min=1e-8)).max(-1).values
    return result


class LocalizationQualityHead(nn.Module):
    def __init__(self, query_dim, feature_dim, bins=8, hidden_dim=128):
        super().__init__()
        if bins < 2:
            raise ValueError('quality_bins must be >= 2')
        self.bins = bins
        self.visual_proj = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 32), nn.ReLU())
        self.query_proj = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, 64), nn.ReLU())
        self.predict = nn.Sequential(nn.Linear(bins * 32 + 64 + 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, queries, boxes, video, valid_mask):
        # Detach all original DDVC tensors: auxiliary training cannot move its
        # boundaries, caption features or encoder through this loss.
        queries, boxes, video = queries.detach(), boxes.detach(), video.detach()
        pooled = []
        offsets = torch.linspace(-0.75, 0.75, self.bins, device=boxes.device, dtype=boxes.dtype)
        for b in range(video.shape[0]):
            frames = video[b][valid_mask[b].bool()]
            if frames.shape[0] == 0:
                raise ValueError('Quality head requires at least one valid frame per video')
            position = (boxes[b, :, :1] + boxes[b, :, 1:] * offsets).clamp(0, 1) * (frames.shape[0] - 1)
            left = position.floor().long()
            right = (left + 1).clamp(max=frames.shape[0] - 1)
            blend = (position - left).unsqueeze(-1)
            pooled.append(frames[left] * (1 - blend) + frames[right] * blend)
        visual = self.visual_proj(torch.stack(pooled)).flatten(2)
        representation = torch.cat((visual, self.query_proj(queries), boxes), -1)
        return self.predict(representation).squeeze(-1).sigmoid()


def quality_loss(scores, boxes, targets):
    target = quality_targets(boxes.detach(), targets)
    return {
        'loss_quality': F.mse_loss(scores, target),
        'quality_mean': scores.detach().mean(),
        'quality_target_mean': target.mean(),
        'quality_mae': (scores.detach() - target).abs().mean(),
    }
