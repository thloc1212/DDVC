"""Optional V1 event-set heads. Temporal overlap alone is not duplication."""
import torch
from torch import nn
from torch.nn import functional as F


class AttentionCounter(nn.Module):
    def __init__(self, feature_dim=512, width=128, max_count=20):
        super().__init__()
        self.project = nn.Linear(feature_dim + 3, width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.pool_score = nn.Linear(width, 1)
        self.classifier = nn.Linear(width, max_count + 1)

    def forward(self, queries, boxes, confidence):
        # Coordinates remain differentiable; confidence is not a GT input.
        x = self.project(torch.cat((queries, boxes, confidence.sigmoid()), -1))
        context, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm(x + context)
        pooled = (x * self.pool_score(x).softmax(1)).sum(1)
        return self.classifier(pooled)


def ordinal_cdf_loss(logits, counts):
    """Ordered-distance auxiliary on a count distribution; not a CORAL head."""
    probabilities = logits.float().softmax(-1)
    thresholds = torch.arange(logits.shape[-1] - 1, device=logits.device)
    target_cdf = (counts[:, None] <= thresholds).float()
    return (probabilities.cumsum(-1)[:, :-1] - target_cdf).abs().mean()


def interval_iou(boxes, targets):
    start, end = boxes[:, 0] - boxes[:, 1] / 2, boxes[:, 0] + boxes[:, 1] / 2
    ts, te = targets[:, 0] - targets[:, 1] / 2, targets[:, 0] + targets[:, 1] / 2
    intersection = (torch.minimum(end[:, None], te) - torch.maximum(start[:, None], ts)).clamp_min(0)
    union = boxes[:, 1, None] + targets[:, 1] - intersection
    return intersection / union.clamp_min(1e-6)


class DuplicateHead(nn.Module):
    def __init__(self, feature_dim=512, width=128):
        super().__init__()
        self.project = nn.Linear(feature_dim, width)
        # Symmetric pair features: semantic difference/product, geometry, scores.
        self.network = nn.Sequential(nn.Linear(width * 2 + 5, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, queries, boxes, confidence):
        h = self.project(queries)
        a, b = h[:, :, None], h[:, None, :]
        c, d = boxes[..., 0], boxes[..., 1].clamp_min(1e-6)
        start, end = c-d/2, c+d/2
        intersection = (torch.minimum(end[:, :, None], end[:, None, :]) -
                        torch.maximum(start[:, :, None], start[:, None, :])).clamp_min(0)
        iou = intersection / (d[:, :, None] + d[:, None, :] - intersection).clamp_min(1e-6)
        geometry = torch.stack(((c[:, :, None]-c[:, None, :]).abs(),
                                (d[:, :, None].log()-d[:, None, :].log()).abs(), iou), -1)
        scores = confidence.sigmoid().squeeze(-1)
        score_pair = torch.stack((torch.minimum(scores[:, :, None], scores[:, None, :]),
                                  torch.maximum(scores[:, :, None], scores[:, None, :])), -1)
        return self.network(torch.cat(((a-b).abs(), a*b, geometry, score_pair), -1)).squeeze(-1)


def duplicate_objective(logits, boxes, targets, min_iou=.5, ambiguity_margin=.1):
    """Many-to-one GT association, detached. Balanced positive/negative pair BCE.

    Different GT identities remain negative even for nested/overlapping GT.
    Ambiguous assignments are ignored instead of providing false duplicate labels.
    """
    positive, negative = [], []
    for batch, target in enumerate(targets):
        gt = target['boxes']
        if not len(gt):
            continue
        with torch.no_grad():
            overlaps = interval_iou(boxes[batch].detach().float(), gt.float())
            best, assigned = overlaps.max(-1)
            runner_up = overlaps.topk(2, dim=-1).values[:, 1] if len(gt) > 1 else torch.zeros_like(best)
            valid = (best >= min_iou) & ((best-runner_up) >= ambiguity_margin)
            pairs = torch.triu(torch.ones_like(logits[batch], dtype=torch.bool), diagonal=1)
            pairs &= valid[:, None] & valid[None, :]
            same = assigned[:, None] == assigned[None, :]
        positive.append(logits[batch][pairs & same])
        negative.append(logits[batch][pairs & ~same])
    positive = torch.cat(positive) if positive else logits.reshape(-1)[:0]
    negative = torch.cat(negative) if negative else logits.reshape(-1)[:0]
    # Do not train a degenerate all-negative classifier before proposals match GT.
    loss = logits.sum() * 0
    if positive.numel() and negative.numel():
        loss = .5 * (F.softplus(-positive.float()).mean() + F.softplus(negative.float()).mean())
    return {'loss_duplicate_pair': loss,
            'duplicate_supervised_batch_fraction': logits.new_tensor(float(bool(positive.numel() and negative.numel()))),
            'duplicate_positive_pairs': logits.new_tensor(positive.numel()),
            'duplicate_negative_pairs': logits.new_tensor(negative.numel())}
