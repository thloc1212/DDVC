"""Learn localization quality and event relations without GT in the forward path.

Relations distinguish duplicate proposals from distinct overlapping/nested events.
The default detached inputs protect the caption/localization representation from
noisy early auxiliary targets; these heads themselves remain trainable.
"""
import torch
from torch import nn
from torch.nn import functional as F

RELATIONS = ('separate', 'overlap', 'contains', 'inside', 'duplicate')


def edges(boxes):
    return torch.stack(((boxes[..., 0] - boxes[..., 1] / 2).clamp(0, 1),
                        (boxes[..., 0] + boxes[..., 1] / 2).clamp(0, 1)), -1)


def pair_geometry(a, b):
    """Broadcasting directional geometry for [...,2] center-duration boxes."""
    ae, be = edges(a), edges(b)
    da = (ae[..., 1] - ae[..., 0]).clamp_min(1e-6)
    db = (be[..., 1] - be[..., 0]).clamp_min(1e-6)
    intersection = (torch.minimum(ae[..., 1], be[..., 1]) - torch.maximum(ae[..., 0], be[..., 0])).clamp_min(0)
    union = (da + db - intersection).clamp_min(1e-6)
    delta = b[..., 0] - a[..., 0]
    return torch.stack(((delta / da).clamp(-10, 10), (delta / db).clamp(-10, 10),
                        (db / da).log().clamp(-10, 10), intersection / union,
                        intersection / da, intersection / db, delta.abs(),
                        (db - da).expand_as(delta)), -1)


def gt_relations(a, b):
    ae, be = edges(a), edges(b)
    intersection = (torch.minimum(ae[..., 1], be[..., 1]) - torch.maximum(ae[..., 0], be[..., 0]))
    overlap = intersection > 0
    equal = (ae - be).abs().amax(-1) < 1e-6
    contains = (ae[..., 0] <= be[..., 0]) & (ae[..., 1] >= be[..., 1]) & ~equal
    inside = (be[..., 0] <= ae[..., 0]) & (be[..., 1] >= ae[..., 1]) & ~equal
    label = torch.zeros_like(intersection, dtype=torch.long)
    label = torch.where(overlap, 1, label)
    label = torch.where(contains & overlap, 2, label)
    return torch.where(inside & overlap, 3, label)


class EventSetHeads(nn.Module):
    def __init__(self, query_dim=512, feature_dim=768, hidden_dim=128, detach_inputs=True):
        super().__init__()
        self.detach_inputs = bool(detach_inputs)
        self.query = nn.Sequential(nn.Linear(2 * query_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.visual = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.quality = nn.Sequential(nn.Linear(2 * hidden_dim + 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.relation = nn.Sequential(nn.Linear(4 * hidden_dim + 8, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 5))
        self.register_buffer('duplicate_jitter', torch.sin(torch.arange(hidden_dim).float()) * .015)

    def relation_logits(self, a, b, box_a, box_b):
        a, b = torch.broadcast_tensors(a, b)
        return self.relation(torch.cat((a, b, a - b, a * b, pair_geometry(box_a, box_b)), -1))

    def forward(self, loc, cap, features, boxes, logits, valid):
        if loc.shape != cap.shape or boxes.shape != (*loc.shape[:2], 2) or valid.shape != features.shape[:2]:
            raise ValueError('Event-set query/box/feature shape mismatch')
        if self.detach_inputs:
            loc, cap, boxes, logits = (v.detach() for v in (loc, cap, boxes, logits))
        features = features.detach()
        valid = valid.to(device=features.device, dtype=torch.bool)
        if not bool(valid.any(-1).all()):
            raise ValueError('Event-set heads need nonempty visual features')
        if not all(bool(torch.isfinite(v).all()) for v in (loc, cap, boxes, logits, features.masked_fill(~valid[..., None], 0))):
            raise ValueError('Nonfinite event-set input')
        q = self.query(torch.cat((loc, cap), -1).float())
        v = self.visual(features.float().masked_fill(~valid[..., None], 0))
        positions = (valid.long().cumsum(-1).float() - .5) / valid.sum(-1, keepdim=True)
        boundary = edges(boxes.float())
        mask = ((positions[:, None] >= boundary[..., :1]) & (positions[:, None] <= boundary[..., 1:])) & valid[:, None]
        # Sub-bin proposals still receive one nearest valid feature, never an empty mean.
        nearest = (positions[:, None] - boxes[..., :1]).abs().masked_fill(~valid[:, None], float('inf')).argmin(-1)
        fallback = F.one_hot(nearest, features.shape[1]).bool() & valid[:, None]
        mask = torch.where(mask.any(-1, keepdim=True), mask, fallback)
        attention = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
        pooled = torch.einsum('bqt,btd->bqd', attention, v)
        embedding = q + pooled
        confidence = logits.float().sigmoid().amax(-1, keepdim=True)
        quality = self.quality(torch.cat((q, pooled, boxes.float(), confidence), -1)).squeeze(-1)
        relation = self.relation_logits(embedding[:, :, None], embedding[:, None, :],
                                        boxes[:, :, None].float(), boxes[:, None, :].float())
        result = dict(quality_logits=quality, relation_logits=relation, boxes=boxes.float())
        if self.training:
            # Training-only duplicate examples: a small deterministic perturbation of
            # the same matched query. No GT boxes/captions enter this forward branch.
            noisy_boxes = torch.stack(((boxes[..., 0] + .01 * boxes[..., 1]).clamp(0, 1),
                                       (boxes[..., 1] * .98).clamp_min(1e-6)), -1).float()
            noisy_embedding = embedding + self.duplicate_jitter
            result['synthetic_duplicate_logits'] = self.relation_logits(embedding, noisy_embedding, boxes.float(), noisy_boxes)
        return result


def event_set_objective(context, targets, indices, count_logits):
    quality, relation, boxes = context['quality_logits'], context['relation_logits'], context['boxes']
    zero = quality.sum() * 0 + relation.sum() * 0
    quality_terms, relation_values, relation_labels = [], [], []
    positive_pairs = 0
    for batch, (target, (query_ids, gt_ids)) in enumerate(zip(targets, indices)):
        gt = target['boxes'].to(boxes.device).detach().float()
        if len(gt) == 0:
            quality_terms.append(F.smooth_l1_loss(quality[batch].sigmoid(), torch.zeros_like(quality[batch])))
            continue
        geometry = pair_geometry(boxes[batch, :, None].detach(), gt[None])
        iou = geometry[..., 3]
        best, assigned = iou.max(-1)
        assigned = assigned.clone()
        if len(gt) > 1:
            top = iou.topk(2, -1).values
            certain = (best >= .7) & ((top[:, 0] - top[:, 1]) >= .15)
        else:
            certain = best >= .7
        assigned[~certain] = -1
        # Matched queries preserve distinct GT identities, even when the intervals
        # overlap or coincide; an uncertain unmatched proposal supplies no label.
        assigned[query_ids] = gt_ids.to(assigned.device)
        err = F.smooth_l1_loss(quality[batch].sigmoid(), best.detach(), reduction='none')
        groups = [err[mask].mean() for mask in (best >= .5, best < .5) if bool(mask.any())]
        quality_terms.append(torch.stack(groups).mean())
        allowed = (assigned[:, None] >= 0) & (assigned[None] >= 0)
        allowed.fill_diagonal_(False)
        same = assigned[:, None] == assigned[None]
        own = iou.gather(-1, assigned.clamp_min(0)[:, None]).squeeze(-1)
        allowed &= (~same) | ((own[:, None] >= .5) & (own[None] >= .5))
        safe = assigned.clamp_min(0)
        labels = gt_relations(gt[safe, None], gt[None, safe])
        labels = torch.where(same, 4, labels)
        if bool(allowed.any()):
            relation_values.append(relation[batch][allowed])
            relation_labels.append(labels[allowed])
            positive_pairs += int(((labels == 4) & allowed).sum())
        if 'synthetic_duplicate_logits' in context and len(query_ids):
            synthetic = context['synthetic_duplicate_logits'][batch, query_ids]
            relation_values.append(synthetic)
            relation_labels.append(torch.full((len(query_ids),), 4, dtype=torch.long, device=quality.device))
    quality_loss = torch.stack(quality_terms).mean() if quality_terms else zero
    if relation_values:
        predicted, labels = torch.cat(relation_values), torch.cat(relation_labels)
        ce = F.cross_entropy(predicted, labels, reduction='none')
        relation_loss = torch.stack([ce[labels == k].mean() for k in range(5) if bool((labels == k).any())]).mean()
    else:
        relation_loss = zero
    # Ordered count objective retains the original count head/loss, adding a
    # final-layer CDF distance so being off by ten is costlier than off by one.
    counts = torch.tensor([len(t['boxes']) for t in targets], device=count_logits.device)
    if bool((counts >= count_logits.shape[-1]).any()):
        raise ValueError('Count head cannot represent all GT events')
    distribution = count_logits.float().softmax(-1)
    target_distribution = F.one_hot(counts, count_logits.shape[-1]).float()
    count_loss = (distribution.cumsum(-1)[..., :-1] - target_distribution.cumsum(-1)[..., :-1]).square().mean()
    with torch.no_grad():
        expected = (distribution * torch.arange(distribution.shape[-1], device=distribution.device)).sum(-1)
    return dict(loss_event_quality=quality_loss, loss_event_relation=relation_loss,
                loss_count_ordinal=count_loss, event_quality_mean=quality.sigmoid().mean().detach(),
                event_relation_real_duplicate_pairs=quality.new_tensor(positive_pairs),
                event_counter_expected_error=(expected - counts).abs().mean().detach())
