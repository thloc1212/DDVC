"""Per-event visual attention masks from predicted normalized center/duration."""
import torch


def event_visual_mask(boxes, valid_frames, context_ratio=0.1, min_frames=4):
    """Return [events, frames] keep masks, including a small boundary context.

    Frame positions use the actual valid feature length, not the padded length.
    Hard selection is detached; the existing localization losses train boundaries.
    This function never reads GT timestamps or captions.
    """
    if boxes.ndim != 2 or boxes.shape[-1] != 2:
        raise ValueError('Expected predicted boxes [events, 2] in normalized center/duration format')
    if context_ratio < 0 or min_frames < 1:
        raise ValueError('context_ratio must be >= 0 and min_frames must be >= 1')
    if valid_frames.ndim != 2 or valid_frames.shape[0] not in (1, len(boxes)):
        raise ValueError('Expected valid_frames [1 or events, frames]')
    with torch.no_grad():
        boxes = boxes.detach().float()
        if not torch.isfinite(boxes).all():
            raise ValueError('Predicted intervals contain NaN/Inf')
        valid = valid_frames.to(device=boxes.device, dtype=torch.bool).expand(len(boxes), -1)
        lengths = valid.sum(-1, keepdim=True)
        if (lengths == 0).any():
            raise ValueError('Cannot caption a video without valid visual frames')
        positions = (valid.long().cumsum(-1).float() - 0.5) / lengths
        center, duration = boxes[:, :1].clamp(0, 1), boxes[:, 1:].clamp(0, 1)
        radius = duration * (0.5 + context_ratio)
        keep = valid & (positions >= center - radius) & (positions <= center + radius)
        # Very short predictions still need visual evidence; never expose padding.
        k = min(int(min_frames), valid.shape[-1])
        distances = (positions - center).abs().masked_fill(~valid, float('inf'))
        nearest = distances.topk(k, largest=False).indices
        fallback = torch.zeros_like(valid).scatter(1, nearest, True) & valid
        return keep | (fallback & (keep.sum(-1, keepdim=True) < lengths.clamp_max(k)))


def query_event_boxes(boxes, query_ids):
    """Preserve query IDs when expanding query-caption pairs or selecting matches."""
    if boxes is None or boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[-1] != 2:
        raise ValueError('Local captioning requires predicted boxes [1, queries, 2]; batch_size=1')
    return boxes[0].index_select(0, query_ids.to(device=boxes.device, dtype=torch.long))
