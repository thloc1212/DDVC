"""Legacy DDVC caption targets, with a guard against empty supervision.

Token 25 is ':' in GPT-2, not '.'. Keep the historical prefix rule for rows
where it leaves targets, but restore a row's original caption targets when a
truncated delimiter/no delimiter would erase every non-padding target.
"""
import torch


def prepare_legacy_caption_targets(labels, truncate_prefix=None, ignore_index=50256):
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError('Caption labels require [captions, sequence_length >= 2]')
    # clone is required: contiguous() alone can alias the caller for one row.
    original = labels[:, 1:].contiguous().clone()
    if original.shape[0] == 0:
        raise ValueError('Caption loss requires at least one caption')
    has_original = original.ne(ignore_index).any(dim=1)
    if not bool(has_original.all()):
        rows = (~has_original).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError('Caption has no non-padding next-token targets: rows '+str(rows))
    if truncate_prefix is None:
        truncate_prefix = bool(original[0, 0].ne(ignore_index) and original[0].eq(25).any())
    if not truncate_prefix:
        return original
    targets = original.clone()
    delimiter = targets.eq(25)
    has_delimiter = delimiter.any(dim=1)
    first = delimiter.to(torch.int64).argmax(dim=1)
    position = torch.arange(targets.shape[1], device=targets.device)[None]
    targets.masked_fill_(~has_delimiter[:, None] | (position <= first[:, None]), ignore_index)
    empty = ~targets.ne(ignore_index).any(dim=1)
    targets[empty] = original[empty]
    return targets
