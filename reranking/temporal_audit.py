"""Post-evaluation temporal diagnostics. GT is never used to select predictions."""
import collections
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(predictions, timestamps):
    a = np.asarray([r['timestamp'] for r in predictions], dtype=float).reshape(-1, 2)
    b = np.asarray(timestamps, dtype=float).reshape(-1, 2)
    overlap = np.maximum(0, np.minimum(a[:, None, 1], b[None, :, 1])
                         - np.maximum(a[:, None, 0], b[None, :, 0]))
    union = (a[:, 1] - a[:, 0])[:, None] + (b[:, 1] - b[:, 0])[None] - overlap
    return overlap / np.maximum(union, 1e-12)


def write_audit(raw, policies, references, folder):
    """Report raw capacity separately from selection and caption quality.

    One-to-one recall uses all GT jointly. Duration/overlap strata use independent
    max-IoU coverage, so a single interval may cover more than one GT there.
    Both are auxiliary diagnostics, not replacements for official DVC metrics.
    """
    totals = collections.defaultdict(collections.Counter)
    stages = dict(raw=raw, **policies)
    for ref_id, reference in enumerate(references):
        gt = json.loads(Path(reference).read_text(encoding='utf8'))
        for vid, annotation in gt.items():
            if vid not in raw:
                raise ValueError('Missing raw candidates for reference video ' + vid)
            stamps = np.asarray(annotation['timestamps'], dtype=float).reshape(-1, 2)
            if not len(stamps):
                continue
            lengths = stamps[:, 1] - stamps[:, 0]
            duration = float(annotation['duration'])
            overlap = np.maximum(0, np.minimum(stamps[:, None, 1], stamps[None, :, 1])
                                 - np.maximum(stamps[:, None, 0], stamps[None, :, 0])) > 0
            np.fill_diagonal(overlap, False)
            groups = dict(all=np.ones(len(stamps), dtype=bool), short_le5s=lengths <= 5,
                          short_le5pct=lengths <= .05 * duration,
                          short_overlap_le10s=(lengths <= 10) & overlap.any(1),
                          long_ge50pct=lengths >= .5 * duration)
            for stage, predictions in stages.items():
                m = iou_matrix(predictions[vid], stamps)
                best = m.max(0) if len(m) else np.zeros(len(stamps))
                for threshold in (.3, .5, .7, .9):
                    one_to_one = 0
                    if m.size:
                        rows, cols = linear_sum_assignment(-((m >= threshold) * (min(m.shape) + 1) + m))
                        one_to_one = int((m[rows, cols] >= threshold).sum())
                    key = (ref_id, stage, threshold, 'all', 'one_to_one_recall')
                    totals[key].update(gt=len(stamps), matched=one_to_one)
                    for group, keep in groups.items():
                        key = (ref_id, stage, threshold, group, 'independent_gt_coverage')
                        totals[key].update(gt=int(keep.sum()), matched=int(((best >= threshold) & keep).sum()))
    path = Path(folder) / 'temporal_audit.tsv'
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['reference', 'stage', 'tiou', 'group', 'measure', 'gt_events', 'matched', 'percent']
        writer = csv.DictWriter(stream, fields, delimiter='\t')
        writer.writeheader()
        for (ref_id, stage, threshold, group, measure), total in sorted(totals.items()):
            writer.writerow(dict(reference=ref_id, stage=stage, tiou=threshold, group=group, measure=measure,
                gt_events=total['gt'], matched=total['matched'],
                percent=100 * total['matched'] / total['gt'] if total['gt'] else ''))
    print('TEMPORAL_AUDIT_READY', path, 'diagnostics only; official metrics remain in summary.tsv', flush=True)
    return path
