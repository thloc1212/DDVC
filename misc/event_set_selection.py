"""GT-free relation-aware selection over all cardinalities, using a soft count prior."""
import copy
import math

DEFAULTS = dict(quality_reference=.3, proposal_weight=.15, caption_weight=.05,
                duplicate_weight=2., count_weight=.2, count_temperature=2., count_uniform_mix=.1)


def select_event_set(rows, duplicate_probs, count_logits, settings=None):
    cfg = dict(DEFAULTS)
    if settings:
        if set(settings) - set(cfg):
            raise ValueError('Unknown event-set selection setting')
        cfg.update(settings)
    if not all(math.isfinite(v) for v in cfg.values()):
        raise ValueError('Nonfinite event-set setting')
    if not (0 < cfg['quality_reference'] < 1 and 0 <= cfg['count_uniform_mix'] <= 1
            and cfg['count_temperature'] > 0 and all(cfg[k] >= 0 for k in
            ('proposal_weight', 'caption_weight', 'duplicate_weight', 'count_weight'))):
        raise ValueError('Invalid event-set setting')
    if len(count_logits) < 2 or not all(math.isfinite(x) for x in count_logits):
        raise ValueError('Invalid counter logits')
    counter_argmax = max(range(len(count_logits)), key=lambda k: count_logits[k])
    if not rows:
        return [], dict(selected_N=0, predicted_N=max(1,counter_argmax), counter_argmax=counter_argmax)
    size = len(duplicate_probs)
    if size == 0 or any(len(r) != size for r in duplicate_probs):
        raise ValueError('Duplicate probabilities must be a square query-ID matrix')
    ids = [int(row['query_id']) for row in rows]
    if len(set(ids)) != len(ids) or any(q < 0 or q >= size for q in ids):
        raise ValueError('Invalid or repeated candidate query IDs')
    if any(not math.isfinite(v) or not 0 <= v <= 1 for r in duplicate_probs for v in r):
        raise ValueError('Invalid duplicate probability')
    for row in rows:
        s, e = row['timestamp']
        values = (s, e, row['event_quality'], row['proposal_score'], row['sentence_score'])
        if not all(math.isfinite(float(v)) for v in values) or e < s or not 0 <= row['event_quality'] <= 1:
            raise ValueError('Nonfinite/invalid candidate')
    # Keep count uncertainty; do not condition selection on argmax N.
    z = [float(x) / cfg['count_temperature'] for x in count_logits]
    maximum = max(z)
    norm = sum(math.exp(x - maximum) for x in z)
    probabilities = [(1 - cfg['count_uniform_mix']) * math.exp(x - maximum) / norm
                     + cfg['count_uniform_mix'] / len(z) for x in z]
    utilities = []
    for row in rows:
        quality = max(1e-6, min(1., float(row['event_quality'])))
        proposal = max(1e-6, min(1., float(row['proposal_score'])))
        # GPT-2 supplies beam sequences_scores, already length-penalized by
        # generation. Do not divide again by decoded word count.
        caption = min(0., float(row['sentence_score']))
        utilities.append(math.log(quality / cfg['quality_reference'])
                         + cfg['proposal_weight'] * math.log(proposal) + cfg['caption_weight'] * caption)
    pending, chosen = set(range(len(rows))), []
    best, best_value, base_score = [], -float('inf'), 0.
    curve = []
    for count in range(1, min(len(rows), len(count_logits) - 1) + 1):
        def gain(i):
            # Symmetric duplicate belief only. Containment itself has no penalty.
            penalty = sum((duplicate_probs[ids[i]][ids[j]] + duplicate_probs[ids[j]][ids[i]]) / 2 for j in chosen)
            return utilities[i] - cfg['duplicate_weight'] * penalty
        index = max(pending, key=lambda i: (gain(i), -ids[i]))
        increment = gain(index)
        base_score += increment
        chosen.append(index)
        pending.remove(index)
        value = base_score + cfg['count_weight'] * math.log(max(probabilities[count], 1e-12))
        curve.append(dict(count=count, score=value))
        if value > best_value:
            best, best_value = chosen[:], value
    final = []
    for index in best:
        row = copy.deepcopy(rows[index])
        row['event_set_utility'] = utilities[index]
        row['selected_event_count'] = len(best)
        final.append(row)
    final.sort(key=lambda r: (r['timestamp'], r['query_id']))
    return final, dict(selected_N=len(best), predicted_N=max(1,counter_argmax), counter_argmax=counter_argmax,
                       objective=best_value, count_probabilities=probabilities, cardinality_curve=curve,
                       settings=cfg, algorithm='greedy-prefix-cardinality-search')
