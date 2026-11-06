"""Method invariants for the finalized C2 model, without changing its recipe."""
BASE_COMMIT = 'ed5494c9dce7371ea30683a3e8a5320ccc19b2a7'
METHOD = dict(learned_proposal_mask=True, proposal_mask_loss_coef=1.0,
              proposal_mask_bins=64, proposal_mask_encoding_dim=128,
              add_cross_attention=True, eval_selection_method='softnms', eval_softnms_sigma=.5)
RETIRED_ENABLED = ('event_concepts', 'concept_condition_prefix', 'local_visual_attention',
                   'bi_interaction', 'semantic_matching', 'narrative_memory',
                   'attention_counter', 'duplicate_head', 'eval_merge_duplicates',
                   'eval_merge_consecutive', 'mask_query_conditioning', 'mask_balanced_bce')


def validate_method(config):
    for key, expected in METHOD.items():
        if config.get(key) != expected:
            raise ValueError(f'C2 requires {key}={expected!r}, got {config.get(key)!r}')
    for key in RETIRED_ENABLED:
        if config.get(key, False):
            raise ValueError('This method was removed from the C2 branch: '+key)
    if config.get('dedup_loss_coef', 0.) != 0.:
        raise ValueError('C2 has no deduplication loss')
    if config.get('mask_target', 'predicted') != 'predicted':
        raise ValueError('C2 mask target is the detached predicted hard window')
    if config.get('mask_prune_threshold', -1.) != -1.:
        raise ValueError('C2 does not prune visual frames')
    return config


def clean_legacy_metadata(config):
    """Drop retired option labels only after checking their methods were disabled."""
    result = dict(config)
    for key in RETIRED_ENABLED:
        if result.get(key, False):
            raise ValueError('Cannot migrate an enabled retired method: '+key)
    if result.get('dedup_loss_coef', 0.) != 0.:
        raise ValueError('Cannot migrate dedup-enabled training')
    for key in list(result):
        if (key.startswith(('concept_', 'dedup_', 'eval_merge_', 'eval_consecutive_', 'local_visual_'))
                or key in RETIRED_ENABLED):
            result.pop(key)
    return result
