"""Historical C1, with no QTS, event-set, concepts or mask auxiliary objective."""
from misc.c2_recipe import clean_legacy_metadata, RETIRED_ENABLED

BASE_COMMIT = 'ed5494c9dce7371ea30683a3e8a5320ccc19b2a7'
METHOD = dict(learned_proposal_mask=True, proposal_mask_loss_coef=0.0,
              proposal_mask_bins=64, proposal_mask_encoding_dim=128,
              add_cross_attention=True, eval_selection_method='softnms', eval_softnms_sigma=.5)


def validate_method(config):
    for key, expected in METHOD.items():
        if config.get(key) != expected:
            raise ValueError(f'C1 requires {key}={expected!r}, got {config.get(key)!r}')
    for key in (*RETIRED_ENABLED, 'query_temporal_support', 'relation_aware_event_set'):
        if config.get(key, False):
            raise ValueError('C1 does not enable '+key)
    if config.get('dedup_loss_coef', 0.) != 0.:
        raise ValueError('C1 has no deduplication loss')
    if config.get('mask_prune_threshold', -1.) != -1.:
        raise ValueError('C1 does not prune visual frames')
    if config.get('mask_target', 'predicted') != 'predicted':
        raise ValueError('C1 keeps the predicted-window mask target')
    return config


def validate_training_method(config):
    if config.get('experiment_recipe') == 'stable_global_baseline':
        from misc.baseline_recipe import validate_method as validate_baseline
        return validate_baseline(config)
    if config.get('experiment_recipe') == 'stable_c1':
        return validate_method(config)
    from misc.c2_recipe import validate_method as validate_c2
    return validate_c2(config)
