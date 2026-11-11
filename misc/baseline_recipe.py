"""Controlled DDVC global baseline; keep the C1 data/training recipe fixed."""
from misc.c1_recipe import METHOD as C1_METHOD, validate_method as validate_c1

METHOD = dict(C1_METHOD, learned_proposal_mask=False)


def validate_method(config):
    if config.get('learned_proposal_mask') is not False:
        raise ValueError('Global baseline requires learned_proposal_mask=False')
    # Validate all other disabled methods and selection settings with the C1 guard.
    validate_c1(dict(config, learned_proposal_mask=True))
    if config.get('online_quality', False):
        raise ValueError('Frozen quality must not be trained inside the baseline')
    return config
