"""Explicit recipe for C1 + query-conditioned temporal support experiments."""
import math

BASE_COMMIT='4ce76e9'
VARIANTS={'baseline':(False,0.,0.),'refine':(True,0.,1.),'support':(True,1.,1.),'set':(True,1.,1.)}


def dataset_settings(dataset):
    if dataset=='yc2':return dict(epoch=15,num_queries=50,max_eseq_length=20,gt_proposal_sample_num=20)
    if dataset=='anet':return dict(epoch=6,num_queries=30,max_eseq_length=30,gt_proposal_sample_num=30)
    raise ValueError('Unknown QTS dataset')


def validate_qts(config):
    from misc.c2_recipe import RETIRED_ENABLED
    variant=config.get('qts_variant','support')
    if variant not in VARIANTS:raise ValueError('Unknown QTS variant')
    enabled,support,refine=VARIANTS[variant]
    expected=dict(learned_proposal_mask=True,proposal_mask_loss_coef=0.,add_cross_attention=True,
        query_temporal_support=enabled,temporal_support_loss_coef=support,temporal_refine_loss_coef=refine,
        temporal_support_dim=128,temporal_support_max_delta=2.,full_event_supervision=True,
        count_loss_coef=1.,counter_prior_source='train_full',
        eval_selection_method='event_set' if variant=='set' else 'softnms',eval_softnms_sigma=.5)
    if variant=='set':
        from misc.event_set_selection import DEFAULTS
        expected.update(relation_aware_event_set=True,event_set_dim=128,event_set_quality_coef=.5,
                        event_set_relation_coef=.2,event_set_count_ordinal_coef=.2,
                        event_set_input_grad=False,event_set_selection_settings=DEFAULTS)
    elif config.get('relation_aware_event_set', False):
        raise ValueError('Relation/quality heads belong to the set variant only')
    for k,v in expected.items():
        if config.get(k)!=v:raise ValueError(f'QTS requires {k}={v!r}, got {config.get(k)!r}')
    if any(config.get(k,False) for k in RETIRED_ENABLED) or config.get('dedup_loss_coef',0)!=0:
        raise ValueError('Retired method enabled in QTS')
    rates=config.get('counter_class_rates')
    if not isinstance(rates,list) or len(rates)!=config['max_eseq_length']+1:
        raise ValueError('Counter prior length must equal count logits')
    if not all(math.isfinite(x) and 0<=x<=1 for x in rates) or abs(sum(rates)-1)>1e-6:
        raise ValueError('Invalid train-only count prior')
    return config


def validate_training_recipe(config):
    if config.get('experiment_recipe')=='query_temporal_support':return validate_qts(config)
    from misc.c2_recipe import validate_method
    return validate_method(config)
