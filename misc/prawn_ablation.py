"""One-factor ablation profiles. Caption-aware matching is never disabled."""
import os

VARIANTS = ('full', 'no_content', 'joint')


def variant():
    value = os.environ.get('PRAWN_ABLATION_VARIANT', 'full')
    if value not in VARIANTS:
        raise ValueError('Unknown PRAWN ablation variant: ' + value)
    return value


def run_name(dataset):
    v = variant()
    name = 'caption_match' if v == 'full' else 'loo_' + v
    return f'{dataset}_{name}_e{15 if dataset == "yc2" else 6}_b1'


def namespace(stage):
    v = variant()
    return ('prawn_caption_match_' if v == 'full' else 'prawn_loo_' + v + '_') + stage


def configure(cfg):
    v = variant()
    if v == 'full':
        return cfg  # Preserve historical Full configuration/signatures exactly.
    cfg.update(prawn_ablation_variant=v, proposal_mask_content_conditioned=v != 'no_content',
               online_quality=v == 'joint')
    if v == 'joint':
        cfg.update(quality_gradient_mode='joint', quality_loss_coef=.1, quality_head_lr=.001,
                   quality_generation_chunk=4, quality_annotation_file=cfg['train_caption_file'])
    return cfg


def validate(cfg):
    v = cfg.get('prawn_ablation_variant', 'full')
    if v not in VARIANTS:
        raise ValueError('Unknown saved ablation profile')
    if cfg.get('proposal_mask_content_conditioned') is not (v != 'no_content'):
        raise ValueError('Content flag differs from ablation profile')
    if cfg.get('proposal_mask_loss_coef') != 0.:
        raise ValueError('Auxiliary mask loss must stay OFF')
    if not cfg.get('caption_match_enabled') or cfg.get('caption_match_weight') != .5:
        raise ValueError('All ablations require caption-aware matching, weight .5')
    if v == 'joint':
        expected = dict(online_quality=True, quality_gradient_mode='joint', quality_loss_coef=.1,
                        quality_head_lr=.001, quality_generation_chunk=4)
        for key, value in expected.items():
            if cfg.get(key) != value:
                raise ValueError('Joint profile changed: ' + key)
    elif cfg.get('online_quality', False):
        raise ValueError('Only Joint may enable online quality')
    return cfg
