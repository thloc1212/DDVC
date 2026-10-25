"""Reject training-recipe drift from the pinned Kaggle learned-mask notebook."""
import argparse
import json
from pathlib import Path


def validate(config, mode, check_data=True):
    expected = json.loads(Path(__file__).with_name('kaggle_mask_recipe.json').read_text())
    mismatches = {key: (config.get(key), value) for key, value in expected.items()
                  if config.get(key) != value}
    if mismatches:
        raise ValueError('Kaggle recipe mismatch (actual, expected): ' + repr(mismatches))
    if config.get('experiment_recipe') != 'kaggle_cvpr2018_mask':
        raise ValueError('Use a fresh Kaggle-recipe config/run, not an old 25-epoch resume')
    experiment=config.get('selection_experiment','baseline')
    allowed={'baseline':(0.,'topk',False),'softnms':(0.,'softnms',False),
             'dedup_softnms':(.1,'softnms',False),'dedup_softnms_merge':(.1,'softnms',True),
             'dedup_softnms_merge_consecutive':(.1,'softnms',True)}
    if experiment not in allowed:raise ValueError('Unknown selection experiment')
    for key,value in zip(['dedup_loss_coef','eval_selection_method','eval_merge_duplicates'],allowed[experiment]):
        if config.get(key,{'dedup_loss_coef':0.,'eval_selection_method':'topk','eval_merge_duplicates':False}[key])!=value:
            raise ValueError('Selection experiment flag mismatch: '+key)
    for key,value in dict(dedup_start_epoch=1,dedup_iou_threshold=.5,dedup_confidence_threshold=.2,
                          eval_softnms_sigma=.5,eval_merge_iou=.7,eval_merge_text_similarity=.9,eval_consecutive_gap=2.,
                          eval_consecutive_max_span=60.,eval_consecutive_max_events=3).items():
        if config.get(key,value)!=value:raise ValueError('Selection recipe mismatch: '+key)
    if config.get('eval_merge_consecutive',False)!=experiment.endswith('_consecutive'):
        raise ValueError('Consecutive merge flag disagrees with experiment')
    milestones = [config['learning_rate_decay_start'] + i * config['learning_rate_decay_every']
                  for i in range(int((config['epoch'] - config['learning_rate_decay_start']) /
                                     config['learning_rate_decay_every']))]
    assert milestones == [8, 11], milestones
    if check_data:
        counts = tuple(len(json.loads(Path(config[key]).read_text()))
                       for key in ('train_caption_file', 'val_caption_file'))
        wanted = {'subset': (371, 131), 'full': (1237, 436)}[mode]
        if counts != wanted:
            raise ValueError(f'{mode} data counts {counts} != {wanted}')
    return milestones


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['subset', 'full'], required=True)
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(Path(args.config).read_text())
    milestones = validate(config, args.mode)
    print('KAGGLE_RECIPE_OK', args.mode, '15 epochs; batch/accumulation 1; LR 5e-5; milestones', milestones)
