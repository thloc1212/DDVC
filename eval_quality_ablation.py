"""Evaluate original/quality ranking on identical saved candidates (no GPU)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import yaml
    from eval_utils_clip import reranking, eval_metrics
    config = yaml.safe_load(Path(args.config).read_text())
    weight = float(config.get('quality_rank_weight', 0))
    if weight <= 0:
        raise ValueError('Comparison requires a trained quality head and positive quality_rank_weight')
    raw = json.loads(Path(args.raw).read_text())
    if any('quality_score' not in item for items in raw['results'].values() for item in items):
        raise ValueError('Raw predictions do not contain quality_score')
    report = {'raw': args.raw, 'config': args.config, 'same_candidates': True, 'infer_mode': True,
              'checkpoint_selection': 'best validation MetaScore using quality ranking; not two independently selected best checkpoints',
              'metrics': {}, 'prediction_files': {}}
    selected = {}
    for label, quality_weight in [('original_ranking', 0.0), ('quality_ranking', weight)]:
        filename = reranking(args.raw, alpha=config['ec_alpha'], cl_score_weight=0,
                             temperature=2.0, quality_weight=quality_weight)
        # train.py currently evaluates with infer_mode=True unconditionally.
        metrics = dict(eval_metrics(filename, config['gt_file_for_eval'], config['gt_file_for_para_eval'], infer_mode=True))
        final = json.loads(Path(filename).read_text())
        final.update(metrics)
        Path(filename).write_text(json.dumps(final))
        selected[label] = final['results']
        report['metrics'][label] = metrics
        report['prediction_files'][label] = str(Path(filename).resolve())
    report['metric_delta'] = {key: value - report['metrics']['original_ranking'][key]
                              for key, value in report['metrics']['quality_ranking'].items()
                              if isinstance(value, (int, float))}
    report['changed_videos'] = sum(
        [p['query_id'] for p in selected['original_ranking'][vid]] !=
        [p['query_id'] for p in selected['quality_ranking'][vid]]
        for vid in raw['results'])
    report['videos'] = len(raw['results'])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
