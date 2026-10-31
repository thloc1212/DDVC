"""Prepare fresh V1 mask ablations; evaluate postprocessing on identical raw queries."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding='utf8'))


def prepare(args):
    import yaml
    source = args.source_config.resolve()
    cfg = load_config(source)
    if cfg.get('event_concepts'):
        raise ValueError('Use a V1 prepared config, not the concept experiment.')
    if float(cfg.get('dedup_loss_coef', 0)) != 0:
        raise ValueError('Expected V1 with dedup loss OFF.')
    if not cfg.get('learned_proposal_mask'):
        raise ValueError('Source must be the original learned-mask V1 config.')
    for key in ('train_caption_file', 'val_caption_file', 'dict_file'):
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(cfg[key])
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg.update(id=args.run_name, exp_name=args.run_name, fixed_run_id=True,
               cfg_path='', start_from='', pretrain=None,
               only_ft_captioner=False, only_ft_class_head=False,
               save_dir=str(args.output_root.resolve()), epoch=args.epochs,
               learned_proposal_mask=args.mask in ('learned', 'learned_no_aux'),
               local_visual_attention=args.mask == 'hard',
               proposal_mask_loss_coef=1.0 if args.mask == 'learned' else 0.0,
               dedup_loss_coef=0.0, eval_selection_method='topk',
               eval_merge_duplicates=False, eval_merge_consecutive=False,
               min_epoch_when_save=0, save_checkpoint_every=1,
               experiment_recipe='v1_mask_ablation', selection_experiment='baseline')
    # Inherit the original data IDs, LR, seed, caption decoder, retrieval and precision.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf8')
    manifest = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    mask=args.mask, run_name=args.run_name,
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print('ABLATION_CONFIG_READY', args.output)


def postprocess(args):
    from misc.proposal_selection import select_proposals
    raw_bytes = args.raw.read_bytes()
    raw = json.loads(raw_bytes)
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    cfg = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Complete 2x2x2 factorial: independent ranking / overlap merge / consecutive merge.
    for method, merge, consecutive in itertools.product(('topk', 'softnms'), (False, True), (False, True)):
        name = f'{method}_overlap{int(merge)}_consecutive{int(consecutive)}'
        path = args.output_dir / (name + '.json')
        if path.exists():
            raise FileExistsError(f'Preserve existing output; choose a new directory: {path}')
        results = {}
        for vid, rows in raw['results'].items():
            if rows and 'pred_event_count' not in rows[0]:
                raise ValueError('Expected raw candidates with pred_event_count, not selected predictions.')
            results[vid] = select_proposals(
                rows, int(rows[0]['pred_event_count']) if rows else 0,
                method, cfg.get('eval_softnms_sigma', .5), cfg.get('ec_alpha', 1.), 2.,
                merge, cfg.get('eval_merge_iou', .7), cfg.get('eval_merge_text_similarity', .9),
                consecutive, cfg.get('eval_consecutive_gap', 2.),
                cfg.get('eval_consecutive_max_span', 60.), cfg.get('eval_consecutive_max_events', 3))
        out = {k: raw[k] for k in ('version', 'external_data') if k in raw}
        out.update(results=results, selection=dict(method=method, merge=merge, merge_consecutive=consecutive),
                   provenance=dict(raw=str(args.raw.resolve()), raw_sha256=raw_sha))
        path.write_text(json.dumps(out), encoding='utf8')
        print('SELECTED', path, flush=True)


def score(args):
    from eval_utils_clip import eval_metrics
    cfg = load_config(args.config)
    rows = []
    for path in sorted(args.input_dir.glob('*.json')):
        if path.name.endswith('.metrics.json'):
            continue
        payload = json.loads(path.read_text(encoding='utf8'))
        if 'results' not in payload:
            continue
        scores = dict(eval_metrics(str(path), cfg['gt_file_for_eval'], cfg['gt_file_for_para_eval'],
                                   dvc_eval_version=cfg.get('eval_tool_version', '2018')))
        path.with_suffix('.metrics.json').write_text(json.dumps(scores, indent=2), encoding='utf8')
        # The official evaluator returns fractions; table uses the conventional x100 scale.
        rows.append(dict(selection=path.stem, videos=len(payload['results']),
                         events=sum(map(len, payload['results'].values())),
                         **{k: 100 * float(scores[k]) for k in
                            ('CIDEr', 'METEOR', 'Bleu_4', 'soda_c', 'Precision', 'Recall', 'MetaScore')}))
    if not rows:
        raise FileNotFoundError('No selected prediction JSON files found.')
    with (args.input_dir / 'summary.tsv').open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    print('SCORES_READY', args.input_dir / 'summary.tsv', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--source-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--mask', choices=('global', 'hard', 'learned', 'learned_no_aux'), required=True)
    p.add_argument('--epochs', type=int, default=15)
    p = sub.add_parser('postprocess')
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p = sub.add_parser('score')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--input-dir', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    {'prepare': prepare, 'postprocess': postprocess, 'score': score}[args.command](args)


if __name__ == '__main__':
    main()
