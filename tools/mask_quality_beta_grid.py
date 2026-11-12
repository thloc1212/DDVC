"""Evaluate saved learned-mask candidates with a frozen quality head."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]

import numpy as np
import torch
import yaml

from reranking.cache import atomic_json, read_shard, sha
from reranking.core import select
from reranking.head import JointQualityHead
from reranking.pipeline import METRICS, SCORING_PROTOCOL, complete_scores, signature, write_table


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def semantics(predictions):
    return {video: [(row['query_id'], row['timestamp'], row['sentence']) for row in rows]
            for video, rows in predictions.items()}


def metric_fingerprint(path, references):
    return signature(dict(file=sha(path), refs=references, scoring_protocol=SCORING_PROTOCOL))


def verified_scores(path, references):
    saved = path.with_suffix('.metrics.json')
    if not path.is_file() or not saved.is_file():
        return None
    cached = read_json(saved)
    if (cached.get('signature') == metric_fingerprint(path, references)
            and cached.get('scoring_protocol') == SCORING_PROTOCOL
            and complete_scores(cached.get('scores'))):
        return cached['scores']
    return None


def reuse_scores(predictions, folder, references):
    for name in ('baseline_softnms', 'reranker_softnms'):
        path = folder / (name + '.json')
        scores = verified_scores(path, references)
        if scores is not None and semantics(read_json(path)['results']) == semantics(predictions):
            print('REUSED_VERIFIED_METRICS', name, flush=True)
            return scores
    return None


def run(args):
    work = args.work.resolve()
    folder = work / 'inputs/reranker/runs' / args.dataset
    output = work / 'runs' / args.dataset / 'beta_grid'
    output.mkdir(parents=True, exist_ok=True)
    epochs = 6 if args.dataset == 'anet' else 15
    config_path = work / 'inputs/base/prepared' / f'{args.dataset}_c1_e{epochs}_b1/config.yml'
    config = yaml.safe_load(config_path.read_text(encoding='utf8'))
    manifest = read_json(folder / 'manifest.json')
    source_signature = manifest['signature']
    if source_signature != signature({k: v for k, v in manifest.items() if k != 'signature'}):
        raise ValueError('Invalid original experiment manifest signature')
    if (manifest.get('base_method') == 'C0_GLOBAL_NO_MASK' or manifest['dataset'] != args.dataset
            or not config.get('learned_proposal_mask') or config.get('proposal_mask_loss_coef') != 0
            or config.get('caption_context', 'global') != 'global'):
        raise ValueError('Expected learned proposal mask, no auxiliary mask loss, and global caption attention')
    if sha(config_path) != manifest['base_config_sha256']:
        raise ValueError('Generator config differs from the config used to train this quality head')

    def restored_path(value):
        relative = Path(value).relative_to('/marimo/DDVC_C1')
        return str(work / 'inputs/base' / relative)

    detection_references = [restored_path(path) for path in config['gt_file_for_eval']]
    paragraph_references = [restored_path(path) for path in config['gt_file_for_para_eval']]
    reference_paths = detection_references + paragraph_references
    references = [sha(path) for path in reference_paths]
    head_path = folder / 'head-best.pt'
    head_hash = sha(head_path)
    checkpoint = torch.load(head_path, map_location='cpu', weights_only=True)
    if checkpoint['signature'] != source_signature or 'beta' not in checkpoint:
        raise ValueError('Quality head is incomplete or belongs to another experiment')
    shard_paths = sorted((folder / 'cache/val').glob('shard_*.npz'))
    if not shard_paths:
        raise FileNotFoundError('Missing validation candidate cache in ' + str(folder / 'cache/val'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = JointQualityHead().to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    baseline_path = folder / 'evaluation/baseline_softnms.json'
    baseline = read_json(baseline_path)['results']
    outputs = {beta: {} for beta in args.betas}
    print('BETA_GRID_START', args.dataset, 'device', device, 'head_epoch', checkpoint['epoch'],
          'original_train_dev_beta', checkpoint['beta'], 'validation_betas', args.betas, flush=True)
    cache_hashes = []
    with torch.inference_mode():
        for index, path in enumerate(shard_paths):
            videos = read_shard(path, source_signature, 'val')
            cache_hashes.append(dict(file=path.name, sha256=sha(path), metadata_sha256=sha(path.with_suffix('.json'))))
            for video in videos:
                video_id = video['video_id']
                if video_id not in baseline or video_id in outputs[args.betas[0]]:
                    raise ValueError('Unexpected or duplicate cached video ID ' + video_id)
                rows, count = video['rows'], int(video['N'])
                if len(baseline[video_id]) != min(count, len(rows)):
                    raise ValueError('Counter budget differs from original selection for ' + video_id)
                features = torch.as_tensor(video['features'].astype(np.float32), device=device)
                quality = model(features).sigmoid().cpu().tolist()
                if not all(math.isfinite(q) and 0 <= q <= 1 for q in quality):
                    raise ValueError('Nonfinite quality prediction for ' + video_id)
                for beta in args.betas:
                    outputs[beta][video_id] = select(rows, count, quality, beta)
            if (index + 1) % 10 == 0 or index + 1 == len(shard_paths):
                print('CACHE_PROGRESS', index + 1, '/', len(shard_paths), flush=True)
    for predictions in outputs.values():
        if set(predictions) != set(baseline):
            raise ValueError('Incomplete validation video coverage')
    if 0.0 in outputs and semantics(outputs[0.0]) != semantics(baseline):
        raise ValueError('Beta zero failed to reproduce baseline Soft-NMS')

    audit = dict(dataset=args.dataset, scope='validation beta ablation',
                 original_experiment_signature=source_signature, head_sha256=head_hash,
                 base_config_sha256=sha(config_path), references=references,
                 baseline_predictions_sha256=sha(baseline_path), cache=cache_hashes,
                 original_train_dev_beta=float(checkpoint['beta']), device=device,
                 betas=args.betas, scoring_protocol=SCORING_PROTOCOL,
                 method='LEARNED_MASK_FROZEN_QUALITY_SOFTNMS',
                 runner_release='learned-mask-quality-beta-grid-20261112',
                 runner_sha256=sha(Path(__file__)))
    atomic_json(output / 'manifest.json', audit)
    summary = []
    os.chdir(ROOT)
    for beta, predictions in outputs.items():
        label = format(beta, 'g').replace('.', 'p')
        path = output / ('softnms_beta_' + label + '.json')
        atomic_json(path, dict(results=predictions, version='VERSION 1.0', external_data={},
                               selection=dict(method='quality_softnms', beta=beta, sigma=0.5)))
        scores = verified_scores(path, references)
        if scores is None:
            scores = reuse_scores(predictions, folder / 'evaluation', references)
        if scores is None and not args.selection_only:
            from eval_utils_clip import eval_metrics
            print('OFFICIAL_SCORING_START', 'beta', beta, flush=True)
            with (output / 'scorer_detail.log').open('a', encoding='utf8') as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    scores = {key: float(value) * 100 for key, value in eval_metrics(
                        str(path), detection_references, paragraph_references,
                        dvc_eval_version='2018', infer_mode=True).items()}
            if not complete_scores(scores):
                raise RuntimeError('Incomplete official scores; inspect ' + str(output / 'scorer_detail.log'))
        row = dict(beta=beta, videos=len(predictions), events=sum(map(len, predictions.values())),
                   status='SCORED' if scores is not None else 'SELECTION_READY')
        if scores is not None:
            atomic_json(path.with_suffix('.metrics.json'), dict(
                signature=metric_fingerprint(path, references), scoring_protocol=SCORING_PROTOCOL, scores=scores))
            row.update({key: scores[key] for key in METRICS})
        summary.append(row)
        write_table(output / 'summary.tsv', summary)
        print('BETA_RESULT', json.dumps(row), flush=True)
        if args.backup and scores is not None:
            from mask_quality_beta_restore import backup
            backup(work, args.dataset)
    if sha(head_path) != head_hash or sha(config_path) != audit['base_config_sha256']:
        raise RuntimeError('Head or config changed during evaluation')
    print('BETA_GRID_COMPLETE', output / 'summary.tsv', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=ROOT.parent)
    parser.add_argument('--dataset', choices=('anet', 'yc2'), required=True)
    parser.add_argument('--betas', type=float, nargs='+', default=[0., .25, .5, 1., 2., 4., 6., 8.])
    parser.add_argument('--selection-only', action='store_true', help='Prepare selections and reuse verified metrics only')
    parser.add_argument('--backup', action='store_true', help='Copy evaluation results to Drive after every scored beta')
    args = parser.parse_args()
    if any(not math.isfinite(beta) or beta < 0 for beta in args.betas):
        parser.error('Betas must be finite and nonnegative')
    args.betas = list(dict.fromkeys(args.betas))
    folder = args.work / 'runs' / args.dataset / 'beta_grid'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'worker.lock').open('a') as lock:
        if os.name == 'posix':
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error('A beta-grid worker is already running for this dataset')
        run(args)


if __name__ == '__main__':
    main()
