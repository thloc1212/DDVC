"""Evaluate a frozen global DDVC baseline on the other dataset's validation split."""

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

import c1_rclone as drive_tools
from eval_utils_clip import eval_metrics
from misc.baseline_recipe import validate_method
from misc.proposal_selection import select_proposals
from misc.runtime_perf import configure_runtime
from misc.utils import set_seed
from opts import parse_opts
from reranking.cache import atomic_json, sha
from reranking.core import feature_pool, select
from reranking.head import JointQualityHead
from reranking.pipeline import METRICS, write_table
from ret_utils import load_clip_memory_bank
from train import build


WORK = drive_tools.WORK
DATASETS = ('yc2', 'anet')
NAMESPACES = {'generator': 'ddvc_baseline', 'quality': 'ddvc_baseline_rerank'}
BASELINE_REVISION = 'b54f081b648b89a7887fb5df651985ef4a0cf9ce'
MODEL_FILES = ('ddvc', 'opts.py', 'train.py', 'ret_utils.py', 'reranking/core.py', 'reranking/head.py',
               'misc/proposal_selection.py', 'data/video_dataset.py')


def compatible_revision(commit):
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        return False
    common = subprocess.run(['git', 'merge-base', '--is-ancestor', BASELINE_REVISION, commit], check=False)
    if common.returncode:
        return False
    diff = subprocess.run(['git', 'diff', '--quiet', commit, 'HEAD', '--', *MODEL_FILES], check=False)
    return diff.returncode == 0


def completed_snapshots(drive, dataset, stage):
    group = drive.path(f'runs/{NAMESPACES[stage]}/{dataset}')
    run_ids = drive.call('lsf', group, '--dirs-only', '--max-depth', '1', quiet=True).splitlines()
    for run_id in sorted((value.rstrip('/') for value in run_ids), reverse=True):
        if not re.fullmatch(r'\d{8}T\d{6}Z_[0-9a-f]{12}', run_id):
            continue
        parent = f'{group}/{run_id}'
        state = drive.optional_json(f'{parent}/RUN_MANIFEST.json')
        if not state or state.get('status') != 'COMPLETE_BACKED_UP':
            continue
        recovery = f'{parent}/recovery'
        pointer = drive.read_json(f'{recovery}/LATEST.json')
        slot = pointer['slot']
        if slot not in (0, 1):
            raise ValueError(f'Invalid recovery slot for {run_id}')
        remote = f'{recovery}/slot-{slot}'
        meta = drive.read_json(f'{remote}/READY.json')
        if (meta['run_id'], meta['dataset'], meta['slot'], meta['created_ns']) != (
                run_id, dataset, slot, pointer['created_ns']):
            raise ValueError(f'Recovery pointer changed for {run_id}. Retry when backup completes.')
        if not compatible_revision(meta['source_commit']):
            print('SKIP_INCOMPATIBLE', dataset, stage, run_id, meta['source_commit'], flush=True)
            continue
        if stage == 'generator' and meta.get('variant') != 'baseline_quality':
            continue
        yield remote, meta


def best_path(dataset):
    return f'runs/{drive_tools.run_name(dataset)}/model-best.pth'


def matched_runs(drive, dataset):
    generators = list(completed_snapshots(drive, dataset, 'generator'))
    by_sha = {}
    for snapshot in generators:
        item = snapshot[1]['files'].get(best_path(dataset))
        if item is not None:
            by_sha.setdefault(item['sha256'], snapshot)
    prefix = f'runs/{dataset}'
    for snapshot in completed_snapshots(drive, dataset, 'quality'):
        meta = snapshot[1]
        needed = [f'{prefix}/manifest.json', f'{prefix}/head-best.pt']
        if not all(name in meta['files'] for name in needed):
            continue
        path = WORK / 'inputs' / 'quality' / meta['run_id'] / str(meta['created_ns'])
        fetch_files(drive, snapshot, path, needed[:1])
        manifest = json.loads((path / needed[0]).read_text(encoding='utf-8'))
        if manifest.get('dataset') != dataset or manifest.get('base_method') != 'C0_GLOBAL_NO_MASK':
            continue
        generator = by_sha.get(manifest['base_checkpoint_sha256'])
        if generator is not None:
            print('MATCHED_RUNS', dataset, generator[1]['run_id'], meta['run_id'], flush=True)
            return generator, snapshot, path, manifest
    raise FileNotFoundError(f'No completed baseline generator/quality pair with matching checkpoint SHA256 for {dataset}')


def fetch_files(drive, snapshot, cache, names):
    remote, meta = snapshot
    missing = set(names) - meta['files'].keys()
    if missing:
        raise FileNotFoundError(f'{remote} is missing {sorted(missing)}')
    pending = {}
    for name in names:
        drive_tools.relative_path(name)
        expected = meta['files'][name]
        path = cache / name
        if (not path.is_file() or path.stat().st_size != expected['bytes']
                or sha(path) != expected['sha256']):
            pending[name] = expected
    if pending:
        cache.mkdir(parents=True, exist_ok=True)
        drive_tools.fetch_verified(drive, remote, {'files': pending}, cache)


def generator_assets(drive, snapshot, dataset, load_weights):
    run = drive_tools.run_name(dataset)
    prepared = f'prepared/{run}'
    config_path = f'{prepared}/config.yml'
    checkpoint_path = best_path(dataset)
    required = [config_path]
    meta = snapshot[1]
    if load_weights:
        required.append(checkpoint_path)
        required.extend(name for name in meta['files'] if name.startswith(f'{prepared}/bank/{dataset}/')
                        and name.endswith('.npy'))
    else:
        required.extend(name for name in meta['files'] if name.startswith(f'{prepared}/data/{dataset}/captiondata/')
                        and name.endswith('.json') and not name.endswith('/train.json'))
    cache = WORK / 'inputs' / 'generator' / meta['run_id'] / str(meta['created_ns'])
    fetch_files(drive, snapshot, cache, required)
    cfg = yaml.safe_load((cache / config_path).read_text(encoding='utf-8'))
    validate_method(cfg)
    if cfg['dataset'] != dataset or cfg['bank_type'] != [dataset]:
        raise ValueError('Generator dataset or caption bank differs from the run ID')

    def local(path):
        relative = Path(path).relative_to(meta['work']).as_posix()
        drive_tools.relative_path(relative)
        result = cache / relative
        if not result.exists():
            raise FileNotFoundError(result)
        return str(result)

    if load_weights:
        cfg['bank_path'] = local(cfg['bank_path'])
        return cfg, cache / config_path, cache / checkpoint_path, meta['files'][checkpoint_path]['sha256']
    cfg['val_caption_file'] = local(cfg['val_caption_file'])
    for key in ('gt_file_for_eval', 'gt_file_for_para_eval'):
        cfg[key] = [local(path) for path in cfg[key]]
    return cfg


def load_target_features(drive, dataset, data_signature):
    root = drive.path(f'data/{dataset}')
    folders = drive.call('lsf', root, '--dirs-only', '--max-depth', '1', quiet=True).splitlines()
    for protocol in (drive_tools.DATA_PROTOCOL, 'c1-base-assets-v1', 'qts-base-assets-v1'):
        if f'{protocol}/' not in folders:
            continue
        parent = f'{root}/{protocol}'
        pointer = drive.optional_json(f'{parent}/LATEST.json')
        if pointer is None:
            continue
        generation = pointer['generation']
        if not re.fullmatch(r'[0-9a-f]{32}', generation):
            raise ValueError(f'{protocol}: invalid data generation')
        remote = f'{parent}/generations/{generation}'
        meta = drive.read_json(f'{remote}/READY.json')
        if meta['protocol'] != protocol or meta['dataset'] != dataset:
            raise ValueError(f'{protocol}: unexpected data manifest')
        cache = WORK / 'inputs' / 'features' / dataset / generation
        fetch_files(drive, (remote, meta), cache, ['config.yml'])
        cfg = yaml.safe_load((cache / 'config.yml').read_text(encoding='utf-8'))
        if cfg['data_signature'] != data_signature:
            continue
        feature_name = Path(cfg['visual_feature_folder'][0]).name
        feature, = [name for name in meta['files'] if Path(name).name == feature_name]
        fetch_files(drive, (remote, meta), cache, [feature])
        return torch.load(cache / feature, map_location='cpu', mmap=True, weights_only=False)
    raise FileNotFoundError(f'No verified {dataset} CLIP bundle matches the target validation recipe')


def infer_video(model, post, bank, opt, video_id, features, duration, tokenizer):
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != opt.feature_dim:
        raise ValueError(f'{video_id}: unexpected CLIP shape {tuple(features.shape)}')
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'{video_id}: invalid duration {duration}')
    video = features.to(device=opt.device, dtype=torch.float32)
    length = video.shape[0]
    batch = {
        'video_tensor': video.unsqueeze(0),
        'video_mask': torch.ones((1, length), dtype=torch.bool, device=video.device),
        'video_length': video.new_tensor([[length, duration, 0]]),
        'video_key': [video_id],
        'cap_raw': [[]],
        'gt_boxes_mask': torch.zeros((1, 0), dtype=torch.bool, device=video.device),
        'video_target': [{}],
    }
    # The eval forward calls the criterion after proposal and caption generation.
    prediction, _ = model(batch, lambda *_: ({}, None, None), None,
                          opt.transformer_input_type, bank, eval_mode=True)
    loader = SimpleNamespace(dataset=SimpleNamespace(tokenizer=tokenizer))
    output = post['bbox'](prediction, batch['video_length'][:, 1], loader, model, None)[0]
    count = int(output['pred_seq_len'])
    rows = []
    for i, (start, end) in enumerate(output['boxes'].cpu().tolist()):
        if end <= start or float(output['scores'][i]) <= 0:
            continue
        rows.append(dict(timestamp=[start, end], query_id=int(output['query_id'][i]),
                         proposal_score=float(output['scores'][i]), sentence=output['captions'][i],
                         sentence_score=float(output['caption_scores'][i]),
                         vid_duration=duration, pred_event_count=count))
    return rows, count, video


def main(source, target):
    drive = drive_tools.Drive('/marimo/secrets/rclone.conf', 'kltnbackup', 'KLTNRUNS')
    source_run, head_run, head_cache, head_manifest = matched_runs(drive, source)
    target_run = next(completed_snapshots(drive, target, 'generator'), None)
    if target_run is None:
        raise FileNotFoundError(f'No completed {target} baseline generator backup for the evaluation split')
    source_cfg, source_path, generator_path, generator_sha = generator_assets(drive, source_run, source, True)
    target_cfg = generator_assets(drive, target_run, target, False)
    head_prefix = f'runs/{source}'
    head_name = f'{head_prefix}/head-best.pt'
    fetch_files(drive, head_run, head_cache, [head_name])
    head_checkpoint = torch.load(head_cache / head_name, map_location='cpu', weights_only=True)
    if (head_manifest['base_checkpoint_sha256'] != generator_sha
            or head_manifest['base_config_sha256'] != sha(source_path)
            or head_checkpoint['signature'] != head_manifest['signature']):
        raise ValueError('The quality head and generator do not belong to the same trained baseline run')
    beta = float(head_checkpoint['beta'])
    if not math.isfinite(beta) or beta < 0:
        raise ValueError(f'Unsupported saved beta: {beta}')

    opt = parse_opts(['--cfg_path', str(source_path)], write_options=False)
    opt.bank_path = source_cfg['bank_path']
    if not opt.text_crossAttn:
        opt.cap_num_feature_levels = 4
    if opt.combined_encoder:
        opt.cap_num_feature_levels = 5
    if opt.caption_decoder_new == 'gpt2':
        opt.caption_decoder_type = 'gpt2'
    if opt.transformer_input_type != 'queries' or opt.ideal_test or opt.retrieval_ablation == 'ideal':
        raise ValueError('The source configuration must support inference using only video and the source caption bank')
    opt.rec_encdec = True
    configure_runtime(opt)
    set_seed(opt.seed)
    generator_checkpoint = torch.load(generator_path, map_location='cpu', weights_only=False)
    if (generator_checkpoint['config']['num_queries'] != opt.num_queries
            or generator_checkpoint['epoch'] != head_manifest['base_epoch']):
        raise ValueError('Source generator config or epoch mismatch')
    model, _, _, post = build(opt)
    model.load_state_dict(generator_checkpoint['model'], strict=True)
    del generator_checkpoint
    model.to(opt.device).eval()
    quality_head = JointQualityHead().to(opt.device).eval()
    quality_head.load_state_dict(head_checkpoint['model'], strict=True)
    bank = load_clip_memory_bank(opt)
    bank['vid_sent_embeds'] = torch.as_tensor(bank['vid_sent_embeds'], device=opt.device)
    bank['clip_gt'] = {}
    caption_head = model.caption_head[-1]
    tokenizer = caption_head.tokenizer
    wte = caption_head.model.transformer.wte

    # Read only IDs and video duration for prediction. GT references enter at scoring.
    durations = {key: float(row['duration']) for key, row in
                 json.loads(Path(target_cfg['val_caption_file']).read_text(encoding='utf-8')).items()}
    references = target_cfg['gt_file_for_eval']
    reference_ids = set().union(*(json.loads(Path(path).read_text(encoding='utf-8')).keys()
                                  for path in references))
    if len(durations) != {'yc2': 436, 'anet': 4917}[target] or durations.keys() != reference_ids:
        raise ValueError(f'{target}: validation videos do not match the complete evaluation references')
    features = load_target_features(drive, target, target_cfg['data_signature'])
    missing = durations.keys() - features.keys()
    if missing:
        raise ValueError(f'{len(missing)} target validation videos have no CLIP features: {sorted(missing)[:5]}')

    out = WORK / 'results' / f'{source}_to_{target}'
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(source=source, target=target, split='validation', queries=opt.num_queries,
                    model='DDVC global no mask', beta=beta, beta_source='source quality checkpoint',
                    generator_run_id=source_run[1]['run_id'], quality_run_id=head_run[1]['run_id'],
                    target_split_run_id=target_run[1]['run_id'], generator_sha256=generator_sha,
                    target_val_sha256=sha(target_cfg['val_caption_file']),
                    target_references_sha256=[sha(path) for path in references],
                    head_sha256=head_run[1]['files'][head_name]['sha256'])
    atomic_json(out / 'protocol.json', protocol)
    raw, softnms, quality = {}, {}, {}
    print(f'INFERENCE {source}->{target} N={len(durations)} queries={opt.num_queries} beta={beta}', flush=True)
    with torch.inference_mode():
        for index, (video_id, duration) in enumerate(durations.items(), 1):
            rows, count, video = infer_video(model, post, bank, opt, video_id, features[video_id], duration, tokenizer)
            raw[video_id] = rows
            softnms[video_id] = select_proposals(rows, count, method='softnms', sigma=.5, alpha=1., temperature=2.)
            if rows:
                pooled = feature_pool(video, rows, duration, wte, tokenizer)
                scores = quality_head(torch.as_tensor(pooled, dtype=torch.float32, device=opt.device))
                quality[video_id] = select(rows, count, scores.sigmoid().cpu().tolist(), beta)
            else:
                quality[video_id] = []
            if index == 1 or index % 50 == 0 or index == len(durations):
                print(f'PROGRESS {index}/{len(durations)} {video_id}', flush=True)

    summary = []
    for method, predictions in [('raw_candidates', raw), ('baseline_softnms', softnms),
                                ('baseline_quality_softnms', quality)]:
        path = out / f'{method}.json'
        atomic_json(path, dict(results=predictions, version='VERSION 1.0', external_data={}))
        if method == 'raw_candidates':
            continue
        print('SCORING', method, flush=True)
        scores = eval_metrics(str(path), references, target_cfg['gt_file_for_para_eval'],
                              dvc_eval_version='2018', infer_mode=True)
        if any(key not in scores or not math.isfinite(float(scores[key])) for key in METRICS):
            raise RuntimeError(f'Incomplete metrics for {method}: {sorted(scores)}')
        precision, recall = (100 * float(scores[k]) for k in ('Precision', 'Recall'))
        summary.append(dict(source=source, target=target, method=method, queries=opt.num_queries,
                            beta=beta if method == 'baseline_quality_softnms' else 0,
                            videos=len(predictions), events=sum(map(len, predictions.values())),
                            **{key: 100 * float(scores[key]) for key in METRICS},
                            F1=2 * precision * recall / (precision + recall) if precision + recall else 0.))
        write_table(out / 'summary.tsv', summary)
    print('CROSS_COMPLETE', out / 'summary.tsv', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=DATASETS)
    parser.add_argument('target', choices=DATASETS)
    args = parser.parse_args()
    if args.source == args.target:
        parser.error('choose different source and target datasets')
    main(args.source, args.target)
