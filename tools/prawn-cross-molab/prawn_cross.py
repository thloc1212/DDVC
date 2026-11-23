"""Evaluate frozen PRAWN on the other dataset's validation split."""

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

import c1_rclone as drive_tools
from c1_pipeline import run_name
from eval_utils_clip import eval_metrics
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
RUNS = {
    'yc2': ('20261117T144626Z_63b81b35a195', '20261118T042233Z_4a8d7e8c4864'),
    'anet': ('20261117T145203Z_6035c03e3ff3', '20261118T044418Z_55fde4ba77d2'),
}


def download_snapshot(drive, dataset, stage, required, include=None):
    run_id = RUNS[dataset][stage == 'quality']
    recovery = drive.path(f'runs/prawn_caption_match_{stage}/{dataset}/{run_id}/recovery')
    parent = WORK / 'inputs' / stage / dataset / run_id
    pointer = drive.read_json(f'{recovery}/LATEST.json')
    slot = pointer['slot']
    if slot not in (0, 1):
        raise ValueError(f'Invalid recovery slot for {run_id}')
    remote = f'{recovery}/slot-{slot}'
    meta = drive.read_json(f'{remote}/READY.json')
    if (meta['run_id'], meta['dataset'], meta['slot'], meta['created_ns']) != (
            run_id, dataset, slot, pointer['created_ns']):
        raise ValueError(f'Recovery pointer changed for {run_id}. Retry when backup completes.')
    missing = set(required) - meta['files'].keys()
    if missing:
        raise FileNotFoundError(f'{run_id} is missing {sorted(missing)}')
    if stage == 'generator' and meta['source_commit'] != os.environ['PRAWN_CROSS_BASE_COMMIT']:
        raise ValueError(f'{run_id} requires generator commit {meta["source_commit"]}')
    cache = parent / str(meta['created_ns'])
    files = {name: expected for name, expected in meta['files'].items()
             if name in required or (include is not None and include(name))}
    pending = {}
    for name, expected in files.items():
        drive_tools.relative_path(name)
        path = cache / name
        if (not path.is_file() or path.stat().st_size != expected['bytes']
                or sha(path) != expected['sha256']):
            pending[name] = expected
    if pending:
        cache.mkdir(parents=True, exist_ok=True)
        drive_tools.fetch_verified(drive, remote, {'files': pending}, cache)
    return cache, meta


def generator_assets(drive, dataset, load_weights):
    run = run_name(dataset)
    prepared = f'prepared/{run}'
    config_path = f'{prepared}/config.yml'
    checkpoint_path = f'runs/{run}/model-best.pth'
    required = [config_path]
    if load_weights:
        required.append(checkpoint_path)
        def include(name):
            return name.startswith(f'{prepared}/bank/{dataset}/') and name.endswith('.npy')
    else:
        def include(name):
            return (name.startswith(f'{prepared}/data/{dataset}/captiondata/')
                    and name.endswith('.json') and not name.endswith('/train.json'))
    cache, meta = download_snapshot(drive, dataset, 'generator', required, include)
    cfg = yaml.safe_load((cache / config_path).read_text(encoding='utf-8'))
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


def load_target_features(drive, dataset):
    for protocol in (drive_tools.DATA_PROTOCOL, 'qts-base-assets-v1'):
        if (drive_tools.base_path(dataset) / 'config.yml').is_file():
            break
        drive_tools.import_qts_data(drive, dataset, protocol)
    return torch.load(drive_tools.base_feature(dataset), map_location='cpu', mmap=True, weights_only=False)


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
    config = Path('/marimo/secrets/rclone.conf')
    if not config.is_file():
        drive_tools.restore_auth(Path('/marimo/KLTN_DRIVE_PRIVATE.env'), config, 'kltnbackup')
    drive = drive_tools.Drive(config, 'kltnbackup', 'KLTNRUNS')
    drive.call('lsf', drive.path(), '--dirs-only', '--max-depth', '1', quiet=True)

    source_cfg, source_path, generator_path, generator_sha = generator_assets(drive, source, True)
    target_cfg = generator_assets(drive, target, False)
    head_prefix = f'runs/{source}'
    cache, _ = download_snapshot(drive, source, 'quality',
                                 [f'{head_prefix}/head-best.pt', f'{head_prefix}/manifest.json'])
    head_manifest = json.loads((cache / head_prefix / 'manifest.json').read_text(encoding='utf-8'))
    head_checkpoint = torch.load(cache / head_prefix / 'head-best.pt', map_location='cpu', weights_only=False)
    if (head_manifest['base_checkpoint_sha256'] != generator_sha
            or head_checkpoint['signature'] != head_manifest['signature']):
        raise ValueError('The supplied head and generator do not belong to the same trained PRAWN run')
    beta = float(head_checkpoint['beta'])

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
    features = load_target_features(drive, target)
    missing = durations.keys() - features.keys()
    if missing:
        raise ValueError(f'{len(missing)} target validation videos have no CLIP features: {sorted(missing)[:5]}')

    out = WORK / 'results' / f'{source}_to_{target}'
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(source=source, target=target, split='validation', queries=opt.num_queries,
                    beta=beta, generator_run_id=RUNS[source][0], quality_run_id=RUNS[source][1],
                    target_split_run_id=RUNS[target][0], generator_sha256=generator_sha,
                    head_sha256=sha(cache / head_prefix / 'head-best.pt'))
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
    for method, predictions in [('raw_candidates', raw), ('generator_softnms', softnms),
                                ('prawn_quality_softnms', quality)]:
        path = out / f'{method}.json'
        atomic_json(path, dict(results=predictions, version='VERSION 1.0', external_data={}))
        if method == 'raw_candidates':
            continue
        print('SCORING', method, flush=True)
        scores = eval_metrics(str(path), target_cfg['gt_file_for_eval'], target_cfg['gt_file_for_para_eval'],
                              dvc_eval_version='2018', infer_mode=True)
        if any(key not in scores or not math.isfinite(float(scores[key])) for key in METRICS):
            raise RuntimeError(f'Incomplete metrics for {method}: {sorted(scores)}')
        precision, recall = (100 * float(scores[k]) for k in ('Precision', 'Recall'))
        summary.append(dict(source=source, target=target, method=method, queries=opt.num_queries,
                            beta=beta if method == 'prawn_quality_softnms' else 0,
                            videos=len(predictions), events=sum(map(len, predictions.values())),
                            **{key: 100 * float(scores[key]) for key in METRICS},
                            F1=2 * precision * recall / (precision + recall) if precision + recall else 0.))
        write_table(out / 'summary.tsv', summary)
    print('CROSS_COMPLETE', out / 'summary.tsv', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=RUNS)
    parser.add_argument('target', choices=RUNS)
    args = parser.parse_args()
    if args.source == args.target:
        parser.error('choose different source and target datasets')
    main(args.source, args.target)
