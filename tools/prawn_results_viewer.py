"""Export two-dataset offline case studies using verified text artifacts only.

Standard library only. Never restores checkpoints, features or training caches.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import sys
from urllib.request import Request, urlopen
import zipfile

RELEASE = 'prawn-results-viewer-20261118-v1'
TRAIN_COMMIT = '31f1eac41841f3a870bb56d9c74a52fc55057466'
AUTH_SHA = '92e7144c6b4c9ffa4d2bce5acd7a8643daf6a2f8cd2197c3147f0eef8959ddfd'
RAW_URL = 'https://raw.githubusercontent.com/thloc1212/DDVC/'
PROFILES = [
    ('matching', 'Caption-matching + content mask', 'prawn_caption_match_generator', 'prawn_caption_match_quality'),
    ('content', 'Content-conditioned mask', 'prawn_content_mask_generator', 'prawn_content_mask_quality'),
    ('original', 'PRAWN learned mask', 'prawn_original_generator', 'prawn_original_quality'),
    ('c1', 'Learned mask (legacy C1)', 'ddvc_c1', 'ddvc_c1_rerank'),
    ('ddvc', 'DDVC global-context baseline', 'ddvc_baseline', 'ddvc_baseline_rerank'),
]
METRICS = ['CIDEr', 'METEOR', 'Bleu_4', 'soda_c', 'Recall', 'Precision', 'F1',
           'Bleu_1', 'Bleu_2', 'Bleu_3', 'ROUGE_L', 'MetaScore']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_url(url, limit, timeout=60):
    with urlopen(Request(url, headers={'User-Agent': 'PRAWN-results-viewer/1.0'}), timeout=timeout) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Unexpectedly large helper download')
    return data


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf8')


def safe_path(value):
    p = PurePosixPath(value)
    if (not value or p.is_absolute() or '..' in p.parts or '\\' in value
            or ':' in value or any(ord(c) < 32 for c in value)):
        raise ValueError('Unsafe artifact path')
    return p


def allowed(path):
    p = safe_path(path)
    blocked = {'cache', 'features', 'toolchains', 'hf-cache', 'secrets', '.git', 'bank', 'caption_bank'}
    if p.suffix not in {'.json', '.tsv', '.yml', '.yaml'} or blocked.intersection(p.parts):
        raise ValueError('Not an allowlisted text artifact: ' + path)
    if p.parts[0] not in {'prepared', 'runs'}:
        raise ValueError('Not a result/annotation artifact: ' + path)
    return p


class Store:
    def __init__(self, drive, output, budget_mb=384):
        self.drive, self.output = drive, Path(output)
        self.limit, self.used = budget_mb * 2**20, 0
        self.records, self.listings = [], {}

    def dirs(self, path):
        if path not in self.listings:
            listing = self.drive.call('lsf', self.drive.path(path), '--dirs-only', '--max-depth', '1', quiet=True)
            self.listings[path] = sorted(x.rstrip('/') for x in listing.splitlines() if x.strip())
        return self.listings[path]

    def snapshots(self, namespace, dataset):
        parent = 'runs/' + namespace
        if namespace not in self.dirs('runs') or dataset not in self.dirs(parent):
            return []
        result = []
        for run_id in reversed(self.dirs(parent + '/' + dataset)):
            safe_path(run_id)
            base = parent + '/' + dataset + '/' + run_id
            if 'recovery' not in self.dirs(base):
                continue
            for slot in self.dirs(base + '/recovery'):
                if slot not in {'slot-0', 'slot-1'}:
                    continue
                remote = base + '/recovery/' + slot
                names = self.drive.call('lsf', self.drive.path(remote), '--files-only', '--max-depth', '1', quiet=True).splitlines()
                if 'READY.json' not in names:
                    continue
                ready = self.drive.read_json(self.drive.path(remote + '/READY.json'))
                if ready.get('dataset') != dataset or ready.get('run_id') != run_id:
                    raise ValueError('Recovery identity mismatch: ' + remote)
                result.append(dict(remote=remote, ready=ready))
        return sorted(result, key=lambda x: (x['ready']['run_id'], x['ready'].get('created_ns', 0)), reverse=True)

    def get(self, snapshot, relative):
        p = allowed(relative)
        meta = snapshot['ready']['files'].get(relative)
        if not isinstance(meta, dict) or not re.fullmatch('[0-9a-f]{64}', meta.get('sha256', '')):
            raise ValueError('Missing file/hash in READY: ' + relative)
        size = int(meta['bytes'])
        if not 0 <= size <= 96 * 2**20 or self.used + size > self.limit:
            raise ValueError('Text-download size limit exceeded: ' + relative)
        local = self.output / 'artifacts' / snapshot['remote'] / Path(*p.parts)
        if not local.exists() or local.stat().st_size != size or sha(local.read_bytes()) != meta['sha256']:
            local.parent.mkdir(parents=True, exist_ok=True)
            temp = local.with_name(local.name + '.partial')
            self.drive.call('copyto', self.drive.path(snapshot['remote'] + '/' + relative), str(temp), quiet=True)
            if temp.stat().st_size != size or sha(temp.read_bytes()) != meta['sha256']:
                temp.unlink(missing_ok=True)
                raise ValueError('Downloaded text failed SHA256 verification: ' + relative)
            os.replace(temp, local)
            self.used += size
            print('TEXT_VERIFIED', relative, f'{size / 2**20:.2f} MiB', flush=True)
        record = dict(remote=snapshot['remote'] + '/' + relative, bytes=size, sha256=meta['sha256'])
        if record not in self.records:
            self.records.append(record)
        return json.loads(local.read_text(encoding='utf-8-sig')) if p.suffix == '.json' else local.read_text(encoding='utf8')


def unique_file(files, suffix):
    matches = [x for x in files if x.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError('Expected exactly one ' + suffix + '; found ' + str(len(matches)))
    return matches[0]


def generator(store, snap):
    files, ready = snap['ready']['files'], snap['ready']
    manifest_path = unique_file(files, '/c1_manifest.json')
    manifest = store.get(snap, manifest_path)
    info_path = unique_file(files, '/info.json')
    info = store.get(snap, info_path)
    cfg = manifest['expected_config']
    if int(info.get('last', {}).get('epoch', -1)) < int(cfg['epoch']) - 1:
        raise ValueError('Generator training is incomplete')
    best = info['best']
    saved = PurePosixPath(best['result_json_path']).name
    final_path = unique_file(files, '/prediction/' + saved)
    raw_path = final_path[:-5] + '.raw.json'
    raw, final = store.get(snap, raw_path), store.get(snap, final_path)
    refs = []
    for reference in cfg['gt_file_for_eval']:
        rel = str(PurePosixPath(reference).relative_to(PurePosixPath(ready['work'])))
        refs.append(store.get(snap, rel))
    model_dir = str(PurePosixPath(info_path).parent)
    checkpoint = files.get(model_dir + '/model-best.pth', {}).get('sha256')
    if not checkpoint:
        raise ValueError('No best-checkpoint identity in READY')
    return dict(raw=raw['results'], final=final, refs=refs, cfg=cfg, snapshot=snap,
                epoch=best['epoch'], checkpoint=checkpoint, raw_sha=files[raw_path]['sha256'],
                config_sha=files[str(PurePosixPath(manifest_path).parent / 'config.yml')]['sha256'])


def quality_matches(g, manifest):
    return (manifest.get('base_checkpoint_sha256') == g['checkpoint']
            and manifest.get('raw_sha256') == g['raw_sha']
            and manifest.get('base_config_sha256') == g['config_sha']
            and manifest.get('base_epoch') == g['epoch'])


def quality(store, namespace, dataset, g):
    for snap in store.snapshots(namespace, dataset):
        files = snap['ready']['files']
        manifest_path = f'runs/{dataset}/manifest.json'
        if manifest_path not in files:
            continue
        manifest = store.get(snap, manifest_path)
        if not quality_matches(g, manifest):
            continue
        modes = ['baseline_topk', 'baseline_softnms', 'reranker_softnms']
        required = [f'runs/{dataset}/evaluation/{m}{s}' for m in modes for s in ['.json', '.metrics.json']]
        if not all(p in files for p in required):
            continue
        try:
            rows = []
            for mode in modes:
                base = f'runs/{dataset}/evaluation/{mode}'
                prediction = store.get(snap, base + '.json')
                scores = store.get(snap, base + '.metrics.json')['scores']
                rows.append((mode, prediction, scores))
            return rows, dict(run_id=snap['ready']['run_id'], source_commit=snap['ready'].get('source_commit'),
                              linked_by='checkpoint + config + raw SHA256 + epoch')
        except (ValueError, KeyError) as error:
            print('QUALITY_SLOT_SKIPPED', snap['remote'], str(error), flush=True)
    return [], None


class Payload:
    def __init__(self):
        self.data = dict(schema='prawn-offline-viewer-v1', built_utc=datetime.now(timezone.utc).isoformat(),
                         strings=[], datasets={})
        self.lookup = {}

    def text(self, s):
        s = str(s)
        if s not in self.lookup:
            self.lookup[s] = len(self.data['strings'])
            self.data['strings'].append(s)
        return self.lookup[s]

    def add(self, dataset, profile, g, policies, qmeta, primary=False):
        key, label = profile[:2]
        ds = self.data['datasets'].setdefault(dataset, dict(refs=[], generators={}, models=[], notes=[]))
        if primary:
            for reference in g['refs']:
                converted = {}
                for vid, r in reference.items():
                    if len(r['timestamps']) != len(r['sentences']):
                        raise ValueError('GT timestamp/caption alignment error')
                    converted[vid] = dict(duration=float(r['duration']), events=[
                        [float(t[0]), float(t[1]), self.text(s)] for t, s in zip(r['timestamps'], r['sentences'])])
                ds['refs'].append(converted)
            ds['primary'] = key
        elif g['refs'] != self._primary_refs:
            raise ValueError('Comparison run uses different reference annotations')
        if primary:
            self._primary_refs = g['refs']
        raw, counts, durations = {}, {}, {}
        for vid, events in g['raw'].items():
            rows, ids = [], set()
            for e in events:
                qid = int(e['query_id'])
                if qid in ids:
                    raise ValueError('Duplicate raw query ID')
                ids.add(qid)
                s, end = map(float, e['timestamp'])
                if not math.isfinite(s + end) or end < s:
                    raise ValueError('Invalid prediction interval')
                rows.append([s, end, self.text(e['sentence']), float(e['proposal_score']), qid,
                             float(e.get('sentence_score', 0))])
            raw[vid] = rows
            counts[vid] = int(events[0]['pred_event_count']) if events else 0
            durations[vid] = float(events[0].get('vid_duration', 0)) if events else 0
        cfg = g['cfg']
        meta = dict(label=label, run_id=g['snapshot']['ready']['run_id'],
                    source_commit=g['snapshot']['ready'].get('source_commit'), epoch=g['epoch'],
                    checkpoint_sha256=g['checkpoint'], raw_sha256=g['raw_sha'],
                    seed=cfg.get('seed'), epochs=cfg.get('epoch'), queries=cfg.get('num_queries'),
                    batch_size=cfg.get('batch_size'), matching_enabled=cfg.get('caption_match_enabled', False))
        ds['generators'][key] = dict(raw=raw, counts=counts, durations=durations, meta=meta)
        if not policies:
            scores = {k: 100 * float(g['final'][k]) for k in METRICS if k in g['final']}
            policies = [('generator_softnms', g['final'], scores)]
            ds['notes'].append(label + ': no completed, hash-matched quality evaluation found; generator final only.')
        for mode, prediction, scores in policies:
            final = {}
            for vid, events in prediction['results'].items():
                lookup = {r[4]: r for r in raw.get(vid, [])}
                selected = []
                for e in events:
                    qid = int(e['query_id'])
                    r = lookup.get(qid)
                    if r is None or r[2] != self.text(e['sentence']) or any(abs(a - b) > 1e-3 for a, b in zip(r[:2], e['timestamp'])):
                        raise ValueError('Final event does not match its raw candidate')
                    selected.append([qid, e.get('selection_score', e.get('joint_score')), e.get('reranker_quality')])
                final[vid] = selected
            clean_scores = {k: float(scores[k]) for k in METRICS if k in scores}
            if 'Precision' in clean_scores and 'Recall' in clean_scores:
                p, r = clean_scores['Precision'], clean_scores['Recall']
                clean_scores['F1'] = 2 * p * r / (p + r) if p + r else 0
            names = {'baseline_topk': 'Top-N, no quality', 'baseline_softnms': 'Soft-NMS, no quality',
                     'reranker_softnms': 'Quality + Soft-NMS', 'generator_softnms': 'Generator final'}
            beta = next((e.get('reranker_beta') for es in prediction['results'].values() for e in es if 'reranker_beta' in e), None)
            ds['models'].append(dict(id=key + '_' + mode, label=label + ' | ' + names[mode],
                                     generator=key, policy=mode, final=final, metrics=clean_scores,
                                     beta=beta, quality_provenance=qmeta))


def render(payload, template, destination):
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    if template.count('__PAYLOAD__') != 1:
        raise ValueError('Invalid viewer template')
    destination.write_text(template.replace('__PAYLOAD__', encoded), encoding='utf8')


def fetch_helper(output):
    target = output / 'tools' / 'c1_rclone_auth.py'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or sha(target.read_bytes()) != AUTH_SHA:
        data = read_url(RAW_URL + TRAIN_COMMIT + '/tools/c1_rclone.py', 128 * 1024)
        if sha(data) != AUTH_SHA:
            raise ValueError('Pinned auth-helper SHA256 mismatch')
        target.write_bytes(data)
    spec = importlib.util.spec_from_file_location('prawn_viewer_auth', target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_rclone(output):
    if shutil.which('rclone'):
        return
    machine = platform.machine().lower()
    arch = {'x86_64': 'amd64', 'amd64': 'amd64', 'aarch64': 'arm64'}.get(machine)
    if sys.platform != 'linux' or not arch:
        raise RuntimeError('Install rclone on this platform, then run the same command again.')
    version = 'v1.68.2'
    name = f'rclone-{version}-linux-{arch}'
    url = 'https://downloads.rclone.org/' + version + '/'
    print('INSTALLING_RCLONE_ONLY', version, '(no training environment)', flush=True)
    checksums = read_url(url + 'SHA256SUMS', 128 * 1024).decode('utf8')
    expected = next((line.split()[0] for line in checksums.splitlines()
                     if len(line.split()) == 2 and line.split()[1].lstrip('*') == name + '.zip'), None)
    if expected is None:
        raise ValueError('rclone archive checksum is missing')
    import io
    data = read_url(url + name + '.zip', 48 * 2**20, timeout=180)
    if sha(data) != expected:
        raise ValueError('rclone archive checksum mismatch')
    folder = output / 'tools' / 'bin'
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        (folder / 'rclone').write_bytes(archive.read(name + '/rclone'))
    os.chmod(folder / 'rclone', 0o700)
    os.environ['PATH'] = str(folder) + os.pathsep + os.environ.get('PATH', '')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/marimo/PRAWN_HTML'))
    parser.add_argument('--env-file', type=Path, default=Path('/marimo/KLTN_DRIVE_PRIVATE.env'))
    parser.add_argument('--config', type=Path, default=Path('/marimo/secrets/rclone.conf'))
    parser.add_argument('--remote', default='kltnbackup')
    parser.add_argument('--root', default='KLTNRUNS')
    parser.add_argument('--datasets', nargs='+', choices=['yc2', 'anet'], default=['yc2', 'anet'])
    parser.add_argument('--primary', choices=['auto'] + [p[0] for p in PROFILES], default='auto')
    parser.add_argument('--upload', action='store_true', help='Save finished ZIP to KLTNRUNS/analysis/html')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ensure_rclone(args.output)
    auth = fetch_helper(args.output)
    auth.restore_auth(args.env_file, args.config, args.remote)
    drive = auth.Drive(args.config, args.remote, args.root)
    store, payload = Store(drive, args.output), Payload()
    namespaces = store.dirs('runs')
    for dataset in args.datasets:
        available = [p for p in PROFILES if p[2] in namespaces and dataset in store.dirs('runs/' + p[2])]
        if args.primary != 'auto':
            primary = next((p for p in available if p[0] == args.primary), None)
        else:
            primary = next((p for p in available if p[0] != 'ddvc'), None)
        if primary is None:
            raise ValueError('No requested result namespace found for ' + dataset)
        comparisons = [p for p in available if p[0] == 'ddvc' and p != primary]
        parent = next((p for p in PROFILES[PROFILES.index(primary) + 1:] if p in available and p[0] != 'ddvc'), None)
        if parent:
            comparisons.append(parent)
        for profile in [primary] + comparisons:
            found = None
            for snap in store.snapshots(profile[2], dataset):
                try:
                    found = generator(store, snap)
                    break
                except (ValueError, KeyError) as error:
                    print('GENERATOR_SLOT_SKIPPED', snap['remote'], str(error), flush=True)
            if found is None:
                if profile == primary:
                    raise ValueError('No complete verified generator snapshot for ' + dataset + '/' + profile[0])
                continue
            policies, qmeta = quality(store, profile[3], dataset, found)
            try:
                if profile[0] == 'matching' and found['cfg'].get('caption_match_enabled') is not True:
                    raise ValueError('Caption-matching namespace has matching disabled in the saved config')
                if profile[0] == 'ddvc' and found['cfg'].get('learned_proposal_mask') is not False:
                    raise ValueError('DDVC baseline namespace does not contain a verified global-context config')
                payload.add(dataset, profile, found, policies, qmeta, profile == primary)
                print('VIEWER_MODEL', dataset, profile[1], found['snapshot']['ready']['run_id'], flush=True)
            except ValueError as error:
                if profile == primary:
                    raise
                print('COMPARISON_SKIPPED', profile[0], str(error), flush=True)
        if not any(p[0] == 'ddvc' for p in comparisons):
            payload.data['datasets'][dataset]['notes'].append('No separate DDVC baseline backup was found. Same-generator selectors are not labelled DDVC.')
    provenance = dict(schema=payload.data['schema'], built_utc=payload.data['built_utc'],
                      downloaded_bytes=store.used, files=store.records, profiles=PROFILES)
    write_json(args.output / 'provenance.json', provenance)
    write_json(args.output / 'viewer_data.json', payload.data)
    template_path = Path(__file__).with_name('prawn_results_viewer.template.html')
    if template_path.exists():
        template = template_path.read_text(encoding='utf8')
    else:
        template = read_url(RAW_URL + RELEASE + '/tools/prawn_results_viewer.template.html', 256 * 1024).decode('utf8')
    render(payload.data, template, args.output / 'index.html')
    (args.output / 'README.txt').write_text('Open index.html in Chrome or Edge. Offline; no web server required.\n'
        'Official saved validation metrics are global. Timeline one-to-one matches are diagnostics only.\n'
        'Reference sets remain separate; F1 is calculated from saved aggregate P/R.\n'
        'No checkpoints, video features, training caches or private credentials included.\n', encoding='utf8')
    dest = args.output / 'PRAWN_YC2_ANET_HTML.zip'
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in ['index.html', 'viewer_data.json', 'provenance.json', 'README.txt']:
            archive.write(args.output / name, 'PRAWN_HTML/' + name)
        for record in store.records:
            path = args.output / 'artifacts' / record['remote']
            archive.write(path, 'PRAWN_HTML/' + path.relative_to(args.output).as_posix())
        archive.write(Path(__file__), 'PRAWN_HTML/tools/prawn_results_viewer.py')
        archive.writestr('PRAWN_HTML/tools/prawn_results_viewer.template.html', template)
    print('HTML_READY', args.output / 'index.html', flush=True)
    print('ZIP_READY', dest, f'({dest.stat().st_size / 2**20:.1f} MiB)', flush=True)
    print('DOWNLOADED_TEXT_MIB', round(store.used / 2**20, 2), flush=True)
    if args.upload:
        remote = drive.path('analysis/html/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + dest.name)
        drive.call('copyto', str(dest), remote, quiet=True)
        print('ZIP_UPLOADED', remote, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('STOPPED', type(error).__name__, str(error), file=sys.stderr, flush=True)
        sys.exit(1)
