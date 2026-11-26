"""Extend the published PRAWN-DVC Full ActivityNet validation sweep."""
import argparse
import csv
import fcntl
import json
from pathlib import Path, PurePosixPath

from c1_rclone import Drive, digest, restore_auth
from prawn_ablation_eval import score, select_fixed, write_table
from prawn_ablation_control import validate_full
from prawn_results_viewer import Store, generator


GENERATOR = '20261117T145203Z_6035c03e3ff3'
HEAD = '20261118T044418Z_55fde4ba77d2'
RAW_SHA = '4874a22f24d136f1f97f5f71514ebb5538a775b29a7322fb8f5527dbe6913d75'
HEAD_SHA = 'a22a10203e372aed9f3cbf9df5004360d55e33c6c4a51d998b46a694f0a6a394'
PRIOR_FILES = {
    'quality_scores.json': 'f58b0f2a370f72d7b179aad5b070515ed875c667b72fc5251082a704982e852f',
    'summary.tsv': '21216d20c20bbd0238a4d98adeddb765473e858faa5905ba9ed43131269aceea',
}
BETAS = (10, 12, 14, 16, 18, 20)


def prior_file(drive, previous, cache, name):
    known = PRIOR_FILES[name]
    for path in (previous / name, cache / name):
        if path.is_file() and digest(path) == known:
            print('REUSE_PREVIOUS', path, flush=True)
            return path
    target = cache / name
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.partial')
    source = drive.path(f'analysis/prawn_loo/anet/full_{GENERATOR}/beta/{name}')
    drive.call('copyto', source, partial, quiet=True)
    if digest(partial) != known:
        raise ValueError(f'{source} differs from the original beta 0–8 export')
    partial.replace(target)
    return target


def restore_generator_text(drive, folder, identity):
    store = Store(drive, folder)
    pinned = folder / 'control_snapshot.json'
    snapshot = None
    if pinned.is_file():
        candidate = json.loads(pinned.read_text(encoding='utf-8'))['generator']
        if candidate['ready']['run_id'] == GENERATOR:
            snapshot = candidate
    if snapshot is None:
        for candidate in store.snapshots('prawn_caption_match_generator', 'anet'):
            ready = candidate['ready']
            if ready['run_id'] == GENERATOR and any(f['sha256'] == RAW_SHA for f in ready['files'].values()):
                snapshot = candidate
                break
    if snapshot is None:
        raise FileNotFoundError(f'Missing original PRAWN generator backup {GENERATOR}')
    saved = generator(store, snapshot)
    if (saved['raw_sha'], saved['checkpoint'], saved['config_sha']) != (
            RAW_SHA, identity['checkpoint_sha256'], identity['config_sha256']):
        raise ValueError('PRAWN generator, raw proposals or config differ from the original beta sweep')
    cfg = dict(saved['cfg'])
    validate_full(cfg, 'anet')
    for key in ('gt_file_for_eval', 'gt_file_for_para_eval'):
        refs = []
        for old in cfg[key]:
            relative = PurePosixPath(old).relative_to(snapshot['ready']['work']).as_posix()
            store.get(snapshot, relative)
            refs.append(str(folder / 'artifacts' / snapshot['remote'] / relative))
        cfg[key] = refs
    return cfg, saved['raw']


def main(work):
    config = Path('/marimo/secrets/rclone.conf')
    env = Path('/marimo/KLTN_DRIVE_PRIVATE.env')
    if not config.is_file() and env.is_file():
        restore_auth(env, config, 'kltnbackup')
    drive = Drive(config, 'kltnbackup', 'KLTNRUNS')
    previous = work / 'results/anet/beta'
    output = work / 'results/anet/beta_10_20'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = output / 'original_inputs'
        payload = json.loads(prior_file(drive, previous, cache, 'quality_scores.json').read_text(encoding='utf-8'))
        identity = payload['identity']
        if (identity['generator_run_id'], identity['quality_run_id'], payload['head_sha256']) != (
                GENERATOR, HEAD, HEAD_SHA):
            raise ValueError('The cached scores do not belong to the original PRAWN Full pair')
        cfg, raw = restore_generator_text(drive, work / 'control/anet', identity)
        identity = dict(identity, head_sha256=HEAD_SHA)
        with prior_file(drive, previous, cache, 'summary.tsv').open(encoding='utf-8-sig', newline='') as stream:
            earlier = list(csv.DictReader(stream, delimiter='\t'))
        if len(raw) != 4917:
            raise ValueError(f'Expected 4917 validation videos, found {len(raw)}')
        print('PRAWN_FULL_ANET_BETA_START', GENERATOR, HEAD, BETAS, flush=True)
        summary = []
        for beta in BETAS:
            predictions = select_fixed(raw, payload['qualities'], beta)
            if sum(map(len, predictions.values())) != 14929:
                raise ValueError('Selection changed the original budget of 14929 events')
            summary.append(score(output, f'beta_{beta}', predictions, cfg, identity, beta))
            write_table(output / 'summary.tsv', summary)
            write_table(output / 'summary_0_20.tsv', earlier + summary)
        print('PRAWN_FULL_ANET_BETA_COMPLETE', output / 'summary_0_20.tsv', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    main(parser.parse_args().work.resolve())
