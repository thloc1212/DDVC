"""Restore the exact inputs used in the DDVC Global ActivityNet beta 0–8 sweep."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1_rclone import Drive, digest, fetch_verified, relative_path, restore_auth


DATASET = 'anet'
GEN_RUN = '20261111T084442Z_449a1a2b6875'
HEAD_RUN = '20261111T115843Z_f6e3a90e088e'
CONFIG_NAME = 'prepared/anet_baseline_e6_b1/config.yml'
CONFIG_SHA = '3b2b645052cd8747867f1824ce3343a82cffafb9c55b8fdf9ce56eb206836c39'
GEN_SHA = '35edfe83a208c23e0c6173b9f24639488068826ac073da379176acdaac7424eb'
HEAD_SHA = '15a0ae1ab691bc785013c4c80b0bc562b8b6ba766cf1ba88798efa34742a6110'
CACHE_SHA = '4d53b6dabd57f1bcdadcfb565d0a4e6bc00dcd375e7570a25157e7a7cfb1eedb'
BASELINE_SHA = 'e34beb6d915b04f3f50b77e518fbcc0ffbd7f73db158b0931c8bc2651d87170c'
FOLDER = 'runs/anet/'


def restore_files(drive, remote, destination, files):
    pending = {}
    for name, expected in files.items():
        relative_path(name)
        local = destination / name
        if local.is_symlink():
            raise ValueError(f'Unexpected symlink at {local}')
        if not local.is_file() or local.stat().st_size != expected['bytes'] or digest(local) != expected['sha256']:
            pending[name] = expected
    if not pending:
        return
    size = sum(spec['bytes'] for spec in pending.values()) / 2**20
    print(f'RESTORE_DOWNLOAD {len(pending)} files, {size:.1f} MiB from {remote}', flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination, prefix='.beta-restore-') as temporary:
        staged = Path(temporary)
        fetch_verified(drive, remote, {'files': pending}, staged)
        for name in pending:
            local = destination / name
            local.parent.mkdir(parents=True, exist_ok=True)
            (staged / name).replace(local)


def snapshot(drive, namespace, run, checks):
    recovery = drive.path(f'runs/{namespace}/{DATASET}/{run}/recovery')
    folders = drive.call('lsf', recovery, '--dirs-only', '--max-depth', '1', quiet=True).splitlines()
    candidates = []
    for slot in (0, 1):
        if f'slot-{slot}/' not in folders:
            continue
        remote = f'{recovery}/slot-{slot}'
        meta = drive.optional_json(f'{remote}/READY.json')
        if not meta or (meta.get('run_id'), meta.get('dataset'), meta.get('slot')) != (run, DATASET, slot):
            continue
        if all(meta['files'].get(name, {}).get('sha256') == sha for name, sha in checks.items()):
            candidates.append((meta['created_ns'], remote, meta))
    if not candidates:
        raise FileNotFoundError(f'No verified snapshot with the historical fingerprints at {recovery}')
    _, remote, meta = max(candidates, key=lambda candidate: candidate[0])
    return remote, meta


def main(work):
    config = Path('/marimo/secrets/rclone.conf')
    if not config.is_file():
        restore_auth('/marimo/KLTN_DRIVE_PRIVATE.env', config, 'kltnbackup')
    drive = Drive(config, 'kltnbackup', 'KLTNRUNS')
    remote, gen = snapshot(drive, 'ddvc_baseline', GEN_RUN, {
        CONFIG_NAME: CONFIG_SHA, 'runs/anet_baseline_e6_b1/model-best.pth': GEN_SHA,
    })
    restore_files(drive, remote, work, {CONFIG_NAME: gen['files'][CONFIG_NAME]})
    cfg = yaml.safe_load((work / CONFIG_NAME).read_bytes())
    original_work = Path(gen['work'])
    ref_paths = [Path(p).relative_to(original_work).as_posix()
                 for p in cfg['gt_file_for_eval'] + cfg['gt_file_for_para_eval']]
    restore_files(drive, remote, work, {name: gen['files'][name] for name in ref_paths})

    head, head_meta = snapshot(drive, 'ddvc_baseline_rerank', HEAD_RUN, {
        FOLDER + 'head-best.pt': HEAD_SHA, FOLDER + 'evaluation/baseline_softnms.json': BASELINE_SHA,
    })
    files = head_meta['files']
    shard_dir = FOLDER + 'cache/val/'
    shards = sorted(n for n in files if re.fullmatch(re.escape(shard_dir) + r'shard_\d{5}\.npz', n))
    pairs = [(n, n.removesuffix('.npz') + '.json') for n in shards]
    fingerprint = [(Path(n).name, files[n]['sha256'], files[m]['sha256']) for n, m in pairs]
    cache_hash = hashlib.sha256(json.dumps(fingerprint, separators=(',', ':')).encode()).hexdigest()
    if cache_hash != CACHE_SHA:
        raise ValueError('The ActivityNet validation cache differs from the previous beta 0–8 sweep')
    names = [FOLDER + n for n in ('head-best.pt', 'manifest.json', 'evaluation/baseline_softnms.json')]
    names.extend(name for pair in pairs for name in pair)
    restore_files(drive, head, work / 'frozen_quality', {name: files[name] for name in names})
    print(f'ANET_BETA_INPUTS_READY {GEN_RUN} {HEAD_RUN} {len(shards)} validation shards', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    main(parser.parse_args().work.resolve())
