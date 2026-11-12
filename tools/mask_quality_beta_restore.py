"""Restore verified evaluation inputs and save beta results through rclone."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import c1_rclone as helpers

RUNS = {'yc2': '20261109T092734Z_05a32edc0bd5', 'anet': '20261110T065806Z_00646d58d36e'}
RELEASE = 'learned-mask-quality-beta-grid-20261112'


def transport():
    config = Path('/marimo/secrets/rclone.conf')
    if not config.exists():
        helpers.restore_auth(Path('/marimo/KLTN_DRIVE_PRIVATE.env'), config, 'kltnbackup')
    helpers.parse_auth(config.read_bytes(), 'kltnbackup')
    drive = helpers.Drive(config, 'kltnbackup', 'KLTNRUNS')
    drive.call('lsf', 'kltnbackup:', '--max-depth', '1', quiet=True)
    return drive


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def snapshot(drive, parent, dataset, run_id):
    ready = []
    for slot in (0, 1):
        try:
            remote = parent + '/recovery/slot-' + str(slot)
            meta = drive.read_json(remote + '/READY.json')
            if meta['dataset'] != dataset or meta['run_id'] != run_id:
                raise ValueError('Drive snapshot identity mismatch')
            ready.append((meta['created_ns'], remote, meta))
        except RuntimeError:
            continue
    if not ready:
        raise FileNotFoundError('No completed Drive snapshot at ' + parent)
    return sorted(ready, reverse=True)


def download_subset(drive, remote, meta, names, destination):
    files = {name: meta['files'][name] for name in names}
    if not files:
        raise ValueError('Empty restore selection')
    for name in files:
        helpers.relative_path(name)
        path = destination / name
        if not path.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Restore path escapes destination')
        if path.exists() and helpers.digest(path) != files[name]['sha256']:
            raise FileExistsError('Existing input differs from snapshot: ' + str(path))
    missing = {name: entry for name, entry in files.items() if not (destination / name).is_file()}
    if missing:
        print('RESTORE_BYTES', sum(entry['bytes'] for entry in missing.values()), remote, flush=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.beta-restore-') as temporary:
            staged = Path(temporary)
            helpers.fetch_verified(drive, remote, dict(meta, files=missing), staged)
            for name in missing:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged / name, target)
    for name, entry in files.items():
        if helpers.digest(destination / name) != entry['sha256']:
            raise ValueError('Restored checksum mismatch: ' + name)
    return files


def restore(work, dataset, run_id):
    drive = transport()
    head_root = work / 'inputs/reranker'
    prefix = 'runs/' + dataset + '/'
    parent = drive.path('runs/ddvc_c1_rerank/' + dataset + '/' + run_id)
    attempts = snapshot(drive, parent, dataset, run_id)
    last_error = None
    for _, remote, meta in attempts:
        if meta.get('work') != '/marimo/DDVC_C1_Rerank' or meta.get('base_work') != '/marimo/DDVC_C1':
            raise ValueError('Expected the learned-mask C1 reranker backup')
        names = [name for name in meta['files'] if
                 name in {prefix + 'manifest.json', prefix + 'head-best.pt', prefix + 'head_history.json',
                          prefix + 'beta_selection.json', 'provenance/rerank_' + dataset + '.json'}
                 or name.startswith(prefix + 'cache/val/')
                 or name.startswith(prefix + 'evaluation/') and Path(name).suffix in {'.json', '.tsv'}]
        required = {prefix + 'manifest.json', prefix + 'head-best.pt',
                    prefix + 'evaluation/baseline_softnms.json', 'provenance/rerank_' + dataset + '.json'}
        if not required.issubset(names):
            continue
        try:
            selected = download_subset(drive, remote, meta, names, head_root)
            break
        except (RuntimeError, ValueError) as error:
            last_error = error
    else:
        raise RuntimeError('Reranker restore failed') from last_error
    state = read(head_root / 'provenance' / ('rerank_' + dataset + '.json'))
    base_state = state['base_c1_drive_session']
    if state['run_id'] != run_id or base_state.get('variant', 'c1') != 'c1':
        raise ValueError('Wrong original generator experiment')
    base_id = base_state['run_id']
    if not re.fullmatch('[A-Za-z0-9_-]+', base_id):
        raise ValueError('Invalid original generator run ID')
    manifest = read(head_root / prefix / 'manifest.json')
    if manifest['dataset'] != dataset or manifest.get('base_method') == 'C0_GLOBAL_NO_MASK':
        raise ValueError('Quality head belongs to a different architecture')
    receipt = dict(dataset=dataset, run_id=run_id, base_run_id=base_id,
                   reranker_remote=remote, reranker_files=selected, release=RELEASE)
    epochs = 15 if dataset == 'yc2' else 6
    base_run = f'{dataset}_c1_e{epochs}_b1'
    base_parent = drive.path('runs/ddvc_c1/' + dataset + '/' + base_id)
    for _, remote, meta in snapshot(drive, base_parent, dataset, base_id):
        best = 'runs/' + base_run + '/model-best.pth'
        if meta['files'].get(best, {}).get('sha256') != manifest['base_checkpoint_sha256']:
            continue
        names = [name for name in meta['files'] if
                 name.startswith('prepared/' + base_run + '/') and Path(name).suffix in {'.json', '.yml', '.yaml'}
                 or name == 'runs/' + base_run + '/info.json']
        receipt.update(base_remote=remote,
                       base_files=download_subset(drive, remote, meta, names, work / 'inputs/base'))
        break
    else:
        raise ValueError('No generator backup matches the quality head checkpoint fingerprint')
    helpers.write_json(work / 'provenance' / ('inputs_' + dataset + '.json'), receipt)
    restore_grid(drive, work, dataset, run_id)
    print('LEARNED_MASK_QUALITY_INPUTS_VERIFIED', dataset, 'head_run', run_id, 'base_run', base_id, flush=True)


def recovery_root(drive, dataset, run_id):
    return drive.path('analysis/learned_mask_quality_beta/' + dataset + '/' + run_id + '/recovery')


def restore_grid(drive, work, dataset, run_id):
    parent = recovery_root(drive, dataset, run_id)
    if (work / 'runs' / dataset / 'beta_grid/manifest.json').exists():
        return
    # The first execution has no analysis recovery directory yet.
    drive.call('mkdir', parent)
    for _, remote, meta in snapshot_optional(drive, parent, dataset, run_id):
        names = [name for name in meta['files'] if name.startswith('runs/' + dataset + '/beta_grid/')]
        try:
            download_subset(drive, remote, meta, names, work)
            print('PREVIOUS_BETA_RESULTS_RESTORED', dataset, flush=True)
            return
        except (RuntimeError, ValueError):
            continue


def snapshot_optional(drive, parent, dataset, run_id):
    try:
        return snapshot(drive, parent.removesuffix('/recovery'), dataset, run_id)
    except FileNotFoundError:
        return []


def backup(work, dataset):
    (work / 'logs').mkdir(exist_ok=True)
    with (work / 'logs' / (dataset + '.backup.lock')).open('a') as lock:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        backup_locked(work, dataset)


def backup_locked(work, dataset):
    drive = transport()
    receipt = read(work / 'provenance' / ('inputs_' + dataset + '.json'))
    run_id = receipt['run_id']
    parent = recovery_root(drive, dataset, run_id)
    drive.call('mkdir', parent)
    pointer = drive.optional_json(parent + '/LATEST.json')
    slot = 1 - int(pointer['slot']) if pointer else 0
    if slot not in (0, 1):
        raise ValueError('Invalid beta recovery slot')
    with tempfile.TemporaryDirectory(dir=work, prefix='.beta-backup-') as temporary:
        staged = Path(temporary)
        folder = work / 'runs' / dataset / 'beta_grid'
        paths = [p for p in folder.rglob('*') if p.is_file() and p.suffix in {'.json', '.tsv', '.log'}]
        paths.append(work / 'provenance' / ('inputs_' + dataset + '.json'))
        progress_log = work / 'logs' / (dataset + '.beta.log')
        if progress_log.is_file():
            paths.append(progress_log)
        for path in paths:
            target = staged / path.relative_to(work)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        inventory = helpers.inventory(staged)
        destination = parent + '/slot-' + str(slot)
        drive.copy(staged, destination)
        drive.call('check', staged, destination, '--one-way')
        meta = dict(dataset=dataset, run_id=run_id, created_ns=time.time_ns(), files=inventory, release=RELEASE)
        drive.put_json(destination + '/READY.json', meta)
        drive.put_json(parent + '/LATEST.json', dict(slot=slot, created_ns=meta['created_ns']))
    print('BETA_BACKUP_VERIFIED', dataset, 'slot', slot, flush=True)


def export(work, dataset):
    output = work.parent / ('LEARNED_MASK_QUALITY_BETA_' + dataset + '_' + time.strftime('%Y%m%d_%H%M%S') + '.zip')
    with zipfile.ZipFile(output.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        folders = [work / 'inputs/reranker/runs' / dataset, work / 'inputs/base/prepared',
                   work / 'runs' / dataset, work / 'provenance', work / 'logs']
        for folder in folders:
            for path in folder.rglob('*'):
                if path.is_file() and not path.is_symlink() and path.suffix in {'.json', '.jsonl', '.yml', '.yaml', '.tsv', '.log', '.npz', '.pt'}:
                    archive.write(path, path.relative_to(work).as_posix())
        for path in list((ROOT / 'reranking').glob('*.py')) + list((ROOT / 'tools').glob('mask_quality_beta*')):
            if path.is_file():
                archive.write(path, 'source/' + path.relative_to(ROOT).as_posix())
        archive.writestr('EXPORT_MANIFEST.json', json.dumps(dict(dataset=dataset, release=RELEASE,
                         source_commit=helpers.revision(), method='LEARNED_MASK_FROZEN_QUALITY_SOFTNMS')))
    os.replace(output.with_suffix('.tmp'), output)
    print('ZIP_READY', output, flush=True)


def worker(work, dataset, run_id):
    restore(work, dataset, run_id)
    subprocess.run([sys.executable, '-u', str(ROOT / 'tools/molab_pipeline.py'), 'java', '--target', str(work / 'tools/java')], check=True)
    env = dict(os.environ, PATH=str(work / 'tools/java/bin') + os.pathsep + os.environ.get('PATH', ''))
    subprocess.run([sys.executable, '-u', str(ROOT / 'tools/mask_quality_beta_grid.py'), '--work', str(work),
                    '--dataset', dataset, '--betas', '0', '0.25', '0.5', '1', '2', '4', '6', '8', '--backup'],
                   cwd=ROOT, env=env, check=True)
    export(work, dataset)
    print('LEARNED_MASK_BETA_COMPLETE', dataset, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('auth', 'restore', 'start', 'worker', 'backup', 'export'))
    parser.add_argument('dataset', choices=('yc2', 'anet'))
    parser.add_argument('--work', type=Path, default=ROOT.parent)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or RUNS[args.dataset]
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        parser.error('Invalid reranker run ID')
    if args.command == 'auth':
        transport()
        print('DRIVE_CONNECTED', flush=True)
    elif args.command == 'restore':
        restore(work, args.dataset, run_id)
    elif args.command == 'backup':
        backup(work, args.dataset)
    elif args.command == 'export':
        export(work, args.dataset)
    elif args.command == 'worker':
        import fcntl
        (work / 'logs').mkdir(exist_ok=True)
        with (work / 'logs' / (args.dataset + '.worker.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker(work, args.dataset, run_id)
    else:
        (work / 'logs').mkdir(exist_ok=True)
        pid_file = work / 'logs' / (args.dataset + '.beta.pid')
        if pid_file.is_file():
            pid = pid_file.read_text().strip()
            process_file = Path('/proc') / pid / 'cmdline'
            if process_file.is_file():
                command = process_file.read_bytes().split(b'\0')
                if b'worker' in command and args.dataset.encode() in command and any(b'mask_quality_beta_restore.py' in value for value in command):
                    print('BETA_WORKER_ALREADY_RUNNING', args.dataset, 'PID', pid, flush=True)
                    return
        with (work / 'logs' / (args.dataset + '.beta.log')).open('a', encoding='utf8') as stream:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', args.dataset,
                                        '--work', str(work), '--run-id', run_id], cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        (work / 'logs' / (args.dataset + '.beta.pid')).write_text(str(process.pid))
        print('BETA_WORKER_STARTED', args.dataset, 'PID', process.pid, flush=True)


if __name__ == '__main__':
    main()
