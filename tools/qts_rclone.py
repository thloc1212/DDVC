"""Terminal-only QTS runs with private rclone auth, shared data and verified recovery.

No model/training defaults are changed here. Credentials stay outside the work root.
Only immutable checkpoint inodes and allowlisted run artifacts enter recovery slots.
"""
import argparse
import base64
import configparser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
DATA_PROTOCOL = 'qts-base-assets-v1'
TEXT = {'.json', '.jsonl', '.yml', '.yaml', '.log', '.txt', '.tsv', '.csv'}


def announce(*values):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *values, flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf8')
    os.replace(temp, path)


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def run_name(dataset):
    return dataset + '_qts_support_e' + ('15' if dataset == 'yc2' else '6') + '_b1'


def session_path(dataset):
    return WORK / 'provenance' / ('drive_session_' + dataset + '.json')


def parse_auth(raw, remote):
    try:
        cfg = configparser.ConfigParser(interpolation=None)
        cfg.read_string(raw.decode('utf8'))
        token = json.loads(cfg[remote]['token'])
        valid = (cfg[remote].get('type') == 'drive'
                 and token.get('refresh_token') and token.get('access_token')
                 and not cfg[remote].get('service_account_file', '').strip())
        if not valid:
            raise ValueError()
        result = configparser.ConfigParser(interpolation=None)
        result[remote] = dict(cfg[remote])
        stream = io.StringIO()
        result.write(stream)
        return stream.getvalue().encode('utf8')
    except Exception:
        raise ValueError('Private file must contain an authenticated Drive remote named ' + remote) from None


def restore_auth(env_file, config, remote):
    """Read one inert base64 assignment; never source shell text or print values."""
    values = []
    for line in Path(env_file).read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if line.startswith('export '):
            line = line[7:].strip()
        if line.startswith('RCLONE_CONFIG_B64='):
            values.append(line.split('=', 1)[1].strip().strip('\"\''))
    if len(values) != 1:
        raise ValueError('Private env file needs exactly one RCLONE_CONFIG_B64 assignment')
    try:
        raw = base64.b64decode(values[0], validate=True)
    except Exception:
        raise ValueError('Invalid private base64 value; do not paste it into chat') from None
    content = parse_auth(raw, remote)
    config = Path(config)
    config.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if config.is_symlink():
        raise ValueError('Refusing a symlink for the private config')
    if config.exists():
        # Preserve tokens refreshed in this session; never overwrite working auth.
        parse_auth(config.read_bytes(), remote)
        old = configparser.ConfigParser(interpolation=None)
        old.read_string(config.read_text())
        new = configparser.ConfigParser(interpolation=None)
        new.read_string(content.decode())
        if old[remote].get('client_id') != new[remote].get('client_id'):
            raise ValueError('Existing config uses a different client; inspect it privately')
        os.chmod(config, 0o600)
        announce('AUTH_CONFIG_PRESENT', config)
        return
    fd, temporary = tempfile.mkstemp(prefix='.rclone-', dir=config.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, config)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    announce('AUTH_RESTORED', config, '(no credential values printed)')


class Drive:
    def __init__(self, config, remote, root):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', remote):
            raise ValueError('Invalid remote name')
        self.config, self.remote = str(config), remote
        self.root = root.strip('/')
        if not self.root or any(p in ('.', '..') for p in self.root.split('/')):
            raise ValueError('Invalid Drive root')

    def path(self, relative=''):
        return self.remote + ':' + self.root + ('/' + relative.strip('/') if relative else '')

    def call(self, *args, quiet=False):
        command = ['rclone', '--config', self.config, '--retries', '2',
                   '--low-level-retries', '3', '--contimeout', '20s', '--timeout', '5m', *map(str, args)]
        if quiet:
            result = subprocess.run(command, capture_output=True, text=True)
        else:
            result = subprocess.run(command)
        if result.returncode:
            # Avoid ever dumping a config, auth command or environment in an error.
            raise RuntimeError('rclone ' + str(args[0]) + ' failed; exit=' + str(result.returncode))
        return result.stdout if quiet else None

    def read_json(self, remote):
        return json.loads(self.call('cat', remote, quiet=True))

    def optional_json(self, remote):
        # An exact listing distinguishes absence from auth/network errors.
        parent, name = remote.rsplit('/', 1)
        listing = self.call('lsf', parent, '--files-only', '--max-depth', '1', quiet=True).splitlines()
        if name not in listing:
            return None
        return self.read_json(remote)

    def copy(self, source, destination, files_from=None):
        announce('COPY_START', source, '->', destination)
        args = ['copy', str(source), str(destination), '--transfers', '2', '--checkers', '4',
                '--stats', '30s', '--stats-log-level', 'NOTICE']
        if files_from:
            args += ['--files-from-raw', str(files_from)]
        self.call(*args)
        announce('COPY_COMPLETE', destination)

    def put_json(self, remote, value):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'value.json'
            write_json(file, value)
            self.call('copyto', str(file), remote)

    def initialize(self):
        parse_auth(Path(self.config).read_bytes(), self.remote)
        self.call('lsf', self.remote + ':', '--max-depth', '1', quiet=True)
        for folder in ('bootstrap', 'code', 'data/yc2', 'data/anet', 'runs/ddvc_qts', 'logs'):
            self.call('mkdir', self.path(folder))
        announce('DRIVE_CONNECTED', self.path())


def relative_path(value):
    p = PurePosixPath(value)
    if (p.is_absolute() or not p.parts or any(x in ('.', '..') for x in p.parts)
            or p.as_posix() != value or '\\' in value or ':' in value or any(ord(c) < 32 for c in value)):
        raise ValueError('Unsafe recovery path')
    return p


def inventory(folder):
    return {p.relative_to(folder).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(Path(folder).rglob('*')) if p.is_file() and not p.is_symlink()}


def fetch_verified(drive, remote, metadata, target, allowed=None):
    files = metadata['files']
    if not files:
        raise ValueError('Empty remote manifest')
    for name in files:
        relative_path(name)
        if allowed and not allowed(name):
            raise ValueError('Unexpected artifact in recovery manifest: ' + name)
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / 'files.txt'
        listing.write_text('\n'.join(files) + '\n', encoding='utf8')
        drive.copy(remote, target, listing)
    for name, expected in files.items():
        path = Path(target) / name
        if (not path.is_file() or path.is_symlink() or path.stat().st_size != expected['bytes']
                or digest(path) != expected['sha256']):
            raise ValueError('Recovery hash mismatch: ' + name)


def base_path(dataset):
    return WORK / 'prepared' / (dataset + '_qts_base_assets')


def base_feature(dataset):
    import yaml
    base = base_path(dataset)
    cfg = yaml.safe_load((base / 'config.yml').read_text(encoding='utf8'))
    files = cfg['visual_feature_folder']
    if len(files) != 1:
        raise ValueError('Expected exactly one cached feature file')
    path = Path(files[0])
    if not path.resolve().is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError('Cached feature path must be inside the base data bundle')
    return path


def data_identity():
    return {name: digest(ROOT / name) for name in ('tools/molab_pipeline.py',
            'tools/activitynet_pipeline.py', 'tools/kaggle_mask_recipe.json')}


def pull_data(drive, dataset):
    base = base_path(dataset)
    local_ready = (base / 'config.yml').is_file()
    remote = drive.path('data/' + dataset + '/' + DATA_PROTOCOL)
    drive.call('mkdir', remote)
    pointer = drive.optional_json(remote + '/LATEST.json')
    if pointer is None:
        if local_ready:
            announce('LOCAL_BASE_DATA_NEEDS_PUBLISH', base)
            return False
        # Optional manually supplied feature file avoids the public gdown quota.
        parent = drive.path('data/' + dataset)
        listing = drive.call('lsf', parent, '--files-only', '--max-depth', '1', quiet=True).splitlines()
        if 'clipvitl14.pth' in listing:
            dest = WORK / 'assets' / dataset / 'clipvitl14.pth'
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                partial = dest.with_name(dest.name + '.download')
                drive.call('copyto', parent + '/clipvitl14.pth', str(partial))
                os.replace(partial, dest)
            announce('OWN_DRIVE_FEATURE_READY', dest)
        announce('DATA_CACHE_EMPTY', dataset, 'original dataset downloader will be used if needed')
        return False
    generation = pointer['generation']
    if not re.fullmatch(r'[0-9a-f]{32}', generation):
        raise ValueError('Invalid data generation')
    source = remote + '/generations/' + generation
    meta = drive.read_json(source + '/READY.json')
    if (meta.get('protocol') != DATA_PROTOCOL or meta.get('dataset') != dataset
            or meta.get('work') != str(WORK) or meta.get('asset_code') != data_identity()):
        raise ValueError('Shared data cache has a different preparation recipe/root')
    if local_ready:
        announce('LOCAL_BASE_DATA_PRESENT_AND_REMOTE_CACHE_READY', base)
        return True
    if base.exists():
        raise FileExistsError('Incomplete local base data; inspect before retrying: ' + str(base))
    with tempfile.TemporaryDirectory(dir=WORK, prefix='.qts-data-') as tmp:
        allowed = lambda n: Path(n).suffix in {'.json', '.yml', '.yaml', '.npy', '.pth'}
        fetch_verified(drive, source, meta, Path(tmp) / 'base', allowed)
        base.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(Path(tmp) / 'base'), base)
    announce('DATA_CACHE_VERIFIED', dataset, len(meta['files']))
    return True


def publish_data(drive, dataset):
    base = base_path(dataset)
    files = inventory(base)
    if 'config.yml' not in files or not any(Path(p).name.startswith('clipvitl14') and p.endswith('.pth') for p in files):
        raise ValueError('Cannot publish an incomplete data bundle')
    if any(Path(p).suffix not in {'.json', '.yml', '.yaml', '.npy', '.pth'} for p in files):
        raise ValueError('Unexpected file in base data bundle')
    generation = uuid.uuid4().hex
    parent = drive.path('data/' + dataset + '/' + DATA_PROTOCOL)
    remote = parent + '/generations/' + generation
    drive.copy(base, remote)
    drive.call('check', str(base), remote, '--one-way')
    meta = dict(protocol=DATA_PROTOCOL, dataset=dataset, work=str(WORK),
                asset_code=data_identity(), files=files, generation=generation)
    drive.put_json(remote + '/READY.json', meta)
    drive.put_json(parent + '/LATEST.json', {'generation': generation})
    announce('DATA_CACHE_PUBLISHED', remote)


def recovery_allowed(name, dataset):
    p = relative_path(name)
    run = run_name(dataset)
    if p.parts[0] == 'prepared' and len(p.parts) > 2 and p.parts[1] == run:
        return p.suffix in {'.json', '.yml', '.yaml', '.npy'}
    if p.parts[0] == 'runs' and len(p.parts) > 2 and p.parts[1] == run:
        return ('backup' not in p.parts and 'tf_summary' not in p.parts and
                (p.suffix in TEXT or p.name in ('model-last.pth', 'model-best.pth')))
    if p.parts[0] == 'logs' and len(p.parts) == 2:
        return p.suffix == '.log' and p.name.startswith((run, dataset + '.drive'))
    if p.parts[0] == 'provenance':
        return p.suffix in TEXT | {'.py', '.cu', '.cpp', '.h'} and not p.name.startswith('drive_session_')
    return False


class Recovery:
    def __init__(self, drive, state):
        self.drive, self.state = drive, state
        self.remote = drive.path('runs/ddvc_qts/' + state['dataset'] + '/' + state['run_id'])
        self.signature = None
        self.lock = threading.Lock()

    def live_logs(self):
        with tempfile.TemporaryDirectory(dir=WORK, prefix='.qts-logs-') as tmp:
            for file in (WORK / 'logs').glob('*.log'):
                if recovery_allowed('logs/' + file.name, self.state['dataset']):
                    shutil.copy2(file, Path(tmp) / file.name)
            self.drive.copy(tmp, self.remote + '/live_logs')

    def sync(self, force=False):
        with self.lock:
            dataset = self.state['dataset']
            last = WORK / 'runs' / run_name(dataset) / 'model-last.pth'
            if not last.is_file():
                announce('BACKUP_WAITING_FOR_FIRST_CHECKPOINT', dataset)
                return False
            stat = last.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature == self.signature and not force:
                return False
            recovery = self.remote + '/recovery'
            self.drive.call('mkdir', recovery)
            pointer = self.drive.optional_json(recovery + '/LATEST.json')
            slot = 1 - pointer['slot'] if pointer else 0
            if slot not in (0, 1):
                raise ValueError('Invalid recovery slot')
            destination = recovery + '/slot-' + str(slot)
            announce('BACKUP_START', dataset, 'slot', slot)
            with tempfile.TemporaryDirectory(dir=WORK, prefix='.qts-recovery-') as tmp:
                staged = Path(tmp)
                candidates = [last]  # Pin the resume checkpoint before copying text/logs.
                for folder in ('prepared/' + run_name(dataset), 'runs/' + run_name(dataset), 'logs', 'provenance'):
                    candidates += sorted((WORK / folder).rglob('*'))
                for file in dict.fromkeys(candidates):
                    if not file.is_file() or file.is_symlink():
                        continue
                    relative = file.relative_to(WORK).as_posix()
                    if not recovery_allowed(relative, dataset):
                        continue
                    target = staged / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if file.name in ('model-last.pth', 'model-best.pth'):
                        try:
                            os.link(file, target)
                        except OSError:
                            shutil.copy2(file, target)
                    else:
                        # Logs may grow; copy an observed prefix, never link mutable text.
                        with file.open('rb') as source, target.open('wb') as dest:
                            remaining = os.fstat(source.fileno()).st_size
                            while remaining:
                                block = source.read(min(remaining, 4 * 1024**2))
                                if not block:
                                    break
                                dest.write(block)
                                remaining -= len(block)
                    if file.suffix == '.json':
                        json.loads(target.read_text(encoding='utf8'))
                files = inventory(staged)
                if not any(n.endswith('/model-last.pth') for n in files):
                    raise ValueError('No resume checkpoint staged')
                metadata = dict(format_version=1, created_ns=time.time_ns(), source_commit=revision(),
                                work=str(WORK), dataset=dataset, run_id=self.state['run_id'],
                                files=files, slot=slot)
                self.drive.copy(staged, destination)
                self.drive.call('check', str(staged), destination, '--one-way')
                self.drive.put_json(destination + '/READY.json', metadata)
                self.drive.put_json(recovery + '/LATEST.json', {'slot': slot, 'created_ns': metadata['created_ns']})
            self.signature = signature
            announce('BACKUP_VERIFIED', self.state['run_id'], 'slot', slot)
            return True

    def restore(self):
        recovery = self.remote + '/recovery'
        self.drive.call('lsf', recovery, '--max-depth', '1', quiet=True)
        candidates = []
        for slot in (0, 1):
            try:
                meta = self.drive.read_json(recovery + '/slot-' + str(slot) + '/READY.json')
                candidates.append((meta['created_ns'], slot, meta))
            except (RuntimeError, ValueError, KeyError):
                pass
        for _, slot, meta in sorted(candidates, reverse=True):
            if (meta.get('dataset') != self.state['dataset'] or meta.get('run_id') != self.state['run_id']
                    or meta.get('work') != str(WORK) or meta.get('source_commit') != revision()):
                raise ValueError('Restore needs the same source commit, dataset and work root')
            with tempfile.TemporaryDirectory(dir=WORK, prefix='.qts-restore-') as tmp:
                staged = Path(tmp) / 'payload'
                try:
                    fetch_verified(self.drive, recovery + '/slot-' + str(slot), meta, staged,
                                   lambda p: recovery_allowed(p, self.state['dataset']))
                    if not any(n.endswith('/model-last.pth') for n in meta['files']):
                        raise ValueError('Snapshot has no resume checkpoint')
                except (RuntimeError, ValueError, OSError):
                    announce('RESTORE_SLOT_INVALID', slot, 'trying older verified slot')
                    continue
                # Verify everything before making any run/prepared artifact visible.
                for relative in meta['files']:
                    target = WORK / relative
                    if not target.resolve().is_relative_to(WORK.resolve()):
                        raise ValueError('Symlink escapes work root')
                    if target.exists():
                        raise FileExistsError('Restore refuses to overwrite: ' + str(target))
                for relative in meta['files']:
                    target = WORK / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(staged / relative), target)
                announce('RESTORE_VERIFIED', self.state['run_id'], 'slot', slot)
                return meta
        raise FileNotFoundError('No complete, hash-valid recovery slot available')


def gpu_preflight():
    # Lazy import: auth/restore/status work even on a CPU-only preparation session.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This Molab session has no CUDA GPU. Select a GPU session before start.')
    announce('GPU_READY', torch.cuda.get_device_name(0), 'torch', torch.__version__, 'CUDA', torch.version.cuda)


def alive(pid):
    if not pid:
        return False
    try:
        command = Path('/proc') / str(int(pid)) / 'cmdline'
        return b'qts_rclone.py' in command.read_bytes() and b'worker' in command.read_bytes()
    except (OSError, ValueError):
        return False


def launch(args):
    gpu_preflight()
    state_file = session_path(args.dataset)
    name = run_name(args.dataset)
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if alive(state.get('worker_pid')):
            raise RuntimeError('Already running; use status/logs')
        if not args.resume and not args.retry:
            raise FileExistsError('Session already exists. Use --retry for failed setup/prepare or --resume for a checkpoint.')
    else:
        if args.resume or args.retry:
            raise FileNotFoundError('Restore the run first, or start a fresh run without flags')
        state = dict(dataset=args.dataset, run_id=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:12],
                     source_commit=revision(), work=str(WORK), remote=args.remote, root=args.root)
    if (state['source_commit'] != revision() or state['remote'] != args.remote or state['root'] != args.root):
        raise ValueError('Session source or Drive destination differs')
    run = WORK / 'runs' / name
    if args.resume:
        if not (run / 'model-last.pth').is_file():
            raise FileNotFoundError('Resume needs a restored model-last.pth')
    elif run.exists():
        raise FileExistsError('Fresh start refuses an existing run directory; use resume if checkpoint exists')
    state['status'] = 'STARTING'
    write_json(state_file, state)
    logs = WORK / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--config', str(args.config),
               '--remote', args.remote, '--root', args.root, 'worker', args.dataset]
    if args.resume:
        command += ['--resume']
    with (logs / (args.dataset + '.drive.log')).open('a') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    # Worker is the only writer after launch; separate PID avoids state races.
    (logs / (args.dataset + '.drive.pid')).write_text(str(process.pid))
    announce('STARTED', args.dataset, 'PID', process.pid, 'run_id', state['run_id'])
    announce('LOG', logs / (args.dataset + '.drive.log'))
    announce('DRIVE_RUN', args.remote + ':' + args.root + '/runs/ddvc_qts/' + args.dataset + '/' + state['run_id'])


def worker(args, drive):
    import fcntl
    lock_path = WORK / 'logs' / 'qts.drive.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A Drive worker already owns this dataset')
        state_file = session_path(args.dataset)
        state = json.loads(state_file.read_text())
        state.update(worker_pid=os.getpid(), status='SETTING_UP')
        write_json(state_file, state)
        backup = Recovery(drive, state)
        stop = threading.Event()
        def monitor():
            while not stop.wait(120):
                try:
                    backup.live_logs()
                    backup.sync()
                except Exception as exc:
                    announce('BACKUP_RETRY_PENDING', type(exc).__name__, str(exc))
        thread = None
        try:
            drive.initialize()
            gpu_preflight()
            (WORK / 'provenance/rclone_version.txt').write_text(drive.call('version', quiet=True), encoding='utf8')
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            launcher = str(ROOT / 'tools/qts_molab.sh')
            if not (WORK / 'provenance/cuda_ready').is_file():
                subprocess.run(['bash', launcher, 'setup'], cwd=ROOT, check=True)
            cached = pull_data(drive, args.dataset)
            if args.resume:
                # Restore feature bytes directly from the verified same-recipe base bundle.
                source = base_feature(args.dataset)
                target = WORK / 'prepared' / run_name(args.dataset) / 'data' / args.dataset / 'features/clipvitl14.pth'
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)
            else:
                subprocess.run(['bash', launcher, 'prepare', args.dataset], cwd=ROOT, check=True)
            if not cached:
                publish_data(drive, args.dataset)
            commit = revision()
            archive = WORK / 'provenance' / ('source_' + commit + '.zip')
            subprocess.run(['git', 'archive', '--format=zip', '--output', str(archive), 'HEAD'], cwd=ROOT, check=True)
            drive.call('copyto', str(archive), drive.path('code/' + commit + '/source.zip'))
            state['status'] = 'TRAINING'
            write_json(state_file, state)
            drive.put_json(backup.remote + '/RUN_MANIFEST.json', state)
            command = [sys.executable, '-u', 'tools/qts_pipeline.py', 'train', '--work', str(WORK),
                       '--dataset', args.dataset, '--variant', 'support']
            if args.resume:
                command += ['--resume']
            # Source the CUDA environment set up by the original launcher in a child shell.
            import shlex
            shell = 'set -e; export PYTHONPATH=' + shlex.quote(str(ROOT) + ':' + str(ROOT / 'ddvc/ops')) + ':"${PYTHONPATH:-}"; '
            shell += 'export HF_HOME=' + shlex.quote(str(WORK / 'hf-cache')) + '; '
            shell += 'export PATH=' + shlex.quote(str(WORK / 'tools/java/bin')) + ':"$PATH"; '
            shell += 'if [ -f ' + shlex.quote(str(WORK / 'cuda130_env.sh')) + ' ]; then source ' + shlex.quote(str(WORK / 'cuda130_env.sh')) + '; fi; '
            shell += 'exec ' + shlex.join(command)
            with (WORK / 'logs' / (run_name(args.dataset) + '.launch.log')).open('a') as stream:
                subprocess.run(['bash', '-c', shell], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            state['status'] = 'TRAIN_COMPLETE_BACKUP_PENDING'
            write_json(state_file, state)
        except BaseException:
            state['status'] = 'FAILED_BACKUP_PENDING'
            write_json(state_file, state)
            raise
        finally:
            stop.set()
            if thread:
                thread.join()  # Let an in-flight verified snapshot finish before final sync.
            try:
                backup.live_logs()
                saved = backup.sync(force=True)
                if saved and state['status'] == 'TRAIN_COMPLETE_BACKUP_PENDING':
                    state['status'] = 'COMPLETE_BACKED_UP'
                elif saved:
                    state['status'] = 'FAILED_CHECKPOINT_BACKED_UP'
                write_json(state_file, state)
                drive.put_json(backup.remote + '/RUN_MANIFEST.json', state)
                announce(state['status'], state['run_id'])
            except Exception as exc:
                announce('FINAL_BACKUP_PENDING', type(exc).__name__, str(exc), 'run backup manually')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=Path('/marimo/secrets/rclone.conf'))
    parser.add_argument('--remote', default='kltnbackup')
    parser.add_argument('--root', default='KLTNRUNS')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('auth')
    p.add_argument('--env', type=Path, default=Path('/marimo/KLTN_DRIVE_PRIVATE.env'))
    for name in ('start', 'worker', 'backup', 'restore', 'status', 'logs'):
        p = sub.add_parser(name)
        p.add_argument('dataset', choices=('yc2', 'anet'))
        if name in ('start', 'worker'):
            p.add_argument('--resume', action='store_true')
        if name == 'start':
            p.add_argument('--retry', action='store_true')
        if name == 'restore':
            p.add_argument('--run-id', required=True)
    args = parser.parse_args()
    drive = Drive(args.config, args.remote, args.root)
    if args.command == 'auth':
        restore_auth(args.env, args.config, args.remote)
        drive.initialize()
    elif args.command == 'start':
        parse_auth(args.config.read_bytes(), args.remote)
        launch(args)
    elif args.command == 'worker':
        worker(args, drive)
    elif args.command == 'logs':
        logs = WORK / 'logs'
        os.execvp('tail', ['tail', '-n', '50', '-F', str(logs / (args.dataset + '.drive.log')),
                  str(logs / (run_name(args.dataset) + '.launch.log')), str(logs / (run_name(args.dataset) + '.log'))])
    elif args.command == 'restore':
        if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
            raise ValueError('Invalid run ID')
        if session_path(args.dataset).exists():
            raise FileExistsError('Restore on a fresh work root; local session state already exists')
        state = dict(dataset=args.dataset, run_id=args.run_id, work=str(WORK), source_commit=revision(),
                     remote=args.remote, root=args.root, status='RESTORED')
        Recovery(drive, state).restore()
        write_json(session_path(args.dataset), state)
    else:
        state = json.loads(session_path(args.dataset).read_text())
        if state['remote'] != args.remote or state['root'] != args.root:
            raise ValueError('Use the Drive destination recorded in the session state')
        if args.command == 'backup':
            pid_file = WORK / 'logs' / (args.dataset + '.drive.pid')
            if pid_file.exists() and alive(pid_file.read_text().strip()):
                raise RuntimeError('Automatic backup worker is running; do not start a second writer')
            import fcntl
            with (WORK / 'logs/qts.drive.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError('Another worker/backup owns the work lock')
                backup = Recovery(drive, state)
                backup.live_logs()
                backup.sync(force=True)
        else:
            print(json.dumps(state, indent=2))
            announce('WORKER_ALIVE', alive(state.get('worker_pid')))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # No traceback locals/environment/credential values are logged by this wrapper.
        announce('STOPPED', type(error).__name__, str(error))
        sys.exit(1)
