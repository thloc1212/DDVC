"""Seed-42 PRAWN generator then frozen quality training with verified Drive backups."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
WORK = ROOT.parent
QUALITY = WORK / 'frozen_quality'


def state_path(dataset, quality=False):
    if quality:
        return QUALITY / 'provenance' / ('rerank_' + dataset + '.json')
    return WORK / 'provenance' / ('drive_session_' + dataset + '.json')


def read_state(path):
    return json.loads(path.read_text(encoding='utf8')) if path.is_file() else {}


def command(script, *args):
    subprocess.run(['bash', str(ROOT / 'tools' / script), *args], cwd=ROOT, check=True)


def chain_alive(dataset):
    path = WORK / 'logs' / (dataset + '.prawn.pid')
    if not path.is_file():
        return False
    try:
        raw = (Path('/proc') / path.read_text().strip() / 'cmdline').read_bytes()
        return b'prawn_seed42.py' in raw and b'worker' in raw and dataset.encode() in raw.split(b'\0')
    except OSError:
        return False


def wait_stage(dataset, quality=False):
    import c1_rclone
    import c1_rerank
    path = state_path(dataset, quality)
    while True:
        state = read_state(path)
        if state.get('status') == 'COMPLETE_BACKED_UP':
            return
        if quality:
            running = c1_rerank.worker_alive(QUALITY, dataset)
        else:
            pid_file = WORK / 'logs' / (dataset + '.drive.pid')
            running = pid_file.is_file() and c1_rclone.alive(pid_file.read_text().strip())
        if not running:
            raise RuntimeError('Stage stopped. Inspect logs and state ' + str(path) + ' status=' + str(state.get('status')))
        time.sleep(10)


def worker(args):
    import fcntl
    with (WORK / 'logs' / (args.dataset + '.prawn.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('PRAWN_RECIPE', args.dataset, 'generator_seed=42 quality_seed=42 head_split_seed=777',
              'generator_epochs=' + ('15' if args.dataset == 'yc2' else '6'),
              'quality_epochs=20 batch=1 accumulation=1 beta=' + ('1' if args.dataset == 'yc2' else '6'), flush=True)
        state = read_state(state_path(args.dataset))
        if state.get('status') != 'COMPLETE_BACKED_UP':
            flags = []
            if args.resume and state:
                epochs = 15 if args.dataset == 'yc2' else 6
                checkpoint = WORK / 'runs' / f'{args.dataset}_c1_e{epochs}_b1' / 'model-last.pth'
                flags = ['--resume' if checkpoint.is_file() else '--retry']
            command('c1_drive.sh', 'start', args.dataset, *flags)
            wait_stage(args.dataset)
        state = read_state(state_path(args.dataset, quality=True))
        if state.get('status') != 'COMPLETE_BACKED_UP':
            flags = ['--resume'] if args.resume and state else []
            command('c1_rerank.sh', 'start', args.dataset, '--work', str(QUALITY), *flags)
            wait_stage(args.dataset, quality=True)
        print('PRAWN_TWO_STAGE_COMPLETE_BACKED_UP', args.dataset, 'seed=42', flush=True)
        print((QUALITY / 'runs' / args.dataset / 'evaluation/summary.tsv').read_text(encoding='utf-8-sig'), flush=True)


def start(args):
    if chain_alive(args.dataset):
        raise RuntimeError('PRAWN chain is already running. Use logs or status.')
    if not args.resume and any(state_path(args.dataset, quality=q).exists() for q in (False, True)):
        raise FileExistsError('This dataset already has state. Inspect it and use resume.')
    command('c1_drive.sh', 'auth')
    logs = WORK / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    launch = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', args.dataset]
    if args.resume:
        launch.append('--resume')
    with (logs / (args.dataset + '.prawn.chain.log')).open('a', encoding='utf8') as stream:
        proc = subprocess.Popen(launch, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    (logs / (args.dataset + '.prawn.pid')).write_text(str(proc.pid), encoding='utf8')
    print('PRAWN_STARTED', args.dataset, 'seed=42 PID', proc.pid, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('auth', 'start', 'worker', 'resume', 'logs', 'status', 'export'))
    parser.add_argument('dataset', choices=('yc2', 'anet'), nargs='?', default='yc2')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'auth':
        command('c1_drive.sh', 'auth')
    elif args.command in ('start', 'resume'):
        args.resume = args.resume or args.command == 'resume'
        start(args)
    elif args.command == 'worker':
        worker(args)
    elif args.command == 'logs':
        epochs = 15 if args.dataset == 'yc2' else 6
        run = f'{args.dataset}_c1_e{epochs}_b1'
        logs = [WORK/'logs'/(args.dataset+'.prawn.chain.log'),
                WORK/'logs'/(args.dataset+'.drive.log'),
                WORK/'logs'/(run+'.launch.log'), WORK/'logs'/(run+'.log'),
                QUALITY/'logs'/(args.dataset+'.rerank.log')]
        for path in logs:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        os.execvp('tail', ['tail', '-n', '30', '-F', *map(str, logs)])
    elif args.command == 'status':
        print(json.dumps(dict(chain_running=chain_alive(args.dataset),
            generator=read_state(state_path(args.dataset)),
            quality=read_state(state_path(args.dataset, quality=True))), indent=2))
    elif args.command == 'export':
        command('c1_drive.sh', 'export', args.dataset)
        command('c1_rerank.sh', 'export', args.dataset, '--work', str(QUALITY))


if __name__ == '__main__':
    main()
