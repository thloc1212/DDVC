"""A PID is live training only if non-zombie argv matches train.py + config."""
import argparse
from pathlib import Path


def active_training_pid(pid_file,config,proc_root=Path('/proc')):
    pid_file=Path(pid_file)
    try:pid=int(pid_file.read_text().strip())
    except (FileNotFoundError,ValueError):return None
    if pid<=0:return None
    process=Path(proc_root)/str(pid)
    try:
        state=(process/'stat').read_text().rsplit(')',1)[1].split()[0]
        argv=[s.decode(errors='replace') for s in (process/'cmdline').read_bytes().split(b'\0') if s]
    except FileNotFoundError:return None
    except (PermissionError,IndexError) as exc:
        raise RuntimeError(f'Cannot safely inspect PID {pid}; inspect manually before relaunch') from exc
    if state in ('Z','X'):return None
    if not any(Path(arg).name=='train.py' for arg in argv):return None
    wanted=str(Path(config).resolve())
    for i,arg in enumerate(argv):
        if arg=='--cfg_path' and i+1<len(argv) and argv[i+1]==wanted:return pid
        if arg=='--cfg_path='+wanted:return pid
    return None


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--pid-file',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    pid=active_training_pid(args.pid_file,args.config)
    if pid is not None:
        print(f'TRAIN_ALREADY_RUNNING PID={pid} config={args.config}',flush=True)
        raise SystemExit(2)
    if args.pid_file.exists():
        args.pid_file.unlink()
        print('CLEARED_STALE_PID_FILE',args.pid_file,'(no process killed)',flush=True)


if __name__=='__main__':main()
