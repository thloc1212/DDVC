"""Background coordinator: fresh baseline, verified backup, then frozen head."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent


def alive(pid,script):
    try:return script.encode() in (Path('/proc')/str(int(pid))/'cmdline').read_bytes()
    except (OSError,ValueError,TypeError):return False


def wait_stage(pid_file,state_file,script):
    pid=pid_file.read_text().strip()
    while alive(pid,script):time.sleep(5)
    state=json.loads(state_file.read_text(encoding='utf8'))
    if state.get('status')!='COMPLETE_BACKED_UP':
        raise RuntimeError('Stage did not finish with verified backup: '+str(state_file)+' status='+str(state.get('status')))


def worker(dataset):
    launcher=str(ROOT/'tools/baseline_quality.sh')
    subprocess.run(['bash',launcher,'start',dataset],cwd=ROOT,check=True)
    wait_stage(WORK/'logs'/(dataset+'.drive.pid'),WORK/'provenance'/('drive_session_'+dataset+'.json'),'c1_rclone.py')
    print('BASELINE_COMPLETE_STARTING_FROZEN_QUALITY',dataset,flush=True)
    subprocess.run(['bash',launcher,'quality',dataset],cwd=ROOT,check=True)
    rank=WORK/'frozen_quality'
    wait_stage(rank/'logs'/(dataset+'.rerank.pid'),rank/'provenance'/('rerank_'+dataset+'.json'),'c1_rerank.py')
    print('TWO_STAGE_COMPLETE_BACKED_UP',dataset,flush=True)
    print((rank/'runs'/dataset/'evaluation/summary.tsv').read_text(encoding='utf-8-sig'),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=('start','worker'))
    parser.add_argument('dataset',choices=('yc2','anet'))
    args=parser.parse_args()
    if os.environ.get('DDVC_RUN_VARIANT')!='baseline_quality':
        raise ValueError('Use baseline_quality.sh run DATASET')
    if args.command=='worker':worker(args.dataset);return
    logs=WORK/'logs';logs.mkdir(parents=True,exist_ok=True)
    pidfile=logs/(args.dataset+'.baseline_quality.chain.pid')
    if pidfile.is_file() and alive(pidfile.read_text().strip(),'baseline_quality_chain.py'):
        raise RuntimeError('Two-stage coordinator already running; use logs')
    if (WORK/'provenance'/('drive_session_'+args.dataset+'.json')).exists():
        raise FileExistsError('An existing run is preserved. Use status and stage-specific resume commands from the guide.')
    with (logs/(args.dataset+'.baseline_quality.chain.log')).open('a',encoding='utf8') as stream:
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker',args.dataset],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    pidfile.write_text(str(proc.pid),encoding='utf8')
    print('TWO_STAGE_STARTED',args.dataset,'coordinator PID',proc.pid,flush=True)


if __name__=='__main__':main()
