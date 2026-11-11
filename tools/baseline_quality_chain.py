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
SETUP_FIX_SOURCES={'b54f081b648b89a7887fb5df651985ef4a0cf9ce',
                   '80c29717b43268423c30150f01f999f7a43eb647'}


def alive(pid,script):
    try:return script.encode() in (Path('/proc')/str(int(pid))/'cmdline').read_bytes()
    except (OSError,ValueError,TypeError):return False


def wait_stage(pid_file,state_file,script):
    pid=pid_file.read_text().strip()
    while alive(pid,script):time.sleep(5)
    state=json.loads(state_file.read_text(encoding='utf8'))
    if state.get('status')!='COMPLETE_BACKED_UP':
        raise RuntimeError('Stage did not finish with verified backup: '+str(state_file)+' status='+str(state.get('status')))


def prepare_setup_retry(dataset,current):
    """Migrate only a failed, pre-training setup; preserve its original evidence."""
    path=WORK/'provenance'/('drive_session_'+dataset+'.json')
    state=json.loads(path.read_text(encoding='utf8'))
    if (state.get('dataset')!=dataset or Path(state.get('work','')).resolve()!=WORK.resolve()
            or state.get('variant')!='baseline_quality'):
        raise ValueError('Setup retry dataset/work/variant mismatch')
    if state.get('source_commit') not in {current,*SETUP_FIX_SOURCES}:
        raise ValueError('Setup retry source is not an approved setup-only fix')
    name=f'{dataset}_baseline_e{15 if dataset=="yc2" else 6}_b1'
    if (state.get('status')!='FAILED_BACKUP_PENDING' or (WORK/'runs'/name).exists()
            or (WORK/'frozen_quality/provenance'/('rerank_'+dataset+'.json')).exists()):
        raise ValueError('Setup/data retry requires a failed worker and no training run or reranker. Existing training is preserved.')
    if alive(state.get('worker_pid'),'c1_rclone.py'):
        raise RuntimeError('Baseline worker is still running; inspect logs first')
    archive=WORK/'provenance/setup_retry_history'/f'{dataset}_{time.time_ns()}.json'
    archive.parent.mkdir(parents=True,exist_ok=True)
    archive.write_text(json.dumps(state,indent=2),encoding='utf8')
    state['source_history']=list(state.get('source_history',[]))+[
        dict(source_commit=state['source_commit'],reason='setup/import-only fix; no training run or checkpoint existed')]
    state.update(source_commit=current,status='SETUP_RETRY_READY')
    temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(state,indent=2),encoding='utf8')
    os.replace(temporary,path)
    print('SETUP_RETRY_READY',dataset,'previous state preserved at',archive,flush=True)


def worker(dataset,retry=False):
    launcher=str(ROOT/'tools/baseline_quality.sh')
    command=['bash',launcher,'start',dataset]
    if retry:command.append('--retry')
    subprocess.run(command,cwd=ROOT,check=True)
    wait_stage(WORK/'logs'/(dataset+'.drive.pid'),WORK/'provenance'/('drive_session_'+dataset+'.json'),'c1_rclone.py')
    print('BASELINE_COMPLETE_STARTING_FROZEN_QUALITY',dataset,flush=True)
    subprocess.run(['bash',launcher,'quality',dataset],cwd=ROOT,check=True)
    rank=WORK/'frozen_quality'
    wait_stage(rank/'logs'/(dataset+'.rerank.pid'),rank/'provenance'/('rerank_'+dataset+'.json'),'c1_rerank.py')
    print('TWO_STAGE_COMPLETE_BACKED_UP',dataset,flush=True)
    print((rank/'runs'/dataset/'evaluation/summary.tsv').read_text(encoding='utf-8-sig'),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=('start','retry','worker'))
    parser.add_argument('dataset',choices=('yc2','anet'))
    parser.add_argument('--retry-setup',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if os.environ.get('DDVC_RUN_VARIANT')!='baseline_quality':
        raise ValueError('Use baseline_quality.sh run DATASET')
    if args.command=='worker':worker(args.dataset,retry=args.retry_setup);return
    logs=WORK/'logs';logs.mkdir(parents=True,exist_ok=True)
    pidfile=logs/(args.dataset+'.baseline_quality.chain.pid')
    if pidfile.is_file() and alive(pidfile.read_text().strip(),'baseline_quality_chain.py'):
        raise RuntimeError('Two-stage coordinator already running; use logs')
    if args.command=='retry':
        current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        prepare_setup_retry(args.dataset,current)
    elif (WORK/'provenance'/('drive_session_'+args.dataset+'.json')).exists():
        raise FileExistsError('An existing run is preserved. Use status and stage-specific resume commands from the guide.')
    with (logs/(args.dataset+'.baseline_quality.chain.log')).open('a',encoding='utf8') as stream:
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker',args.dataset]
        if args.command=='retry':command.append('--retry-setup')
        proc=subprocess.Popen(command,
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    pidfile.write_text(str(proc.pid),encoding='utf8')
    print('TWO_STAGE_STARTED',args.dataset,'coordinator PID',proc.pid,flush=True)


if __name__=='__main__':main()
