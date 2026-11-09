"""Terminal runner for a separate, frozen-C1 reranker experiment."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from reranking.cache import atomic_json,sha
from reranking.pipeline import Workflow


def revision():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def announce(*x):print(time.strftime('%Y-%m-%d %H:%M:%S'),*x,flush=True)
def state_path(work,ds):return work/'provenance'/('rerank_'+ds+'.json')
def pid_path(work,ds):return work/'logs'/(ds+'.rerank.pid')


def worker_alive(work,ds):
    path=pid_path(work,ds)
    return alive(path.read_text().strip()) if path.is_file() else False


def drive(args):
    spec=importlib.util.spec_from_file_location('reranker_private_drive',ROOT/'tools/c1_rclone.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=Path('/marimo/secrets/rclone.conf')
    if not config.is_file():module.restore_auth(Path('/marimo/KLTN_DRIVE_PRIVATE.env'),config,'kltnbackup')
    module.parse_auth(config.read_bytes(),'kltnbackup')
    result=module.Drive(config,'kltnbackup','KLTNRUNS')
    result.call('lsf','kltnbackup:', '--max-depth','1',quiet=True)
    return result,module


def alive(pid):
    try:
        text=(Path('/proc')/str(int(pid))/'cmdline').read_bytes()
        return b'c1_rerank.py' in text and b'worker' in text
    except (OSError,TypeError,ValueError):return False


class Backup:
    def __init__(self,transport,helpers,work,state):
        self.drive,self.helpers,self.work,self.state=transport,helpers,work,state
        self.remote=transport.path('runs/ddvc_c1_rerank/'+state['dataset']+'/'+state['run_id'])
        self.lock=threading.Lock()

    def files(self):
        folder=self.work/'runs'/self.state['dataset'];files=[]
        if folder.exists():
            for p in folder.rglob('*'):
                if not p.is_file() or p.is_symlink() or p.name.endswith('.tmp'):continue
                if 'cache' in p.relative_to(folder).parts:
                    # Shard NPZ is committed first; JSON ready marker is last.
                    if p.suffix=='.json' and p.with_suffix('.npz').is_file():
                        meta=json.loads(p.read_text(encoding='utf8'))
                        if sha(p.with_suffix('.npz'))!=meta['sha256']:raise ValueError('Cache changed before backup')
                        files.extend([p,p.with_suffix('.npz')])
                elif p.suffix in ('.json','.jsonl','.tsv','.log','.pt'):files.append(p)
        files+=list((self.work/'logs').glob(self.state['dataset']+'.rerank.log'))
        files+=[state_path(self.work,self.state['dataset'])]
        return [p for p in dict.fromkeys(files) if p.is_file() and not p.is_symlink()]

    def sync(self):
        with self.lock,tempfile.TemporaryDirectory(dir=self.work,prefix='.rerank-backup-') as temporary:
            staged=Path(temporary)
            for p in self.files():
                relative=p.relative_to(self.work);target=staged/relative;target.parent.mkdir(parents=True,exist_ok=True)
                if p.suffix in ('.pt','.npz') or 'cache' in relative.parts:
                    try:os.link(p,target)
                    except OSError:shutil.copy2(p,target)
                else:
                    with p.open('rb') as source,target.open('wb') as output:
                        remaining=os.fstat(source.fileno()).st_size
                        while remaining:
                            block=source.read(min(remaining,4*1024**2))
                            if not block:break
                            output.write(block);remaining-=len(block)
            files=self.helpers.inventory(staged)
            parent=self.remote+'/recovery';self.drive.call('mkdir',parent)
            pointer=self.drive.optional_json(parent+'/LATEST.json');slot=1-pointer['slot'] if pointer else 0
            if slot not in (0,1):raise ValueError('Invalid recovery slot')
            destination=parent+'/slot-'+str(slot)
            meta=dict(files=files,source_commit=revision(),work=str(self.work),base_work=self.state['base_work'],
                      dataset=self.state['dataset'],run_id=self.state['run_id'],created_ns=time.time_ns(),slot=slot)
            self.drive.copy(staged,destination);self.drive.call('check',str(staged),destination,'--one-way')
            self.drive.put_json(destination+'/READY.json',meta)
            self.drive.put_json(parent+'/LATEST.json',dict(slot=slot,created_ns=meta['created_ns']))
            self.drive.put_json(self.remote+'/RUN_MANIFEST.json',self.state)
            announce('RERANK_BACKUP_VERIFIED',self.state['run_id'],'slot',slot)


def launch(args):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Use the existing GPU C1 session')
    Workflow(args.work,args.base,args.dataset)  # Catch missing/mismatched C1 before backgrounding.
    path=state_path(args.work,args.dataset)
    if path.is_file():
        state=json.loads(path.read_text(encoding='utf8'))
        if worker_alive(args.work,args.dataset):raise RuntimeError('Reranker is already running; use logs')
        if not args.resume:raise FileExistsError('Run already exists; use start DATASET --resume')
        if state['source_commit']!=revision() or state['base_work']!=str(args.base):raise ValueError('Resume source/base work changed')
    else:
        if args.resume:raise FileNotFoundError('Restore a reranker backup before resume')
        state=dict(dataset=args.dataset,work=str(args.work),base_work=str(args.base),source_commit=revision(),
                   run_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:12])
        base_state=args.base/'provenance'/('drive_session_'+args.dataset+'.json')
        if base_state.is_file():state['base_c1_drive_session']=json.loads(base_state.read_text(encoding='utf8'))
    transport,_=drive(args)
    parent=transport.path('runs/ddvc_c1_rerank/'+args.dataset+'/'+state['run_id']);transport.call('mkdir',parent)
    logs=args.work/'logs';logs.mkdir(parents=True,exist_ok=True)
    state['status']='STARTING';atomic_json(path,state)
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker',args.dataset,'--base',str(args.base),'--work',str(args.work)]
    with (logs/(args.dataset+'.rerank.log')).open('a',encoding='utf8') as f:
        proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    # Only the worker updates state after launch: avoid overwriting its progress.
    pid_path(args.work,args.dataset).write_text(str(proc.pid),encoding='utf8')
    announce('STARTED_FROZEN_C1_RERANK',proc.pid,'dataset',args.dataset)
    announce('LOG',logs/(args.dataset+'.rerank.log'));announce('DRIVE_RUN',parent)


def worker(args):
    import fcntl
    state=json.loads(state_path(args.work,args.dataset).read_text(encoding='utf8'))
    transport,helpers=drive(args);backup=Backup(transport,helpers,args.work,state)
    stop=threading.Event()
    def monitor():
        while not stop.wait(120):
            try:backup.sync()
            except Exception as error:announce('BACKUP_RETRY_PENDING',str(error))
    with (args.work/'logs/gpu.lock').open('a') as own_lock,(args.base/'logs/c1_gpu.lock').open('a') as base_lock:
        for lock in (own_lock,base_lock):
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('C1 training/probe or another reranker still owns the GPU')
        state['pid']=os.getpid();state['status']='EXTRACTING';atomic_json(state_path(args.work,args.dataset),state)
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        try:
            workflow=Workflow(args.work,args.base,args.dataset)
            code=args.work/'provenance'/('source_'+revision()+'.zip');code.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','archive','--format=zip','--output',str(code),'HEAD'],cwd=ROOT,check=True)
            transport.call('copyto',str(code),transport.path('code/'+revision()+'/source.zip'))
            workflow.extract()
            state['status']='TRAINING_HEAD_ONLY';atomic_json(state_path(args.work,args.dataset),state)
            workflow.fit()
            state['status']='EVALUATING_FROZEN_HEAD';atomic_json(state_path(args.work,args.dataset),state)
            workflow.evaluate();state['status']='COMPLETE_BACKUP_PENDING'
        except BaseException:
            state['status']='FAILED_BACKUP_PENDING';raise
        finally:
            stop.set();thread.join()
            atomic_json(state_path(args.work,args.dataset),state)
            try:
                backup.sync()
                state['status']='COMPLETE_BACKED_UP' if state['status']=='COMPLETE_BACKUP_PENDING' else 'FAILED_BACKED_UP'
                atomic_json(state_path(args.work,args.dataset),state)
                transport.put_json(backup.remote+'/RUN_MANIFEST.json',state)
                announce(state['status'],state['run_id'])
            except Exception as error:announce('FINAL_BACKUP_PENDING',str(error))


def restore(args):
    transport,helpers=drive(args)
    if not args.run_id or not re.fullmatch('[A-Za-z0-9_-]+',args.run_id):raise ValueError('Restore requires --run-id ID')
    if state_path(args.work,args.dataset).exists():raise FileExistsError('Restore into a fresh reranker work directory')
    remote=transport.path('runs/ddvc_c1_rerank/'+args.dataset+'/'+args.run_id+'/recovery')
    candidates=[]
    for slot in (0,1):
        try:
            meta=transport.read_json(remote+'/slot-'+str(slot)+'/READY.json');candidates.append((meta['created_ns'],slot,meta))
        except (RuntimeError,KeyError,ValueError):pass
    for _,slot,meta in sorted(candidates,reverse=True):
        if meta['source_commit']!=revision() or meta['work']!=str(args.work) or meta['base_work']!=str(args.base):
            raise ValueError('Restore needs the same reranker commit and base/work paths')
        if meta['dataset']!=args.dataset or meta['run_id']!=args.run_id:raise ValueError('Restore dataset/run mismatch')
        with tempfile.TemporaryDirectory(dir=args.work,prefix='.restore-') as temporary:
            staged=Path(temporary)/'files'
            def allowed(name):
                p=helpers.relative_path(name)
                return ((p.parts[:2]==('runs',args.dataset) and p.suffix in ('.pt','.npz','.json','.jsonl','.tsv','.log'))
                        or name=='logs/'+args.dataset+'.rerank.log' or name=='provenance/rerank_'+args.dataset+'.json')
            try:helpers.fetch_verified(transport,remote+'/slot-'+str(slot),meta,staged,allowed)
            except (RuntimeError,ValueError,OSError):continue
            for name in meta['files']:
                target=args.work/name
                if target.exists() or not target.resolve().is_relative_to(args.work):raise FileExistsError('Refusing restore overwrite/escape: '+str(target))
            for name in meta['files']:
                target=args.work/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(staged/name),target)
            announce('RERANK_RESTORE_VERIFIED',args.run_id,'slot',slot)
            return
    raise FileNotFoundError('No hash-valid reranker backup slot')


def export(args):
    target=args.work/('FROZEN_RERANK_ANALYSIS_'+args.dataset+'_'+time.strftime('%Y%m%d_%H%M%S')+'.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        folder=args.work/'runs'/args.dataset
        for p in folder.rglob('*'):
            if not p.is_file() or p.name.endswith('.tmp'):continue
            relative=p.relative_to(folder)
            if p.suffix in ('.json','.jsonl','.tsv','.log') or p.name=='head-best.pt' or (p.suffix=='.npz' and relative.parts[:2]==('cache','val')):
                z.write(p,str(p.relative_to(args.work)))
        for p in (args.work/'logs').glob(args.dataset+'.rerank.log'):z.write(p,str(p.relative_to(args.work)))
        z.write(state_path(args.work,args.dataset),'provenance/session.json')
        for name in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines():
            if Path(name).suffix in ('.py','.sh','.md','.json') and (ROOT/name).is_file():z.write(ROOT/name,'source/'+name)
        z.writestr('EXPORT_MANIFEST.json',json.dumps(dict(method='FROZEN_C1_JOINT_RERANKER',dataset=args.dataset,
                    source_commit=revision(),base_C1_weights_included=False,small_ranker_weights_included=True,
                    validation_feature_cache_included=True,training_feature_cache_included=False)))
    announce('ZIP_READY',target)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('command',choices=('start','worker','logs','status','backup','restore','export'))
    parser.add_argument('dataset',choices=('yc2','anet'));parser.add_argument('--work',type=Path,default=ROOT.parent)
    parser.add_argument('--base',type=Path,default=Path('/marimo/DDVC_C1'));parser.add_argument('--resume',action='store_true')
    parser.add_argument('--run-id');args=parser.parse_args();args.work=args.work.resolve();args.base=args.base.resolve()
    args.work.mkdir(parents=True,exist_ok=True)
    if args.command=='start':launch(args)
    elif args.command=='worker':worker(args)
    elif args.command=='restore':restore(args)
    elif args.command=='export':export(args)
    elif args.command=='logs':
        file=args.work/'logs'/(args.dataset+'.rerank.log')
        if not file.is_file():raise FileNotFoundError('Start the reranker first')
        os.execvp('tail',['tail','-n','60','-F',str(file)])
    else:
        state=json.loads(state_path(args.work,args.dataset).read_text(encoding='utf8'))
        if args.command=='status':print(json.dumps(dict(state,worker_alive=worker_alive(args.work,args.dataset)),indent=2))
        else:
            if worker_alive(args.work,args.dataset):raise RuntimeError('Automatic backup worker is running; do not start another writer')
            import fcntl
            with (args.work/'logs/gpu.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:raise RuntimeError('Reranker worker is active')
                transport,helpers=drive(args);Backup(transport,helpers,args.work,state).sync()


if __name__=='__main__':main()
