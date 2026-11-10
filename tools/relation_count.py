"""Fresh-session Molab workflow: restore cached C1, learn relation/count heads, evaluate paired ablations."""
import argparse
import json
import os
from pathlib import Path
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
from event_selection.storage import prepare,candidates,fetch,valid_id


def revision():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def say(*args):print(time.strftime('%Y-%m-%d %H:%M:%S'),*args,flush=True)
def state_file(work,ds):return work/'provenance'/('relation_count_'+ds+'.json')
def pid_file(work,ds):return work/'logs'/(ds+'.relation_count.pid')


def connection():
    import c1_rclone as helper
    conf=Path('/marimo/secrets/rclone.conf')
    if not conf.is_file():helper.restore_auth(Path('/marimo/KLTN_DRIVE_PRIVATE.env'),conf,'kltnbackup')
    helper.parse_auth(conf.read_bytes(),'kltnbackup')
    drive=helper.Drive(conf,'kltnbackup','KLTNRUNS');drive.call('lsf','kltnbackup:','--max-depth','1',quiet=True)
    return drive,helper


def alive(work,ds):
    try:
        pid=int(pid_file(work,ds).read_text().strip());command=(Path('/proc')/str(pid)/'cmdline').read_bytes()
        return b'relation_count.py' in command and b'worker' in command
    except (OSError,ValueError):return False


class Recovery:
    def __init__(self,drive,helper,work,state):
        self.drive,self.helper,self.work,self.state=drive,helper,work,state
        self.remote=drive.path('runs/ddvc_c1_relation_count/'+state['dataset']+'/'+state['run_id'])
        self.lock=threading.Lock()

    def sync(self):
        with self.lock,tempfile.TemporaryDirectory(dir=self.work,prefix='.relation-snapshot-') as temp:
            staged=Path(temp);ds=self.state['dataset'];folder=self.work/'runs'/ds
            files=[p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.tsv','.log','.pt')]
            files += [state_file(self.work,ds),self.work/'logs'/(ds+'.relation_count.log'),
                      self.work/'provenance'/('input_receipt_'+ds+'.json'),self.work/'provenance/native_runtime.json']
            for p in files:
                if not p.is_file() or p.is_symlink():continue
                dest=staged/p.relative_to(self.work);dest.parent.mkdir(parents=True,exist_ok=True)
                if p.suffix=='.pt':
                    try:os.link(p,dest)
                    except OSError:shutil.copy2(p,dest)
                else:
                    with p.open('rb') as src,dest.open('wb') as out:
                        left=os.fstat(src.fileno()).st_size
                        while left:
                            block=src.read(min(left,4*1024**2))
                            if not block:break
                            out.write(block);left-=len(block)
            inventory=self.helper.inventory(staged);parent=self.remote+'/recovery';self.drive.call('mkdir',parent)
            old=self.drive.optional_json(parent+'/LATEST.json');slot=1-old['slot'] if old else 0
            if slot not in (0,1):raise ValueError('Invalid backup slot')
            destination=parent+'/slot-'+str(slot)
            meta=dict(dataset=ds,run_id=self.state['run_id'],work=str(self.work),source_commit=revision(),
                      input_receipt_sha256=self.state['input_receipt_sha256'],files=inventory,slot=slot,created_ns=time.time_ns())
            self.drive.copy(staged,destination);self.drive.call('check',str(staged),destination,'--one-way')
            self.drive.put_json(destination+'/READY.json',meta);self.drive.put_json(parent+'/LATEST.json',dict(slot=slot,created_ns=meta['created_ns']))
            self.drive.put_json(self.remote+'/RUN_MANIFEST.json',self.state)
            say('RELATION_COUNT_BACKUP_VERIFIED',self.state['run_id'],'slot',slot)


def start(args):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Choose a GPU Molab session')
    if alive(args.work,args.dataset):raise RuntimeError('A relation/count worker is already running; use logs')
    receipt_path=args.work/'inputs'/args.dataset/'READY.json'
    if not receipt_path.is_file():raise FileNotFoundError('Run prepare '+args.dataset+' first')
    receipt=json.loads(receipt_path.read_text());path=state_file(args.work,args.dataset)
    if path.is_file():
        if not args.resume:raise FileExistsError('Existing experiment: use start DATASET --resume')
        state=json.loads(path.read_text())
        if state['source_commit']!=revision() or state['input_receipt_sha256']!=sha(receipt_path):raise ValueError('Resume source/input identity changed')
    else:
        if args.resume:raise FileNotFoundError('Restore a saved relation/count run before resume')
        state=dict(dataset=args.dataset,run_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:12],
                   work=str(args.work),source_commit=revision(),input_receipt_sha256=sha(receipt_path),
                   source_reranker_run=receipt['source_reranker_run'],status='STARTING')
    drive,_=connection();drive.call('mkdir',drive.path('runs/ddvc_c1_relation_count/'+args.dataset+'/'+state['run_id']))
    (args.work/'logs').mkdir(parents=True,exist_ok=True);atomic_json(path,state)
    with (args.work/'logs'/(args.dataset+'.relation_count.log')).open('a',encoding='utf8') as log:
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker',args.dataset,'--work',str(args.work)]
        proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pid_file(args.work,args.dataset).write_text(str(proc.pid))
    say('RELATION_COUNT_STARTED',proc.pid,'run',state['run_id']);say('LOG',args.work/'logs'/(args.dataset+'.relation_count.log'))


def worker(args):
    import fcntl
    from event_selection.pipeline import Experiment
    ds=args.dataset;path=state_file(args.work,ds);state=json.loads(path.read_text());drive,helper=connection()
    recovery=Recovery(drive,helper,args.work,state);stop=threading.Event()
    def monitor():
        while not stop.wait(120):
            try:recovery.sync()
            except Exception as error:say('BACKUP_RETRY_PENDING',str(error))
    with (args.work/'logs/gpu.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another relation/count worker owns the GPU')
        state.update(pid=os.getpid(),status='VERIFYING_CACHED_INPUTS');atomic_json(path,state)
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        try:
            archive=args.work/'provenance'/('source_'+revision()+'.zip')
            subprocess.run(['git','archive','--format=zip','--output',str(archive),'HEAD'],cwd=ROOT,check=True)
            drive.call('copyto',str(archive),drive.path('code/'+revision()+'/source.zip'))
            experiment=Experiment(args.work,ds);experiment.smoke();experiment.scorer_smoke()
            state['status']='TRAINING_RELATION_AND_COUNT_HEADS';atomic_json(path,state)
            experiment.fit();state['status']='EVALUATING_FROZEN_POLICIES';atomic_json(path,state)
            experiment.evaluate();state['status']='COMPLETE_BACKUP_PENDING'
        except BaseException:
            state['status']='FAILED_BACKUP_PENDING';raise
        finally:
            stop.set();thread.join();atomic_json(path,state)
            try:
                recovery.sync();state['status']='COMPLETE_BACKED_UP' if state['status']=='COMPLETE_BACKUP_PENDING' else 'FAILED_BACKED_UP'
                atomic_json(path,state);drive.put_json(recovery.remote+'/RUN_MANIFEST.json',state);say(state['status'],state['run_id'])
            except Exception as error:say('FINAL_BACKUP_PENDING',str(error))


def restore(args):
    drive,helper=connection();valid_id(args.run_id);path=state_file(args.work,args.dataset)
    if path.exists():raise FileExistsError('Restore needs a fresh work root without existing experiment state')
    remote=drive.path('runs/ddvc_c1_relation_count/'+args.dataset+'/'+args.run_id+'/recovery')
    for _,source,meta in candidates(drive,remote):
        if meta.get('source_commit')!=revision() or meta.get('work')!=str(args.work) or meta.get('dataset')!=args.dataset or meta.get('run_id')!=args.run_id:
            raise ValueError('Restore needs the same commit/work/dataset/run')
        with tempfile.TemporaryDirectory(dir=args.work,prefix='.relation-restore-') as temp:
            staged=Path(temp)
            def allowed(n):
                p=helper.relative_path(n)
                return ((p.parts[:2]==('runs',args.dataset) and p.suffix in ('.pt','.json','.tsv','.log'))
                        or n in ['provenance/relation_count_'+args.dataset+'.json','provenance/input_receipt_'+args.dataset+'.json',
                                 'provenance/native_runtime.json','logs/'+args.dataset+'.relation_count.log'])
            try:helper.fetch_verified(drive,source,meta,staged,allowed)
            except (RuntimeError,ValueError,OSError):continue
            receipt=json.loads((staged/('provenance/input_receipt_'+args.dataset+'.json')).read_text())
            prepare(args.work,args.dataset,drive,helper,receipt['source_reranker_run'])
            if sha(args.work/'inputs'/args.dataset/'READY.json')!=meta['input_receipt_sha256']:raise ValueError('Original frozen inputs differ')
            for name in meta['files']:
                dest=args.work/name
                if not dest.resolve().is_relative_to(args.work):raise ValueError('Recovery path escapes work root')
                if dest.exists() and name!='provenance/native_runtime.json':raise FileExistsError('Refusing to overwrite '+str(dest))
            for name in meta['files']:
                if name=='provenance/native_runtime.json':continue # Keep the NEW session runtime evidence.
                dest=args.work/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(staged/name),dest)
            say('RELATION_COUNT_RESTORED',args.run_id);return
    raise FileNotFoundError('No valid recovery snapshot')


def export(args):
    target=args.work/('RELATION_COUNT_ANALYSIS_'+args.dataset+'_'+time.strftime('%Y%m%d_%H%M%S')+'.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for folder in (args.work/'runs'/args.dataset,args.work/'logs',args.work/'provenance'):
            for p in folder.rglob('*'):
                if p.is_file() and p.suffix in ('.json','.tsv','.log','.pt') and (folder.name!='logs' or p.name.startswith(args.dataset)):
                    z.write(p,p.relative_to(args.work).as_posix())
        z.write(args.work/'inputs'/args.dataset/'READY.json','inputs/READY.json')
        for name in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines():
            if Path(name).suffix in ('.py','.sh','.md','.txt') and (ROOT/name).is_file():z.write(ROOT/name,'source/'+name)
        z.writestr('EXPORT_MANIFEST.json',json.dumps(dict(dataset=args.dataset,source_commit=revision(),
                    DDVC_weights_included=False,CLIP_features_included=False,derived_head_weights_included=True)))
    say('ZIP_READY',target)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=('prepare','start','worker','logs','status','backup','restore','export','auth'))
    parser.add_argument('dataset',choices=('yc2','anet'));parser.add_argument('--work',type=Path,default=ROOT.parent)
    parser.add_argument('--rank-run');parser.add_argument('--run-id');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();args.work=args.work.resolve();args.work.mkdir(parents=True,exist_ok=True)
    if args.command=='prepare':
        drive,helper=connection();receipt=prepare(args.work,args.dataset,drive,helper,args.rank_run)
        atomic_json(args.work/'provenance'/('input_receipt_'+args.dataset+'.json'),receipt)
    elif args.command=='start':start(args)
    elif args.command=='worker':worker(args)
    elif args.command=='restore':restore(args)
    elif args.command=='export':export(args)
    elif args.command=='auth':connection();say('DRIVE_CONNECTED_NO_CREDENTIALS_PRINTED')
    elif args.command=='logs':os.execvp('tail',['tail','-n','80','-F',str(args.work/'logs'/(args.dataset+'.relation_count.log'))])
    else:
        state=json.loads(state_file(args.work,args.dataset).read_text())
        if args.command=='status':print(json.dumps(dict(state,worker_alive=alive(args.work,args.dataset)),indent=2))
        else:
            if alive(args.work,args.dataset):raise RuntimeError('Worker already performs automatic backups')
            drive,helper=connection();Recovery(drive,helper,args.work,state).sync()


if __name__=='__main__':main()
