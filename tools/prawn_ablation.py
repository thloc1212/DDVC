"""Separate Full-control, no-content, frozen-beta and Joint experiments for two accounts."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
WORK=ROOT.parent
PARTS=('no-content','quality','beta','joint')


def revision():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf8');os.replace(temporary,path)


def packages():
    command=[sys.executable,'-c','import json,torch,torchvision; print(json.dumps([torch.__version__,torchvision.__version__]))']
    before=subprocess.check_output(command,text=True)
    subprocess.run(['uv','pip','install','--python',sys.executable,'-r',str(ROOT/'tools/molab-requirements.txt')],check=True)
    if subprocess.check_output(command,text=True)!=before:raise RuntimeError('Native Torch/Torchvision changed')


def alive(path):
    try:
        raw=(Path('/proc')/Path(path).read_text().strip()/'cmdline').read_bytes()
        return b'prawn_ablation.py' in raw and b'worker' in raw
    except (OSError,ValueError):return False


def auth():
    from prawn_results_viewer import ensure_rclone
    ensure_rclone(WORK)
    import c1_rclone
    config=Path('/marimo/secrets/rclone.conf')
    c1_rclone.restore_auth(Path('/marimo/KLTN_DRIVE_PRIVATE.env'),config,'kltnbackup')
    drive=c1_rclone.Drive(config,'kltnbackup','KLTNRUNS')
    drive.call('lsf',drive.path(), '--max-depth','1',quiet=True)
    return drive


def child_source(part):
    name='no_content' if part=='no-content' else part
    work=WORK/('train_'+name);source=work/'source'
    if source.exists():
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
        if actual!=revision():raise ValueError('Existing ablation checkout differs; preserve it and use its fixed release')
        changed=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=source,text=True).splitlines()
        runtime={'data/video_dataset.py','ddvc/CaptioningHead/gpt2.py','transformers/dependency_versions_table.py',
                 'ddvc/ops/src/cuda/ms_deform_attn_cuda.cu','eval_utils_clip.py'}
        for name in changed:
            receipt=work/'provenance/runtime_sources'/name
            if (name not in runtime or not receipt.is_file() or
                    hashlib.sha256((source/name).read_bytes()).digest()!=hashlib.sha256(receipt.read_bytes()).digest()):
                raise ValueError('Ablation source drift outside verified runtime patches: '+name)
    else:
        work.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','worktree','add','--detach',str(source),revision()],cwd=ROOT,check=True)
    return source


def evaluation_setup():
    java=WORK/'tools/java'
    subprocess.run([sys.executable,str(ROOT/'tools/molab_pipeline.py'),'java','--target',str(java)],cwd=ROOT,check=True)
    os.environ['PATH']=str(java/'bin')+os.pathsep+os.environ.get('PATH','')


def exists_remote(drive,relative):
    parent=''
    for part in relative.split('/'):
        names=drive.call('lsf',drive.path(parent),'--dirs-only','--max-depth','1',quiet=True).splitlines()
        if part+'/' not in names:return False
        parent=parent+'/'+part if parent else part
    return True


def sync_output(drive,folder,remote):
    drive.call('copy',str(folder),drive.path(remote),'--transfers','2',quiet=True)
    drive.call('check',str(folder),drive.path(remote),'--one-way',quiet=True)
    print('ABLATION_OUTPUT_BACKUP_VERIFIED',drive.path(remote),flush=True)


def worker(args):
    import fcntl
    logs=WORK/'logs';logs.mkdir(exist_ok=True)
    state_path=WORK/'provenance'/f'{args.dataset}_{args.part}.json'
    with (logs/'ablation.gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        drive=auth()
        state=dict(dataset=args.dataset,part=args.part,source_commit=revision(),status='RUNNING',
                   caption_matching=True,mask_loss=0.,pid=os.getpid())
        atomic_json(state_path,state)
        try:
            packages()  # Install the locked legacy dependencies before importing scipy/yaml/model helpers.
            if args.part in ('no-content','joint'):
                source=child_source(args.part)
                env=dict(os.environ,PRAWN_ABLATION_VARIANT='no_content' if args.part=='no-content' else 'joint')
                env.update(DDVC_C1_WORK=str(source.parent),DDVC_HF_HOME=str(source.parent/'hf-cache'))
                env['PYTHONPATH']=os.pathsep.join(map(str,[source/'tools',source,source/'ddvc/ops']))+os.pathsep+os.environ.get('PYTHONPATH','')
                # Child stays attached; the verified generator/head workers own their snapshots.
                subprocess.run([sys.executable,'-u',str(source/'tools/prawn_ablation_train.py'),args.dataset],cwd=source,env=env,check=True)
                if args.part=='joint':
                    from misc.prawn_ablation import run_name
                    old=os.environ.get('PRAWN_ABLATION_VARIANT');os.environ['PRAWN_ABLATION_VARIANT']='joint'
                    try:folder=source.parent/'runs'/run_name(args.dataset)/'joint_evaluation'
                    finally:
                        if old is None:os.environ.pop('PRAWN_ABLATION_VARIANT',None)
                        else:os.environ['PRAWN_ABLATION_VARIANT']=old
                else:folder=source.parent/'frozen_quality/runs'/args.dataset/'evaluation'
                remote=f'analysis/prawn_loo/{args.dataset}/{args.part}/{revision()}'
                sync_output(drive,folder,remote)
                state['summary']=str(folder/'summary.tsv')
            else:
                from prawn_ablation_control import restore_control,quality_ablation,beta_ablation
                control=restore_control(drive,args.dataset,WORK/'control'/args.dataset)
                folder=WORK/'results'/args.dataset/args.part;folder.mkdir(parents=True,exist_ok=True)
                run=control['identity']['generator_run_id']
                remote=f'analysis/prawn_loo/{args.dataset}/full_{run}/{args.part}'
                if exists_remote(drive,remote):
                    drive.call('copy',drive.path(remote),str(folder),quiet=True)
                if args.part=='quality':quality_ablation(control,args.dataset,folder)
                else:
                    evaluation_setup()
                    beta_ablation(control,args.dataset,folder,lambda:sync_output(drive,folder,remote))
                sync_output(drive,folder,remote)
                state.update(summary=str(folder/'summary.tsv'),control=control['identity'])
            state['status']='COMPLETE_BACKED_UP'
            print('ABLATION_COMPLETE_BACKED_UP',args.dataset,args.part,state['summary'],flush=True)
        except BaseException:
            state['status']='FAILED';raise
        finally:atomic_json(state_path,state)


def launch(args):
    logs=WORK/'logs';logs.mkdir(parents=True,exist_ok=True)
    pid=logs/(args.dataset+'.active.pid')
    if alive(pid):raise RuntimeError('Another experiment is running for this dataset. Use logs/status; wait for completion.')
    auth()
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker',args.dataset,'--part',args.part]
    log=logs/f'{args.dataset}.{args.part}.log'
    with log.open('a',encoding='utf8') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL,start_new_session=True)
    pid.write_text(str(process.pid),encoding='utf8')
    print('ABLATION_STARTED',args.dataset,args.part,'PID',process.pid,'LOG',log,flush=True)


def export(args):
    stamp=time.strftime('%Y%m%d_%H%M%S')
    target=WORK/f'PRAWN_LOO_ANALYSIS_{args.dataset}_{stamp}.zip'
    report=WORK/'results'/args.dataset/'ablation_summary.tsv'
    report.parent.mkdir(parents=True,exist_ok=True)
    combined=[]
    inputs=[(WORK/'results'/args.dataset/'quality/summary.tsv',None),
            (WORK/'results'/args.dataset/'beta/summary.tsv','beta'),
            (WORK/'train_no_content/frozen_quality/runs'/args.dataset/'evaluation/summary.tsv','no-content')]
    epochs=15 if args.dataset=='yc2' else 6
    inputs.append((WORK/'train_joint/runs'/f'{args.dataset}_loo_joint_e{epochs}_b1'/'joint_evaluation/summary.tsv','joint'))
    for path,part in inputs:
        if not path.is_file():continue
        for row in csv.DictReader(path.open(encoding='utf-8-sig'),delimiter='\t'):
            if part=='no-content' and row['method']!='reranker_softnms':continue
            if part=='no-content':row['method']='w_o_content'
            if part=='joint':row['method']='Joint'
            row.update(dataset=args.dataset,caption_matching=True,content_conditioned=part!='no-content',
                       quality_enabled=row['method']!='w_o_quality',mask_loss=0.,
                       regime='joint' if part=='joint' else 'two-stage')
            combined.append(row)
    if combined:
        with report.open('w',encoding='utf-8-sig',newline='') as stream:
            columns=list(dict.fromkeys(k for row in combined for k in row))
            writer=csv.DictWriter(stream,columns,delimiter='\t');writer.writeheader();writer.writerows(combined)
        print('COMBINED_SUMMARY',report,flush=True)
    roots=[WORK/'results'/args.dataset,WORK/'control'/args.dataset/'control_identity.json',
           WORK/'provenance',WORK/'logs']
    for part in ('no_content','joint'):
        base=WORK/('train_'+part)
        roots += [base/'runs',base/'prepared',base/'logs',base/'provenance',base/'frozen_quality/runs'/args.dataset/'evaluation',
                  base/'frozen_quality/runs'/args.dataset/'manifest.json',base/'frozen_quality/logs']
    allowed={'.json','.jsonl','.yml','.yaml','.tsv','.csv','.log','.txt'}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        seen=set()
        for root in roots:
            for p in ([root] if root.is_file() else root.rglob('*') if root.is_dir() else []):
                if p.is_file() and p.suffix in allowed and not {'backup','cache','secrets'}.intersection(p.relative_to(WORK).parts):
                    name=p.relative_to(WORK).as_posix()
                    if name not in seen:z.write(p,name);seen.add(name)
        for name in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines():
            if Path(name).suffix in {'.py','.sh','.md','.json'}:z.write(ROOT/name,'source/'+name)
        z.writestr('EXPERIMENTS.json',json.dumps(dict(dataset=args.dataset,source_commit=revision(),caption_matching=True,
                 auxiliary_mask_loss=False,variants=['Full','w/o Content','w/o Quality','beta grid','Joint'],
                 training_checkpoint_weights_included=False)))
    drive=auth();remote=f'analysis/prawn_loo/{args.dataset}/exports/{target.name}'
    drive.call('copyto',str(target),drive.path(remote),quiet=True)
    print('ZIP_READY',target,flush=True);print('ZIP_ON_DRIVE',drive.path(remote),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=PARTS+('worker','logs','status','export','auth'))
    parser.add_argument('dataset',choices=('yc2','anet'),nargs='?',default='yc2')
    parser.add_argument('--part',choices=PARTS,default='no-content')
    args=parser.parse_args()
    os.environ['PRAWN_ABLATION_VARIANT']='full'
    if args.command in PARTS:args.part=args.command;launch(args)
    elif args.command=='worker':worker(args)
    elif args.command=='auth':auth()
    elif args.command=='export':export(args)
    elif args.command=='status':
        for part in PARTS:
            p=WORK/'provenance'/f'{args.dataset}_{part}.json'
            print(part,p.read_text(encoding='utf8') if p.is_file() else 'NOT_STARTED')
        print('WORKER_ALIVE',alive(WORK/'logs'/(args.dataset+'.active.pid')))
    else:
        paths=[WORK/'logs'/f'{args.dataset}.{args.part}.log']
        if args.part in ('no-content','joint'):
            v='no_content' if args.part=='no-content' else args.part
            base=WORK/('train_'+v)
            name=f'{args.dataset}_loo_{v}_e{15 if args.dataset=="yc2" else 6}_b1'
            paths += [base/'logs'/(args.dataset+'.drive.log'),base/'logs'/(name+'.launch.log'),base/'logs'/(name+'.log')]
            if args.part=='no-content':paths += [base/'frozen_quality/logs'/(args.dataset+'.rerank.log')]
        os.execvp('tail',['tail','-n','25','-F',*map(str,paths)])


if __name__=='__main__':main()
