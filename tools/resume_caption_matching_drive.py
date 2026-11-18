"""Standalone Drive recovery for the immutable caption-matching v1 checkout.

Download this file outside source and run with native Molab Python. It clones
the fixed training release if absent, never pulls an existing training checkout,
and never starts a fresh generator.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

TRAIN_COMMIT='31f1eac41841f3a870bb56d9c74a52fc55057466'
TRAIN_TAG='prawn-caption-matching-20261117-v1'


def bootstrap_source(source):
    if (source/'tools/prawn_caption_match.sh').is_file():return
    if source.exists():
        raise FileExistsError('Source directory is incomplete; preserve/inspect it before restoring: '+str(source))
    source.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','-c','advice.detachedHead=false','clone','--single-branch','--branch',TRAIN_TAG,
                    'https://github.com/thloc1212/DDVC.git',str(source)],check=True)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def directories(drive,path):
    return drive.call('lsf',path,'--dirs-only','--max-depth','1',quiet=True).splitlines()


def snapshots(drive,parent,dataset,work,quality=False):
    result=[]
    for folder in sorted(directories(drive,parent),reverse=True):
        run_id=folder.rstrip('/')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id):continue
        remote=parent+'/'+run_id
        if 'recovery/' not in directories(drive,remote):continue
        recovery=remote+'/recovery'
        slots=directories(drive,recovery)
        for slot in ('slot-0','slot-1'):
            if slot+'/' not in slots:continue
            meta=drive.optional_json(recovery+'/'+slot+'/READY.json')
            if meta is None:continue
            expected_work=work/'frozen_quality' if quality else work
            if (meta.get('dataset')!=dataset or meta.get('run_id')!=run_id
                    or meta.get('source_commit')!=TRAIN_COMMIT or meta.get('work')!=str(expected_work)):
                continue
            if quality:
                if meta.get('base_work')!=str(work):continue
                state_name='provenance/rerank_'+dataset+'.json'
                if state_name not in meta.get('files',{}):continue
                state=drive.read_json(recovery+'/'+slot+'/'+state_name)
                base_id=state.get('base_c1_drive_session',{}).get('run_id')
                if not base_id:continue
            else:
                checkpoint='runs/'+dataset+'_caption_match_e'+('15' if dataset=='yc2' else '6')+'_b1/model-last.pth'
                if (meta.get('variant')!='prawn-caption-matching-b1'
                        or checkpoint not in meta.get('files',{})):continue
                base_id=run_id
            result.append(dict(run_id=run_id,base_id=base_id,created_ns=int(meta['created_ns']),slot=slot))
    return sorted(result,key=lambda r:(r['run_id'],r['created_ns']),reverse=True)


def choose(generator,quality,run_id=None):
    candidates=[r for r in generator if not run_id or r['run_id']==run_id]
    if not candidates:raise FileNotFoundError('No matching, checkpoint-bearing generator backup was found')
    selected=candidates[0]
    head=next((r for r in quality if r['base_id']==selected['run_id']),None)
    return selected,head


def preserve_conflicts(work,dataset,run):
    """Keep existing local run artifacts; never move source or delete a file."""
    work=work.resolve()
    saved=work/('.before_drive_restore_'+dataset+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    targets=[work/'prepared'/run,work/'runs'/run,work/'logs',work/'provenance',
             work/'frozen_quality/runs'/dataset,work/'frozen_quality/logs',work/'frozen_quality/provenance']
    moved=[]
    for path in targets:
        if not path.exists():continue
        resolved=path.resolve()
        if not resolved.is_relative_to(work) or resolved==work or resolved.is_relative_to(work/'source'):
            raise ValueError('Refusing a local path outside the experiment: '+str(path))
        if path.is_symlink():raise ValueError('Refusing symlink artifact: '+str(path))
        destination=saved/path.relative_to(work)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(path),str(destination));moved.append(str(path.relative_to(work)))
    if moved:print('LOCAL_ARTIFACTS_PRESERVED',saved,flush=True)
    return saved


def generator_complete(checkpoint,budget):
    return (checkpoint.get('format_version')==2 and checkpoint.get('phase')=='epoch_complete'
            and int(checkpoint.get('next_epoch',-1))>=budget)


def rank_alive(work,dataset):
    pid=work/'frozen_quality/logs'/(dataset+'.rerank.pid')
    if not pid.is_file():return False
    try:
        command=(Path('/proc')/pid.read_text().strip()/'cmdline').read_bytes()
        return b'c1_rerank.py' in command and b'worker' in command
    except OSError:return False


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('dataset',choices=('yc2','anet'))
    parser.add_argument('--source',type=Path,default=Path('/marimo/PRAWN_Caption_Match/source'))
    parser.add_argument('--run-id')
    parser.add_argument('--config',type=Path,default=Path('/marimo/secrets/rclone.conf'))
    parser.add_argument('--env',type=Path,default=Path('/marimo/KLTN_DRIVE_PRIVATE.env'))
    parser.add_argument('--no-tail',action='store_true')
    args=parser.parse_args();source=args.source.resolve();work=source.parent
    if not args.env.is_file():raise FileNotFoundError('Upload the private file first: '+str(args.env))
    if not shutil.which('rclone'):raise FileNotFoundError('rclone is required in this Molab terminal')
    bootstrap_source(source)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if revision!=TRAIN_COMMIT:raise ValueError('Keep the exact training checkout at '+TRAIN_COMMIT+'; helper does not change source')
    os.environ.update(LC_ALL='C',LANG='C',PYTHONUNBUFFERED='1',DDVC_PYTHON=sys.executable,DDVC_C1_WORK=str(work))
    sys.path.insert(0,str(source));sys.path.insert(0,str(source/'tools'))
    q=load('matching_recovery_drive',source/'tools/c1_rclone.py')
    chain=load('matching_recovery_chain',source/'tools/prawn_content_mask.py')
    training=load('matching_recovery_pid',source/'tools/training_pid.py')
    for ds in ('yc2','anet'):
        state=q.session_path(ds)
        local=json.loads(state.read_text()) if state.is_file() else {}
        active=training.active_training_pid(work/'logs'/(q.run_name(ds)+'.pid'),
            work/'prepared'/q.run_name(ds)/'config.yml')
        if chain.chain_alive(ds) or q.alive(local.get('worker_pid')) or rank_alive(work,ds) or active is not None:
            raise RuntimeError('An experiment worker is already active here; use its logs')
    q.gpu_preflight()
    q.restore_auth(args.env,args.config,'kltnbackup')
    drive=q.Drive(args.config,'kltnbackup','KLTNRUNS');drive.initialize()
    roots=directories(drive,drive.path('runs'))
    generators=snapshots(drive,drive.path('runs/prawn_caption_match_generator/'+args.dataset),args.dataset,work)
    quality=[]
    if 'prawn_caption_match_quality/' in roots:
        parent=drive.path('runs/prawn_caption_match_quality')
        if args.dataset+'/' in directories(drive,parent):
            quality=snapshots(drive,parent+'/'+args.dataset,args.dataset,work,quality=True)
    selected,head=choose(generators,quality,args.run_id)
    print('RESTORE_GENERATOR',args.dataset,selected['run_id'],flush=True)
    print('RESTORE_QUALITY',head['run_id'] if head else 'not started; will fit after generator finishes',flush=True)
    preserve_conflicts(work,args.dataset,q.run_name(args.dataset))
    state=dict(dataset=args.dataset,run_id=selected['run_id'],work=str(work),source_commit=TRAIN_COMMIT,
               remote='kltnbackup',root='KLTNRUNS',status='RESTORED',variant=q.RUN_VARIANT)
    q.Recovery(drive,state).restore()
    q.write_json(q.session_path(args.dataset),state)
    # New sessions need the native operator/dependencies rebuilt before the chain
    # imports the quality workflow. This never trains or changes model weights.
    subprocess.run(['bash',str(source/'tools/c1_molab.sh'),'setup',args.dataset],cwd=source,check=True)
    import torch
    path=work/'runs'/q.run_name(args.dataset)/'model-last.pth'
    checkpoint=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    budget=15 if args.dataset=='yc2' else 6
    complete=generator_complete(checkpoint,budget)
    print('RESTORED_CHECKPOINT',checkpoint.get('phase'),'next_epoch',checkpoint.get('next_epoch'),'budget',budget,flush=True)
    del checkpoint
    if head and not complete:
        raise ValueError('Quality backup exists but restored generator is incomplete; refusing to alter its cached generator')
    if complete:
        # Do not resume/rewrite a finished generator's YAML: its exact bytes are
        # part of the restored quality cache signature. Rebuild environment only.
        if not q.pull_data(drive,args.dataset):raise FileNotFoundError('Verified shared data cache missing; source/cache remains preserved')
        import c1_pipeline
        target=work/'prepared'/q.run_name(args.dataset)/'data'/args.dataset/'features/clipvitl14.pth'
        c1_pipeline.stage_file(q.base_feature(args.dataset),target)
        c1_pipeline.check(c1_pipeline.config_path(work,args.dataset))
        state['status']='COMPLETE_BACKED_UP';q.write_json(q.session_path(args.dataset),state)
    if head:
        rank=load('matching_recovery_rank',source/'tools/c1_rerank.py')
        rank.restore(SimpleNamespace(work=work/'frozen_quality',base=work,dataset=args.dataset,run_id=head['run_id']))
    command=['bash',str(source/'tools/prawn_caption_match.sh')]
    subprocess.run(command+['resume',args.dataset],cwd=source,check=True)
    if not args.no_tail:
        os.execvp('bash',command+['logs',args.dataset])


if __name__=='__main__':
    try:main()
    except Exception as error:
        print('RECOVERY_STOPPED',type(error).__name__,str(error),flush=True)
        sys.exit(1)
