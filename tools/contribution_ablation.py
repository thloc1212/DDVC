"""Controlled V1/V2 contribution study. Full splits; Soft-NMS only."""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = {
    'C0': ('global', False, False, False),
    'C1': ('learned', False, False, False),
    'C2': ('learned', True, False, False),
    'C3': ('learned', True, True, False),
    'C4': ('learned', True, True, True),
    'C5': ('learned', False, True, True),
}
METRICS = ('CIDEr', 'METEOR', 'Bleu_4', 'soda_c', 'Precision', 'Recall', 'MetaScore')


def helper(name):
    spec = importlib.util.spec_from_file_location('contribution_'+name, ROOT/'tools'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf8')


def epochs(dataset):
    return 15 if dataset == 'yc2' else 8


def run_name(dataset, case):
    return f'{dataset}_contribution_{case.lower()}_e{epochs(dataset)}_b1'


def config_path(work, dataset, case):
    return Path(work)/'prepared'/run_name(dataset, case)/'config.yml'


def make_config(base, work, dataset, case):
    context, mask_loss, concepts, prefix = CASES[case]
    cfg = copy.deepcopy(base)
    name = run_name(dataset, case)
    cfg.update(id=name, exp_name=name, fixed_run_id=True, epoch=epochs(dataset),
               cfg_path='', save_dir=str(Path(work)/'runs'), start_from='',
               start_from_mode='last', pretrain=None, only_ft_captioner=False,
               only_ft_class_head=False, training_scheme='all',
               learned_proposal_mask=context=='learned', local_visual_attention=False,
               proposal_mask_loss_coef=1.0 if mask_loss else 0.0,
               event_concepts=concepts, concept_condition_prefix=prefix,
               concept_loss_coef=.1, dedup_loss_coef=0.,
               eval_selection_method='softnms', eval_softnms_sigma=.5,
               eval_merge_duplicates=False, eval_merge_consecutive=False,
               min_epoch_when_save=0, save_checkpoint_every=1,
               experiment_recipe='v1_v2_contribution_'+case.lower(),
               selection_experiment='softnms', contribution_case=case)
    cfg.pop('current_lr', None)
    return cfg


def prepare(args):
    work = args.work.resolve()
    concept = helper('concept_pipeline')
    base_name = concept.run_name(args.dataset, 'control', epochs(args.dataset))
    base_path = work/'prepared'/base_name/'config.yml'
    source = args.source_config
    if source:
        source = source.resolve()
        original = yaml.safe_load(source.read_text(encoding='utf8'))
        if original.get('attention_counter') or original.get('duplicate_head'):
            raise ValueError('Do not reuse counter/duplicate-head experiment configs for V1/V2 ablation')
        # Reuse complete concept-run assets; reset only concept method before
        # asking the established V1 preparation code to validate/stage them.
        if original.get('event_concepts'):
            original.update(event_concepts=False, concept_condition_prefix=False)
            source = work/'prepared/shared_assets_source.yml'
            if source.exists() and yaml.safe_load(source.read_text()) != original:
                raise FileExistsError('Another shared source config is already registered')
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(yaml.safe_dump(original, sort_keys=False), encoding='utf8')
    concept.prepare(argparse.Namespace(work=work, dataset=args.dataset, variant='control',
                                      source_config=source, epochs=epochs(args.dataset)))
    base = concept.check(base_path)
    for key in ('bi_interaction', 'semantic_matching', 'narrative_memory', 'attention_counter', 'duplicate_head'):
        if base.get(key):
            raise ValueError(f'Unrelated module must be disabled: {key}')
    hashes = {key:sha(base[key]) for key in ('train_caption_file','val_caption_file','dict_file','concept_vocab_path')}
    shared_files = set(base['gt_file_for_eval'] + base['gt_file_for_para_eval'])
    for folder in {base['bank_folder'],'clip'}:
        shared_files.update(str(p) for p in (Path(base['bank_path'])/args.dataset/folder).glob('*.npy'))
    shared_hashes = {str(p):sha(p) for p in sorted(shared_files)}
    for case in args.cases:
        path = config_path(work, args.dataset, case)
        if path.exists():
            check(path)
            continue
        cfg = make_config(base, work, args.dataset, case)
        if (work/'runs'/cfg['id']).exists():
            raise FileExistsError('Existing output without registered config: '+cfg['id'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf8')
        manifest = dict(expected_config=cfg, base_config=str(base_path), base_sha256=sha(base_path),
                        source_commit=concept.load_helper('molab_pipeline').revision(),
                        input_sha256=hashes, shared_file_sha256=shared_hashes, dataset=args.dataset, case=case,
                        note='Only the specified mask/concept factors differ; Soft-NMS only; fresh weights.')
        dump(path.parent/'contribution_manifest.json', manifest)
        concept.make_smoke(cfg, path.parent, work)
        check(path)
        print('CONFIG_READY', path, flush=True)
    dump(work/'prepared'/f'{args.dataset}_contribution_plan.json',
         dict(dataset=args.dataset, epochs=epochs(args.dataset), cases=CASES,
              input_sha256=hashes, base_config=str(base_path)))


def check(path, require_features=True):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding='utf8'))
    manifest = json.loads((path.parent/'contribution_manifest.json').read_text(encoding='utf8'))
    expected = manifest['expected_config']
    permitted = {'start_from','start_from_mode','current_lr'}
    if {k:v for k,v in cfg.items() if k not in permitted} != {k:v for k,v in expected.items() if k not in permitted}:
        raise ValueError('Contribution recipe drift: '+str(path))
    dataset, case = manifest['dataset'], manifest['case']
    expected_method = make_config(expected, Path(cfg['save_dir']).parent, dataset, case)
    for key in ('learned_proposal_mask','local_visual_attention','proposal_mask_loss_coef',
                'event_concepts','concept_condition_prefix','concept_loss_coef','dedup_loss_coef',
                'eval_selection_method','eval_softnms_sigma','eval_merge_duplicates','eval_merge_consecutive'):
        if cfg[key] != expected_method[key]:
            raise ValueError('Incorrect matrix factor '+key)
    assert cfg['seed']==777 and cfg['batch_size']==1 and cfg['gradient_accumulation_steps']==1
    assert cfg['epoch']==epochs(dataset) and cfg['num_queries']==(50 if dataset=='yc2' else 10)
    assert cfg['precision']=='fp32' and not cfg['allow_tf32'] and cfg['able_ret']
    assert cfg['lr']==5e-5 and cfg['weight_decay']==1e-4
    for key, digest in manifest['input_sha256'].items():
        if sha(cfg[key]) != digest:raise ValueError('Input bytes changed: '+key)
    for shared_path, digest in manifest.get('shared_file_sha256',{}).items():
        if sha(shared_path) != digest:raise ValueError('Reference/bank bytes changed: '+shared_path)
    train = json.loads(Path(cfg['train_caption_file']).read_text(encoding='utf8'))
    val = json.loads(Path(cfg['val_caption_file']).read_text(encoding='utf8'))
    if set(train)&set(val):raise ValueError('Train/validation overlap')
    expected_counts = (1237,436) if dataset=='yc2' else (10009,4917)
    if (len(train),len(val)) != expected_counts:raise ValueError('Wrong full-data split size')
    references = [json.loads(Path(p).read_text(encoding='utf8')) for p in cfg['gt_file_for_eval']]
    if set().union(*(set(r) for r in references)) != set(val):raise ValueError('Reference IDs differ')
    for p in cfg['gt_file_for_para_eval'] + (cfg['visual_feature_folder'] if require_features else []):
        if not Path(p).is_file():raise FileNotFoundError(p)
    for folder in {cfg['bank_folder'],'clip'}:
        for suffix in ('scene_sentences','clip_token_embeds','scene_videoID'):
            p=Path(cfg['bank_path'])/dataset/folder/f'{dataset}_{suffix}.npy'
            if not p.is_file():raise FileNotFoundError(p)
    print('CONTRIBUTION_OK',dataset,case,'Soft-NMS ONLY; epochs',cfg['epoch'],flush=True)
    return cfg


def logged(command, log, pid_file=None):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    print('COMMAND_LOG',log,flush=True)
    with Path(log).open('a', encoding='utf8') as stream:
        if pid_file is None:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        else:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            Path(pid_file).write_text(str(process.pid),encoding='utf8')
            rc=process.wait()
            if rc:raise subprocess.CalledProcessError(rc,command)


def fingerprint(path):
    # Include the actual training/evaluation code, not only launcher/config bytes.
    sources=[path]+[ROOT/p for p in ('train.py','opts.py','eval_utils_clip.py',
        'ddvc/ddvc_ret_encdec_clip_capQ.py','ddvc/CaptioningHead/gpt2.py',
        'ddvc/CaptioningHead/event_concepts.py','ddvc/CaptioningHead/proposal_mask.py',
        'misc/proposal_selection.py','misc/gradient_guard.py','transformers/caption_targets.py',
        'ddvc/temporal_pooling.py','tools/contribution_ablation.py')]
    return {str(p):sha(p) for p in sources}


def smoke(work, dataset, case, checkpoint=False):
    path=config_path(work,dataset,case)
    cfg=check(path)
    command=[sys.executable,'-u','tools/molab_pipeline.py','smoke','--config',str(path if checkpoint else path.parent/'smoke/config.yml')]
    if checkpoint:command+=['--checkpoint',str(work/'runs'/cfg['id']/'model-last.pth')]
    logged(command,work/'logs'/(cfg['id']+'.smoke.log'))
    dump(work/'provenance'/(cfg['id']+'_contribution_smoke.json'),fingerprint(path))


def completed(work, cfg):
    run=work/'runs'/cfg['id']
    info_path=run/'info.json'
    if not info_path.is_file():return False
    info=json.loads(info_path.read_text(encoding='utf8'))
    previous=info.get('last',{}).get('opt',{})
    for key in ('contribution_case','data_signature','epoch','eval_selection_method',
                'eval_merge_duplicates','eval_merge_consecutive','event_concepts','concept_condition_prefix',
                'learned_proposal_mask','proposal_mask_loss_coef'):
        if previous.get(key)!=cfg.get(key):raise ValueError('Existing run differs: '+key)
    return info.get('last',{}).get('epoch',-1)>=cfg['epoch']-1 and (run/'model-last.pth').is_file()


def queue(args):
    # Linux kernel releases this lock on session/process loss; no stale PID lock.
    import fcntl
    work=args.work.resolve()
    (work/'logs').mkdir(parents=True,exist_ok=True)
    with (work/'logs/contribution_gpu.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another contribution queue holds this work/GPU lock')
        pid_guard=helper('training_pid')
        for dataset in ('yc2','anet'):
            for case in CASES:
                path=config_path(work,dataset,case)
                pid=pid_guard.active_training_pid(work/'logs'/(run_name(dataset,case)+'.pid'),path)
                if pid is not None:raise RuntimeError(f'Training PID {pid} is already active; inspect before relaunch')
        print('QUEUE_PREFLIGHT',args.dataset,args.cases,flush=True)
        for case in args.cases:
            check(config_path(work,args.dataset,case))
        for case in args.cases:
            path=config_path(work,args.dataset,case)
            cfg=check(path)
            if completed(work,cfg):
                print('SKIP_COMPLETED',cfg['id'],flush=True)
                continue
            run=work/'runs'/cfg['id']
            if run.exists():
                if not args.resume:
                    raise FileExistsError(f'{run} exists. Use queue --resume to continue; never overwrite.')
                if not (run/'model-last.pth').is_file():
                    raise FileNotFoundError('Partial run without checkpoint: preserve it and inspect '+str(run))
                helper('molab_pipeline').resume_config(argparse.Namespace(config=str(path)))
                smoke(work,args.dataset,case,checkpoint=True)
            else:
                if cfg.get('start_from'):raise ValueError('Fresh run has resume state but no output')
                smoke(work,args.dataset,case)
            marker=work/'provenance'/(cfg['id']+'_contribution_smoke.json')
            if json.loads(marker.read_text())!=fingerprint(path):raise RuntimeError('Smoke signature changed')
            print('TRAIN_START',cfg['id'],time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
            import shutil
            print('DISK_FREE_GIB',round(shutil.disk_usage(work).free/2**30,2),flush=True)
            logged([sys.executable,'-u','train.py','--cfg_path',str(path)],work/'logs'/(cfg['id']+'.log'),
                   work/'logs'/(cfg['id']+'.pid'))
            if not completed(work,check(path)):raise RuntimeError('Training exited without a complete final checkpoint')
            print('TRAIN_COMPLETE',cfg['id'],flush=True)
            report(argparse.Namespace(work=work,dataset=args.dataset))
        print('QUEUE_COMPLETE',args.dataset,args.cases,flush=True)


def report(args):
    rows=[]
    for case in CASES:
        run=args.work/'runs'/run_name(args.dataset,case)
        row=dict(case=case,run=run.name,status='not_run',best_epoch='',last_epoch='',**{k:'' for k in METRICS})
        if (run/'info.json').is_file():
            info=json.loads((run/'info.json').read_text(encoding='utf8'))
            best,last=info.get('best',{}),info.get('last',{})
            row.update(best_epoch=best.get('epoch',''),last_epoch=last.get('epoch',''),status='partial')
            if last.get('epoch',-1)>=epochs(args.dataset)-1:row['status']='complete'
            basename=str(best.get('result_json_path','')).replace('\\','/').split('/')[-1]
            path=run/'prediction'/basename
            if basename and path.is_file():
                result=json.loads(path.read_text(encoding='utf8'))
                row.update({k:100*float(result[k]) for k in METRICS if k in result})
        rows.append(row)
    target=args.work/'analysis'/f'{args.dataset}_contribution_summary.tsv'
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    print('SUMMARY_READY',target,flush=True)


def rescore(args):
    from misc.proposal_selection import select_proposals
    from eval_utils_clip import eval_metrics
    cfg=yaml.safe_load(args.config.read_text(encoding='utf8'))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw_paths=sorted(args.raw_dir.glob('*.raw.json'))
    if not raw_paths:raise FileNotFoundError('No *.raw.json under '+str(args.raw_dir))
    summary=[]
    for raw_path in raw_paths:
        destination=args.output_dir/(raw_path.stem.replace('.raw','')+'.softnms_only.json')
        metrics_path=destination.with_suffix('.metrics.json')
        provenance=dict(raw_sha256=sha(raw_path),config_sha256=sha(args.config),policy='softnms_only_sigma0.5')
        if destination.exists():
            old=json.loads(destination.read_text(encoding='utf8'))
            if old.get('provenance')!=provenance:raise FileExistsError('Different input for '+str(destination))
        else:
            raw=json.loads(raw_path.read_text(encoding='utf8'))
            refs=set().union(*(set(json.loads(Path(p).read_text(encoding='utf8'))) for p in cfg['gt_file_for_eval']))
            if set(raw['results'])!=refs:raise ValueError('Raw prediction/reference video IDs differ')
            selected={}
            for video,rows in raw['results'].items():
                if rows and 'pred_event_count' not in rows[0]:raise ValueError('Raw lacks event count')
                if any(r.get('duplicate_selection') for r in rows):raise ValueError('Not a V1/V2 raw prediction')
                selected[video]=select_proposals(rows,int(rows[0]['pred_event_count']) if rows else 0,
                    method='softnms',sigma=.5,alpha=cfg.get('ec_alpha',1.),temperature=2.,
                    merge=False,merge_consecutive=False)
            dump(destination,dict(results=selected,version=raw.get('version','VERSION 1.0'),
                                  external_data=raw.get('external_data',{}),provenance=provenance))
        if not metrics_path.exists():
            scores=dict(eval_metrics(str(destination),cfg['gt_file_for_eval'],cfg['gt_file_for_para_eval'],
                                     dvc_eval_version=cfg.get('eval_tool_version','2018')))
            scores['MetaScore']=float(scores['METEOR'])+float(scores['soda_c'])
            dump(metrics_path,scores)
        scores=json.loads(metrics_path.read_text(encoding='utf8'))
        summary.append(dict(prediction=destination.name,**{k:100*float(scores[k]) for k in METRICS if k in scores}))
    with (args.output_dir/'summary.tsv').open('w',encoding='utf8',newline='') as stream:
        fields=['prediction']+list(METRICS)
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(summary)
    best=max(summary,key=lambda r:r.get('METEOR',0)+r.get('soda_c',0))
    dump(args.output_dir/'best_softnms_only.json',best)
    print('RESCORE_BEST_METEOR_PLUS_SODA',best,flush=True)


def export(args):
    work=args.work.resolve()
    path=work/f'CONTRIBUTION_{args.dataset}_{time.strftime("%Y%m%d_%H%M%S")}.zip'
    extensions={'.json','.jsonl','.yml','.yaml','.log','.txt','.tsv','.csv'}
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for folder in ('runs','prepared','logs','provenance','analysis'):
            parent=work/folder
            if parent.exists():
                for p in parent.rglob('*'):
                    if p.is_file() and p.suffix.lower() in extensions and 'backup' not in p.parts:
                        z.write(p,str(p.relative_to(work)))
        tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
        for relative in tracked:
            p=ROOT/relative
            if p.is_file() and p.suffix in {'.py','.sh','.md','.txt','.json'}:z.write(p,'source/'+relative)
        z.writestr('EXPORT_MANIFEST.json',json.dumps(dict(dataset=args.dataset,checkpoint_weights=False,
            source_commit=helper('molab_pipeline').revision(),note='Analysis export; feature maps, banks and checkpoints excluded.'),indent=2))
    print('ZIP_READY',path,'MiB',round(path.stat().st_size/2**20,2),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','queue','smoke','report','export'):
        p=sub.add_parser(name)
        p.add_argument('--work',type=Path,required=True)
        p.add_argument('--dataset',choices=['yc2','anet'],required=True)
        if name in ('prepare','queue'):
            p.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
        if name=='prepare':p.add_argument('--source-config',type=Path)
        if name=='queue':p.add_argument('--resume',action='store_true')
        if name=='smoke':p.add_argument('--case',choices=list(CASES),required=True)
    p=sub.add_parser('check');p.add_argument('--config',type=Path,required=True)
    p=sub.add_parser('rescore');p.add_argument('--config',type=Path,required=True)
    p.add_argument('--raw-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='check':check(args.config)
    elif args.command=='smoke':smoke(args.work.resolve(),args.dataset,args.case)
    else:globals()[args.command](args)


if __name__=='__main__':main()
