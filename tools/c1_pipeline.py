"""Historical C1 model/recipe: YouCook2 15 epochs, ANET 6 epochs by request."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from misc.c1_recipe import BASE_COMMIT, METHOD, clean_legacy_metadata, validate_method

METRICS = ('CIDEr','METEOR','Bleu_4','soda_c','Precision','Recall','MetaScore')


def helper(name):
    spec = importlib.util.spec_from_file_location('c1_'+name, ROOT/'tools'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def epochs(dataset):
    return 15 if dataset=='yc2' else 6


def run_name(dataset):
    return f'{dataset}_c1_e{epochs(dataset)}_b1'


def config_path(work, dataset):
    return Path(work)/'prepared'/run_name(dataset)/'config.yml'


def stage_file(source, target):
    source, target = Path(source).resolve(), Path(target)
    if not source.is_file():raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.samefile(target) or (source.stat().st_size==target.stat().st_size and sha(source)==sha(target)):return
        raise FileExistsError('Refusing to replace input: '+str(target))
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)


def ensure_assets(args):
    if args.source_config:
        path=args.source_config.resolve()
        if not path.is_file():raise FileNotFoundError(path)
        return path
    name=f'{args.dataset}_c1_base_assets'
    path=args.work/'prepared'/name/'config.yml'
    if path.is_file():return path
    if args.dataset=='anet':
        print('ANET CLIP features: about 2.52 GiB; train-only caption bank.',flush=True)
        command=[sys.executable,'-u','tools/activitynet_pipeline.py','prepare','--work',str(args.work),
                 '--mode','full','--run-name',name,'--data-root',str(args.work/'assets/anet')]
        if args.features:command+=['--features',str(args.features.resolve())]
    else:
        print('YouCook2 CLIP features: about 872 MiB; train-only caption bank.',flush=True)
        if args.features:stage_file(args.features,args.work/'assets/yc2/clipvitl14.pth')
        command=[sys.executable,'-u','tools/molab_pipeline.py','prepare','--mode','full',
                 '--run-name',name,'--epochs','15','--experiment','c2',
                 '--data-root',str(args.work/'assets/yc2'),'--prepared-root',str(args.work/'prepared'),
                 '--output-root',str(args.work/'runs'),'--log-root',str(args.work/'logs')]
    subprocess.run(command,cwd=ROOT,check=True)
    return path


def make_config(base, work, dataset):
    cfg=dict(base)
    if cfg.get('query_temporal_support') or cfg.get('relation_aware_event_set'):
        raise ValueError('Supply a base data bundle, not a trained QTS/event-set run config')
    for key in list(cfg):
        if key.startswith(('temporal_support_', 'event_set_')) or key in (
                'query_temporal_support','relation_aware_event_set','qts_variant',
                'full_event_supervision','counter_class_rates','counter_prior_source'):
            cfg.pop(key)
    cfg.update(json.loads((ROOT/"tools/c1_frozen_options.json").read_text(encoding="utf8"))[dataset])
    cfg=clean_legacy_metadata(cfg)
    name=run_name(dataset)
    cfg.update(METHOD)
    cfg.update(seed=42,id=name,exp_name=name,fixed_run_id=True,cfg_path='',save_dir=str(Path(work)/'runs'),
               epoch=epochs(dataset),start_from='',start_from_mode='last',pretrain=None,
               only_ft_captioner=False,only_ft_class_head=False,training_scheme='all',
               experiment_recipe='stable_c1',selection_experiment='softnms',
               min_epoch_when_save=0,save_checkpoint_every=1,save_all_checkpoint=False)
    cfg.pop('current_lr',None)
    validate_method(cfg)
    return cfg


def make_smoke(cfg, target):
    tiny=copy.deepcopy(cfg);tiny.update(id=cfg['id']+'_smoke',exp_name=cfg['id']+'_smoke',epoch=2)
    folder=target/'smoke';folder.mkdir(parents=True,exist_ok=True)
    for key in ('train_caption_file','val_caption_file'):
        rows=json.loads(Path(cfg[key]).read_text(encoding='utf8'))
        path=folder/(key+'.json');dump(path,{k:rows[k] for k in list(rows)[:2]});tiny[key]=str(path)
    ids=set(json.loads(Path(tiny['val_caption_file']).read_text(encoding='utf8')))
    for key in ('gt_file_for_eval','gt_file_for_para_eval'):
        tiny[key]=[]
        for i,path in enumerate(cfg[key]):
            rows=json.loads(Path(path).read_text(encoding='utf8'))
            output=folder/f'{key}_{i}.json';dump(output,{k:v for k,v in rows.items() if k in ids});tiny[key].append(str(output))
    tiny['gt_file_for_auc']=tiny['gt_file_for_eval'][:]
    tiny['data_signature']=hashlib.sha256(b''.join(Path(tiny[k]).read_bytes() for k in
                                  ('train_caption_file','val_caption_file','dict_file'))).hexdigest()
    (folder/'config.yml').write_text(yaml.safe_dump(tiny,sort_keys=False),encoding='utf8')


def prepare(args):
    args.work=args.work.resolve();path=config_path(args.work,args.dataset)
    if path.is_file():check(path);return
    if (args.work/'runs'/run_name(args.dataset)).exists():raise FileExistsError('Output already exists; inspect before preparing')
    source=ensure_assets(args);base=yaml.safe_load(source.read_text(encoding='utf8'))
    if base.get('dataset')!=args.dataset or base.get('bank_type')!=[args.dataset]:
        raise ValueError('Source config must use the same dataset and bank')
    if base.get('num_queries')!=(50 if args.dataset=='yc2' else 10):raise ValueError('Wrong source query budget')
    # Base bundle supplies data only. Model options are frozen from historical C1 below.
    signature=hashlib.sha256(b''.join(Path(base[k]).read_bytes() for k in
                                    ('train_caption_file','val_caption_file','dict_file'))).hexdigest()
    if signature!=base.get('data_signature'):raise ValueError('Source data signature mismatch')
    cfg=make_config(base,args.work,args.dataset);target=path.parent;data=target/'data'/args.dataset
    for key,relative in [('train_caption_file','captiondata/train.json'),('val_caption_file','captiondata/val.json'),('dict_file','vocabulary.json')]:
        dest=data/relative;stage_file(base[key],dest);cfg[key]=str(dest)
    for key,folder in [('gt_file_for_eval','references'),('gt_file_for_para_eval','para')]:
        cfg[key]=[]
        for i,original in enumerate(base[key]):
            dest=data/'captiondata'/folder/f'reference_{i}.json';stage_file(original,dest);cfg[key].append(str(dest))
    cfg['gt_file_for_auc']=cfg['gt_file_for_eval'][:]
    if len(base['visual_feature_folder'])!=1:raise ValueError('Expected one CLIP features file')
    feature=data/'features/clipvitl14.pth';stage_file(base['visual_feature_folder'][0],feature)
    cfg['visual_feature_folder']=[str(feature)]
    for folder in {cfg['bank_folder'],'clip'}:
        bank=Path(base['bank_path'])/args.dataset/folder
        if not bank.is_dir():raise FileNotFoundError(bank)
        for original in bank.glob('*.npy'):stage_file(original,target/'bank'/args.dataset/folder/original.name)
    cfg['bank_path']=str(target/'bank')
    inputs={key:sha(cfg[key]) for key in ('train_caption_file','val_caption_file','dict_file')}
    shared=set(cfg['gt_file_for_eval']+cfg['gt_file_for_para_eval'])
    shared.update(str(p) for p in (target/'bank').rglob('*.npy'))
    manifest=dict(expected_config=cfg,source_config=str(source),source_sha256=sha(source),
                  source_commit=helper('molab_pipeline').revision(),ablation_base_commit=BASE_COMMIT,
                  dataset=args.dataset,case='C1',input_sha256=inputs,
                  shared_file_sha256={p:sha(p) for p in sorted(shared)},
                  note='Same C1 method/recipe. Assets copied byte-for-byte; fresh model and optimizer.')
    dump(target/'c1_manifest.json',manifest)
    path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf8')
    make_smoke(cfg,target);check(path)
    print('CONFIG_READY',path,flush=True)


def check(path, require_features=True):
    path=Path(path);cfg=yaml.safe_load(path.read_text(encoding='utf8'));validate_method(cfg)
    manifest=json.loads((path.parent/'c1_manifest.json').read_text(encoding='utf8'))
    expected=manifest['expected_config'];mutable={'start_from','start_from_mode','current_lr'}
    if {k:v for k,v in cfg.items() if k not in mutable}!={k:v for k,v in expected.items() if k not in mutable}:
        raise ValueError('C1 config drift: '+str(path))
    ds=manifest['dataset']
    invariants=dict(seed=42,batch_size=1,gradient_accumulation_steps=1,precision='fp32',allow_tf32=False,
                    lr=5e-5,weight_decay=1e-4,epoch=epochs(ds),num_queries=50 if ds=='yc2' else 10,
                    gt_proposal_sample_num=10,max_eseq_length=20,count_loss_coef=.5,
                    able_ret=True,aux_loss=True,learning_rate_decay_start=8,
                    learning_rate_decay_every=3,learning_rate_decay_rate=.5)
    for key,value in invariants.items():
        if cfg.get(key)!=value:raise ValueError('C1 training recipe changed: '+key)
    for key,value in manifest['input_sha256'].items():
        if sha(cfg[key])!=value:raise ValueError('Input bytes changed: '+key)
    for file,digest in manifest['shared_file_sha256'].items():
        if sha(file)!=digest:raise ValueError('Reference/bank changed: '+file)
    train,val=[json.loads(Path(cfg[k]).read_text(encoding='utf8')) for k in ('train_caption_file','val_caption_file')]
    if set(train)&set(val):raise ValueError('Train/val leakage')
    if (len(train),len(val))!=((1237,436) if ds=='yc2' else (10009,4917)):raise ValueError('Wrong full split sizes')
    words=json.loads(Path(cfg['dict_file']).read_text(encoding='utf8'))['word_to_ix']
    if len(words)!=cfg['vocab_size']:raise ValueError('Vocabulary mismatch')
    refs=[json.loads(Path(p).read_text(encoding='utf8')) for p in cfg['gt_file_for_eval']]
    if set().union(*(set(r) for r in refs))!=set(val):raise ValueError('Reference/loader IDs differ')
    for folder in {cfg['bank_folder'],'clip'}:
        for suffix in ('scene_sentences','clip_token_embeds','scene_videoID'):
            if not (Path(cfg['bank_path'])/ds/folder/f'{ds}_{suffix}.npy').is_file():raise FileNotFoundError('CLIP bank incomplete')
    if require_features:
        for p in cfg['visual_feature_folder']:
            if not Path(p).is_file():raise FileNotFoundError(p)
    print('C1_OK',ds,len(train),'train',len(val),'val;',cfg['epoch'],'epochs; mask loss 0; Soft-NMS only',flush=True)
    return cfg


def fingerprint(path):
    files=[Path(path)]+[ROOT/p for p in ('train.py','opts.py','eval_utils_clip.py','misc/c1_recipe.py',
           'ddvc/ddvc_ret_encdec_clip_capQ.py','ddvc/CaptioningHead/gpt2.py',
           'ddvc/CaptioningHead/proposal_mask.py','misc/proposal_selection.py',
           'misc/gradient_guard.py','transformers/caption_targets.py','ddvc/temporal_pooling.py','tools/c1_pipeline.py','tools/c1_frozen_options.json')]
    return {str(p):sha(p) for p in files}


def logged(command, log, pid=None):
    log.parent.mkdir(parents=True,exist_ok=True);print('COMMAND_LOG',log,flush=True)
    with log.open('a',encoding='utf8') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if pid:pid.write_text(str(process.pid),encoding='utf8')
        status=process.wait()
        if status:
            print(log.read_text(encoding='utf8',errors='replace')[-14000:],flush=True)
            raise subprocess.CalledProcessError(status,command)


def smoke(args):
    path=config_path(args.work,args.dataset);cfg=check(path)
    logged([sys.executable,'-u','tools/prepare_gpt2_tokenizer.py'],args.work/'logs'/(cfg['id']+'.tokenizer.log'))
    command=[sys.executable,'-u','tools/molab_pipeline.py','smoke','--config',str(path.parent/'smoke/config.yml')]
    if getattr(args,'checkpoint',False):
        command=[sys.executable,'-u','tools/molab_pipeline.py','smoke','--config',str(path),
                 '--checkpoint',str(args.work/'runs'/cfg['id']/'model-last.pth')]
    logged(command,args.work/'logs'/(cfg['id']+'.smoke.log'))
    dump(args.work/'provenance'/(cfg['id']+'_smoke.json'),fingerprint(path))


def train(args):
    import fcntl
    args.work=args.work.resolve();logs=args.work/'logs';logs.mkdir(parents=True,exist_ok=True)
    with (logs/'c1_gpu.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another C1 training owns this work/GPU lock')
        path=config_path(args.work,args.dataset);cfg=check(path)
        for ds in ('yc2','anet'):
            pid=helper('training_pid').active_training_pid(logs/(run_name(ds)+'.pid'),config_path(args.work,ds))
            if pid is not None:raise RuntimeError(f'Training is already running: PID={pid}')
        run=args.work/'runs'/cfg['id']
        if args.resume:
            if not (run/'model-last.pth').is_file():raise FileNotFoundError('Resume requires model-last.pth')
            helper('molab_pipeline').resume_config(argparse.Namespace(config=str(path)))
            args.checkpoint=True
        elif run.exists():
            raise FileExistsError('Run exists; inspect it or use resume. Never overwrite '+str(run))
        elif cfg.get('start_from') or cfg.get('pretrain'):
            raise ValueError('Fresh run cannot load weights')
        smoke(args)
        marker=args.work/'provenance'/(cfg['id']+'_smoke.json')
        if json.loads(marker.read_text(encoding='utf8'))!=fingerprint(path):raise RuntimeError('Smoke fingerprint changed')
        print('TRAIN_START',cfg['id'],flush=True)
        logged([sys.executable,'-u','train.py','--cfg_path',str(path)],logs/(cfg['id']+'.log'),logs/(cfg['id']+'.pid'))
        info=json.loads((run/'info.json').read_text(encoding='utf8'))
        if info.get('last',{}).get('epoch',-1)<cfg['epoch']-1:raise RuntimeError('Training did not finish all epochs')
        print('TRAIN_COMPLETE',cfg['id'],flush=True);report(args)


def report(args):
    cfg=check(config_path(args.work,args.dataset),require_features=False)
    helper('molab_pipeline').report(argparse.Namespace(config=str(config_path(args.work,args.dataset))))


def restage(args):
    path=config_path(args.work,args.dataset);cfg=check(path,require_features=False)
    if all(Path(p).is_file() for p in cfg['visual_feature_folder']):print('FEATURES_ALREADY_READY');return
    if args.dataset=='anet':
        adapter=helper('activitynet_pipeline')
        source=adapter.ensure_feature_file(args.features or args.work/'assets/anet/clipvitl14.pth')
        features=adapter.load_features(source)
        ids=set(json.loads(Path(cfg['train_caption_file']).read_text()))|set(json.loads(Path(cfg['val_caption_file']).read_text()))
        if ids-set(features):raise ValueError('Missing feature IDs')
        adapter.save_features(features,ids,cfg['visual_feature_folder'][0])
    else:
        if args.features:stage_file(args.features,args.work/'assets/yc2/clipvitl14.pth')
        helper('molab_pipeline').restage(argparse.Namespace(config=str(path),data_root=str(args.work/'assets/yc2')))
    check(path)


def export(args):
    args.work=args.work.resolve();target=args.work/f'C1_ANALYSIS_{args.dataset}_{time.strftime("%Y%m%d_%H%M%S")}.zip'
    extensions={'.json','.jsonl','.yml','.yaml','.log','.txt','.tsv','.csv'}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name in ('runs','prepared','logs','provenance'):
            for p in (args.work/name).rglob('*'):
                if p.is_file() and p.suffix.lower() in extensions and 'backup' not in p.parts:
                    z.write(p,str(p.relative_to(args.work)))
        tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
        for relative in tracked:
            p=ROOT/relative
            if p.is_file() and p.suffix in {'.py','.sh','.md','.txt','.json'}:z.write(p,'source/'+relative)
        z.writestr('EXPORT_MANIFEST.json',json.dumps(dict(dataset=args.dataset,method='C1',
             source_commit=helper('molab_pipeline').revision(),checkpoint_weights=False)))
    print('ZIP_READY',target,flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','train','smoke','report','export','restage'):
        p=sub.add_parser(name);p.add_argument('--work',type=Path,required=True)
        p.add_argument('--dataset',choices=['yc2','anet'],required=True)
        if name=='prepare':p.add_argument('--source-config',type=Path)
        if name in ('prepare','restage'):p.add_argument('--features',type=Path)
        if name=='train':p.add_argument('--resume',action='store_true')
        if name=='smoke':p.add_argument('--checkpoint',action='store_true')
    p=sub.add_parser('check');p.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='check':check(args.config)
    else:globals()[args.command](args)


if __name__=='__main__':main()
