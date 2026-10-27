"""Prepare an 8-epoch, opt-in event-concept experiment on either DDVC dataset."""
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
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from misc.event_concepts import make_vocabulary,load_vocabulary


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')


def stage_file(source,target):
    source,target=Path(source).resolve(),Path(target)
    if not source.is_file():raise FileNotFoundError(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if source.samefile(target):return
        if source.stat().st_size==target.stat().st_size and sha(source)==sha(target):return
        raise FileExistsError('Refusing to replace staged input: '+str(target))
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)


def run_name(dataset,variant):
    suffix={'full':'event_concept','aux':'concept_aux','control':'concept_control'}[variant]
    return f'{dataset}_full_{suffix}_e8_b1'


def load_helper(name):
    spec=importlib.util.spec_from_file_location('concept_'+name,ROOT/'tools'/f'{name}.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);return helper


def ensure_base(work,dataset,source_config):
    if source_config:
        source_config=source_config.resolve()
        if not source_config.is_file():raise FileNotFoundError(source_config)
        return source_config
    name=f'{dataset}_full_concept_base_assets'
    target=work/'prepared'/name/'config.yml'
    if target.is_file():return target
    if dataset=='anet':
        print('ASSETS: official ANET feature download is about 2.52 GiB; bank uses training captions only.',flush=True)
        command=[sys.executable,'-u','tools/activitynet_pipeline.py','prepare','--work',str(work),
                 '--mode','full','--run-name',name]
    else:
        print('ASSETS: official YouCook2 feature download is about 872 MiB; train-only bank.',flush=True)
        command=[sys.executable,'-u','tools/molab_pipeline.py','prepare','--mode','full',
                 '--run-name',name,'--epochs','15','--experiment','no_dedup_same_merges',
                 '--data-root',str(work/'assets'),'--prepared-root',str(work/'prepared'),
                 '--output-root',str(work/'runs'),'--log-root',str(work/'logs')]
    subprocess.run(command,cwd=ROOT,check=True)
    return target


def prepare(args):
    work=args.work.resolve();work.mkdir(parents=True,exist_ok=True)
    name=run_name(args.dataset,args.variant);target=work/'prepared'/name
    if (target/'config.yml').exists():check(target/'config.yml');return
    if (work/'runs'/name).exists():raise FileExistsError('Training output already exists; do not prepare over it')
    source=ensure_base(work,args.dataset,args.source_config)
    original=yaml.safe_load(source.read_text(encoding='utf8'))
    if original.get('dataset')!=args.dataset or original.get('bank_type')!=[args.dataset]:
        raise ValueError('Source config must be from the same dataset and its own bank')
    expected_queries=10 if args.dataset=='anet' else 50
    if original.get('num_queries')!=expected_queries:
        raise ValueError(f'Expected {expected_queries} queries; select the proper baseline source config')
    if original.get('event_concepts') or original.get('ideal_test') or original.get('dedup_loss_coef',0)!=0:
        raise ValueError('Expected the current no-dedup, non-oracle baseline before adding concepts')
    original_signature=hashlib.sha256(b''.join(Path(original[k]).read_bytes()
        for k in ('train_caption_file','val_caption_file','dict_file'))).hexdigest()
    if original.get('data_signature')!=original_signature:
        raise ValueError('Source baseline data signature does not match its files')
    cfg=copy.deepcopy(original);data=target/'data'/args.dataset;anno=data/'captiondata'
    changed_paths={}
    for key,relative in [('train_caption_file','captiondata/train.json'),
                         ('val_caption_file','captiondata/val.json'),('dict_file','vocabulary.json')]:
        dest=data/relative;stage_file(cfg[key],dest);changed_paths[key]=str(dest)
    for key,subfolder in [('gt_file_for_eval','references'),('gt_file_for_para_eval','para')]:
        changed_paths[key]=[]
        for index,path in enumerate(cfg[key]):
            dest=anno/subfolder/f'reference_{index}.json';stage_file(path,dest)
            changed_paths[key].append(str(dest))
    changed_paths['gt_file_for_auc']=changed_paths['gt_file_for_eval'][:]
    if len(cfg['visual_feature_folder'])!=1:raise ValueError('Expected one pre-extracted CLIP feature file')
    feature=data/'features/clipvitl14.pth';stage_file(cfg['visual_feature_folder'][0],feature)
    changed_paths['visual_feature_folder']=[str(feature)]
    bank=target/'bank'/args.dataset/cfg['bank_folder']
    for folder in {cfg['bank_folder'],'clip'}:
        source_bank=Path(cfg['bank_path'])/args.dataset/folder
        if not source_bank.is_dir():raise FileNotFoundError(source_bank)
        for path in source_bank.glob('*.npy'):
            stage_file(path,target/'bank'/args.dataset/folder/path.name)
    changed_paths['bank_path']=str(target/'bank')
    records=json.loads(Path(changed_paths['train_caption_file']).read_text(encoding='utf8'))
    vocab=make_vocabulary(records,max_concepts=512,min_frequency=5)
    vocab.update(dataset=args.dataset,training_caption_sha256=sha(changed_paths['train_caption_file']))
    vocab_path=target/'concept_vocabulary.json';dump(vocab_path,vocab)
    cfg.update(changed_paths)
    cfg.update(id=name,exp_name=name,fixed_run_id=True,save_dir=str(work/'runs'),cfg_path='',
               start_from='',start_from_mode='last',pretrain=None,epoch=8,min_epoch_when_save=0,
               save_checkpoint_every=1,save_all_checkpoint=False,
               only_ft_captioner=False,only_ft_class_head=False,training_scheme='all',
               experiment_recipe='event_concept_e8_v1',event_concepts=args.variant!='control',
               concept_condition_prefix=args.variant=='full',concept_loss_coef=.1,
               concept_hidden_dim=256,concept_embedding_dim=128,
               concept_vocab_path=str(vocab_path),concept_vocab_sha256=sha(vocab_path))
    cfg.pop('current_lr',None)
    cfg['data_signature']=hashlib.sha256(b''.join(Path(cfg[k]).read_bytes()
                          for k in ('train_caption_file','val_caption_file','dict_file'))).hexdigest()
    # Preserve every unrelated training/decode option from the supplied baseline.
    allowed=set(changed_paths)|{'id','exp_name','fixed_run_id','save_dir','cfg_path','start_from',
        'start_from_mode','pretrain','epoch','min_epoch_when_save','save_checkpoint_every',
        'save_all_checkpoint','only_ft_captioner','only_ft_class_head','training_scheme',
        'experiment_recipe','event_concepts','concept_condition_prefix','concept_loss_coef',
        'concept_hidden_dim','concept_embedding_dim','concept_vocab_path','concept_vocab_sha256',
        'current_lr','data_signature'}
    assert all(cfg.get(k)==v for k,v in original.items() if k not in allowed)
    manifest=dict(source_config=str(source),source_config_sha256=sha(source),variant=args.variant,
        dataset=args.dataset,base_config=original,expected_config=cfg,
        train_videos=len(records),validation_videos=len(json.loads(Path(cfg['val_caption_file']).read_text())),
        concepts=len(vocab['concepts']),note='Fresh DDVC/optimizer; input files reused by hardlink/copy only. '
        'Concept labels are training content words, not POS-annotated action/object classes. '
        'Original temperature, loss weights, mask, retrieval and decoding remain unchanged.')
    dump(target/'concept_manifest.json',manifest)
    (target/'config.yml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf8')
    make_smoke(cfg,target,work)
    check(target/'config.yml')
    print('CONCEPT_CONFIG_READY',target/'config.yml',flush=True)


def make_smoke(cfg,target,work):
    tiny=copy.deepcopy(cfg);tiny['id']=cfg['id']+'_smoke';tiny['exp_name']=tiny['id'];tiny['epoch']=2
    smoke=target/'smoke';smoke.mkdir(parents=True,exist_ok=True)
    train=json.loads(Path(cfg['train_caption_file']).read_text(encoding='utf8'))
    val=json.loads(Path(cfg['val_caption_file']).read_text(encoding='utf8'))
    for key,rows in [('train_caption_file',train),('val_caption_file',val)]:
        dest=smoke/(key+'.json');dump(dest,{k:rows[k] for k in list(rows)[:2]});tiny[key]=str(dest)
    keep=set(json.loads(Path(tiny['val_caption_file']).read_text()))
    for key in ('gt_file_for_eval','gt_file_for_para_eval'):
        tiny[key]=[]
        for i,path in enumerate(cfg[key]):
            ref=json.loads(Path(path).read_text(encoding='utf8'))
            dest=smoke/f'{key}_{i}.json';dump(dest,{k:v for k,v in ref.items() if k in keep});tiny[key].append(str(dest))
    tiny['gt_file_for_auc']=tiny['gt_file_for_eval'][:]
    tiny['data_signature']=hashlib.sha256(b''.join(Path(tiny[k]).read_bytes()
                           for k in ('train_caption_file','val_caption_file','dict_file'))).hexdigest()
    (smoke/'config.yml').write_text(yaml.safe_dump(tiny,sort_keys=False),encoding='utf8')


def check(path,require_features=True):
    path=Path(path).resolve();cfg=yaml.safe_load(path.read_text(encoding='utf8'))
    manifest=json.loads((path.parent/'concept_manifest.json').read_text(encoding='utf8'))
    expected=manifest['expected_config']
    differences={k:(cfg.get(k),v) for k,v in expected.items()
                 if k not in {'start_from','start_from_mode','current_lr'} and cfg.get(k)!=v}
    if differences:raise ValueError('Concept experiment recipe drift: '+repr(differences))
    if set(cfg)-set(expected)-{'current_lr'}:raise ValueError('Unexpected extra config keys')
    assert cfg['epoch']==8 and cfg['batch_size']==1 and cfg['gradient_accumulation_steps']==1
    assert cfg['num_queries']==(10 if cfg['dataset']=='anet' else 50)
    assert cfg['precision']=='fp32' and not cfg['allow_tf32'] and cfg['seed']==777
    assert cfg['aux_loss'] and cfg['learned_proposal_mask'] and cfg['able_ret']
    signature=hashlib.sha256(b''.join(Path(cfg[k]).read_bytes()
                            for k in ('train_caption_file','val_caption_file','dict_file'))).hexdigest()
    if signature!=cfg['data_signature']:raise ValueError('Train/validation/vocabulary bytes changed')
    vocab=load_vocabulary(cfg['concept_vocab_path'],cfg['concept_vocab_sha256'])
    if vocab['training_caption_sha256']!=sha(cfg['train_caption_file']):
        raise ValueError('Concept vocabulary was not fitted to these training captions')
    train=json.loads(Path(cfg['train_caption_file']).read_text());val=json.loads(Path(cfg['val_caption_file']).read_text())
    if set(train)&set(val):raise ValueError('Train/validation IDs overlap')
    expected_counts=(10009,4917) if cfg['dataset']=='anet' else (1237,436)
    if (len(train),len(val))!=expected_counts:
        raise ValueError('Expected full feature-matched baseline splits '+str(expected_counts))
    words=json.loads(Path(cfg['dict_file']).read_text())['word_to_ix']
    if len(words)!=cfg['vocab_size']:raise ValueError('Baseline caption vocabulary size mismatch')
    references=[json.loads(Path(p).read_text()) for p in cfg['gt_file_for_eval']]
    if set().union(*(set(r) for r in references))!=set(val):
        raise ValueError('Validation loader/reference IDs differ')
    for folder in {cfg['bank_folder'],'clip'}:
        for suffix in ('scene_sentences','clip_token_embeds','scene_videoID'):
            if not (Path(cfg['bank_path'])/cfg['dataset']/folder/f"{cfg['dataset']}_{suffix}.npy").is_file():
                raise FileNotFoundError('Training CLIP bank missing')
    required=cfg['gt_file_for_eval']+cfg['gt_file_for_para_eval']
    if require_features:required+=cfg['visual_feature_folder']
    for p in required:
        if not Path(p).is_file():raise FileNotFoundError(p)
    print('CONCEPT_RECIPE_OK',cfg['dataset'],len(train),'train',len(val),'validation',len(vocab['concepts']),
          'concepts; epochs=8; variant='+manifest['variant'],flush=True)
    return cfg


def restage(args):
    cfg=check(args.config,require_features=False)
    if all(Path(p).is_file() for p in cfg['visual_feature_folder']):
        print('FEATURES_ALREADY_READY',cfg['visual_feature_folder'],flush=True);return
    work=args.work.resolve()
    if cfg['dataset']=='anet':
        helper=load_helper('activitynet_pipeline')
        source=helper.ensure_feature_file(args.features.resolve() if args.features else work/'assets/clipvitl14.pth')
        features=helper.load_features(source)
        ids=set(json.loads(Path(cfg['train_caption_file']).read_text()))|set(json.loads(Path(cfg['val_caption_file']).read_text()))
        if ids-set(features):raise ValueError('Downloaded features are missing restored annotation IDs')
        helper.save_features(features,ids,cfg['visual_feature_folder'][0])
    else:
        from types import SimpleNamespace
        if args.features:stage_file(args.features,work/'assets/clipvitl14.pth')
        load_helper('molab_pipeline').restage(SimpleNamespace(config=str(args.config),data_root=str(work/'assets')))
    check(args.config)
    print('RESTAGE_COMPLETE',flush=True)


def reuse_runtime(args):
    source=args.source_work.resolve();work=args.work.resolve()
    if source==work:raise ValueError('Use a separate work folder to preserve the completed baseline')
    destination=ROOT/'ddvc/ops'
    candidates=list((source/'source/ddvc/ops').glob('MultiScaleDeformableAttention*.so'))
    if len(candidates)!=1:raise FileNotFoundError('Need exactly one already-built CUDA extension in '+str(source))
    destination.mkdir(parents=True,exist_ok=True);shutil.copy2(candidates[0],destination/candidates[0].name)
    for relative in ('cuda130_env.sh','provenance/runtime.json','provenance/resolved_packages.txt'):
        original=source/relative
        if original.is_file():
            target=work/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target)
    for relative in ('tools/java','hf-cache'):
        original=source/relative;target=work/relative
        if original.is_dir() and not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(original,target_is_directory=True)
    subprocess.run([sys.executable,'tools/molab_pipeline.py','patch'],cwd=ROOT,check=True)
    # The shell wrapper sources the toolchain and runs an actual CUDA forward/backward test next.
    print('RUNTIME_STAGED; CUDA test must pass before marking it ready',flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--work',type=Path,required=True)
    p.add_argument('--dataset',choices=['anet','yc2'],required=True)
    p.add_argument('--variant',choices=['full','aux','control'],default='full');p.add_argument('--source-config',type=Path)
    p=sub.add_parser('check');p.add_argument('--config',type=Path,required=True)
    p=sub.add_parser('reuse-runtime');p.add_argument('--work',type=Path,required=True);p.add_argument('--source-work',type=Path,required=True)
    p=sub.add_parser('restage');p.add_argument('--work',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--features',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='check':check(args.config)
    elif args.command=='restage':restage(args)
    else:reuse_runtime(args)


if __name__=='__main__':main()
