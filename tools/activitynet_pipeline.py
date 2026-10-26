"""ActivityNet data adapter for the current learned-mask/no-dedup DDVC model.

No model, loss, optimizer, or selection implementation is changed here.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FEATURE_ID='1v08rs9Hwqh3XIM-8u8rRcFZc3HWUtJo_'
ANNOTATION_COMMIT='0b025c84f42fe27da51c312e8871c4b19628a04c'
ANNOTATION_BASE=f'https://raw.githubusercontent.com/ttengwang/PDVC/{ANNOTATION_COMMIT}/data/anet/captiondata/'
ANNOTATION_FILES=('train_modified.json','val_1.json','val_2.json')


def load_helper():
    spec=importlib.util.spec_from_file_location('ddvc_native_pipeline',ROOT/'tools/molab_pipeline.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False),encoding='utf8')


def canonical_video_id(value):
    value=str(value)
    if value.endswith('.mp4'):value=value[:-4]
    if value.startswith('v_') and len(value)==13:return value
    if len(value)==11:return 'v_'+value
    raise ValueError(f'Unexpected ActivityNet video ID: {value!r}')


def canonical_features(features):
    if isinstance(features,dict) and isinstance(features.get('features'),dict):features=features['features']
    if not isinstance(features,dict):raise ValueError('Expected a video-ID -> feature tensor mapping')
    result={}
    for key,value in features.items():
        cid=canonical_video_id(key)
        if cid in result:raise ValueError('Ambiguous feature aliases: '+cid)
        if getattr(value,'ndim',None)!=2 or value.shape[0]<1 or value.shape[1]!=768:
            raise ValueError(f'Invalid CLIP feature shape for {cid}: {getattr(value,"shape",None)}')
        result[cid]=value
    return result


def validate_annotations(records,train=False):
    result={};dropped=[]
    for key,record in records.items():
        vid=canonical_video_id(key)
        if vid in result:raise ValueError('Duplicate annotation alias: '+vid)
        if not math.isfinite(record['duration']) or record['duration']<=0:raise ValueError('Invalid duration: '+vid)
        if len(record['sentences'])!=len(record['timestamps']):raise ValueError('Caption/timestamp mismatch: '+vid)
        sentences=[];timestamps=[]
        for i,(sentence,span) in enumerate(zip(record['sentences'],record['timestamps'])):
            if len(span)!=2 or not all(math.isfinite(x) for x in span):raise ValueError('Invalid timestamps: '+vid)
            if span[0]<0 or span[1]<=span[0] or not isinstance(sentence,str) or not sentence.strip():
                if not train:raise ValueError(f'Invalid validation event: {vid}/{i}')
                dropped.append(dict(video=vid,event=i,span=span));continue
            sentences.append(sentence);timestamps.append(span)
        if sentences:result[vid]=dict(duration=record['duration'],sentences=sentences,timestamps=timestamps)
    return result,dropped


def select_splits(train,ref1,ref2,feature_ids,mode,seed):
    val_all=dict(ref2);val_all.update(ref1) # one loader record/video; evaluator keeps both references
    if set(train)&set(val_all):raise ValueError('ActivityNet train/validation IDs overlap')
    fraction={'smoke':.01,'subset':.30,'full':1.}[mode]
    selected=[];coverage={}
    for split,records,offset in [('train',train,0),('validation',val_all,1)]:
        eligible=sorted(set(records)&set(feature_ids));random.Random(seed+offset).shuffle(eligible)
        if not eligible:raise ValueError('No feature-matched '+split+' videos')
        ids=eligible[:max(1,round(len(eligible)*fraction))] if fraction<1 else eligible
        selected.append({vid:records[vid] for vid in ids})
        coverage[split]=dict(annotated=len(records),feature_matched=len(eligible),used=len(ids),
            missing_feature=len(set(records)-set(feature_ids)),missing_examples=sorted(set(records)-set(feature_ids))[:20])
    train_used,val_used=selected
    refs=[{vid:r[vid] for vid in val_used if vid in r} for r in (ref1,ref2)]
    return train_used,val_used,refs,coverage


def vocabulary(train):
    def words(sentence):
        for char in [',',':','!','_',';','-','.','?','/','"','\\n','\\']:
            sentence=sentence.lower().replace(char,' ')
        return sentence.split()
    counts=collections.Counter(w for r in train.values() for s in r['sentences'] for w in words(s))
    vocab=[w for w,n in counts.items() if n>=2]+['<bos>','<eos>','UNK']
    w2i={word:i+1 for i,word in enumerate(vocab)}
    return dict(word_to_ix=w2i,ix_to_word={str(i):word for word,i in w2i.items()})


def make_config(prepared,run_id,save_root,mode,vocab_size,signature):
    helper=load_helper()
    cfg=json.loads((ROOT/'tools/kaggle_mask_recipe.json').read_text())
    cfg.update(helper.experiment_options('no_dedup_same_merges'))
    data=prepared/'data/anet';anno=data/'captiondata'
    cfg.update(experiment_recipe='activitynet_current_model',dataset='anet',target_domain='anet',bank_type=['anet'],
        cfg_path='',id=run_id,exp_name=run_id,fixed_run_id=True,save_dir=str(save_root),
        start_from='',pretrain=None,train_caption_file=str(anno/'train.json'),val_caption_file=str(anno/'val.json'),
        visual_feature_folder=[str(data/'features/clipvitl14.pth')],
        gt_file_for_eval=[str(anno/'val_1.json'),str(anno/'val_2.json')],
        gt_file_for_auc=[str(anno/'val_1.json'),str(anno/'val_2.json')],
        gt_file_for_para_eval=[str(anno/'para/val_1.json'),str(anno/'para/val_2.json')],
        dict_file=str(data/'vocabulary_activitynet.json'),vocab_size=vocab_size,bank_path=str(prepared/'bank'),
        epoch=2 if mode=='smoke' else 15,min_epoch_when_save=0,save_checkpoint_every=1,save_all_checkpoint=False,
        seed=777,random_seed=False,debug=False,disable_tqdm=False,gpu_id=[0],device='cuda',
        nthreads=0,pin_memory=False,progress_log_every=10,data_signature=signature,feature_mmap=True,
        hardware_profile='legacy-cvpr2018',eval_enable_grounding=False,contrastive_loss_coef=.5,
        mask_gate_min=0.,mask_gate_max=1.,mask_context_ratio=0.,mask_min_frames=1)
    return cfg


def validate_config(cfg):
    expected=make_config(Path('/unused'),'unused',Path('/unused'),'full',0,'unused')
    variable={'cfg_path','id','exp_name','save_dir','train_caption_file','val_caption_file','visual_feature_folder',
        'gt_file_for_eval','gt_file_for_auc','gt_file_for_para_eval','dict_file','vocab_size','bank_path',
        'epoch','data_signature','start_from','start_from_mode','current_lr'}
    differences={k:(cfg.get(k),v) for k,v in expected.items() if k not in variable and cfg.get(k)!=v}
    if differences:raise ValueError('Current model recipe drift: '+repr(differences))
    if cfg['epoch'] not in (2,15):raise ValueError('Expected 2 smoke or 15 training epochs')
    for key in ('train_caption_file','val_caption_file','dict_file'):
        if not Path(cfg[key]).is_file():raise FileNotFoundError(cfg[key])
    signature=hashlib.sha256(b''.join(Path(cfg[k]).read_bytes() for k in ('train_caption_file','val_caption_file','dict_file'))).hexdigest()
    if signature!=cfg['data_signature']:raise ValueError('Annotation/vocabulary signature changed')
    vocab=json.loads(Path(cfg['dict_file']).read_text())
    if len(vocab['word_to_ix'])!=cfg['vocab_size']:raise ValueError('Vocabulary size mismatch')
    train=json.loads(Path(cfg['train_caption_file']).read_text());val=json.loads(Path(cfg['val_caption_file']).read_text())
    if set(train)&set(val):raise ValueError('Train/validation leakage')
    refs=[json.loads(Path(p).read_text()) for p in cfg['gt_file_for_eval']]
    if set().union(*(set(r) for r in refs))!=set(val):raise ValueError('Validation reference IDs differ from loader IDs')
    print('ACTIVITYNET_CURRENT_MODEL_OK',len(train),'train',len(val),'validation; 50 queries; dedup OFF; mask/Soft-NMS/merges ON',flush=True)


def ensure_feature_file(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file():return path
    import gdown
    print('Downloading DDVC ActivityNet CLIP features to',path,'; existing YouCook2 features are not reused',flush=True)
    partial=path.with_name(path.name+'.download')
    result=gdown.download(id=FEATURE_ID,output=str(partial),quiet=False)
    if result is None or not partial.is_file() or partial.stat().st_size==0:raise RuntimeError('Feature download failed')
    os.replace(partial,path);return path


def load_features(path):
    import torch
    try:features=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    except RuntimeError as exc:
        if 'mmap' not in str(exc):raise
        print('Legacy feature archive: loading without mmap',flush=True)
        features=torch.load(path,map_location='cpu',weights_only=False)
    return canonical_features(features)


def save_features(features,ids,target):
    import torch
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    selected={vid:torch.as_tensor(features[vid]).detach().cpu().float() for vid in sorted(ids)}
    partial=target.with_name(target.name+'.tmp');torch.save(selected,partial);os.replace(partial,target)


def build_bank(train,destination,batch=64):
    import clip
    import torch
    import numpy as np
    sentences=[];ids=[]
    for vid,record in train.items():
        for sentence in record['sentences']:sentences.append(sentence.strip());ids.append(vid)
    model,_=clip.load('ViT-L/14',device='cuda:0',jit=False);model.eval();encoded=[]
    with torch.inference_mode():
        for start in range(0,len(sentences),batch):
            tokens=clip.tokenize(sentences[start:start+batch],truncate=True).to('cuda:0')
            encoded.append(model.encode_text(tokens).float().cpu())
            print('BANK_PROGRESS',min(start+batch,len(sentences)),'/',len(sentences),flush=True)
    embedding=torch.cat(encoded).numpy()
    if embedding.shape!=(len(sentences),768) or not np.isfinite(embedding).all():raise ValueError('Invalid CLIP bank')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for name,array in [('anet_scene_sentences.npy',np.asarray(sentences,dtype=str)),
                       ('anet_scene_videoID.npy',np.asarray(ids,dtype=str)),('anet_clip_token_embeds.npy',embedding)]:
        partial=destination/(name+'.tmp')
        with partial.open('wb') as stream:np.save(stream,array)
        os.replace(partial,destination/name)
    del model,encoded,embedding;torch.cuda.empty_cache()


def prepare(args):
    import torch,yaml
    if not torch.cuda.is_available():raise RuntimeError('CUDA required to encode training caption bank')
    work=args.work.resolve();prepared=work/'prepared'/args.run_name
    if (prepared/'config.yml').exists():raise FileExistsError('Reuse existing config or select a new run name')
    assets=args.data_root.resolve() if args.data_root else work/'assets'
    anno=assets/'annotations';anno.mkdir(parents=True,exist_ok=True)
    for name in ANNOTATION_FILES:
        path=anno/name
        if not path.exists():
            partial=path.with_name(name+'.download');urllib.request.urlretrieve(ANNOTATION_BASE+name,partial);os.replace(partial,path)
    train,dropped=validate_annotations(json.loads((anno/'train_modified.json').read_text()),train=True)
    ref1,_=validate_annotations(json.loads((anno/'val_1.json').read_text()))
    ref2,_=validate_annotations(json.loads((anno/'val_2.json').read_text()))
    source=ensure_feature_file(args.features.resolve() if args.features else assets/'clipvitl14.pth')
    features=load_features(source)
    train,val,refs,coverage=select_splits(train,ref1,ref2,features,args.mode,777)
    print('COVERAGE',json.dumps(coverage),'dropped invalid training events',len(dropped),flush=True)
    data=prepared/'data/anet';caption=data/'captiondata'
    dump(caption/'train.json',train);dump(caption/'val.json',val)
    for i,ref in enumerate(refs,1):
        dump(caption/f'val_{i}.json',ref)
        dump(caption/f'para/val_{i}.json',{vid:' '.join(s.strip().rstrip('.')+'.' for _,s in sorted(zip(r['timestamps'],r['sentences']),key=lambda pair:pair[0][0])) for vid,r in ref.items()})
    vocab=vocabulary(train);dump(data/'vocabulary_activitynet.json',vocab)
    save_features(features,set(train)|set(val),data/'features/clipvitl14.pth');del features
    build_bank(train,prepared/'bank/anet/clip')
    signature=hashlib.sha256(b''.join(p.read_bytes() for p in [caption/'train.json',caption/'val.json',data/'vocabulary_activitynet.json'])).hexdigest()
    cfg=make_config(prepared,args.run_name,work/'runs',args.mode,len(vocab['word_to_ix']),signature)
    validate_config(cfg)
    dump(prepared/'run_manifest.json',dict(source_commit=load_helper().revision(),dataset='ActivityNet Captions',
        annotation_repo='ttengwang/PDVC',annotation_commit=ANNOTATION_COMMIT,
        annotation_hashes={name:hashlib.sha256((anno/name).read_bytes()).hexdigest() for name in ANNOTATION_FILES},
        feature_drive_id=FEATURE_ID,feature_source=str(source),feature_bytes=source.stat().st_size,
        coverage=coverage,dropped_invalid_training_events=dropped,train_ids=list(train),val_ids=list(val),
        reference_counts=[len(r) for r in refs],data_signature=signature,
        protocol='Same current model/50 queries; val_1 loader with val_2 fallback; both references evaluated separately',
        note='No YouCook2 checkpoint, vocabulary or memory bank reused; not original paper ActivityNet query setting'))
    path=prepared/'config.yml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    print('CONFIG_READY',path,flush=True)


def restage(args):
    import yaml
    cfg=yaml.safe_load(args.config.read_text());validate_config(cfg)
    source=ensure_feature_file(args.features.resolve() if args.features else args.work.resolve()/'assets/clipvitl14.pth')
    features=load_features(source)
    ids=set(json.loads(Path(cfg['train_caption_file']).read_text()))|set(json.loads(Path(cfg['val_caption_file']).read_text()))
    if ids-set(features):raise ValueError('Restored data IDs missing from downloaded feature file')
    save_features(features,ids,cfg['visual_feature_folder'][0])
    print('RESTAGED',len(ids),'ActivityNet videos',flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--work',type=Path,required=True)
    p.add_argument('--run-name',required=True);p.add_argument('--mode',choices=['smoke','subset','full'],default='full')
    p.add_argument('--features',type=Path);p.add_argument('--data-root',type=Path)
    p=sub.add_parser('check');p.add_argument('--config',type=Path,required=True)
    p=sub.add_parser('restage');p.add_argument('--work',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--features',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_name):raise ValueError('Unsafe run-name')
        prepare(args)
    elif args.command=='restage':restage(args)
    else:
        import yaml
        validate_config(yaml.safe_load(args.config.read_text()))


if __name__=='__main__':main()
