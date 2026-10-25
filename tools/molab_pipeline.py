"""Molab native-environment CLI. No notebook, venv, git clone or root needed at runtime."""
import argparse
import ast
import collections
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def revision():
    manifest=ROOT.parent/'BUNDLE_MANIFEST.json'
    if manifest.exists():return json.loads(manifest.read_text())['source_commit']
    if (ROOT/'.git').exists():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    return 'unversioned-export'

def patch_runtime(args):
    for rel in ['data/video_dataset.py','ddvc/CaptioningHead/gpt2.py']:
        p=ROOT/rel;s=p.read_text()
        s=s.replace("GPT2Tokenizer.from_pretrained('/DATA_DISK/plm_cache/gpt2')", "GPT2Tokenizer.from_pretrained('gpt2')")
        p.write_text(s)
    p=ROOT/'transformers/dependency_versions_table.py';s=p.read_text()
    s=s.replace('"tokenizers": "tokenizers>=0.19,<0.20",','"tokenizers": "tokenizers>=0.19,<0.24",')
    if 'tokenizers>=0.19,<0.24' not in s:raise RuntimeError('Unexpected vendored tokenizers constraint')
    p.write_text(s)
    p=ROOT/'ddvc/ops/src/cuda/ms_deform_attn_cuda.cu';s=p.read_text()
    s=s.replace('AT_DISPATCH_FLOATING_TYPES(value.type(),','AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),').replace('.data<','.data_ptr<')
    p.write_text(s)
    p=ROOT/'eval_utils_clip.py';s=p.read_text()
    old="    scores_g = eval_metrics_grounding(dvc_json_path + '.grounding.json', gt_filename=opt.eval_gt_file_for_grounding)\n    aux_scores_g = eval_metrics_grounding(dvc_json_path + '_aux.grounding.json', gt_filename=opt.eval_gt_file_for_grounding)\n"
    new="    if getattr(opt, 'eval_enable_grounding', True):\n        scores_g = eval_metrics_grounding(dvc_json_path + '.grounding.json', gt_filename=opt.eval_gt_file_for_grounding)\n        aux_scores_g = eval_metrics_grounding(dvc_json_path + '_aux.grounding.json', gt_filename=opt.eval_gt_file_for_grounding)\n    else:\n        scores_g, aux_scores_g = {}, {}\n"
    if old in s:s=s.replace(old,new,1)
    elif new not in s:raise RuntimeError('Could not patch unrelated ActivityNet grounding evaluation')
    p.write_text(s)
    provenance=ROOT.parent/'provenance/runtime_sources'
    for rel in ['data/video_dataset.py','ddvc/CaptioningHead/gpt2.py',
                'transformers/dependency_versions_table.py','ddvc/ops/src/cuda/ms_deform_attn_cuda.cu','eval_utils_clip.py']:
        target=provenance/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,target)
    print('Applied explicit runtime compatibility patches; GPU torch unchanged.',flush=True)

def build_cuda(args):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    from torch.utils.cpp_extension import CUDA_HOME
    nvcc=Path(CUDA_HOME)/'bin/nvcc' if CUDA_HOME else None
    if not nvcc or not nvcc.exists():raise RuntimeError('CUDA development toolkit / nvcc missing. Do not start training; select a runtime with CUDA build tools.')
    cccl=Path(CUDA_HOME)/'include/cccl'
    if (cccl/'nv/target').is_file():
        print('CUDA CCCL headers:',cccl,flush=True)
    major,minor=torch.cuda.get_device_capability(0)
    env=os.environ.copy();env.update(TORCH_CUDA_ARCH_LIST=f'{major}.{minor}',MAX_JOBS=str(args.jobs))
    if (cccl/'nv/target').is_file():
        env['CPATH']=str(cccl)+(os.pathsep+env['CPATH'] if env.get('CPATH') else '')
    print('Building for GPU',torch.cuda.get_device_name(0),'arch',env['TORCH_CUDA_ARCH_LIST'],flush=True)
    # Build locally: no root privileges, package installation or torchvision replacement.
    subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=ROOT/'ddvc/ops',env=env,check=True,timeout=1800)
    env['PYTHONPATH']=str(ROOT/'ddvc/ops')+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,'-c',
        'import torch; import MultiScaleDeformableAttention as op; print("EXTENSION_LOAD_PASS",op.__file__)'],
        cwd=ROOT,env=env,check=True)

def inspect_env(args):
    import torch
    print(json.dumps({'python':sys.executable,'version':sys.version,'torch':torch.__version__,
        'torch_cuda':torch.version.cuda,'cuda':torch.cuda.is_available(),
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'vram_gib':torch.cuda.get_device_properties(0).total_memory/2**30 if torch.cuda.is_available() else None,
        'compute_capability':torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        'compiled_arches':torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        'nvcc':shutil.which('nvcc'),'java':shutil.which('java'),'source':revision()},indent=2))
    if not torch.cuda.is_available():raise RuntimeError('Selected interpreter has no CUDA GPU')

def install_java(args):
    if shutil.which('java'):
        subprocess.run(['java','-version'],check=True);return
    import platform,tarfile,urllib.request,tempfile
    target=Path(args.target).expanduser().resolve()
    if (target/'bin/java').exists():print('JAVA_READY',target);return
    arch={'x86_64':'x64','aarch64':'aarch64'}.get(platform.machine())
    if not arch:raise RuntimeError('Unsupported Java architecture: '+platform.machine())
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        archive=Path(temporary)/'jre.tar.gz'
        url=f'https://api.adoptium.net/v3/binary/latest/17/ga/linux/{arch}/jre/hotspot/normal/eclipse'
        print('Downloading JRE (~50 MB):',url,flush=True)
        urllib.request.urlretrieve(url,archive)
        with tarfile.open(archive) as tar:
            tar.extractall(Path(temporary)/'extracted',filter='data')
        roots=list((Path(temporary)/'extracted').iterdir())
        if len(roots)!=1 or not (roots[0]/'bin/java').exists():raise RuntimeError('Unexpected JRE archive')
        shutil.move(str(roots[0]),target)
    print('JAVA_READY',target,flush=True)

def restage(args):
    """Restore downloadable video features after Drive restored configs/banks/checkpoints."""
    import torch,yaml,gdown
    cfg=yaml.safe_load(Path(args.config).read_text())
    source=Path(args.data_root).expanduser().resolve()/'clipvitl14.pth'
    source.parent.mkdir(parents=True,exist_ok=True)
    if not source.exists():
        gdown.download(id='17H_lxSKFve57kHpkAD7pcYS7ijHIy4-M',output=str(source),quiet=False)
    data=torch.load(source,map_location='cpu',weights_only=False)
    if 'features' in data and isinstance(data['features'],dict):data=data['features']
    ids=set(json.loads(Path(cfg['train_caption_file']).read_text()))|set(json.loads(Path(cfg['val_caption_file']).read_text()))
    missing=ids-set(data)
    if missing:raise ValueError('Restored run IDs lack video features: '+str(sorted(missing)[:10]))
    target=Path(cfg['visual_feature_folder'][0]);target.parent.mkdir(parents=True,exist_ok=True)
    torch.save({k:torch.as_tensor(data[k]).detach().cpu().float() for k in ids},target)
    print('RESTAGED',len(ids),'videos to',target,flush=True)

def smoke(args):
    import torch,yaml
    from opts import parse_opts
    from torch.utils.data import DataLoader
    from train import build,PropSeqDataset,collate_fn,load_clip_memory_bank,load_clip_gt
    from misc.training_state import schedule_contrastive,validate_resume
    # Follow train.py: parser defaults first, then YAML overrides. opts.json is historical
    # and intentionally does not contain every option added to the current source.
    opt=parse_opts(['--cfg_path',str(Path(args.config).resolve())],write_options=False)
    if not opt.text_crossAttn:opt.cap_num_feature_levels=4
    if opt.combined_encoder:opt.cap_num_feature_levels=5
    if opt.feature_dim==512 and opt.decoder_query_ablation:opt.num_queries=opt.decoder_query_num
    if opt.caption_decoder_new=='gpt2':opt.caption_decoder_type='gpt2'
    opt.rec_encdec=True
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    ds=PropSeqDataset(opt.train_caption_file,opt.visual_feature_folder,opt.dict_file,True,'gt',opt)
    from misc.runtime_perf import configure_runtime,amp_context,vram_metrics
    print('RUNTIME',configure_runtime(opt),flush=True)
    model,criterion,contrastive,postprocessors=build(opt);model.translator=ds.translator;model.to(opt.device)
    if args.checkpoint:
        cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        validate_resume(vars(opt),cp['config']);model.load_state_dict(cp['model'],strict=True)
        print('CHECKPOINT_LOAD_PASS',cp.get('phase'),cp.get('next_epoch'),flush=True)
    else:print('Fresh training: random decoder; no pretrained caption checkpoint requested.',flush=True)
    bank=load_clip_memory_bank(opt);bank['vid_sent_embeds']=torch.as_tensor(bank['vid_sent_embeds'],device=opt.device)
    bank['clip_gt']=load_clip_gt(opt)
    dt=next(iter(DataLoader(ds,batch_size=1,num_workers=0,collate_fn=collate_fn)))
    dt={k:v.to(opt.device) if isinstance(v,torch.Tensor) else v for k,v in dt.items()}
    dt['video_target']=[{k:v.to(opt.device) if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in dt['video_target']]
    dt=collections.defaultdict(lambda:None,dt)
    schedule_contrastive(criterion.weight_dict,0,opt.cl_schedule_time,opt.cl_schedule_val,opt.corrected_query_contrastive)
    if args.benchmark_steps<1:raise ValueError('benchmark-steps must be positive')
    model.train();torch.cuda.reset_peak_memory_stats();timings=[]
    for step in range(args.benchmark_steps+1):
        model.zero_grad(set_to_none=True);torch.cuda.synchronize();started=time.perf_counter()
        with amp_context(opt):
            output,loss=model(dt,criterion,contrastive,opt.transformer_input_type,bank)
            total=sum(v.float()*criterion.weight_dict[k] for k,v in loss.items() if k in criterion.weight_dict)
        if not torch.isfinite(total):raise RuntimeError('Non-finite smoke loss')
        total.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError('Non-finite smoke gradients')
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        if step:timings.append(elapsed)
        print('SMOKE_STEP',step,'loss',float(total.detach()),'seconds',elapsed,flush=True)
        del output,loss,total
    model.zero_grad(set_to_none=True);model.eval()
    with torch.no_grad(),amp_context(opt):
        output,loss=model(dt,criterion,contrastive,opt.transformer_input_type,bank,eval_mode=True)
    loader=DataLoader(ds,batch_size=1,num_workers=0,collate_fn=collate_fn)
    postprocessors['bbox'](output,dt['video_length'][:,1],loader,model,None)
    postprocessors['bbox'].forward_grounding(output,dt['video_length'][:,1],dt['video_target'])
    report=dict(precision=opt.precision,benchmark_steps=args.benchmark_steps,
                mean_forward_backward_seconds=sum(timings)/len(timings),
                benchmark='same training video, no optimizer update; excludes first warmup',**vram_metrics(opt.device))
    dest=Path(args.config).parent/'smoke_performance.json';dest.write_text(json.dumps(report,indent=2))
    print('REAL_MODEL_FORWARD_BACKWARD_PASS',json.dumps(report),flush=True)
    print('GENERATION_POSTPROCESS_PASS',dest,flush=True)


def resume_config(args):
    import yaml
    p=Path(args.config);cfg=yaml.safe_load(p.read_text());run=Path(cfg['save_dir'])/cfg['id']
    if not (run/'model-last.pth').exists():raise FileNotFoundError(run/'model-last.pth')
    cfg['start_from']=cfg['id'];cfg['start_from_mode']='last'
    p.write_text(yaml.safe_dump(cfg,sort_keys=False));print('RESUME_CONFIG_READY',p,flush=True)

def report(args):
    import yaml
    cfg=yaml.safe_load(Path(args.config).read_text());run=Path(cfg['save_dir'])/cfg['id']
    print('Run:',run)
    for name in ['model-last.pth','model-best.pth','info.json']:
        p=run/name;print(name,p.stat().st_size if p.exists() else 'missing')
    perf=run/'performance.jsonl'
    if perf.exists() and perf.stat().st_size:print('Latest performance:',perf.read_text().splitlines()[-1])
    p=run/'info.json'
    if p.exists():
        info=json.loads(p.read_text());best=info.get('best',{})
        print('Best:',json.dumps({k:best.get(k) for k in ['epoch','best_val_score','result_json_path']},ensure_ascii=False))
        for epoch,row in info.get('history',{}).get('val_result_history',{}).items():
            print('epoch',epoch,json.dumps(row.get('eval_score',{})))

def resolve_profile(args):
    defaults=dict(accumulation=1,precision='fp32',caption_chunk=16,workers=0,
                  bank_batch=64,allow_tf32=False,pin_memory=False)
    for key,value in defaults.items():
        if getattr(args,key,None) is None:setattr(args,key,value)
    if args.accumulation!=1 or args.precision!='fp32' or args.allow_tf32:
        raise ValueError('Legacy reproduction requires batch 1, accumulation 1, FP32, TF32 disabled')
    if args.caption_chunk!=16 or args.workers!=0 or args.bank_batch!=64 or args.pin_memory:
        raise ValueError('Legacy reproduction keeps caption chunk 16, workers 0, bank batch 64, no pinned memory')
    return args


def prepare(args):
    resolve_profile(args)
    from types import SimpleNamespace
    from misc.runtime_perf import configure_runtime
    print("RUNTIME_PREFLIGHT",configure_runtime(SimpleNamespace(device="cuda",precision=args.precision,allow_tf32=args.allow_tf32)),flush=True)
    import torch
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_name):raise ValueError('Use a simple run-name: letters, digits, underscores, hyphens')
    if args.epochs<1 or args.accumulation<1 or args.log_every<1:raise ValueError('Epochs/accumulation/log interval must be positive')
    if args.mask_loss!=1.0:raise ValueError('Legacy recipe keeps proposal mask loss coefficient 1.0')
    if not torch.cuda.is_available():
        raise RuntimeError('Select a Molab GPU before CLIP-bank preparation')
    REPO = ROOT
    DATA_ROOT = Path(args.data_root).expanduser().resolve()
    PREPARED = Path(args.prepared_root).expanduser().resolve()/args.run_name
    if (PREPARED/'config.yml').exists():
        raise FileExistsError('Prepared run already exists; reuse its config or select a fresh --run-name')
    PREPARED.mkdir(parents=True,exist_ok=True)
    RUN_MODE, SEED, EPOCHS = args.mode, args.seed, args.epochs
    ACTIVE_FRACTION = {'smoke':.01,'subset':.30,'full':1.}[RUN_MODE]
    RUN_ID, RESUME_RUN_ID, MASK_MODE = args.run_name, '', args.mask_mode
    SAVE_DIR = str(Path(args.output_root).expanduser().resolve())
    Path(SAVE_DIR).mkdir(parents=True,exist_ok=True)
    LOG_PATH = Path(args.log_root).expanduser().resolve()/(RUN_ID+'.log')
    SOURCE_COMMIT = revision()
    MASK_BINS,MASK_ENCODING_DIM,MASK_LOSS_COEF=64,128,args.mask_loss
    MASK_TARGET='predicted'
    QUERY_CONDITIONING=False
    PRUNE_THRESHOLD=-1.0
    GATE_MIN,GATE_MAX=0.0,1.0
    MASK_CONTEXT_RATIO,MASK_MIN_FRAMES=0.0,1
    BALANCED_MASK_BCE=False
    CORRECTED_QUERY_CONTRASTIVE=False
    LOCAL_CONTEXT_RATIO,LOCAL_MIN_FRAMES=.1,4
    random.seed(SEED);torch.manual_seed(SEED)
    import numpy as np
    np.random.seed(SEED)
    import json, tarfile, requests

    DATA_ROOT.mkdir(parents=True, exist_ok=True)


    def download_file(url, destination):
        destination=Path(destination)
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        print('Downloading:', url)
        partial=destination.with_name(destination.name+'.part')
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with partial.open('wb') as f:
                for chunk in response.iter_content(chunk_size=8*1024*1024):
                    if chunk:
                        f.write(chunk)
        partial.replace(destination)
        return destination


    def convert_youcook2(raw):
        database=raw.get('database', raw)
        splits={'training':{}, 'validation':{}}
        for video_id, item in database.items():
            subset=item.get('subset')
            if subset not in splits:
                continue
            annotations=[a for a in item.get('annotations', [])
                         if a.get('sentence') and isinstance(a.get('segment'), (list, tuple)) and len(a['segment']) == 2]
            if not annotations:
                continue
            splits[subset][video_id]={
                'duration': float(item['duration']),
                'sentences': [a['sentence'] for a in annotations],
                'timestamps': [[float(a['segment'][0]), float(a['segment'][1])] for a in annotations],
            }
        return splits['training'], splits['validation']


    # Official YouCook2 combined train/validation annotations.
    try:
        ann_tar=download_file(
            'https://youcook2.eecs.umich.edu/static/YouCookII/youcookii_annotations_trainval.tar.gz',
            DATA_ROOT/'youcookii_annotations_trainval.tar.gz')
        with tarfile.open(ann_tar, 'r:gz') as archive:
            members=[m for m in archive.getmembers() if Path(m.name).name == 'youcookii_annotations_trainval.json' and m.isfile()]
            if len(members) != 1:
                raise FileNotFoundError('Official annotation archive did not contain exactly one youcookii_annotations_trainval.json')
            with archive.extractfile(members[0]) as f:
                raw_annotations=json.load(f)
    except requests.exceptions.SSLError:
        # Kaggle mirror of the same official annotation file, used only if the UMich TLS chain fails.
        print('Official host SSL verification failed; using the Kaggle mirror of the official annotation file.')
        ann_json=download_file(
            'https://www.kaggle.com/api/v1/datasets/download/ravishkumarcs23m055/annotation-youcook2/youcookii_annotations_trainval.json',
            DATA_ROOT/'youcookii_annotations_trainval.json')
        with ann_json.open(encoding='utf-8') as f:
            raw_annotations=json.load(f)

    train_all,val_all=convert_youcook2(raw_annotations)
    del raw_annotations
    assert train_all and val_all, 'Official YouCook2 train/validation annotations did not load.'
    print('Official annotations:',len(train_all),'train videos,',len(val_all),'validation videos')

    # DDVC's pre-extracted CLIP ViT-L/14 features (linked from its official repository README).
    feature_src=DATA_ROOT/'clipvitl14.pth'
    if not feature_src.exists() or feature_src.stat().st_size < 100*1024*1024:
        if feature_src.exists(): feature_src.unlink()
        import gdown
        downloaded=gdown.download(
            id='17H_lxSKFve57kHpkAD7pcYS7ijHIy4-M',
            output=str(feature_src), quiet=False)
        if downloaded is None or not feature_src.exists() or feature_src.stat().st_size < 100*1024*1024:
            raise RuntimeError('CLIP feature download failed or is incomplete. Enable Kaggle Internet and retry; expected the DDVC clipvitl14.pth file.')
    print('DDVC CLIP features:',feature_src,round(feature_src.stat().st_size/1024**2),'MiB')

    for split,data in [('train',train_all),('validation',val_all)]:
        assert isinstance(data,dict) and data, f'{split} annotation split is empty'
        assert {'duration','sentences','timestamps'} <= set(next(iter(data.values()))), f'{split}.json has an unexpected format'
    try:
        feature_map=torch.load(feature_src,map_location='cpu',weights_only=False)
    except TypeError:
        feature_map=torch.load(feature_src,map_location='cpu')
    if 'features' in feature_map and isinstance(feature_map['features'],dict):
        feature_map=feature_map['features']
    assert isinstance(feature_map,dict), f'Expected a video-ID to feature mapping in {feature_src}'
    feature_ids=set(feature_map)


    def choose(data,n,seed,split_name):
        valid_ids=[k for k,v in data.items() if k in feature_ids and v.get('sentences') and len(v['sentences'])==len(v['timestamps'])]
        missing_ids=set(data)-feature_ids
        random.Random(seed).shuffle(valid_ids)
        if not valid_ids:
            raise ValueError(f'No {split_name} videos match the downloaded CLIP feature IDs.')
        selected_ids=valid_ids if n is None else valid_ids[:min(n,len(valid_ids))]
        selected={k:data[k] for k in selected_ids}
        events_all=sum(len(v['sentences']) for v in data.values())
        events_selected=sum(len(v['sentences']) for v in selected.values())
        print(f'{split_name}: {len(data):,} official annotated videos / {len(valid_ids):,} with features / {len(selected):,} used; {events_all:,} annotations / {events_selected:,} used')
        print(f'  Missing feature for {len(missing_ids):,} annotated videos; examples: {sorted(missing_ids)[:8]}')
        print(f'  Feature-matched but omitted by subset limit: {len(set(valid_ids)-set(selected_ids)):,}')
        return selected

    train_available=sum(k in feature_ids for k in train_all)
    val_available=sum(k in feature_ids for k in val_all)
    TRAIN_VIDEOS=None if RUN_MODE=='full' else max(1, round(train_available*ACTIVE_FRACTION))
    VAL_VIDEOS=None if RUN_MODE=='full' else max(1, round(val_available*ACTIVE_FRACTION))
    train=choose(train_all,TRAIN_VIDEOS,SEED,'train')
    val=choose(val_all,VAL_VIDEOS,SEED+1,'validation')
    assert not (set(train)&set(val)), 'Train/validation video IDs overlap.'

    REPO_DATA=PREPARED/'data'/'yc2'
    ANNO=REPO_DATA/'captiondata'
    ANNO.mkdir(parents=True,exist_ok=True)
    train_json,val_json=ANNO/'train.json',ANNO/'val.json'
    train_json.write_text(json.dumps(train),encoding='utf-8')
    val_json.write_text(json.dumps(val),encoding='utf-8')

    # Upstream paragraph evaluator expects video_id -> concatenated reference captions.
    para_dir=ANNO/'para'
    para_dir.mkdir(parents=True,exist_ok=True)
    para_json=para_dir/'para_yc2_val.json'
    para_refs={}
    for vid,record in val.items():
        ordered=sorted(zip(record['timestamps'],record['sentences']),key=lambda pair:float(pair[0][0]))
        para_refs[vid]=' '.join(sentence.strip().rstrip('.')+'.' for _,sentence in ordered)
    para_json.write_text(json.dumps(para_refs),encoding='utf-8')
    print('Wrote full matched validation references:',para_json,'videos:',len(para_refs))

    small={k:torch.as_tensor(feature_map[k]).detach().cpu().float() for k in set(train)|set(val)}
    features_path=REPO_DATA/'features'/f'clipvitl14_{RUN_MODE}.pth'
    features_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(small,features_path)
    del feature_map,small
    print('Prepared selected data:',len(train),'train,',len(val),'validation videos; feature file:',features_path)

    def words(s):
        for ch in [',',':','!','_',';','-','.','?','/','"','\\n','\\']:
            s=s.lower().replace(ch,' ')
        return [x for x in s.split() if x]
    counts=collections.Counter(w for v in train.values() for s in v['sentences'] for w in words(s))
    vocab=[w for w,n in counts.items() if n>=2]+['<bos>','<eos>','UNK']
    word_to_ix={w:i+1 for i,w in enumerate(vocab)}
    vocab_path=REPO_DATA/'vocabulary_youcook2.json'
    vocab_path.write_text(json.dumps({'word_to_ix':word_to_ix,'ix_to_word':{str(i):w for w,i in word_to_ix.items()}}))
    import clip
    sentences=[]; video_ids=[]
    for vid,record in train.items():
        for sent in record['sentences']:
            sentences.append(sent.strip()); video_ids.append(vid)
    model_clip,_=clip.load('ViT-L/14',device='cuda:0',jit=False); model_clip.eval()
    encoded=[]
    with torch.inference_mode():
        for j in range(0,len(sentences),args.bank_batch):
            tokens=clip.tokenize(sentences[j:j+args.bank_batch],truncate=True).to('cuda:0')
            encoded.append(model_clip.encode_text(tokens).float().cpu())
            print('BANK_PROGRESS encoded',min(j+args.bank_batch,len(sentences)),'/',len(sentences),flush=True)
    emb=torch.cat(encoded).numpy()
    assert emb.shape==(len(sentences),768), emb.shape
    for folder in ['clip','clip_arr_split']:
        out=PREPARED/'bank'/'yc2'/folder; out.mkdir(parents=True,exist_ok=True)
        import numpy as np
        np.save(out/'yc2_scene_sentences.npy',np.asarray(sentences,dtype=str))
        np.save(out/'yc2_clip_token_embeds.npy',emb)
        np.save(out/'yc2_scene_videoID.npy',np.asarray(video_ids,dtype=str))
    print('CLIP memory entries:',len(sentences),emb.shape)
    del model_clip,encoded,emb; torch.cuda.empty_cache()
    import yaml
    opts=json.loads((REPO/'opts.json').read_text())
    opts['experiment_recipe']='kaggle_cvpr2018_mask'
    opts.update({'lr':5e-5,'optimizer_type':'adam','weight_decay':1e-4,
                 'learning_rate_decay_start':8,'learning_rate_decay_every':3,'learning_rate_decay_rate':0.5})
    import hashlib
    data_signature = hashlib.sha256(train_json.read_bytes()+val_json.read_bytes()+vocab_path.read_bytes()).hexdigest()
    opts.update({
     'cfg_path':'','target_domain':'yc2','dataset':'yc2','bank_type':['yc2'],
     'train_caption_file':str(train_json),
     'val_caption_file':str(val_json),
     'gt_file_for_eval':[str(val_json)],
     'gt_file_for_auc':[str(val_json)],
     'gt_file_for_para_eval':[str(para_json)],
     'visual_feature_folder':[str(features_path)],
     'visual_feature_type':['clip'],'feature_dim':768,'frame_embedding_num':200,
     'dict_file':str(vocab_path),'vocab_size':len(word_to_ix),
     'bank_path':str(PREPARED/'bank'),'bank_folder':'clip','able_ret':True,
     'gpu_id':[0],'device':'cuda','batch_size':1,'batch_size_for_eval':1,
     'nthreads':args.workers,'pin_memory':args.pin_memory,'precision':args.precision,'allow_tf32':args.allow_tf32,'hardware_profile':args.hardware_profile,'fixed_run_id':True,'gradient_accumulation_steps':args.accumulation,'progress_log_every':args.log_every,'epoch':EPOCHS,'min_epoch_when_save':0,'save_checkpoint_every':1,
     'save_all_checkpoint':False,'save_dir':SAVE_DIR,'id':RUN_ID,
     'exp_name':f'{RUN_MODE}{len(train)}_e{EPOCHS}_{MASK_MODE}_mask_'+time.strftime('%Y%m%d_%H%M%S'),'start_from':RESUME_RUN_ID,'start_from_mode':'last','pretrain':None,
     'seed':SEED,'random_seed':False,'debug':False,'disable_tqdm':False,
     'train_proposal_sample_num':30,'gt_proposal_sample_num':10,
     'soft_k':20,'window_size':80,'ret_encoder':'multiscale_project',
     'max_caption_len':30,'gpt_beam_size':3,'num_queries':50,'decoder_query_ablation':False,
     'caption_decoder_type':'gpt2','caption_decoder_new':'none',
     'eval_enable_grounding':False,
     'with_box_refine':True,
     # Keep DDVC objectives; mask target and corrected contrastive recipe are explicit below.
     'bbox_loss_coef':0.0,'set_cost_bbox':0.0,
     'local_visual_attention':MASK_MODE=='hard_local',
     'learned_proposal_mask':MASK_MODE=='learned_gated',
     'proposal_mask_bins':MASK_BINS,'proposal_mask_encoding_dim':MASK_ENCODING_DIM,
     'proposal_mask_loss_coef':MASK_LOSS_COEF,
     'mask_target':MASK_TARGET,'mask_query_conditioning':QUERY_CONDITIONING,
     'mask_prune_threshold':PRUNE_THRESHOLD,'mask_gate_min':GATE_MIN,'mask_gate_max':GATE_MAX,
     'mask_context_ratio':MASK_CONTEXT_RATIO,'mask_min_frames':MASK_MIN_FRAMES,
     'mask_balanced_bce':BALANCED_MASK_BCE,'corrected_query_contrastive':CORRECTED_QUERY_CONTRASTIVE,
     'data_signature':data_signature,
     'local_visual_context_ratio':LOCAL_CONTEXT_RATIO,'local_visual_min_frames':LOCAL_MIN_FRAMES,
     'gpt_layer':2,'add_cross_attention':True,
     # Make the existing reranking defaults explicit for reproducibility.
     'ec_alpha':1.0,'count_loss_coef':0.5,'eval_matching_score_weight':0.0,
     'caption_cost_chunk_size':args.caption_chunk
    })
    if RUN_MODE != 'smoke':
        # Load our sibling by absolute path: imported dependencies can alter sys.path.
        import importlib.util
        recipe_spec = importlib.util.spec_from_file_location('ddvc_kaggle_recipe', ROOT/'tools'/'kaggle_recipe.py')
        recipe_module = importlib.util.module_from_spec(recipe_spec)
        recipe_spec.loader.exec_module(recipe_module)
        recipe_module.validate(opts, RUN_MODE)
    PREPARED.mkdir(parents=True,exist_ok=True)
    cfg=PREPARED/'config.yml'
    cfg.write_text(yaml.safe_dump(opts,sort_keys=False))
    print(cfg, '| selected train/val:',len(train),len(val),'| vocab:',len(word_to_ix),'| epochs:',EPOCHS)
    print('DDVC mask mode:',MASK_MODE,'| random 2-layer GPT-2 | one GPU')
    print('New objective:',{k:opts[k] for k in ['learned_proposal_mask','proposal_mask_loss_coef','proposal_mask_bins']})
    print("Reranking:", {k: opts[k] for k in ("ec_alpha", "count_loss_coef", "eval_matching_score_weight")})

    # Preserve exactly what was configured and the actual selected data IDs.
    run_manifest = {
        'gradient_accumulation_steps':args.accumulation,'maximum_effective_batch':args.accumulation,
        'source_commit':SOURCE_COMMIT,'legacy_model_commit':'4918a0cb89bac0966f74c39e7320d44f0a780430','reference_repo':'https://github.com/bhubanendra-mishra/dense-video-cap',
        'reference_commit':'43914e17769701b9cf98eda203ae4c465b315fab',
        'mask_mode':MASK_MODE,'seed':SEED,'epochs':EPOCHS,
        'runtime_profile':{k:opts[k] for k in ['hardware_profile','precision','allow_tf32','nthreads','pin_memory','caption_cost_chunk_size']},
        'train_ids':list(train),'val_ids':list(val),
        'mask_target':MASK_TARGET, 'data_signature':data_signature,
        'mask_settings':{k:opts[k] for k in opts if k.startswith('mask_') or k=='corrected_query_contrastive'},
        'mask_placement':'legacy feature gating before GPT-2 cross-attention; no attention pruning',
    }
    manifest_path = PREPARED/'run_manifest.json'
    manifest_path.write_text(json.dumps(run_manifest,indent=2),encoding='utf-8')

    print('Action-aware mask recipe:',run_manifest['mask_settings'])
    print("CONFIG_READY",cfg,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('patch');sub.add_parser('env')
    p=sub.add_parser('java');p.add_argument('--target',required=True)
    p=sub.add_parser('restage');p.add_argument('--config',required=True);p.add_argument('--data-root',required=True)
    p=sub.add_parser('build-cuda');p.add_argument('--jobs',type=int,default=2)
    p=sub.add_parser('prepare')
    p.add_argument('--data-root',required=True);p.add_argument('--prepared-root',required=True)
    p.add_argument('--output-root',required=True);p.add_argument('--log-root',required=True)
    p.add_argument('--run-name',required=True);p.add_argument('--mode',choices=['smoke','subset','full'],default='full')
    p.add_argument('--epochs',type=int,default=15);p.add_argument('--accumulation',type=int,default=None)
    p.add_argument('--hardware-profile',choices=['legacy-cvpr2018'],default='legacy-cvpr2018')
    p.add_argument('--precision',choices=['fp32','bf16'],default=None)
    p.add_argument('--caption-chunk',type=int,default=None);p.add_argument('--workers',type=int,default=None)
    p.add_argument('--bank-batch',type=int,default=None)
    p.add_argument('--allow-tf32',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--pin-memory',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--seed',type=int,default=777);p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--mask-mode',choices=['global','hard_local','learned_gated'],default='learned_gated')
    p.add_argument('--mask-loss',type=float,default=1.0)
    for name in ['smoke','resume-config','report']:
        p=sub.add_parser(name);p.add_argument('--config',required=True)
        if name=='smoke':
            p.add_argument('--checkpoint',default='');p.add_argument('--benchmark-steps',type=int,default=3)
    args=parser.parse_args()
    os.chdir(ROOT)
    {'prepare':prepare,'patch':patch_runtime,'env':inspect_env,'build-cuda':build_cuda,
     'smoke':smoke,'resume-config':resume_config,'report':report,
     'java':install_java,'restage':restage}[args.command](args)

if __name__=='__main__':main()
