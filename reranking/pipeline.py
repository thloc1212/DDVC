"""Frozen-C1 extraction, train-only head fitting and fixed-candidate evaluation."""
import collections
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from .cache import atomic_json,sha,write_shard,read_shard
from .core import SCHEMA,fit_idf,targets,feature_pool,select,dev_reward
from .head import JointQualityHead,objective

ROOT=Path(__file__).resolve().parents[1]
BETAS=(0.,.25,.5,1.,2.)
TRAINING_SEED=2016
HEAD_SPLIT_SEED=777
FIXED_BETAS={'yc2':1.,'anet':6.}
METRICS=('CIDEr','METEOR','Bleu_4','soda_c','Precision','Recall','MetaScore')
SCORING_PROTOCOL='ddvc-2018-full-language-infer-mode-v1'


def complete_scores(scores):
    return isinstance(scores,dict) and all(
        key in scores and isinstance(scores[key],(float,int)) and np.isfinite(scores[key]) for key in METRICS)


def write_table(path,rows):
    import csv
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(rows)


def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');torch.save(value,temp);os.replace(temp,path)


def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


class Workflow:
    def __init__(self,work,base,dataset):
        self.work=Path(work).resolve();self.base=Path(base).resolve();self.dataset=dataset
        self.folder=self.work/'runs'/dataset;self.folder.mkdir(parents=True,exist_ok=True)
        # Prefer the exact baseline checkout and its already-compiled native operator.
        for path in (self.base/'source/ddvc/ops',self.base/'source/tools',self.base/'source'):
            sys.path.insert(0,str(path))
        import c1_pipeline
        self.c1=c1_pipeline
        self.config_path=c1_pipeline.config_path(self.base,dataset)
        self.cfg=c1_pipeline.check(self.config_path)
        self.base_run=Path(self.cfg['save_dir'])/self.cfg['id']
        info=json.loads((self.base_run/'info.json').read_text(encoding='utf8'))
        if info['last']['epoch']<self.cfg['epoch']-1:raise ValueError('Finish C1 training before fitting this head')
        self.epoch=info['best']['epoch'];self.checkpoint=self.base_run/'model-best.pth'
        self.raw_path=self.base_run/'prediction'/Path(info['best']['result_json_path']).name.replace('.json','.raw.json')
        self.final_path=self.raw_path.with_name(self.raw_path.name.replace('.raw.json','.json'))
        self.annotations=json.loads(Path(self.cfg['train_caption_file']).read_text(encoding='utf8'))
        val=json.loads(Path(self.cfg['val_caption_file']).read_text(encoding='utf8'))
        if set(val)&set(self.annotations):raise ValueError('Baseline train/validation ID leakage')
        ids=sorted(self.annotations);order=np.random.default_rng(HEAD_SPLIT_SEED).permutation(len(ids))
        dev={ids[i] for i in order[:max(1,round(.1*len(ids))) ]};fit=[i for i in ids if i not in dev]
        self.folds={vid:('head_dev' if vid in dev else 'head_fit') for vid in ids}
        self.idf=fit_idf(self.annotations,fit)
        raw=json.loads(self.raw_path.read_text(encoding='utf8'))['results']
        if set(raw)!=set(val):raise ValueError('Saved validation candidates do not cover the configured validation split')
        self.manifest=dict(schema=SCHEMA,dataset=dataset,base_checkpoint_sha256=sha(self.checkpoint),
            base_epoch=self.epoch,base_config_sha256=sha(self.config_path),raw_sha256=sha(self.raw_path),
            data_signature=self.cfg['data_signature'],fit_ids=fit,dev_ids=sorted(dev),
            idf_sha256=signature(self.idf),epochs=20,head_lr=.001,batch_videos=8,beta_grid=list(BETAS),
            training_seed=TRAINING_SEED,head_split_seed=HEAD_SPLIT_SEED,fixed_beta=FIXED_BETAS[dataset],
            feature_source='local/context CLIP and existing frozen C1 GPT2 mean token embeddings',
            target='max over SAME GT: temporal IoU * fit-IDF weighted caption unigram F1')
        self.sig=signature(self.manifest);self.manifest['signature']=self.sig
        path=self.folder/'manifest.json'
        if path.is_file() and json.loads(path.read_text(encoding='utf8'))!=self.manifest:
            raise ValueError('Frozen model/data/recipe fingerprint changed; use a new reranker work directory')
        atomic_json(path,self.manifest);atomic_json(self.folder/'fit_idf.json',self.idf)
        print('FROZEN_C1_READY',dataset,'base_epoch',self.epoch,'head_fit',len(fit),'head_dev',len(dev),flush=True)

    def chunks(self,split):
        if split=='train':ids=sorted(self.annotations)
        else:ids=sorted(json.loads(self.raw_path.read_text(encoding='utf8'))['results'])
        return [(ids[i:i+32],self.folder/'cache'/split/f'shard_{i//32:05d}.npz') for i in range(0,len(ids),32)]

    def load(self,split):
        result=[]
        for ids,path in self.chunks(split):
            rows=read_shard(path,self.sig,split)
            if [r['video_id'] for r in rows]!=ids:raise ValueError('Cached video IDs/order changed')
            result.extend(rows)
        return result

    def load_base(self):
        if not torch.cuda.is_available():raise RuntimeError('Candidate extraction requires the working C1 GPU environment')
        os.chdir(self.base/'source')
        from opts import parse_opts
        from train import build,PropSeqDataset,collate_fn,load_clip_memory_bank,load_clip_gt
        from misc.runtime_perf import configure_runtime
        from misc.training_state import schedule_contrastive
        opt=parse_opts(['--cfg_path',str(self.config_path)],write_options=False)
        if not opt.text_crossAttn:opt.cap_num_feature_levels=4
        if opt.combined_encoder:opt.cap_num_feature_levels=5
        if opt.caption_decoder_new=='gpt2':opt.caption_decoder_type='gpt2'
        opt.rec_encdec=True;configure_runtime(opt)
        random.seed(TRAINING_SEED);np.random.seed(TRAINING_SEED);torch.manual_seed(TRAINING_SEED)
        train=PropSeqDataset(opt.train_caption_file,opt.visual_feature_folder,opt.dict_file,False,'gt',opt)
        val=PropSeqDataset(opt.val_caption_file,opt.visual_feature_folder,opt.dict_file,False,'gt',opt)
        model,criterion,contrastive,post=build(opt);model.translator=train.translator
        cp=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        if cp.get('epoch')!=self.epoch:raise ValueError('C1 best checkpoint epoch differs from info.json')
        model.load_state_dict(cp['model'],strict=True);del cp
        model.to(opt.device).eval()
        for p in model.parameters():p.requires_grad_(False)
        schedule_contrastive(criterion.weight_dict,self.epoch,opt.cl_schedule_time,opt.cl_schedule_val,False)
        bank=load_clip_memory_bank(opt);bank['vid_sent_embeds']=torch.as_tensor(bank['vid_sent_embeds'],device=opt.device)
        bank['clip_gt']=load_clip_gt(opt)
        def infer(ds,index):
            np.random.seed(TRAINING_SEED+index);item=ds[index]
            dt=collate_fn([item]);dt={k:v.to(opt.device) if isinstance(v,torch.Tensor) else v for k,v in dt.items()}
            dt['video_target']=[{k:v.to(opt.device) if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in dt['video_target']]
            dt=collections.defaultdict(lambda:None,dt)
            with torch.no_grad():
                pred,_=model(dt,criterion,contrastive,opt.transformer_input_type,bank,eval_mode=True)
                output=post['bbox'](pred,dt['video_length'][:,1],SimpleNamespace(dataset=ds),model,None)[0]
            rows=[]
            for i,box in enumerate(output['boxes']):
                if float(box[1])<=float(box[0]):continue
                rows.append(dict(timestamp=box.float().cpu().tolist(),query_id=int(output['query_id'][i]),
                    proposal_score=float(output['scores'][i]),sentence=output['captions'][i],
                    sentence_score=float(output['caption_scores'][i]),pred_event_count=int(output['pred_seq_len'])))
            if not rows:raise ValueError('No valid candidates')
            return rows,torch.as_tensor(item[0],device=opt.device),float(item[6])
        return model,train,val,infer

    def extract(self):
        # Completed shards are read and hash-checked before deciding whether GPU work is needed.
        missing=[]
        for split in ('train','val'):
            for ids,path in self.chunks(split):
                if path.is_file() and path.with_suffix('.json').is_file():
                    existing=read_shard(path,self.sig,split)
                    if [v['video_id'] for v in existing]!=ids:raise ValueError('Shard video order changed')
                else:missing.append((split,ids,path))
        if not missing:print('EXTRACTION_ALREADY_COMPLETE',flush=True);return
        model,train,val,infer=self.load_base();raw=json.loads(self.raw_path.read_text(encoding='utf8'))['results']
        train_index={str(v):i for i,v in enumerate(train.keys)};val_index={str(v):i for i,v in enumerate(val.keys)}
        head=model.caption_head[-1];tokenizer=head.tokenizer;wte=head.model.transformer.wte
        # Actual full-model GPU smoke, compared to the frozen saved validation candidates.
        smoke=[]
        for vid in sorted(raw)[:2]:
            rows,_,duration=infer(val,val_index[vid]);a={r['query_id']:r for r in rows};b={r['query_id']:r for r in raw[vid]}
            if set(a)!=set(b):raise ValueError('C1 candidate IDs changed during smoke')
            delta=max(abs(x-y)/duration for q in a for x,y in zip(a[q]['timestamp'],b[q]['timestamp']))
            if delta>1e-5 or any(a[q]['sentence']!=b[q]['sentence'] for q in a):
                raise ValueError('C1 GPU smoke does not reproduce saved candidates; stop before head training')
            smoke.append(dict(video_id=vid,max_normalized_box_delta=delta,captions_identical=True))
        atomic_json(self.folder/'gpu_smoke.json',dict(passed=True,base_gradients_disabled=True,checks=smoke))
        print('REAL_FROZEN_C1_GPU_SMOKE_PASS',json.dumps(smoke),flush=True)
        completed=0
        for split,ids,path in missing:
            videos=[];arrays=[];started=time.time()
            for vid in ids:
                if split=='train':rows,features,duration=infer(train,train_index[vid])
                else:
                    rows=raw[vid];features=val.viz_feats[vid].detach().to(wte.weight.device).float()
                    duration=float(rows[0]['vid_duration'])
                arrays.append(feature_pool(features,rows,duration,wte,tokenizer))
                v=dict(video_id=vid,rows=rows,N=int(rows[0]['pred_event_count']))
                if split=='train':
                    _,reward=targets(rows,self.annotations[vid],self.idf)
                    v.update(fold=self.folds[vid],gt_captions=self.annotations[vid]['sentences'],reward=reward)
                videos.append(v);completed+=1
                print('CANDIDATE',split,vid,'rows',len(rows),'new_videos',completed,flush=True)
            write_shard(path,videos,arrays,self.sig,split)
            print('SHARD_COMMITTED',path.name,split,len(ids),'seconds',round(time.time()-started,1),flush=True)
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError('Frozen C1 gradient guard failed')
        if sha(self.checkpoint)!=self.manifest['base_checkpoint_sha256']:raise RuntimeError('C1 checkpoint changed')
        del model,train,val,infer;gc.collect();torch.cuda.empty_cache()
        self.load('train');self.load('val')
        print('EXTRACTION_COMPLETE_C1_UNCHANGED',flush=True)

    @staticmethod
    def batch(videos,device):
        width=max(len(v['features']) for v in videos);dim=videos[0]['features'].shape[-1]
        x=torch.zeros(len(videos),width,dim,device=device);y=torch.zeros(len(videos),width,device=device)
        valid=torch.zeros_like(y,dtype=torch.bool)
        for i,v in enumerate(videos):
            if v.get('fold') not in ('head_fit','head_dev') or 'quality' not in v:raise ValueError('TRAIN labels required; validation cache rejected')
            n=len(v['features']);x[i,:n]=torch.as_tensor(v['features'].astype(np.float32),device=device)
            y[i,:n]=torch.as_tensor(v['quality'],device=device);valid[i,:n]=True
        return x,y,valid

    def fit(self):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        all_train=self.load('train');fit=[v for v in all_train if v['fold']=='head_fit'];dev=[v for v in all_train if v['fold']=='head_dev']
        if set(v['video_id'] for v in fit)&set(v['video_id'] for v in dev):raise ValueError('Head fit/dev leakage')
        torch.manual_seed(TRAINING_SEED);model=JointQualityHead().to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        last=self.folder/'head-last.pt';best=self.folder/'head-best.pt';start=0;minimum=float('inf');history=[]
        if last.is_file():
            cp=torch.load(last,map_location=device,weights_only=False)
            if cp['signature']!=self.sig:raise ValueError('Head resume signature mismatch')
            model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);start=cp['next_epoch'];minimum=cp['best_dev_loss'];history=cp['history']
        if start==20 and best.is_file():
            frozen=torch.load(best,map_location='cpu',weights_only=False)
            if frozen['signature']!=self.sig:raise ValueError('Frozen head signature mismatch')
            if 'beta' in frozen:
                print('HEAD_AND_BETA_ALREADY_FROZEN',frozen['epoch'],frozen['beta'],flush=True)
                return
        # There is no reference to the C1 model in optimizer.param_groups.
        print('HEAD_ONLY_TRAINING',sum(p.numel() for p in model.parameters()),'parameters','epochs',20,
              'training_seed',TRAINING_SEED,'head_split_seed',HEAD_SPLIT_SEED,'fixed_beta',FIXED_BETAS[self.dataset],flush=True)
        for epoch in range(start,20):
            torch.manual_seed(TRAINING_SEED+epoch);order=np.random.default_rng(TRAINING_SEED+epoch).permutation(len(fit))
            model.train();training=[]
            for lo in range(0,len(order),8):
                batch=[fit[i] for i in order[lo:lo+8]];x,y,valid=self.batch(batch,device)
                optimizer.zero_grad(set_to_none=True);loss=objective(model(x),y,valid)
                if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite head loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite head gradients')
                optimizer.step();training.append(float(loss.detach()))
            model.eval();values=[]
            with torch.no_grad():
                for v in dev:
                    x,y,valid=self.batch([v],device);values.append(float(objective(model(x),y,valid)))
            value=float(np.mean(values));row=dict(epoch=epoch,train_loss=float(np.mean(training)),head_dev_loss=value)
            if not np.isfinite(value):raise RuntimeError('Nonfinite head-dev objective')
            history.append(row)
            if value<minimum:
                minimum=value;atomic_torch(best,dict(signature=self.sig,model=model.state_dict(),epoch=epoch,head_dev_loss=value))
            atomic_torch(last,dict(signature=self.sig,model=model.state_dict(),optimizer=optimizer.state_dict(),
                                  next_epoch=epoch+1,best_dev_loss=minimum,history=history))
            atomic_json(self.folder/'head_history.json',history);print('HEAD_EPOCH',json.dumps(row),flush=True)
        cp=torch.load(best,map_location=device,weights_only=False);model.load_state_dict(cp['model']);model.eval()
        qualities=[]
        with torch.no_grad():
            for v in dev:qualities.append(model(torch.as_tensor(v['features'].astype(np.float32),device=device)).sigmoid().cpu().tolist())
        grid=[];winner=0.;winning=-float('inf')
        for beta in BETAS:
            value=float(np.mean([dev_reward(v['rows'],select(v['rows'],v['N'],q,beta),v['reward']) for v,q in zip(dev,qualities)]))
            grid.append(dict(beta=beta,head_dev_reward=value))
            if value>winning+1e-8:winning=value;winner=beta
        beta=FIXED_BETAS[self.dataset]
        scope='Fixed before seed-2016 training from previous seed-777 experiments; ANET beta 6 was validation-tuned'
        cp.update(beta=beta,beta_selection=scope,beta_grid=grid,head_dev_beta=winner,
                  training_seed=TRAINING_SEED,head_split_seed=HEAD_SPLIT_SEED)
        atomic_torch(best,cp);atomic_json(self.folder/'beta_selection.json',
            dict(beta=beta,head_dev_beta=winner,grid=grid,scope=scope))
        print('HEAD_FROZEN_BEFORE_VALIDATION','epoch',cp['epoch'],'beta',beta,flush=True)

    def evaluate(self):
        from misc.proposal_selection import select_proposals
        os.chdir(self.base/'source')
        from eval_utils_clip import eval_metrics
        device='cuda' if torch.cuda.is_available() else 'cpu';model=JointQualityHead().to(device)
        cp=torch.load(self.folder/'head-best.pt',map_location=device,weights_only=False)
        if cp['signature']!=self.sig or 'beta' not in cp:raise ValueError('Freeze head and beta before validation')
        model.load_state_dict(cp['model']);model.eval();beta=cp['beta'];head_sha=sha(self.folder/'head-best.pt')
        val=self.load('val');final=json.loads(self.final_path.read_text(encoding='utf8'))
        outputs={m:{} for m in ('baseline_topk','baseline_softnms','reranker_softnms')}
        semantics=lambda rows:[(r['query_id'],r['timestamp'],r['sentence']) for r in rows]
        with torch.no_grad():
            for v in val:
                rows=v['rows'];n=v['N'];vid=v['video_id']
                q=model(torch.as_tensor(v['features'].astype(np.float32),device=device)).sigmoid().cpu().tolist()
                for mode,method in [('baseline_topk','topk'),('baseline_softnms','softnms')]:
                    outputs[mode][vid]=select_proposals(rows,n,method=method,sigma=.5,alpha=1.,temperature=2.)
                outputs['reranker_softnms'][vid]=select(rows,n,q,beta)
                if semantics(outputs['baseline_softnms'][vid])!=semantics(final['results'][vid]):
                    raise ValueError('Baseline selection mismatch')
                if len(outputs['reranker_softnms'][vid])!=len(outputs['baseline_softnms'][vid]):
                    raise ValueError('Counter budget changed')
                source={r['query_id']:r for r in rows}
                for r in outputs['reranker_softnms'][vid]:
                    old=source[r['query_id']]
                    if r['timestamp']!=old['timestamp'] or r['sentence']!=old['sentence']:raise ValueError('Frozen output mutated')
        folder=self.folder/'evaluation';folder.mkdir(exist_ok=True);summary=[]
        for mode,predictions in outputs.items():
            path=folder/(mode+'.json');atomic_json(path,dict(results=predictions,version='VERSION 1.0',external_data={},
                                                          selection=dict(method=mode,beta=beta if mode=='reranker_softnms' else 0)))
            fingerprint=signature(dict(file=sha(path),refs=[sha(p) for p in self.cfg['gt_file_for_eval']+self.cfg['gt_file_for_para_eval']],
                                       scoring_protocol=SCORING_PROTOCOL))
            saved=folder/(mode+'.metrics.json');cached=json.loads(saved.read_text(encoding='utf8')) if saved.is_file() else {}
            if cached.get('signature')==fingerprint and complete_scores(cached.get('scores')):scores=cached['scores']
            else:
                print('OFFICIAL_SCORING',mode,flush=True)
                with (folder/'scorer_detail.log').open('a',encoding='utf8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    # In the vendored DDVC evaluator, infer_mode=True enables all
                    # language scorers. False silently returns METEOR only.
                    scores={k:float(v)*100 for k,v in eval_metrics(str(path),self.cfg['gt_file_for_eval'],
                            self.cfg['gt_file_for_para_eval'],dvc_eval_version='2018',infer_mode=True).items()}
                if not complete_scores(scores):
                    raise RuntimeError('Incomplete/nonfinite official metrics: required='+str(METRICS)+
                                       ' returned='+str(sorted(scores))+'; inspect '+str(folder/'scorer_detail.log'))
                atomic_json(saved,dict(signature=fingerprint,scoring_protocol=SCORING_PROTOCOL,scores=scores))
            if mode=='baseline_softnms':
                for k in METRICS:
                    if abs(scores[k]-100*final[k])>1e-4:raise ValueError('Stored C1 metric not reproduced: '+k)
            row=dict(method=mode,beta=beta if mode=='reranker_softnms' else 0.,events=sum(map(len,predictions.values())),**{k:scores[k] for k in METRICS})
            summary.append(row);write_table(folder/'summary.tsv',summary);print('RESULT',json.dumps(row),flush=True)
        if sha(self.folder/'head-best.pt')!=head_sha or sha(self.checkpoint)!=self.manifest['base_checkpoint_sha256']:
            raise RuntimeError('Head/base checkpoint changed during evaluation')
        print('RERANK_EVALUATION_COMPLETE',folder/'summary.tsv',flush=True)
