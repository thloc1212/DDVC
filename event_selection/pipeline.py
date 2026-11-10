"""No DDVC imports/forward/training: reuse audited candidate features and frozen quality head."""
import contextlib
import collections
import gc
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from reranking.cache import atomic_json,sha,read_shard
from reranking.pipeline import atomic_torch,signature,write_table,complete_scores,METRICS
from reranking.head import JointQualityHead
from reranking.core import select,dev_reward,iou_matrix,words
from misc.proposal_selection import select_proposals
from .core import SCHEMA,RECIPE,pair_targets,geometry,normalize_features,relation_select,calibrated_count,median_count,temporal_matches
from .heads import RelationHead,CountHead,relation_loss,count_loss

ROOT=Path(__file__).resolve().parents[1]


def official_scores(prediction,refs,para):
    # The same vendored scorer and tIoU thresholds as C1; avoid importing the DDVC model.
    for path in (ROOT,ROOT/'densevid_eval3',ROOT/'densevid_eval3/SODA'):
        sys.path.insert(0,str(path))
    from densevid_eval3.eval_dvc import eval_dvc
    from densevid_eval3.eval_soda import eval_soda
    from densevid_eval3.eval_para import eval_para
    scores={k:float(np.mean(v)) for k,v in eval_dvc(str(prediction),refs,version='2018',infer_mode=True).items()}
    scores.update(eval_soda(str(prediction),ref_list=refs));scores.update(eval_para(str(prediction),referneces=para))
    scores['MetaScore']=scores['METEOR']+scores['soda_c']
    return {k:float(v)*100 for k,v in scores.items()}


class Experiment:
    def __init__(self,work,dataset,device=None):
        self.work=Path(work).resolve();self.dataset=dataset;self.input=self.work/'inputs'/dataset
        self.folder=self.work/'runs'/dataset;self.folder.mkdir(parents=True,exist_ok=True)
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.receipt=json.loads((self.input/'READY.json').read_text(encoding='utf8'))
        # Inputs have already been downloaded through a complete READY snapshot. Recheck bytes before use.
        for name,entry in self.receipt['files'].items():
            p=self.input/name
            if p.is_symlink() or not p.is_file() or p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:
                raise ValueError('Restored input hash mismatch: '+name)
        self.cache=self.input/self.receipt['reranker_prefix'];self.source=json.loads((self.cache/'manifest.json').read_text())
        self.quality_path=self.cache/'head-best.pt';self.quality_sha=sha(self.quality_path)
        cp=torch.load(self.quality_path,map_location='cpu',weights_only=False)
        if cp.get('signature')!=self.source['signature'] or 'beta' not in cp:raise ValueError('Need a completed frozen quality head/beta')
        self.beta=float(cp['beta']);self.quality=JointQualityHead().to(self.device)
        self.quality.load_state_dict(cp['model'],strict=True);self.quality.eval();self.quality.requires_grad_(False)
        self.train_annotation=json.loads((self.input/self.receipt['train_annotation']).read_text())
        # Only IDs and duration enter feature construction. Validation count/GT is read only inside evaluate().
        self.val_metadata=self.receipt['validation_video_metadata'];self.q=int(self.receipt['queries'])
        self.manifest=dict(schema=SCHEMA,dataset=dataset,recipe=RECIPE,source_reranker_signature=self.source['signature'],
                           source_C1_checkpoint_sha256=self.source['base_checkpoint_sha256'],quality_head_sha256=self.quality_sha,
                           input_receipt_sha256=sha(self.input/'READY.json'),queries=self.q,quality_beta=self.beta,
                           note='Same C1 rerun candidates; relation/count heads only; no original DDVC weights loaded')
        self.sig=signature(self.manifest);self.manifest['signature']=self.sig
        path=self.folder/'manifest.json'
        if path.is_file() and json.loads(path.read_text())!=self.manifest:raise ValueError('Experiment input/recipe changed')
        atomic_json(path,self.manifest)
        self.fit_ids=set(self.source['fit_ids']);self.dev_ids=set(self.source['dev_ids'])
        if self.fit_ids&self.dev_ids or self.fit_ids|self.dev_ids!=set(self.train_annotation):raise ValueError('Train holdout mismatch')
        self._train=None
        print('FROZEN_CACHE_READY',dataset,'Q',self.q,'base_epoch',self.source['base_epoch'],
              'checkpoint_sha',self.source['base_checkpoint_sha256'],flush=True)

    def load(self,split):
        if split=='train' and self._train is not None:return self._train
        result=[]
        paths=sorted((self.cache/'cache'/split).glob('shard_*.npz'))
        if not paths:raise FileNotFoundError('No cached '+split+' candidates')
        for path in paths:result.extend(read_shard(path,self.source['signature'],split))
        expected=self.fit_ids|self.dev_ids if split=='train' else set(self.val_metadata)
        if {v['video_id'] for v in result}!=expected or len(result)!=len(expected):raise ValueError('Missing/duplicate cached video IDs')
        for v in result:
            vid=v['video_id'];rows=v['rows']
            if not rows or len(rows)>self.q or len({r['query_id'] for r in rows})!=len(rows):raise ValueError('Invalid candidate query IDs')
            if v['features'].shape!=(len(rows),2312):raise ValueError('Invalid frozen candidate feature shape')
            if split=='train':
                annotation=self.train_annotation[vid]
                if v['fold']!=('head_fit' if vid in self.fit_ids else 'head_dev'):raise ValueError('Head train/dev leakage')
                if annotation['sentences']!=v['gt_captions']:raise ValueError('GT identity differs from original quality targets')
                v['duration']=annotation['duration'];v['full_count']=len(annotation['sentences'])
                v['pair_labels'],v['pair_valid'],v['pair_hard']=pair_targets(rows,annotation,v['reward'])
                v['count_target']=min(v['full_count'],len(rows))
            else:
                if any(k in v for k in ('quality','reward','fold','gt_captions')):raise ValueError('Validation labels leaked into candidate input')
                v['duration']=self.val_metadata[vid]['duration']
            with torch.no_grad():
                feature=torch.as_tensor(v['features'].astype(np.float32),device=self.device)
                v['predicted_quality']=self.quality(feature).sigmoid().cpu().numpy()
            v['x']=normalize_features(v['features'],rows,v['predicted_quality'])
            v['geometry']=geometry(rows,v['duration'],v['features'],v['predicted_quality'])
        if split=='train':self._train=result
        return result

    def tensors(self,v):
        return (torch.as_tensor(v['x'],device=self.device),torch.as_tensor(v['geometry'],device=self.device))

    def models(self):return RelationHead().to(self.device),CountHead(self.q).to(self.device)

    def scorer_smoke(self):
        """Check Java/all metric keys on synthetic data, never DVC validation labels."""
        folder=self.folder/'scorer_smoke';folder.mkdir(exist_ok=True)
        sentences=['slice the onion into small pieces','boil the water in a large pot']
        annotation={'synthetic':dict(duration=10.,timestamps=[[0.,5.],[5.,10.]],sentences=sentences)}
        ref=folder/'synthetic_reference.json';para=folder/'synthetic_paragraph.json';pred=folder/'synthetic_prediction.json'
        atomic_json(ref,annotation);atomic_json(para,{'synthetic':' '.join(sentences)})
        atomic_json(pred,dict(version='VERSION 1.0',external_data={},results={'synthetic':[
            dict(timestamp=interval,sentence=caption) for interval,caption in zip(annotation['synthetic']['timestamps'],sentences)]}))
        with (folder/'scorer.log').open('w',encoding='utf8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            scores=official_scores(pred,[str(ref)],[str(para)])
        if not complete_scores(scores):raise RuntimeError('Synthetic official scorer smoke failed before head training')
        atomic_json(folder/'PASS.json',dict(passed=True,synthetic_only=True,required_metrics=list(METRICS)))
        print('FULL_OFFICIAL_SCORER_SMOKE_PASS_SYNTHETIC',flush=True)

    def smoke(self,require_cuda=True):
        if require_cuda and self.device!='cuda':raise RuntimeError('Select a GPU Molab session')
        videos=self.load('train');sample=next((v for v in videos if v['pair_valid'].any()),None)
        if sample is None:raise ValueError('No reliable TRAIN pairs: inspect label coverage')
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if self.device=='cuda' else []):
            torch.manual_seed(777);relation,count=self.models();x,g=self.tensors(sample)
            labels=torch.as_tensor(sample['pair_labels'],device=self.device)
            valid=torch.as_tensor(sample['pair_valid'],device=self.device);hard=torch.as_tensor(sample['pair_hard'],device=self.device)
            loss=relation_loss(relation(x,g),labels,valid,hard)+count_loss(count(x,sample['N']),sample['count_target'],x.new_ones(self.q+1))
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for m in (relation,count) for p in m.parameters()):
                raise RuntimeError('New-head forward/backward smoke failed')
        if any(p.requires_grad or p.grad is not None for p in self.quality.parameters()):raise RuntimeError('Frozen quality head received gradients')
        if sha(self.quality_path)!=self.quality_sha:raise RuntimeError('Frozen quality head changed')
        atomic_json(self.folder/'gpu_smoke.json',dict(device=self.device,passed=True,loss=float(loss.detach()),
                    actual_models='RelationHead + CountHead on restored real C1 candidate features',DDVC_forward_required=False))
        print('REAL_RELATION_COUNT_FORWARD_BACKWARD_PASS',self.device,flush=True)

    def fit(self):
        train=self.load('train');fit=[v for v in train if v['fold']=='head_fit'];dev=[v for v in train if v['fold']=='head_dev']
        if not fit or not dev:raise ValueError('Need disjoint nonempty TRAIN fit/dev')
        coverage={name:dict(videos=len(vs),pairs=int(sum(v['pair_valid'].sum() for v in vs)),
                   positive_pairs=int(sum((v['pair_valid']&(v['pair_labels']>.5)).sum() for v in vs)),
                   hard_negative_pairs=int(sum((v['pair_valid']&(v['pair_labels']<.5)&v['pair_hard']).sum() for v in vs)),
                   count_capacity_capped=sum(v['full_count']>len(v['rows']) for v in vs))
                   for name,vs in [('head_fit',fit),('head_dev',dev)]}
        atomic_json(self.folder/'label_coverage.json',coverage)
        if any(not c['positive_pairs'] or c['pairs']==c['positive_pairs'] for c in coverage.values()):
            raise ValueError('Insufficient positive/negative relation labels; inspect label_coverage.json')
        torch.manual_seed(777);relation,count=self.models()
        ro=torch.optim.AdamW(relation.parameters(),lr=.001,weight_decay=.0001)
        co=torch.optim.AdamW(count.parameters(),lr=.001,weight_decay=.0001)
        frequencies=np.bincount([v['count_target'] for v in fit],minlength=self.q+1)
        weights=np.ones(self.q+1,np.float32);present=frequencies>0
        weights[present]=1/np.sqrt(frequencies[present]);weights[present]/=weights[present].mean();weights=np.clip(weights,.5,3.)
        weights=torch.as_tensor(weights,device=self.device)
        last=self.folder/'heads-last.pt';start=0;best_rel=float('inf');best_count=(float('inf'),float('inf'));history=[]
        if last.is_file():
            cp=torch.load(last,map_location=self.device,weights_only=False)
            if cp['signature']!=self.sig:raise ValueError('New-head resume signature changed')
            relation.load_state_dict(cp['relation']);count.load_state_dict(cp['count']);ro.load_state_dict(cp['relation_optimizer']);co.load_state_dict(cp['count_optimizer'])
            start=cp['next_epoch'];best_rel=cp['best_relation'];best_count=tuple(cp['best_count']);history=cp['history']
        if start==RECIPE['epochs'] and (self.folder/'policy.json').is_file():
            self.policy();print('RELATION_COUNT_ALREADY_FROZEN_NO_TRAINING',flush=True);return
        for epoch in range(start,RECIPE['epochs']):
            torch.manual_seed(777+epoch);order=np.random.default_rng(777+epoch).permutation(len(fit));relation.train();count.train();rl=[];cl=[]
            for lo in range(0,len(order),8):
                ro.zero_grad(set_to_none=True);co.zero_grad(set_to_none=True);rterms=[];cterms=[]
                for index in order[lo:lo+8]:
                    v=fit[index];x,g=self.tensors(v)
                    if v['pair_valid'].any():
                        rterms.append(relation_loss(relation(x,g),torch.as_tensor(v['pair_labels'],device=self.device),
                             torch.as_tensor(v['pair_valid'],device=self.device),torch.as_tensor(v['pair_hard'],device=self.device)))
                    cterms.append(count_loss(count(x,v['N']),v['count_target'],weights))
                total=torch.stack(cterms).mean()
                if rterms:total=total+torch.stack(rterms).mean()
                if not torch.isfinite(total):raise RuntimeError('Nonfinite new-head loss')
                total.backward()
                for model,opt in ((relation,ro),(count,co)):
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    if not torch.isfinite(norm):raise RuntimeError('Nonfinite new-head gradients')
                    opt.step()
                if rterms:rl.append(float(torch.stack(rterms).mean().detach()))
                cl.append(float(torch.stack(cterms).mean().detach()))
            relation.eval();count.eval();dev_rel=[];count_errors=[];dev_count_loss=[]
            with torch.no_grad():
                for v in dev:
                    x,g=self.tensors(v)
                    if v['pair_valid'].any():dev_rel.append(float(relation_loss(relation(x,g),torch.as_tensor(v['pair_labels'],device=self.device),
                               torch.as_tensor(v['pair_valid'],device=self.device),torch.as_tensor(v['pair_hard'],device=self.device))))
                    logits=count(x,v['N']);pred=calibrated_count(v['N'],median_count(logits.softmax(-1).cpu().numpy()),1.,len(v['rows']))
                    count_errors.append(abs(pred-v['full_count']));dev_count_loss.append(float(count_loss(logits,v['count_target'],weights)))
            rvalue=float(np.mean(dev_rel));cvalue=(float(np.mean(count_errors)),float(np.mean(dev_count_loss)))
            if not np.isfinite([rvalue,*cvalue]).all():raise RuntimeError('Nonfinite head-dev evaluation')
            row=dict(epoch=epoch,train_relation_loss=float(np.mean(rl)),train_count_loss=float(np.mean(cl)),
                     head_dev_relation_loss=rvalue,head_dev_count_MAE=cvalue[0]);history.append(row)
            if rvalue<best_rel:
                best_rel=rvalue;atomic_torch(self.folder/'relation-best.pt',dict(signature=self.sig,model=relation.state_dict(),epoch=epoch))
            if cvalue<best_count:
                best_count=cvalue;atomic_torch(self.folder/'count-best.pt',dict(signature=self.sig,model=count.state_dict(),epoch=epoch,queries=self.q))
            atomic_torch(last,dict(signature=self.sig,next_epoch=epoch+1,relation=relation.state_dict(),count=count.state_dict(),
                     relation_optimizer=ro.state_dict(),count_optimizer=co.state_dict(),best_relation=best_rel,best_count=best_count,history=history))
            atomic_json(self.folder/'history.json',history);print('RELATION_COUNT_EPOCH',json.dumps(row),flush=True)
        self.choose_policy(dev)
        if sha(self.quality_path)!=self.quality_sha:raise RuntimeError('Frozen quality checkpoint changed')

    def trained_models(self):
        relation,count=self.models()
        for model,name in ((relation,'relation-best.pt'),(count,'count-best.pt')):
            cp=torch.load(self.folder/name,map_location=self.device,weights_only=False)
            if cp['signature']!=self.sig:raise ValueError('Head signature changed')
            model.load_state_dict(cp['model']);model.eval()
        return relation,count

    def choose_policy(self,dev):
        relation,count=self.trained_models();rs=[];cs=[]
        with torch.no_grad():
            for v in dev:
                x,g=self.tensors(v);rs.append(relation(x,g).sigmoid().cpu().numpy());cs.append(median_count(count(x,v['N']).softmax(-1).cpu().numpy()))
        options=[('softnms',0.)]+[('learned_relation',v) for v in RECIPE['relation_lambdas']]
        grid=[];best=-float('inf');winner=options[0]
        for name,penalty in options:
            rewards=[]
            for v,d in zip(dev,rs):
                chosen=(select(v['rows'],v['N'],v['predicted_quality'],self.beta) if name=='softnms' else
                        relation_select(v['rows'],v['N'],v['predicted_quality'],self.beta,d,penalty))
                rewards.append(dev_reward(v['rows'],chosen,v['reward']))
            score=float(np.mean(rewards));grid.append(dict(selector=name,penalty=penalty,head_dev_reward=score))
            if score>best+1e-8:best=score;winner=(name,penalty)
        cgrid=[];error=float('inf');blend=0.
        for candidate in RECIPE['count_blends']:
            value=float(np.mean([abs(calibrated_count(v['N'],n,candidate,len(v['rows']))-v['full_count']) for v,n in zip(dev,cs)]))
            cgrid.append(dict(blend=candidate,head_dev_count_MAE=value))
            if value<error-1e-8:error=value;blend=candidate
        policy=dict(signature=self.sig,selector=winner[0],penalty=winner[1],count_blend=blend,relation_grid=grid,count_grid=cgrid,
                    relation_sha256=sha(self.folder/'relation-best.pt'),count_sha256=sha(self.folder/'count-best.pt'),
                    selection_scope='TRAIN head_dev only; frozen before DVC validation')
        atomic_json(self.folder/'policy.json',policy);print('POLICY_FROZEN_BEFORE_VALIDATION',json.dumps(policy),flush=True)

    def policy(self):
        p=json.loads((self.folder/'policy.json').read_text())
        if p['signature']!=self.sig or p['relation_sha256']!=sha(self.folder/'relation-best.pt') or p['count_sha256']!=sha(self.folder/'count-best.pt'):
            raise ValueError('Frozen selector/count policy changed')
        return p

    def evaluate(self):
        policy=self.policy();relation,count=self.trained_models();val=self.load('val')
        modes=('C1_softnms','quality_topk','quality_softnms','quality_relation','quality_softnms_count','quality_relation_count')
        outputs={name:{} for name in modes};count_rows=[]
        with torch.no_grad():
            for v in val:
                x,g=self.tensors(v);duplicate=relation(x,g).sigmoid().cpu().numpy()
                head_n=median_count(count(x,v['N']).softmax(-1).cpu().numpy())
                n=calibrated_count(v['N'],head_n,policy['count_blend'],len(v['rows']));q=v['predicted_quality'];rows=v['rows'];vid=v['video_id']
                outputs['C1_softnms'][vid]=select_proposals(rows,v['N'])
                outputs['quality_topk'][vid]=relation_select(rows,v['N'],q,self.beta,duplicate,0.)
                outputs['quality_softnms'][vid]=select(rows,v['N'],q,self.beta)
                outputs['quality_softnms_count'][vid]=select(rows,n,q,self.beta)
                for mode,budget in [('quality_relation',v['N']),('quality_relation_count',n)]:
                    outputs[mode][vid]=(select(rows,budget,q,self.beta) if policy['selector']=='softnms' else
                                       relation_select(rows,budget,q,self.beta,duplicate,policy['penalty']))
                count_rows.append(dict(video_id=vid,old_N=v['N'],head_N=head_n,calibrated_N=n,available_queries=len(rows)))
                byid={r['query_id']:r for r in rows}
                for mode in modes:
                    if len(outputs[mode][vid])!=min(n if mode.endswith('_count') else v['N'],len(rows)):raise ValueError('Selection budget mismatch')
                    for row in outputs[mode][vid]:
                        old=byid[row['query_id']]
                        if row['timestamp']!=old['timestamp'] or row['sentence']!=old['sentence']:raise ValueError('Frozen caption/timestamp changed')
                        row['base_pred_event_count']=v['N']
                        row['pred_event_count']=len(outputs[mode][vid])
        # Validation annotations become available only after every inference output and policy is frozen.
        annotation=json.loads((self.input/self.receipt['val_annotation']).read_text());folder=self.folder/'evaluation';folder.mkdir(exist_ok=True)
        refs=[str(self.input/p) for p in self.receipt['references']];para=[str(self.input/p) for p in self.receipt['paragraph_references']]
        summary=[];strata=[];saved_baselines=self.receipt['baseline_scores']
        for mode,predictions in outputs.items():
            path=folder/(mode+'.json');atomic_json(path,dict(results=predictions,version='VERSION 1.0',external_data={},selection=dict(method=mode,policy=policy)))
            fingerprint=signature(dict(prediction=sha(path),references=[sha(p) for p in refs+para],scoring='ddvc2018-all-language-v1'))
            score_path=folder/(mode+'.metrics.json');cached=json.loads(score_path.read_text()) if score_path.is_file() else {}
            if cached.get('signature')==fingerprint and complete_scores(cached.get('scores')):scores=cached['scores']
            else:
                print('OFFICIAL_SCORING',mode,flush=True)
                with (folder/'scorer.log').open('a',encoding='utf8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    scores=official_scores(path,refs,para)
                if not complete_scores(scores):raise RuntimeError('Incomplete/nonfinite official metrics; inspect scorer.log')
                atomic_json(score_path,dict(signature=fingerprint,scores=scores))
            if mode in saved_baselines:
                for k in METRICS:
                    if abs(scores[k]-saved_baselines[mode][k])>1e-4:raise ValueError('Frozen baseline not reproduced: '+mode+' '+k)
            duplicate_events=sum(len(rs)-len({tuple(words(r['sentence'])) for r in rs}) for rs in predictions.values())
            groups=collections.defaultdict(collections.Counter)
            for vid,rows in predictions.items():
                gt=annotation[vid]['timestamps'];matched=temporal_matches(rows,gt)
                for j,(s,e) in enumerate(gt):
                    labels=['all','short_le10s' if e-s<=10 else 'long_gt10s']
                    if e-s<=5:labels.append('short_le5s')
                    if e-s>30:labels.append('long_gt30s')
                    if len(gt)>10:labels.append('video_gt_count_gt10')
                    if any(k!=j and min(e,b)>max(s,a) for k,(a,b) in enumerate(gt)):labels.append('overlapping_gt')
                    for label in labels:groups[label]['gt']+=1;groups[label]['matched']+=int(j in matched)
            strata.extend(dict(method=mode,group=label,**stats,diagnostic_recall=100*stats['matched']/stats['gt']) for label,stats in groups.items())
            row=dict(method=mode,events=sum(map(len,predictions.values())),count_MAE=float(np.mean([abs(len(predictions[v])-len(annotation[v]['sentences'])) for v in predictions])),
                     duplicate_events=duplicate_events,**{k:scores[k] for k in METRICS})
            summary.append(row);write_table(folder/'summary.tsv',summary);print('RESULT',json.dumps(row),flush=True)
        for row in count_rows:row['GT_count_diagnostic']=len(annotation[row['video_id']]['sentences'])
        atomic_json(folder/'count_diagnostics.json',dict(reference='configured val annotation; diagnostic only',
                    policy=policy['count_blend'],MAE={key:float(np.mean([abs(r[key]-r['GT_count_diagnostic']) for r in count_rows]))
                    for key in ('old_N','head_N','calibrated_N')},
                    histograms={key:dict(collections.Counter(str(r[key]) for r in count_rows))
                    for key in ('old_N','head_N','calibrated_N','GT_count_diagnostic')}))
        write_table(folder/'counts.tsv',count_rows);write_table(folder/'strata.tsv',strata);self.policy()
        if sha(self.quality_path)!=self.quality_sha:raise RuntimeError('Frozen quality checkpoint changed')
        print('RELATION_COUNT_COMPLETE',folder/'summary.tsv',flush=True)
