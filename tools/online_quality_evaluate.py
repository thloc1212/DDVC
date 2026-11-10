"""Freeze beta using TRAIN head-dev, then score fixed best-checkpoint validation candidates."""
import collections
import contextlib
import copy
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from reranking.cache import atomic_json,sha
from reranking.core import targets,select,joint_score,dev_reward
from reranking.online import partition
from reranking.pipeline import BETAS,METRICS,signature,complete_scores,write_table


def official_scores(prediction,references,paragraphs):
    # Identical DDVC 2018 scorer calls; do not import unrelated TAL/plot utilities.
    sys.path[:0]=[str(ROOT/'densevid_eval3'),str(ROOT/'densevid_eval3/SODA')]
    from densevid_eval3.eval_dvc import eval_dvc
    from densevid_eval3.eval_soda import eval_soda
    from densevid_eval3.eval_para import eval_para
    result={k:float(np.mean(v)) for k,v in eval_dvc(str(prediction),references,version='2018',infer_mode=True).items()}
    result.update(eval_soda(str(prediction),ref_list=references))
    result.update(eval_para(str(prediction),referneces=paragraphs))
    result['MetaScore']=result['METEOR']+result['soda_c']
    return {k:float(v)*100 for k,v in result.items()}


def scorer_smoke(folder):
    """Use the actual Java-backed scorer before full training; synthetic data only."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    captions=['slice the onion into small pieces','boil the water in a large pot']
    annotation={'synthetic':dict(duration=10.,timestamps=[[0.,5.],[5.,10.]],sentences=captions)}
    reference=folder/'reference.json';paragraph=folder/'paragraph.json';prediction=folder/'prediction.json'
    atomic_json(reference,annotation);atomic_json(paragraph,{'synthetic':' '.join(captions)})
    atomic_json(prediction,dict(version='VERSION 1.0',external_data={},results={'synthetic':[
        dict(timestamp=t,sentence=c) for t,c in zip(annotation['synthetic']['timestamps'],captions)]}))
    with (folder/'scorer.log').open('w',encoding='utf8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        scores=official_scores(prediction,[str(reference)],[str(paragraph)])
    if not complete_scores(scores):raise RuntimeError('Synthetic official scorer preflight failed')
    atomic_json(folder/'PASS.json',dict(passed=True,synthetic_only=True,required_metrics=list(METRICS)))
    print('FULL_OFFICIAL_SCORER_SMOKE_PASS_SYNTHETIC',flush=True)


def select_modes(raw,beta):
    from misc.proposal_selection import select_proposals
    outputs={m:{} for m in ('online_base_topk','online_base_softnms','online_quality_topk','online_quality_softnms')}
    for vid,rows in raw.items():
        if not rows:raise ValueError('Empty validation candidates')
        n=int(rows[0]['pred_event_count']);q=[float(r['reranker_quality']) for r in rows]
        for mode,method in [('online_base_topk','topk'),('online_base_softnms','softnms')]:
            outputs[mode][vid]=select_proposals(rows,n,method=method,sigma=.5,alpha=1.,temperature=2.)
        rescored=[]
        for row,quality in zip(rows,q):
            row=copy.deepcopy(row);row.update(joint_score=joint_score(row),reranker_beta=float(beta),
                                             selection_score=joint_score(row)+beta*quality)
            rescored.append(row)
        outputs['online_quality_topk'][vid]=sorted(sorted(rescored,key=lambda r:r['selection_score'],reverse=True)[:n],key=lambda r:r['timestamp'])
        outputs['online_quality_softnms'][vid]=select(rows,n,q,beta)
        source={r['query_id']:r for r in rows}
        for mode,pred in outputs.items():
            if len(pred[vid])!=min(n,len(rows)):raise ValueError('Counter budget changed')
            for r in pred[vid]:
                old=source[r['query_id']]
                if r['timestamp']!=old['timestamp'] or r['sentence']!=old['sentence']:
                    raise ValueError('Selection modified timestamp/caption')
    return outputs


def choose_beta(dev):
    grid=[];winner=0.;best=-float('inf')
    for beta in BETAS:
        value=float(np.mean([dev_reward(v['rows'],select(v['rows'],v['N'],v['quality'],beta),np.array(v['reward'])) for v in dev]))
        grid.append(dict(beta=beta,head_dev_reward=value))
        if value>best+1e-8:best=value;winner=beta
    return winner,grid


def extract_dev(cfg,checkpoint,dev_ids,cache,expected_signature):
    from opts import parse_opts
    from train import build,PropSeqDataset,collate_fn,load_clip_memory_bank,load_clip_gt
    from misc.runtime_perf import configure_runtime
    from misc.training_state import schedule_contrastive
    from torch.utils.data import DataLoader
    # Caller provides the exact config path; never infer it from the cache depth.
    config_file=Path(cfg['_config_path'])
    opt=parse_opts(['--cfg_path',str(config_file)],write_options=False)
    if not opt.text_crossAttn:opt.cap_num_feature_levels=4
    if opt.combined_encoder:opt.cap_num_feature_levels=5
    if opt.caption_decoder_new=='gpt2':opt.caption_decoder_type='gpt2'
    opt.rec_encdec=True;configure_runtime(opt)
    random.seed(777);np.random.seed(777);torch.manual_seed(777)
    if not torch.cuda.is_available():raise RuntimeError('TRAIN-dev calibration requires GPU inference')
    ds=PropSeqDataset(opt.train_caption_file,opt.visual_feature_folder,opt.dict_file,False,'gt',opt)
    model,criterion,contrastive,post=build(opt);model.translator=ds.translator
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(cp['model'],strict=True);epoch=cp['epoch'];del cp
    model.to(opt.device).eval()
    for p in model.parameters():p.requires_grad_(False)
    schedule_contrastive(criterion.weight_dict,epoch,opt.cl_schedule_time,opt.cl_schedule_val,False)
    bank=load_clip_memory_bank(opt);bank['vid_sent_embeds']=torch.as_tensor(bank['vid_sent_embeds'],device=opt.device)
    bank['clip_gt']=load_clip_gt(opt)
    index={str(v):i for i,v in enumerate(ds.keys)}
    loader=DataLoader(ds,batch_size=1,num_workers=0,collate_fn=collate_fn)
    cache.mkdir(parents=True,exist_ok=True);result=[]
    for offset in range(0,len(dev_ids),32):
        ids=dev_ids[offset:offset+32];path=cache/f'dev_{offset//32:05d}.json'
        ready=path.with_name(path.stem+'.ready.json')
        if path.is_file() and ready.is_file():
            payload=json.loads(path.read_text());receipt=json.loads(ready.read_text())
            if receipt!={'signature':expected_signature,'sha256':sha(path)} or [v['video_id'] for v in payload]!=ids:
                raise ValueError('TRAIN-dev calibration shard identity changed')
            result.extend(payload);continue
        shard=[]
        for vid in ids:
            np.random.seed(777+index[vid])
            dt=collate_fn([ds[index[vid]]]);dt={k:v.to(opt.device) if isinstance(v,torch.Tensor) else v for k,v in dt.items()}
            dt['video_target']=[{k:v.to(opt.device) if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in dt['video_target']]
            dt=collections.defaultdict(lambda:None,dt)
            with torch.no_grad():
                out,_=model(dt,criterion,contrastive,opt.transformer_input_type,bank,eval_mode=True)
                pred=post['bbox'](out,dt['video_length'][:,1],loader,model,None)[0]
            rows=[];quality=[]
            for i,box in enumerate(pred['boxes']):
                qid=int(pred['query_id'][i])
                if float(box[1])<=float(box[0]):continue
                rows.append(dict(query_id=qid,timestamp=box.float().cpu().tolist(),sentence=pred['captions'][i],
                     proposal_score=float(pred['scores'][i]),sentence_score=float(pred['caption_scores'][i])))
                quality.append(float(out['pred_quality'][0,qid]))
            _,reward=targets(rows,model._quality_annotations[vid],model._quality_idf)
            shard.append(dict(video_id=vid,rows=rows,quality=quality,N=int(pred['pred_seq_len']),reward=reward.tolist()))
            print('ONLINE_CALIBRATION_TRAIN_DEV',offset+len(shard),'/',len(dev_ids),vid,flush=True)
        atomic_json(path,shard);atomic_json(ready,dict(signature=expected_signature,sha256=sha(path)))
        result.extend(shard)
    return result


def finalize(args):
    from online_quality_pipeline import check,config_path
    work=args.work.resolve();path=config_path(work,args.dataset);cfg=check(path)
    run=work/'runs'/cfg['id'];info=json.loads((run/'info.json').read_text())
    if info['last']['epoch']<cfg['epoch']-1:raise ValueError('Finish the full requested DDVC budget before calibration')
    checkpoint=run/'model-best.pth';saved=run/'prediction'/Path(info['best']['result_json_path']).name
    raw_path=saved.with_name(saved.stem+'.raw.json')
    stable_cfg={k:v for k,v in cfg.items() if k not in ('start_from','start_from_mode','current_lr')}
    frozen=dict(source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                config_semantic_sha256=signature(stable_cfg),checkpoint_sha256=sha(checkpoint),raw_sha256=sha(raw_path),base_epoch=info['best']['epoch'])
    sig=signature(frozen);folder=run/'quality_evaluation';folder.mkdir(exist_ok=True)
    policy_path=folder/'policy.json'
    if policy_path.is_file():
        policy=json.loads(policy_path.read_text())
        if policy['signature']!=sig:raise ValueError('Frozen beta/checkpoint/input identity changed')
        print('ONLINE_BETA_ALREADY_FROZEN_NO_TRAINING',policy['beta'],flush=True)
    else:
        train=json.loads(Path(cfg['train_caption_file']).read_text());fit,dev=partition(train)
        frozen['head_fit_ids']=fit;frozen['head_dev_ids']=dev
        cfg['_config_path']=str(path)
        candidates=extract_dev(cfg,checkpoint,dev,folder/'calibration',sig)
        beta,grid=choose_beta(candidates)
        policy=dict(signature=sig,beta=beta,beta_grid=grid,scope='TRAIN head_dev ONLY; no quality optimizer here',
                    mode=cfg['quality_gradient_mode'],identity=frozen)
        atomic_json(policy_path,policy)
        print('ONLINE_POLICY_FROZEN_BEFORE_VALIDATION',json.dumps(dict(beta=beta,scope=policy['scope'])),flush=True)
    # No validation annotation is used until the TRAIN-dev policy is immutable.
    raw=json.loads(raw_path.read_text())['results'];outputs=select_modes(raw,policy['beta'])
    summary=[]
    for mode,predictions in outputs.items():
        target=folder/(mode+'.json');atomic_json(target,dict(results=predictions,version='VERSION 1.0',external_data={},
            selection=dict(method=mode,beta=policy['beta'] if 'quality_' in mode else 0.,sigma=.5)))
        fingerprint=signature(dict(prediction=sha(target),policy=sha(policy_path),
            refs=[sha(p) for p in cfg['gt_file_for_eval']+cfg['gt_file_for_para_eval']],scoring='ddvc2018-full-language-v1'))
        metric_path=folder/(mode+'.metrics.json');cached=json.loads(metric_path.read_text()) if metric_path.is_file() else {}
        if cached.get('signature')==fingerprint and complete_scores(cached.get('scores')):scores=cached['scores']
        else:
            print('OFFICIAL_SCORING',mode,flush=True)
            with (folder/'scorer.log').open('a',encoding='utf8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                scores=official_scores(target,cfg['gt_file_for_eval'],cfg['gt_file_for_para_eval'])
            if not complete_scores(scores):raise RuntimeError('Official scorer returned incomplete/nonfinite metrics')
            atomic_json(metric_path,dict(signature=fingerprint,scores=scores))
        if mode=='online_base_softnms':
            original=json.loads(saved.read_text())
            if any(abs(scores[k]-100*original[k])>1e-4 for k in METRICS):raise ValueError('Online model baseline metric mismatch')
        summary.append(dict(method=mode,gradient_mode=cfg['quality_gradient_mode'],beta=policy['beta'] if 'quality_' in mode else 0.,
            events=sum(map(len,predictions.values())),**{k:scores[k] for k in METRICS}))
        write_table(folder/'summary.tsv',summary);print('ONLINE_RESULT',json.dumps(summary[-1]),flush=True)
    if sha(checkpoint)!=frozen['checkpoint_sha256']:raise RuntimeError('DDVC checkpoint changed during calibration/evaluation')
    print('ONLINE_QUALITY_EVALUATION_COMPLETE',folder/'summary.tsv',flush=True)
