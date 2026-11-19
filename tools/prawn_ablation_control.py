"""Restore Full's verified text/results and, only for beta, its small head + VAL cache."""
import json
from pathlib import Path, PurePosixPath
import os
import numpy as np
import torch

from prawn_results_viewer import Store, generator, quality_matches
from prawn_ablation_eval import FIXED, METRICS, BETAS, score, select_fixed, with_f1
from reranking.cache import atomic_json, sha, read_shard
from reranking.head import JointQualityHead
from reranking.pipeline import write_table


def validate_full(cfg, dataset):
    expected=dict(caption_match_enabled=True,caption_match_weight=.5,proposal_mask_content_conditioned=True,
                  proposal_mask_loss_coef=0.,seed=777,batch_size=1,epoch=15 if dataset=='yc2' else 6,
                  num_queries=50 if dataset=='yc2' else 10,gradient_accumulation_steps=1)
    for key,value in expected.items():
        if cfg.get(key)!=value:raise ValueError('Saved Full differs from agreed control: '+key)
    if cfg.get('online_quality',False):raise ValueError('Full control must be TWO-STAGE, not Joint')


def fetch_binary(drive, snap, name, output):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError('Unsafe recovery path')
    if not (p.name=='head-best.pt' or ('cache','val')==p.parts[2:4] and p.suffix in ('.npz','.json')):
        raise ValueError('Only the frozen head and VAL feature shards may be restored')
    meta=snap['ready']['files'][name]
    if meta['bytes']>128*2**20:raise ValueError('Unexpected large quality artifact')
    target=Path(output)/Path(*p.parts);target.parent.mkdir(parents=True,exist_ok=True)
    if not target.is_file() or sha(target)!=meta['sha256'] or target.stat().st_size!=meta['bytes']:
        partial=target.with_name(target.name+'.partial')
        drive.call('copyto',drive.path(snap['remote']+'/'+name),str(partial),quiet=True)
        if partial.stat().st_size!=meta['bytes'] or sha(partial)!=meta['sha256']:
            raise ValueError('Quality artifact hash mismatch: '+name)
        os.replace(partial,target)
        print('FROZEN_ARTIFACT_VERIFIED',name,round(meta['bytes']/2**20,2),'MiB',flush=True)
    return target


def restore_control(drive, dataset, folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    store=Store(drive,folder)
    pinned=folder/'control_snapshot.json'
    if pinned.is_file():
        selected=json.loads(pinned.read_text(encoding='utf8'))
        g=generator(store,selected['generator'])
        validate_full(g['cfg'],dataset)
        qs=selected['quality'];qm=store.get(qs,f'runs/{dataset}/manifest.json')
        if not quality_matches(g,qm):raise ValueError('Pinned head/generator linkage changed')
    else:
        g=None;qs=None;qm=None
        quality_snaps=store.snapshots('prawn_caption_match_quality',dataset)
        for gs in store.snapshots('prawn_caption_match_generator',dataset):
            try:
                candidate=generator(store,gs);validate_full(candidate['cfg'],dataset)
            except (ValueError,KeyError) as error:
                print('CONTROL_SLOT_SKIPPED',str(error),flush=True);continue
            for snapshot in quality_snaps:
                files=snapshot['ready']['files'];mp=f'runs/{dataset}/manifest.json'
                if mp not in files:continue
                manifest=store.get(snapshot,mp)
                required=[f'runs/{dataset}/evaluation/{m}{s}' for m in ('baseline_softnms','reranker_softnms') for s in ('.json','.metrics.json')]
                if quality_matches(candidate,manifest) and all(n in files for n in required):
                    g,qs,qm=candidate,snapshot,manifest;break
            if g is not None:break
        if g is None:raise FileNotFoundError('No completed Full + hash-matched quality control on Drive')
        atomic_json(pinned,dict(generator=g['snapshot'],quality=qs))
    cfg=dict(g['cfg'])
    for key in ('gt_file_for_eval','gt_file_for_para_eval'):
        local=[]
        for old in cfg[key]:
            relative=str(PurePosixPath(old).relative_to(g['snapshot']['ready']['work']))
            store.get(g['snapshot'],relative)
            local.append(str(folder/'artifacts'/g['snapshot']['remote']/relative))
        cfg[key]=local
    policies={}
    for mode in ('baseline_softnms','reranker_softnms'):
        base=f'runs/{dataset}/evaluation/{mode}'
        pred=store.get(qs,base+'.json')['results']
        metrics=with_f1(store.get(qs,base+'.metrics.json')['scores'])
        for vid,rows in pred.items():
            old={r['query_id']:r for r in g['raw'][vid]}
            if len(rows)!=min(len(old),int(g['raw'][vid][0]['pred_event_count'])):raise ValueError('Saved Full counter budget differs')
            if any(r['sentence']!=old[r['query_id']]['sentence'] or r['timestamp']!=old[r['query_id']]['timestamp'] for r in rows):
                raise ValueError('Saved selection changed raw captions/timestamps')
        policies[mode]=dict(predictions=pred,scores=metrics)
    identity=dict(generator_run_id=g['snapshot']['ready']['run_id'],quality_run_id=qs['ready']['run_id'],
        checkpoint_sha256=g['checkpoint'],raw_sha256=g['raw_sha'],config_sha256=g['config_sha'],
        caption_matching=True,content_conditioned=True,mask_loss=0.,seed=777)
    atomic_json(folder/'control_identity.json',identity)
    return dict(store=store,g=g,qs=qs,qm=qm,cfg=cfg,policies=policies,identity=identity)


def quality_ablation(control, dataset, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[]
    for name,mode,beta in [('Full','reranker_softnms',FIXED[dataset]),('w_o_quality','baseline_softnms',0.)]:
        item=control['policies'][mode]
        atomic_json(output/(name+'.json'),dict(results=item['predictions'],identity=control['identity']))
        atomic_json(output/(name+'.metrics.json'),dict(scores=item['scores'],identity=control['identity']))
        rows.append(dict(method=name,beta=beta,events=sum(map(len,item['predictions'].values())),
                         **{k:item['scores'][k] for k in METRICS if k in item['scores']}))
    write_table(output/'summary.tsv',rows)
    print('QUALITY_ABLATION_NO_TRAINING',json.dumps(rows),flush=True)
    return output


def beta_ablation(control, dataset, output, sync=lambda:None):
    qs,qm=control['qs'],control['qm'];store=control['store'];drive=store.drive
    cache=Path(output).parent/'frozen_input';cache.mkdir(parents=True,exist_ok=True)
    head_path=fetch_binary(drive,qs,f'runs/{dataset}/head-best.pt',cache)
    cp=torch.load(head_path,map_location='cpu',weights_only=False)
    if cp.get('signature')!=qm['signature'] or cp.get('beta')!=FIXED[dataset]:raise ValueError('Frozen Full head identity/beta mismatch')
    before=sha(head_path);device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32=False
    model=JointQualityHead().to(device);model.load_state_dict(cp['model'],strict=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    qualities={};raw=control['g']['raw']
    shard_names=sorted(n for n in qs['ready']['files'] if n.startswith(f'runs/{dataset}/cache/val/') and n.endswith('.npz'))
    if not shard_names:raise FileNotFoundError('Completed validation feature cache is missing')
    for n in shard_names:
        path=fetch_binary(drive,qs,n,cache)
        fetch_binary(drive,qs,str(PurePosixPath(n).with_suffix('.json')),cache)
        videos=read_shard(path,qm['signature'],'val')
        with torch.no_grad():
            for video in videos:
                vid=video['video_id'];rows=video['rows']
                if rows!=raw[vid]:raise ValueError('VAL cache candidate rows differ from saved Full raw')
                values=model(torch.as_tensor(video['features'].astype(np.float32),device=device)).sigmoid().cpu().tolist()
                qualities[vid]={str(r['query_id']):float(q) for r,q in zip(rows,values)}
    if set(qualities)!=set(raw):raise ValueError('Incomplete VAL quality pool')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'quality_scores.json',dict(identity=control['identity'],head_sha256=before,qualities=qualities))
    summary=[]
    for beta in BETAS:
        name='beta_'+format(beta,'g').replace('.','p')
        pred=select_fixed(raw,qualities,beta)
        if beta==0.:
            expected=control['policies']['baseline_softnms']['predictions']
            if any([r['query_id'] for r in pred[v]]!=[r['query_id'] for r in expected[v]] for v in raw):
                raise ValueError('beta=0 does not reproduce Full without quality')
        row=score(output,name,pred,control['cfg'],dict(control['identity'],head_sha256=before),beta)
        if beta in (0.,FIXED[dataset]):
            expected=control['policies']['baseline_softnms' if beta==0. else 'reranker_softnms']['scores']
            if any(abs(row[k]-expected[k])>1e-4 for k in ('CIDEr','METEOR','Bleu_4','soda_c','Precision','Recall')):
                raise ValueError('Frozen beta evaluation does not reproduce saved Full metrics')
        summary.append(row);write_table(output/'summary.tsv',summary);sync()
    if sha(head_path)!=before or any(p.grad is not None for p in model.parameters()):raise RuntimeError('Frozen head changed')
    return output
