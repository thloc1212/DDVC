"""No-update paired temporal intervention on the real C1 best checkpoint.

Cyclic feature shift preserves content but creates a join. Events crossing that
join are excluded from BOTH paired GT sets. This is a diagnostic, not a benchmark
result or proof of invariance, and never modifies training data or saved weights.
"""
import argparse
import collections
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from c1_diagnostics import dump,table,matrix,matching


def shift_intervals(intervals,duration,fraction):
    if duration<=0 or not 0<fraction<1:raise ValueError('Invalid duration/shift')
    cut=duration*(1-fraction);delta=duration*fraction;retained=[]
    for index,(s,e) in enumerate(intervals):
        if not 0<=s<e<=duration+max(1.,duration)*1e-5:raise ValueError('Invalid source interval')
        e=min(e,duration)
        if e<=cut:
            retained.append((index,[s+delta,e+delta]))
        elif s>=cut:
            retained.append((index,[s+delta-duration,e+delta-duration]))
    return retained


def paired_items(item,steps):
    import torch
    features,featstamps,labels,tokens,token_mask,stamps,duration,captions,key=item
    length=len(features)
    if not 0<steps<length:raise ValueError('Shift must be inside the feature sequence')
    keep=shift_intervals(stamps,float(duration),steps/length)
    if not keep:return None
    ids=[i for i,_ in keep];indices=torch.tensor(ids,dtype=torch.long)
    original=[stamps[i] for i in ids];shifted=[v for _,v in keep]
    def assemble(f,ts):
        feat=np.clip((length*np.asarray(ts)/duration).astype(int),0,length-1).tolist()
        lab=np.asarray(labels)[ids].tolist()
        return (f,feat,lab,tokens.index_select(0,indices),token_mask.index_select(0,indices),
                ts,duration,[captions[i] for i in ids],key)
    return assemble(features.copy(),original),assemble(np.roll(features,steps,axis=0).copy(),shifted)


def run(args):
    import fcntl,torch
    import c1_pipeline as pipeline
    from opts import parse_opts
    from train import build,PropSeqDataset,collate_fn,load_clip_memory_bank,load_clip_gt
    from misc.runtime_perf import configure_runtime,amp_context
    from misc.training_state import schedule_contrastive
    from scipy.optimize import linear_sum_assignment
    work=args.work.resolve();cfg=pipeline.check(pipeline.config_path(work,args.dataset))
    if not torch.cuda.is_available():raise RuntimeError('Real checkpoint probe requires CUDA')
    if not 0<args.shift<1 or args.videos<1:raise ValueError('Invalid sample size or shift')
    with (work/'logs/c1_gpu.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Training/probe still owns the C1 GPU lock')
        run_folder=Path(cfg['save_dir'])/cfg['id'];checkpoint=run_folder/'model-best.pth'
        before=pipeline.sha(checkpoint)
        info=json.loads((run_folder/'info.json').read_text(encoding='utf8'))
        opt=parse_opts(['--cfg_path',str(pipeline.config_path(work,args.dataset))],write_options=False)
        if not opt.text_crossAttn:opt.cap_num_feature_levels=4
        if opt.combined_encoder:opt.cap_num_feature_levels=5
        if opt.caption_decoder_new=='gpt2':opt.caption_decoder_type='gpt2'
        opt.rec_encdec=True;configure_runtime(opt)
        random.seed(777);np.random.seed(777);torch.manual_seed(777)
        ds=PropSeqDataset(opt.val_caption_file,opt.visual_feature_folder,opt.dict_file,False,'gt',opt)
        model,criterion,contrastive,post=build(opt);model.translator=ds.translator
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if cp.get('epoch')!=info['best']['epoch']:raise ValueError('Best weights/info epoch mismatch')
        model.load_state_dict(cp['model'],strict=True);del cp
        model.to(opt.device);model.eval()
        schedule_contrastive(criterion.weight_dict,info['best']['epoch'],opt.cl_schedule_time,opt.cl_schedule_val,False)
        bank=load_clip_memory_bank(opt);bank['vid_sent_embeds']=torch.as_tensor(bank['vid_sent_embeds'],device=opt.device)
        bank['clip_gt']=load_clip_gt(opt)
        loader=SimpleNamespace(dataset=ds)
        output_folder=run_folder/'diagnostics/visual_probe';output_folder.mkdir(parents=True,exist_ok=True)
        order=np.random.default_rng(777).permutation(len(ds));records=[];summaries=[]
        def infer(item):
            batch=collate_fn([item])
            batch={k:v.to(opt.device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            batch['video_target']=[{k:v.to(opt.device) if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in batch['video_target']]
            batch=collections.defaultdict(lambda:None,batch)
            with torch.no_grad(),amp_context(opt):
                prediction,_=model(batch,criterion,contrastive,opt.transformer_input_type,bank,eval_mode=True)
                result=post['bbox'](prediction,batch['video_length'][:,1],loader,model,None)[0]
            rows=[]
            for i,box in enumerate(result['boxes']):
                if float(box[1])<=float(box[0]):continue
                rows.append(dict(timestamp=box.float().cpu().tolist(),query_id=int(result['query_id'][i]),
                                 proposal_score=float(result['scores'][i]),sentence=result['captions'][i],
                                 sentence_score=float(result['caption_scores'][i]),pred_event_count=int(result['pred_seq_len'])))
            if not rows:raise RuntimeError('Model returned no valid temporal candidates')
            return rows
        for index in order:
            # Identical GT subsampling for each paired input; no random query order comparisons.
            np.random.seed(777+int(index));item=ds[int(index)];length=len(item[0])
            if length<2:continue
            steps=max(1,min(length-1,round(length*args.shift)))
            paired=paired_items(item,steps)
            if paired is None:continue
            normal,shifted=paired;a=infer(normal);b=infer(shifted)
            gt_a=normal[5];gt_b=shifted[5];duration=float(item[6])
            def distance(x,y):
                x=np.asarray(x).reshape(-1,2)/duration;y=np.asarray(y).reshape(-1,2)/duration
                if not len(x) or not len(y):return None
                cost=np.abs(x[:,None,:]-y[None,:,:]).mean(-1);ii,jj=linear_sum_assignment(cost)
                return float(cost[ii,jj].mean())
            retained=shift_intervals([r['timestamp'] for r in a],duration,steps/length)
            expected=[v for i,v in retained];fixed=[a[i]['timestamp'] for i,v in retained]
            rolled=[r['timestamp'] for r in b]
            summary=dict(video_id=item[-1],retained_gt=len(gt_a),shift_fraction=steps/length,
                         normal_matched=len(matching(matrix([r['timestamp'] for r in a],gt_a))),
                         shifted_matched=len(matching(matrix(rolled,gt_b))),
                         shifted_against_unshifted_gt=len(matching(matrix(rolled,gt_a))),
                         distance_to_expected_shift=distance(expected,rolled),distance_to_fixed_prior=distance(fixed,rolled),
                         normal_counter=a[0]['pred_event_count'],shifted_counter=b[0]['pred_event_count'])
            summaries.append(summary);records.append(dict(**summary,normal_gt=gt_a,shifted_gt=gt_b,normal=a,shifted=b))
            print('VISUAL_PROBE',len(records),'/',args.videos,json.dumps(summary),flush=True)
            if len(records)%10==0:dump(output_folder/'pairs.json',records);table(output_folder/'per_video.tsv',summaries)
            if len(records)>=args.videos:break
        if not records:raise RuntimeError('No valid non-wrapping probe pairs')
        dump(output_folder/'pairs.json',records);table(output_folder/'per_video.tsv',summaries)
        total=sum(r['retained_gt'] for r in summaries)
        report=dict(videos=len(records),checkpoint_epoch=info['best']['epoch'],checkpoint_sha256=before,
                    shift=args.shift,normal_recall=100*sum(r['normal_matched'] for r in summaries)/total,
                    shifted_recall=100*sum(r['shifted_matched'] for r in summaries)/total,
                    shifted_vs_unshifted_gt_recall=100*sum(r['shifted_against_unshifted_gt'] for r in summaries)/total,
                    warning='Intervention diagnostic only. Circular join events excluded from both GT sets; not official benchmark metrics.')
        if pipeline.sha(checkpoint)!=before:raise RuntimeError('Checkpoint changed during no-update probe')
        dump(output_folder/'summary.json',report);print('VISUAL_PROBE_READY',json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--dataset',choices=('yc2','anet'),required=True)
    parser.add_argument('--videos',type=int,default=200);parser.add_argument('--shift',type=float,default=.25)
    run(parser.parse_args())
