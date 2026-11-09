"""Diagnostic oracles over frozen C1 captions/boxes; no training or checkpoint load.

GT-assisted cases are upper-bound diagnostics, never deployable method results.
Temporal matching uses reference_0 and maximum-cardinality matching at tIoU >= .5.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from misc.proposal_selection import select_proposals


def matrix(a,b):
    a=np.asarray(a,dtype=float).reshape(-1,2);b=np.asarray(b,dtype=float).reshape(-1,2)
    inter=np.maximum(0,np.minimum(a[:,None,1],b[None,:,1])-np.maximum(a[:,None,0],b[None,:,0]))
    union=np.maximum(0,a[:,1]-a[:,0])[:,None]+np.maximum(0,b[:,1]-b[:,0])[None,:]-inter
    return inter/np.maximum(union,1e-12)


def matching(m,threshold=.5):
    if not m.size:return []
    a,b=linear_sum_assignment(-((m>=threshold)*(min(m.shape)+1)+m))
    return [(int(i),int(j)) for i,j in zip(a,b) if m[i,j]>=threshold]


def oracle_rank(rows,gt,n):
    """One-to-one temporal oracle, then fill unused slots; preserves caption strings."""
    if not rows:return []
    m=matrix([r['timestamp'] for r in rows],gt)
    pairs=sorted(matching(m),key=lambda ij:-m[ij])
    ids=[i for i,j in pairs][:n]
    ids+=sorted((i for i in range(len(rows)) if i not in ids),key=lambda i:-m[i].max())[:max(0,n-len(ids))]
    return sorted((dict(rows[i]) for i in ids),key=lambda r:(r['timestamp'],r['query_id']))


def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')


def table(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(rows)


def diagnose(work,dataset,score=False):
    import c1_pipeline as pipeline
    cfg=pipeline.check(pipeline.config_path(work,dataset),require_features=False)
    run=Path(cfg['save_dir'])/cfg['id'];info=json.loads((run/'info.json').read_text(encoding='utf8'))
    epoch=info['best']['epoch'];pred=list((run/'prediction').glob(f'num*_epoch{epoch}.raw.json'))
    if len(pred)!=1:raise FileNotFoundError('Expected exactly one best-epoch raw prediction')
    raw=json.loads(pred[0].read_text(encoding='utf8'))
    final=json.loads(pred[0].with_name(pred[0].name.replace('.raw.json','.json')).read_text(encoding='utf8'))
    gt=json.loads(Path(cfg['gt_file_for_eval'][0]).read_text(encoding='utf8'))
    folder=run/'diagnostics';folder.mkdir(exist_ok=True)
    modes=('C1_softnms','oracle_N_ref0','oracle_rank_fixed_N','oracle_N_and_rank_ref0')
    outputs={k:{} for k in modes};totals={k:collections.Counter() for k in modes};per_video=[]
    groups=collections.defaultdict(collections.Counter);counter_hist=collections.Counter();gt_hist=collections.Counter()
    for vid,rows in raw['results'].items():
        stamps=gt[vid]['timestamps'];ng=len(stamps);n=int(rows[0]['pred_event_count']) if rows else 0
        counter_hist[n]+=1;gt_hist[ng]+=1
        outputs['C1_softnms'][vid]=select_proposals(rows,n,method='softnms',sigma=.5,alpha=cfg['ec_alpha'],temperature=2.)
        outputs['oracle_N_ref0'][vid]=select_proposals(rows,ng,method='softnms',sigma=.5,alpha=cfg['ec_alpha'],temperature=2.)
        outputs['oracle_rank_fixed_N'][vid]=oracle_rank(rows,stamps,n)
        outputs['oracle_N_and_rank_ref0'][vid]=oracle_rank(rows,stamps,ng)
        semantic=lambda r:[(v['query_id'],v['timestamp'],v['sentence']) for v in r]
        if semantic(outputs['C1_softnms'][vid])!=semantic(final['results'][vid]):
            raise ValueError('Selection does not reproduce stored C1 final: '+vid)
        raw_matches=len(matching(matrix([r['timestamp'] for r in rows],stamps)))
        for mode in modes:
            selected=outputs[mode][vid];pairs=matching(matrix([r['timestamp'] for r in selected],stamps))
            matched=len(pairs);matched_gt={j for _,j in pairs}
            totals[mode].update(videos=1,events=len(selected),gt=ng,matched=matched,raw_matched=raw_matches,
                                count_abs_error=abs(len(selected)-ng),gt_exceeds_Q=int(ng>len(rows)))
            per_video.append(dict(video_id=vid,mode=mode,gt_events=ng,events=len(selected),counter_N=n,
                                  matched_tiou05=matched,raw_oracle_tiou05=raw_matches))
            for j,(s,e) in enumerate(stamps):
                overlap=any(k!=j and min(e,b)>max(s,a) for k,(a,b) in enumerate(stamps))
                child=any(k!=j and a<=s and b>=e and (a<s or b>e) for k,(a,b) in enumerate(stamps))
                labels=['all','short_le10s' if e-s<=10 else 'long_gt10s']
                if overlap:labels.append('overlap')
                if overlap and e-s<=10:labels.append('short_overlap')
                if child:labels.append('nested_child')
                for label in labels:
                    groups[(mode,label)]['gt']+=1;groups[(mode,label)]['matched']+=j in matched_gt
    summary=[]
    for mode in modes:
        document=dict(results=outputs[mode],version='VERSION 1.0',external_data=raw.get('external_data',{}))
        dest=folder/(mode+'.json');dump(dest,document)
        stats=totals[mode]
        result=dict(case=mode,checkpoint_epoch=epoch,GT_ASSISTED=mode!='C1_softnms',
                    events=stats['events'],events_per_video=stats['events']/stats['videos'],
                    count_mae_ref0=stats['count_abs_error']/stats['videos'],
                    recall_one_to_one_tiou05_ref0=100*stats['matched']/stats['gt'],
                    raw_oracle_recall_tiou05_ref0=100*stats['raw_matched']/stats['gt'],
                    videos_GT_exceeds_Q=stats['gt_exceeds_Q'])
        signature=hashlib.sha256(dest.read_bytes()+b''.join(Path(p).read_bytes() for p in cfg['gt_file_for_eval']+cfg['gt_file_for_para_eval'])).hexdigest()
        metric_path=folder/(mode+'.metrics.json')
        if score:
            cached=json.loads(metric_path.read_text(encoding='utf8')) if metric_path.is_file() else {}
            if cached.get('signature')==signature:
                scores=cached['scores']
            else:
                from eval_utils_clip import eval_metrics
                print('OFFICIAL_SCORING',mode,'GT_ASSISTED',result['GT_ASSISTED'],flush=True)
                scores={k:float(v)*100 for k,v in eval_metrics(str(dest),cfg['gt_file_for_eval'],cfg['gt_file_for_para_eval']).items()}
                dump(metric_path,dict(signature=signature,scores=scores))
            result.update({k:scores[k] for k in pipeline.METRICS})
            if mode=='C1_softnms':
                for k in pipeline.METRICS:
                    if abs(scores[k]-100*final[k])>1e-4:raise ValueError('Stored official metric not reproduced: '+k)
        summary.append(result);table(folder/'summary.tsv',summary)
        print('DIAGNOSTIC',json.dumps(result),flush=True)
    table(folder/'per_video.tsv',per_video)
    table(folder/'strata.tsv',[dict(case=mode,group=group,**stats,recall_pct=100*stats['matched']/stats['gt'])
                              for (mode,group),stats in groups.items()])
    dump(folder/'count_histograms.json',dict(counter=dict(counter_hist),ground_truth_ref0=dict(gt_hist)))
    dump(folder/'manifest.json',dict(checkpoint_epoch=epoch,source_raw=str(pred[0]),source_sha256=pipeline.sha(pred[0]),
                                  reference_for_oracles=cfg['gt_file_for_eval'][0],
                                  warning='GT-assisted selection is a diagnostic upper bound, not an inference method.'))
    print('DIAGNOSTICS_READY',folder,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--dataset',choices=('yc2','anet'),required=True);parser.add_argument('--score',action='store_true')
    args=parser.parse_args();diagnose(args.work.resolve(),args.dataset,args.score)
