"""Fixed-policy scoring. No quality/generator optimizer is constructed here."""
import contextlib
import json
from pathlib import Path
import re
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from reranking.cache import atomic_json, sha
from reranking.core import select
from reranking.pipeline import complete_scores, signature, write_table

BETAS = (0., .25, .5, 1., 2., 4., 6., 8.)
FIXED = {'yc2': 1., 'anet': 6.}
METRICS = ('CIDEr','METEOR','Bleu_4','soda_c','Precision','Recall','F1',
           'Bleu_1','Bleu_2','Bleu_3','ROUGE_L','MetaScore')


def with_f1(scores):
    result = {k: float(v) for k,v in scores.items() if isinstance(v,(float,int,np.number))}
    p,r = result['Precision'],result['Recall']
    result['F1'] = 2*p*r/(p+r) if p+r else 0.
    if not complete_scores(result) or not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Missing or nonfinite official metrics')
    return result


def official_scores(path, references, paragraphs):
    sys.path[:0] = [str(ROOT/'densevid_eval3'),str(ROOT/'densevid_eval3/SODA')]
    from densevid_eval3.eval_dvc import eval_dvc
    from densevid_eval3.eval_soda import eval_soda
    from densevid_eval3.eval_para import eval_para
    result = {k:float(np.mean(v)) for k,v in eval_dvc(str(path),references,version='2018',infer_mode=True).items()}
    result.update(eval_soda(str(path),ref_list=references))
    result.update(eval_para(str(path),referneces=paragraphs))
    result['MetaScore'] = result['METEOR']+result['soda_c']
    return with_f1({k:float(v)*100 for k,v in result.items()})


def score(folder, name, predictions, cfg, identity, beta):
    folder = Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path = folder/(name+'.json')
    atomic_json(path,dict(version='VERSION 1.0',external_data={},results=predictions,
                          selection=dict(method='gaussian_softnms',sigma=.5,beta=beta,identity=identity)))
    sig = signature(dict(prediction=sha(path),refs=[sha(p) for p in cfg['gt_file_for_eval']+cfg['gt_file_for_para_eval']],
                         protocol='ddvc-2018-full-language-infer-mode-v1'))
    target=folder/(name+'.metrics.json')
    saved=json.loads(target.read_text(encoding='utf8')) if target.is_file() else {}
    if saved.get('signature')==sig and complete_scores(saved.get('scores')):
        scores=with_f1(saved['scores'])
    else:
        print('OFFICIAL_SCORING',name,'beta',beta,flush=True)
        with (folder/'scorer.log').open('a',encoding='utf8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            scores=official_scores(path,cfg['gt_file_for_eval'],cfg['gt_file_for_para_eval'])
        atomic_json(target,dict(signature=sig,scores=scores,identity=identity))
    row=dict(method=name,beta=beta,events=sum(map(len,predictions.values())),**{k:scores[k] for k in METRICS if k in scores})
    print('ABLATION_RESULT',json.dumps(row),flush=True)
    return row


def select_fixed(raw, qualities, beta):
    if set(raw)!=set(qualities):raise ValueError('Quality and raw video pools differ')
    outputs={}
    for vid,rows in raw.items():
        byid=qualities[vid]
        if set(map(int,byid))!={int(r['query_id']) for r in rows}:raise ValueError('Quality query pool changed')
        q=[float(byid[str(r['query_id'])]) for r in rows]
        outputs[vid]=select(rows,int(rows[0]['pred_event_count']),q,beta)
        if len(outputs[vid])!=min(len(rows),int(rows[0]['pred_event_count'])):raise ValueError('Counter budget changed')
    return outputs


def score_joint_epoch(raw_path, opt):
    raw_path=Path(raw_path)
    raw=json.loads(raw_path.read_text(encoding='utf8'))['results']
    qualities={vid:{str(r['query_id']):float(r['reranker_quality']) for r in rows} for vid,rows in raw.items()}
    beta=FIXED[opt.dataset]
    folder=raw_path.parent.parent/'quality_epochs'/raw_path.name.replace('.raw.json','')
    folder.mkdir(parents=True,exist_ok=True)
    cfg=vars(opt)
    identity=dict(regime='joint',caption_matching=True,content_conditioned=True,mask_loss=0.,
                  beta_fixed_before_evaluation=True,checkpoint_selection='generator MetaScore; unchanged')
    row=score(folder,'joint_quality_softnms',select_fixed(raw,qualities,beta),cfg,identity,beta)
    write_table(folder/'summary.tsv',[row])
    # Readable history is supplementary; original generator metrics choose the checkpoint.
    history=raw_path.parent.parent/'quality_epoch_history.json'
    rows=json.loads(history.read_text(encoding='utf8')) if history.is_file() else {}
    rows[raw_path.stem]=row;atomic_json(history,rows)
    print('JOINT_VALIDATION_EPOCH',raw_path.name,json.dumps(row),flush=True)
    return row


def finalize_joint(work, dataset):
    from misc.prawn_ablation import run_name
    work=Path(work);run=work/'runs'/run_name(dataset)
    info=json.loads((run/'info.json').read_text(encoding='utf8'))
    name=Path(info['best']['result_json_path']).name.replace('.json','')
    folder=run/'quality_epochs'/name
    if not (folder/'summary.tsv').is_file():raise FileNotFoundError('Missing best-epoch joint evaluation: '+str(folder))
    target=run/'joint_evaluation';target.mkdir(exist_ok=True)
    for p in folder.iterdir():
        if p.is_file():shutil.copy2(p,target/p.name)
    atomic_json(target/'provenance.json',dict(best_generator_epoch=info['best']['epoch'],
        selection='Same generator MetaScore checkpoint rule as two-stage',
        quality_head_epochs='Trained jointly during generator epochs; no head-only stage',
        budget_caveat='Two-stage additionally trains frozen-generator head for 20 epochs'))
    print('JOINT_BEST_SUMMARY',target/'summary.tsv',flush=True)
    print((target/'summary.tsv').read_text(encoding='utf-8-sig'),flush=True)
    return target
