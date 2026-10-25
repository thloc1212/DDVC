"""Actual confidence gradients, normalization, overlapping-GT protection and selection."""
import ast
import importlib.util
import json
import math
import sys
import tempfile
import collections
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from ddvc.dedup_loss import normalized_dedup_loss
from misc.proposal_selection import select_proposals,merge_proposals
from misc.training_state import validate_resume


def main():
    torch.manual_seed(777)
    logits=torch.tensor([[[2.],[1.],[1.5],[-3.]]],requires_grad=True)
    boxes=torch.tensor([[[.3,.2],[.31,.2],[.8,.2],[.9,.05]]],requires_grad=True)
    targets=[dict(boxes=torch.tensor([[.3,.2],[.8,.2]]),labels=torch.zeros(2,dtype=torch.long))]
    matching=([(torch.tensor([0,2]),torch.tensor([0,1]))],[])
    outputs=dict(pred_logits=logits,pred_boxes=boxes,pred_count=torch.zeros(1,6))
    loss=normalized_dedup_loss(outputs,targets,matching)
    assert loss['diag_dedup_queries']==1
    assert torch.allclose(loss['loss_dedup'],F.softplus(logits[0,1,0]))
    loss['loss_dedup'].backward();assert logits.grad[0,1,0]>0
    assert logits.grad[0,[0,2,3]].abs().sum()==0 and boxes.grad is None
    # Repeating identical duplicate negatives doesn't multiply this normalized objective.
    extra=dict(pred_logits=logits.detach()[:,[0,1,2,3,1]],pred_boxes=boxes.detach()[:,[0,1,2,3,1]])
    assert torch.allclose(normalized_dedup_loss(extra,targets,matching)['loss_dedup'],loss['loss_dedup'])
    no_duplicates=dict(pred_logits=logits.detach().clone().requires_grad_(),pred_boxes=boxes.detach())
    no_matching=([(torch.arange(4),torch.tensor([0,0,1,1]))],[])
    zero=normalized_dedup_loss(no_duplicates,targets,no_matching)['loss_dedup'];zero.backward()
    assert zero==0 and no_duplicates['pred_logits'].grad.abs().sum()==0
    # Both matched GTs may legitimately overlap: never repel their positive queries.
    overlapping=[dict(boxes=torch.tensor([[.3,.3],[.35,.3]]),labels=torch.zeros(2,dtype=torch.long))]
    overlap_out=dict(pred_logits=torch.ones(1,3,1,requires_grad=True),pred_boxes=torch.tensor([[[.3,.3],[.35,.3],[.325,.3]]]))
    protected=normalized_dedup_loss(overlap_out,overlapping,([(torch.tensor([0,1]),torch.tensor([0,1]))],[]))
    assert protected['diag_dedup_queries']==0 and protected['diag_dedup_ambiguous_skipped']==1
    # Exercise the actual criterion forward, not a mirrored implementation.
    root=Path(__file__).parent
    tree=ast.parse((root/'ddvc/criterion.py').read_text())
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SetCriterion_cl')
    env=dict(torch=torch,nn=nn,F=F,normalized_dedup_loss=normalized_dedup_loss,
             is_dist_avail_and_initialized=lambda:False,get_world_size=lambda:1)
    exec(compile(ast.Module(body=[klass],type_ignores=[]),'<actual criterion>','exec'),env)
    criterion=env['SetCriterion_cl'].__new__(env['SetCriterion_cl']);nn.Module.__init__(criterion)
    criterion.opt=SimpleNamespace(dedup_loss_coef=.1,dedup_iou_threshold=.5,dedup_confidence_threshold=.2)
    criterion.losses=['cardinality'];criterion.matcher=lambda output,target:matching
    active=dict(pred_logits=logits.detach().clone().requires_grad_(),pred_boxes=boxes.detach(),pred_count=torch.tensor([[0.,0.,5.,0.,0.,0.]]))
    losses,_=criterion(active,targets)
    assert losses['counter_count_error']==0 and losses['cardinality_error']==1
    optimizer=torch.optim.SGD([active['pred_logits']],lr=.1)
    previous=active['pred_logits'].detach().clone();(.1*losses['loss_dedup']).backward();optimizer.step()
    assert active['pred_logits'][0,1,0]<previous[0,1,0]
    assert torch.equal(active['pred_logits'].detach()[0,[0,2,3]],previous[0,[0,2,3]])
    criterion.opt.dedup_loss_coef=0.;assert 'loss_dedup' not in criterion(active,targets)[0]
    rows=[dict(timestamp=[0.,10.],sentence='chop the onion',proposal_score=.9,sentence_score=-.1,query_id=0),
          dict(timestamp=[.5,10.5],sentence='Chop the onion.',proposal_score=.88,sentence_score=-.1,query_id=1),
          dict(timestamp=[20.,30.],sentence='add oil',proposal_score=.8,sentence_score=-.1,query_id=2)]
    snapshot=json.dumps(rows,sort_keys=True)
    assert [r['query_id'] for r in select_proposals(rows,2)]==[0,1]
    assert [r['query_id'] for r in select_proposals(rows,2,'softnms')]==[0,2]
    assert json.dumps(rows,sort_keys=True)==snapshot
    merged=select_proposals(rows,2,'softnms',merge=True)
    fused=next(r for r in merged if 'merge_size' in r)
    assert fused['merge_size']==2 and 0<fused['timestamp'][0]<.5 and 10<fused['timestamp'][1]<10.5
    assert len(merged)==2 and fused['sentence']==rows[0]['sentence']
    separate=[dict(r, timestamp=[20+20*i,30+20*i]) for i,r in enumerate(rows[:2])]
    assert len(select_proposals(separate,2,'softnms',merge=True))==2
    different=[rows[0],dict(rows[1],sentence='boil pasta in water')]
    assert len(select_proposals(different,2,'softnms',merge=True))==2
    assert select_proposals([],0,'softnms')==[] and select_proposals(rows,0,'softnms')==[]
    negative=[dict(r,proposal_score=-1.) for r in rows]
    assert all(r['selection_score']<=r['joint_score'] for r in select_proposals(negative,3,'softnms'))
    for key,value in [('dedup_loss_coef',.1),('eval_selection_method','softnms'),('eval_merge_duplicates',True)]:
        try:validate_resume({key:value},{})
        except ValueError:pass
        else:raise AssertionError('Resume accepted changed recipe: '+key)
    # Execute the actual evaluate() save/selection/metric tail with an evaluator spy.
    source=ast.parse((root/'eval_utils_clip.py').read_text())
    definitions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ['save_dvc_json','reranking']]
    evaluation=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    start=next(i for i,n in enumerate(evaluation.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw_dvc_json_path' for t in n.targets))
    runner=ast.FunctionDef(name='run_tail',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='skip_lang_eval')],kwonlyargs=[],kw_defaults=[],defaults=[ast.Constant(False)]),body=evaluation.body[start:],decorator_list=[])
    runner=ast.fix_missing_locations(runner)
    with tempfile.TemporaryDirectory() as tmp:
        path=str(Path(tmp)/'prediction.json')
        opts=SimpleNamespace(count_loss_coef=.5,eval_matching_score_weight=0.,eval_selection_method='softnms',
            eval_softnms_sigma=.5,eval_merge_duplicates=False,eval_merge_iou=.7,eval_merge_text_similarity=.9,
            eval_gt_file_for_grounding='',only_ft_class_head=False,gt_file_for_eval=[],gt_file_for_para_eval=[])
        observed=[]
        def score_selected(filename,**kwargs):
            saved=json.loads(Path(filename).read_text());observed.append(saved)
            assert [p['query_id'] for p in saved['results']['video']]==[0,2]
            return dict(CIDEr=.4,METEOR=.06,soda_c=.07,MetaScore=.13)
        scope=dict(json=json,np=np,math=math,select_proposals=select_proposals,
            opt=opts,dvc_json_path=path,out_json=dict(results={'video':[dict(r,pred_event_count=2) for r in rows]},version='1',external_data={}),
            out_json_g=dict(results={'video':[]}),aux_out_json_g=dict(results={'video':[]}),
            loss_sum={},loader=[1],logger=None,alpha=1.,skip_lang_eval=False,dvc_eval_version='2018',infer_mode=False,
            eval_metrics=score_selected,eval_metrics_grounding=lambda *a,**k:{},plot_proposal_distribution=lambda *a:None)
        exec(compile(ast.fix_missing_locations(ast.Module(body=definitions+[runner],type_ignores=[])),'<actual evaluate selection/save>','exec'),scope)
        scores,_=scope['run_tail']();final=json.loads(Path(path).read_text());raw_saved=json.loads(Path(path[:-5]+'.raw.json').read_text())
        assert len(raw_saved['results']['video'])==3 and len(final['results']['video'])==2
        assert final['CIDEr']==scores['CIDEr']==.4 and final['selection']['method']=='softnms' and len(observed)==1
    train_tree=ast.parse((root/'train.py').read_text())
    warmup=next(n for n in ast.walk(train_tree) if isinstance(n,ast.If)
                and "weight_dict['loss_dedup']" in ast.unparse(n)
                and isinstance(n.test,ast.Compare))
    for epoch,expected_weight in [(0,0.),(1,.1),(14,.1)]:
        state=dict(opt=SimpleNamespace(dedup_loss_coef=.1,dedup_start_epoch=1),epoch=epoch,weight_dict={'loss_dedup':.1})
        exec(compile(ast.Module(body=[warmup],type_ignores=[]),'<actual train dedup warmup>','exec'),state)
        assert state['weight_dict']['loss_dedup']==expected_weight
    # Optional exact parity with audited real candidates, no scoring or GT required.
    data=root.parent/'kiemtra'
    if data.exists():
        for epoch in [11,14]:
            raw=data/f'prediction_extracted/prediction/num436_epoch{epoch}.raw.json'
            expected=data/f'analysis/epoch{epoch}_softnms_sigma0.5.json'
            if raw.exists() and expected.exists():
                candidates=json.loads(raw.read_text())['results'];reference=json.loads(expected.read_text())['results']
                for vid,items in candidates.items():
                    actual=select_proposals(items,len(reference[vid]),'softnms')
                    assert [p['query_id'] for p in actual]==[p['query_id'] for p in reference[vid]],(epoch,vid)
                original=json.loads(raw.with_name(f'num436_epoch{epoch}.json').read_text())['results']
                for vid,items in candidates.items():
                    assert [p['query_id'] for p in select_proposals(items,len(original[vid]))]==[p['query_id'] for p in original[vid]],(epoch,vid)
    print('PASS dedup actual-criterion backward/update, normalized repeats, empty sets, protected/ambiguous GT, counter logging, Soft-NMS audit parity, conservative merge and resume guards')


if __name__=='__main__':main()
