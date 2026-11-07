"""Offline (or --cuda) relation/quality supervision and selection failure-case checks."""
import argparse
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from ddvc.event_set import EventSetHeads, event_set_objective, gt_relations
from misc.event_set_selection import select_event_set

ROOT = Path(__file__).resolve().parent


def heads_test(device):
    torch.manual_seed(41)
    model = EventSetHeads(8, 6, 16).to(device)
    # One long event, one nested event and a duplicate of the nested candidate.
    boxes = torch.tensor([[[.5,.8],[.5,.1],[.5,.1],[.15,.03]]],device=device,requires_grad=True)
    loc = torch.randn(1,4,8,device=device,requires_grad=True)
    cap = torch.randn_like(loc,requires_grad=True)
    features = torch.randn(1,15,6,device=device,requires_grad=True)
    valid = torch.tensor([[True]*12+[False]*3],device=device)
    logits = torch.zeros(1,4,1,device=device,requires_grad=True)
    count = torch.zeros(1,31,device=device,requires_grad=True)
    targets = [{'boxes': boxes[0,:2].detach()}]
    pairs = [(torch.tensor([1,0],device=device),torch.tensor([1,0],device=device))]
    context = model(loc,cap,features,boxes,logits,valid)
    assert context['relation_logits'].shape == (1,4,4,5)
    objective = event_set_objective(context,targets,pairs,count)
    assert all(bool(torch.isfinite(x).all()) for x in objective.values())
    assert float(objective['event_relation_real_duplicate_pairs']) >= 2
    loss = sum(v for k,v in objective.items() if k.startswith('loss_'))
    loss.backward()
    assert model.quality[-1].weight.grad.abs().sum()>0
    assert model.relation[-1].weight.grad.abs().sum()>0
    assert count.grad.abs().sum()>0
    assert all(v.grad is None for v in (loc,cap,features,boxes,logits))
    model.eval()
    changed=features.detach().clone();changed[:,12:]=1e9
    with torch.no_grad():
        first=model(loc,cap,features,boxes,logits,valid)
        second=model(loc,cap,changed,boxes,logits,valid)
    for key in ('quality_logits','relation_logits'):
        torch.testing.assert_close(first[key],second[key])
    assert 'synthetic_duplicate_logits' not in first
    assert gt_relations(targets[0]['boxes'][:1],targets[0]['boxes'][1:]).item()==2
    assert gt_relations(targets[0]['boxes'][1:],targets[0]['boxes'][:1]).item()==3
    # Different GT IDs at identical intervals are overlap, not duplicates.
    coincident=torch.tensor([[[.5,.4],[.5,.4]]],device=device)
    ctx=model(loc[:,:2],cap[:,:2],features,coincident,logits[:,:2],valid)
    labels_seen=[]
    original=F.cross_entropy
    def record(predicted,labels,**kwargs):
        labels_seen.extend(labels.tolist());return original(predicted,labels,**kwargs)
    with patch('ddvc.event_set.F.cross_entropy',side_effect=record):
        event_set_objective(ctx,[{'boxes':coincident[0]}],
                            [(torch.arange(2,device=device),torch.arange(2,device=device))],count.detach())
    assert labels_seen and set(labels_seen)=={1},labels_seen
    # One-event and empty-target batches stay finite; no pairwise divide-by-zero.
    for n in (0,1):
        ls=event_set_objective(first,[{'boxes':boxes[0,:n].detach()}],
             [(torch.arange(n,device=device),torch.arange(n,device=device))],count.detach())
        assert all(bool(torch.isfinite(v).all()) for v in ls.values())
    optimizer=torch.optim.Adam(model.parameters(),lr=.02)
    initial=None
    model.train()
    for _ in range(35):
        optimizer.zero_grad()
        ls=event_set_objective(model(loc,cap,features,boxes,logits,valid),targets,pairs,count.detach())
        total=ls['loss_event_quality']+ls['loss_event_relation']
        if initial is None:initial=float(total.detach())
        total.backward();optimizer.step()
    assert float(total.detach())<initial*.75,(float(total.detach()),initial)
    # Checkpoint carries the new heads and can be loaded strictly.
    clone=EventSetHeads(8,6,16).to(device);clone.load_state_dict(model.state_dict(),strict=True)
    print('PASS heads:',device,'quality/relation/ordinal gradients, detached inputs, nested/coincident GT, padding, singleton, learning, state dict')


def selection_test():
    def row(q,span,quality=.95,sentence='a person performs an action'):
        return dict(query_id=q,timestamp=span,event_quality=quality,proposal_score=.9,
                    sentence_score=-5.,sentence=sentence,pred_event_count=1)
    rows=[row(0,[0,100]),row(1,[40,50]),row(2,[41,51]),row(3,[110,115],.02)]
    duplicate=[[0.]*4 for _ in range(4)];duplicate[1][2]=duplicate[2][1]=1.
    # A strong count=1 prior can be overridden by two high-quality distinct events.
    before=copy.deepcopy(rows)
    final,diag=select_event_set(rows,duplicate,[-20,20,-20,-20,-20])
    assert {r['query_id'] for r in final}=={0,1},final
    assert diag['predicted_N']==1 and diag['selected_N']==2
    assert rows==before
    # Query IDs, not score-sorted row positions, index the pair matrix.
    permuted,_=select_event_set([rows[i] for i in [3,2,0,1]],duplicate,[-20,20,-20,-20,-20])
    assert {r['query_id'] for r in permuted}=={0,1}
    # Similar/identical captions alone must not suppress distinct events.
    separate=[row(0,[0,10],sentence='the men continue to play'),row(1,[70,80],sentence='the men continue to play')]
    final,_=select_event_set(separate,[[0,0],[0,0]],[0,0,0])
    assert len(final)==2
    # Poor candidates do not all get selected merely because Q grew.
    poor=[row(i,[i,i+1],.01) for i in range(30)]
    final,_=select_event_set(poor,[[0.]*30 for _ in range(30)],[0.]*31)
    assert len(final)==1
    final,diag=select_event_set([],[],[0.,0.]);assert final==[] and diag['selected_N']==0
    try:select_event_set([dict(rows[0],event_quality=float('nan'))],duplicate,[0,0])
    except ValueError:pass
    else:raise AssertionError('NaN quality accepted')
    print('PASS selection: nested events retained, duplicate suppressed, soft count overridable, row-order invariance, no caption-only merge, Q30 restraint')


def criterion_test(device):
    tree=ast.parse((ROOT/'ddvc/criterion.py').read_text(encoding='utf8'))
    nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in
           ('sigmoid_focal_loss','cross_entropy_with_gaussian_mask')) or
           (isinstance(n,ast.ClassDef) and n.name=='SetCriterion_cl')]
    namespace=dict(torch=torch,nn=nn,F=F,is_dist_avail_and_initialized=lambda:False,get_world_size=lambda:1)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_criterion','exec'),namespace)
    pairs=[(torch.tensor([2,0],device=device),torch.tensor([0,1],device=device))]
    opt=SimpleNamespace(max_eseq_length=30,counter_class_rates=[0.,0.,1.]+[0.]*28,lloss_gau_mask=1,lloss_beta=1)
    criterion=namespace['SetCriterion_cl'](1,lambda out,gt:(pairs,pairs),{},['labels'],opt=opt)
    model=EventSetHeads(8,6,16).to(device)
    boxes=torch.tensor([[[.6,.3],[.5,.2],[.15,.1]]],device=device)
    logits=torch.zeros(1,3,1,device=device,requires_grad=True)
    context=model(torch.randn(1,3,8,device=device),torch.randn(1,3,8,device=device),
                  torch.randn(1,12,6,device=device),boxes,logits,torch.ones(1,12,dtype=torch.bool,device=device))
    outputs=dict(pred_logits=logits,pred_count=torch.zeros(1,31,device=device,requires_grad=True),pred_boxes=boxes,event_set=context)
    gt=[dict(boxes=boxes[0,[2,0]],labels=torch.zeros(2,dtype=torch.long,device=device))]
    losses,_=criterion(outputs,gt)
    assert {'loss_event_quality','loss_event_relation','loss_count_ordinal'}<=set(losses)
    sum(v for k,v in losses.items() if k.startswith('loss_')).backward()
    assert model.quality[-1].weight.grad.abs().sum()>0 and model.relation[-1].weight.grad.abs().sum()>0
    print('PASS actual criterion: one-to-one IDs, quality/relation targets, final-only ordinal loss and gradients')


def saved_selection_test():
    # Execute the real JSON reranker, including matrix serialization/removal and
    # original Top-N/Soft-NMS controls, without loading Java/evaluator packages.
    tree=ast.parse((ROOT/'eval_utils_clip.py').read_text(encoding='utf8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('save_dvc_json','reranking')]
    from misc.proposal_selection import select_proposals
    import math
    ns=dict(json=json,np=np,math=math,select_event_set=select_event_set,select_proposals=select_proposals)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_json_reranking','exec'),ns)
    rows=[dict(query_id=q,timestamp=span,event_quality=.95,proposal_score=.9,
               sentence_score=-5.,sentence='a person performs an action',pred_event_count=1)
          for q,span in [(2,[41,51]),(0,[0,100]),(1,[40,50])]]
    pairs=[[0.]*3 for _ in range(3)];pairs[1][2]=pairs[2][1]=1.
    raw=dict(results={'video':rows},counter_predictions={'video':dict(logits=[-20,20,-20,-20],predicted_N=1)},
             event_set_predictions={'video':dict(duplicate_probs=pairs)})
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'num1_epoch0.raw.json';path.write_text(json.dumps(raw))
        before=path.read_bytes()
        for method,n in [('topk',1),('softnms',1),('event_set',2)]:
            selected=ns['reranking'](str(path),alpha=.3,cl_score_weight=0.,temperature=2.,method=method)
            final=json.loads(Path(selected).read_text())
            assert len(final['results']['video'])==n
            assert 'event_set_predictions' not in final
            if method=='event_set':
                assert final['counter_predictions']['video']['selected_N']==2
                assert {r['query_id'] for r in final['results']['video']}=={0,1}
            assert path.read_bytes()==before
    print('PASS actual saved JSON: all three selectors, original query IDs, raw unchanged, final count diagnostics, no repeated pair matrices')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cuda',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2)
    if args.cuda and not torch.cuda.is_available():raise RuntimeError('CUDA required for requested smoke')
    device='cuda' if args.cuda else 'cpu'
    heads_test(device);selection_test();criterion_test(device);saved_selection_test()
