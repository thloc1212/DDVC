"""Offline CPU checks with actual DDVC caption methods and random tiny GPT-2."""
import ast
import copy
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel


def main():
    torch.set_num_threads(2)
    torch.manual_seed(777)
    root = Path(__file__).resolve().parent
    package = types.ModuleType('_proposal_caption_test')
    package.__path__ = [str(root/'ddvc/CaptioningHead')]
    sys.modules[package.__name__] = package
    def read(name):
        spec = importlib.util.spec_from_file_location(package.__name__+'.'+name, Path(package.__path__[0])/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    pm, local = read('proposal_mask'), read('local_visual')
    Head = read('gpt2').GPT2CaptionModel
    head = Head.__new__(Head)
    nn.Module.__init__(head)
    head.opt = SimpleNamespace(add_cross_attention=True, learned_proposal_mask=True,
        local_visual_attention=False, max_caption_len=6, gpt_beam_size=2)
    head.fc1, head.fc2, head.relu = nn.Linear(8,16), nn.Linear(16,768), nn.ReLU()
    head.model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=768, n_layer=1, n_head=12,
        n_inner=32, n_positions=32, add_cross_attention=True, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.,
        bos_token_id=0, eos_token_id=1, pad_token_id=0))
    head.proposal_mask = pm.LearnedGatedProposalMask(bins=16,encoding_dim=32)
    head.prefix_token = torch.zeros(1,1,dtype=torch.long)
    head.tokenizer = SimpleNamespace(encode=lambda _: [1],pad_token_id=0)
    boxes = torch.tensor([[.25,.2],[.75,.2]],requires_grad=True)
    anchors = torch.tensor([[.3,.3],[.7,.3]])
    confidence = torch.tensor([.25,.7],requires_grad=True)
    valid = torch.tensor([[True]*8+[False]*2])
    details = head.proposal_mask(boxes,valid,confidence,anchors)
    assert details['weights'].shape == (2,10)
    assert not details['weights'][:,8:].any()
    assert (details['weights']>=0).all() and (details['weights']<=1).all()
    hard = head.proposal_mask(boxes,valid,torch.ones(2),anchors)
    assert torch.equal(hard['weights'],hard['hard'])
    soft = head.proposal_mask(boxes,valid,torch.zeros(2),anchors)
    assert torch.allclose(soft['weights'][:,:8],soft['logits'][:,:8].sigmoid())
    expanded = head.proposal_mask(boxes,torch.tensor([[True]*8+[False]*5]),confidence,anchors)
    assert torch.allclose(details['weights'][:,:8],expanded['weights'][:,:8])
    singleton = head.proposal_mask(torch.tensor([[.5,0.]]),torch.tensor([[True,False]]),torch.tensor([.5]))
    assert singleton['hard'].tolist()==[[1.,0.]]
    print('PASS: mask bounds, confidence-gate endpoints, padding/length invariance and sub-frame events')

    hs = torch.randn(2,1,8,requires_grad=True)
    video = torch.randn(1,10,768,requires_grad=True)
    labels = torch.tensor([[0,3,4,1],[0,5,6,1]])
    kwargs = dict(event_boxes=boxes,video_mask=valid,event_scores=confidence,anchor_boxes=anchors)
    head.eval()
    out = head(hs,labels,torch.ones_like(labels),{},labels,video,**kwargs)
    # Caption CE, without auxiliary mask loss, must train the mask AND geometry.
    out.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.proposal_mask.parameters())
    assert boxes.grad is not None and boxes.grad.abs().sum()>0
    assert confidence.grad is not None and confidence.grad.abs().sum()>0
    assert video.grad is None
    assert torch.isfinite(boxes.grad).all()
    print('PASS: real caption CE gradients reach learned mask, predicted timestamps and confidence; CLIP features remain frozen')

    changed = video.detach().clone();changed[:,8:]=1e5
    with torch.no_grad():
        original=head(hs,labels,torch.ones_like(labels),{},labels,video,**kwargs).logits
        padded=head(hs,labels,torch.ones_like(labels),{},labels,changed,**kwargs).logits
    torch.testing.assert_close(original,padded)
    seen=[]
    def hook(_,args,kw):seen.append(kw['encoder_hidden_states'].detach().clone())
    handle=head.model.register_forward_pre_hook(hook,with_kwargs=True)
    generated=head.generate(hs,None,{},video,**kwargs)
    handle.remove()
    assert generated.sequences.shape[0]==2 and len(seen)>1
    expected,_=head.visual_context(hs,video,**kwargs)
    for item in seen:torch.testing.assert_close(item,expected.repeat_interleave(2,0))
    print('PASS: generation keeps per-event gated CLIP context through beam expansion and cached decoding')

    model_path=root/'ddvc/ddvc_ret_encdec_clip_capQ.py'
    tree=ast.parse(model_path.read_text(encoding='utf8'))
    original_class=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='CM2')
    wanted=['decode_event_boxes','predict_event_num','parallel_prediction_matched','parallel_prediction_full',
            'caption_prediction','caption_prediction_no_grad','caption_prediction_eval']
    methods=[copy.deepcopy(x) for x in original_class.body if isinstance(x,ast.FunctionDef) and x.name in wanted]
    small_class=ast.ClassDef(name='TinyDDVC',bases=[ast.Attribute(value=ast.Name(id='nn',ctx=ast.Load()),attr='Module',ctx=ast.Load())],
                            keywords=[],body=methods,decorator_list=[])
    def inverse_sigmoid(x):
        x=x.clamp(1e-5,1-1e-5);return (x/(1-x)).log()
    namespace=dict(torch=torch,nn=nn,F=F,inverse_sigmoid=inverse_sigmoid,
        CrossEntropyLoss=nn.CrossEntropyLoss,query_event_boxes=local.query_event_boxes,
        proposal_confidence=pm.proposal_confidence,proposal_anchors=pm.proposal_anchors,
        mask_layer_context=pm.mask_layer_context)
    for name in ['event_query_cl','contrastive_loss_cosine_single_direction']:
        function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(model_path),'exec'),namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[small_class],type_ignores=[])),str(model_path),'exec'),namespace)
    model=namespace['TinyDDVC']()
    model.opt=SimpleNamespace(caption_loss_coef=2,caption_decoder_type='gpt2',caption_cost_type='loss',
        caption_cost_chunk_size=1,capQ_addP='after',contrastive_loss_temperature=.1,disable_mid_caption_heads=False)
    model.aux_loss=True
    model.two_layer_mlp=nn.Linear(8,768)
    model.class_head=nn.ModuleList([nn.Linear(768,1),nn.Linear(768,1)])
    model.count_head=nn.ModuleList([nn.Linear(768,20),nn.Linear(768,20)])
    model.bbox_head=nn.ModuleList([nn.Linear(8,2),nn.Linear(8,2)])
    model.caption_head=nn.ModuleList([head,head])
    q=3
    dt=dict(cap_tensor=labels,cap_mask=torch.ones_like(labels),gt_boxes_mask=torch.ones(1,2,dtype=torch.bool),
            video_tensor=video.detach(),video_mask=valid,gt_timestamp=[[[0,2],[4,7]]],
            gt_featstamps=[[0,2],[4,7]],cap_raw=[['one','two']],video_target=[{'boxes':boxes.detach()}])
    namespace['clip_gt']={'one':torch.randn(768),'two':torch.randn(768)}
    others=dict(proposals_mask=torch.ones(1,q,dtype=torch.bool))
    h=torch.randn(2,1,q,8,requires_grad=True)
    hc=torch.randn(2,1,q,8,requires_grad=True)
    init=torch.tensor([[[.2],[.5],[.8]]])
    inter=torch.tensor([[[[.2,.3],[.5,.3],[.8,.3]]]])
    pairs=[(torch.tensor([2,0]),torch.tensor([0,1]))]
    class Criterion:
        def __call__(self,out,targets):
            assert out['caption_losses'].shape==(3,2) if 'caption_losses' in out else True
            return {'loss_giou':out['pred_boxes'].sum()*0},[pairs],[[pairs]]
    model.train()
    output,losses=model.parallel_prediction_matched(dt,Criterion(),None,h,hc,None,init,inter,others,False)
    assert all(torch.isfinite(v).all() for v in losses.values())
    context=pm.mask_layer_context(others,output['pred_logits'],inter[0],output['pred_boxes'])
    losses.update(head.proposal_mask.objective(output['pred_boxes'][0],valid,pm.proposal_confidence(context),pm.proposal_anchors(context)))
    sum(v for k,v in losses.items() if k.startswith('loss_')).backward()
    assert hc.grad is not None and h.grad is not None
    assert head.proposal_mask.mask_model[-1].weight.grad is not None
    # Actual all-pair matcher code: confidence and reference must follow pair IDs.
    pair_ids=torch.tensor([2,2,0,0,1,1]);captions=torch.tensor([0,1,0,1,0,1])
    assert torch.equal(pm.proposal_confidence(context,pair_ids),pm.proposal_confidence(context)[pair_ids])
    with torch.no_grad():
        head.eval();model.train();head.eval()
        model.opt.caption_cost_chunk_size=1
        c1=model.caption_prediction_no_grad(head,dt,hc[-1],None,context,'gpt2',[(pair_ids,captions)],event_boxes=output['pred_boxes'])[0]
        model.opt.caption_cost_chunk_size=4
        c2=model.caption_prediction_no_grad(head,dt,hc[-1],None,context,'gpt2',[(pair_ids,captions)],event_boxes=output['pred_boxes'])[0]
        torch.testing.assert_close(c1,c2)
    print('PASS: actual DDVC two-layer matched training/backward, final/aux caption paths, mask objective, nontrivial query pairing and chunk equivalence')
    model.eval()
    with torch.no_grad():
        predicted,_=model.parallel_prediction_full(dt,Criterion(),None,h,hc,None,init,inter,others,False)
    assert predicted['seq'].shape[:2]==(1,q)
    assert predicted['pred_boxes'].shape==(1,q,2)
    print('PASS: actual DDVC full prediction method generates all query captions with aligned layer contexts')
    saved=copy.deepcopy(head.state_dict())
    head.load_state_dict(saved)
    print('PASS: mask parameters are included in the caption-head checkpoint state')

    # Upgraded experiment: real matched DDVC path, query-conditioned masks and GT-only objective.
    head.proposal_mask = pm.LearnedGatedProposalMask(16,32,query_dim=8,
        gate_min=.1,gate_max=.85,context_ratio=.1,min_frames=4)
    head.opt.mask_prune_threshold=.05
    model.opt.learned_proposal_mask=True
    model.opt.mask_target='matched_gt'
    model.opt.mask_balanced_bce=True
    model.opt.corrected_query_contrastive=True
    probe_a=torch.randn(2,1,4,8,requires_grad=True);probe_b=torch.randn_like(probe_a)
    cold=namespace['event_query_cl'](probe_a,probe_b,.1,corrected=True)
    warm=namespace['event_query_cl'](probe_a,probe_b,1.,corrected=True)
    assert any(not torch.allclose(x,y) for x,y in zip(cold,warm))
    assert all(torch.isfinite(x).all() for x in cold)
    single=namespace['event_query_cl'](probe_a[:,:,:1],probe_b[:,:,:1],.1,corrected=True)
    assert all(torch.allclose(x,torch.zeros_like(x)) for x in single), 'Self-negative was not removed'
    model.train()
    model.zero_grad(set_to_none=True)
    h2=h.detach().clone().requires_grad_();hc2=hc.detach().clone().requires_grad_()
    calls=[]
    objective=head.proposal_mask.objective
    def checked_objective(*args,**kwargs):
        calls.append((args[0].detach().clone(),kwargs['target_boxes'].detach().clone(),kwargs['query'].shape))
        return objective(*args,**kwargs)
    head.proposal_mask.objective=checked_objective
    upgraded,losses=model.parallel_prediction_matched(dt,Criterion(),None,h2,hc2,None,init,inter,others,False)
    assert len(calls)==1 and calls[0][0].shape==(2,2)
    torch.testing.assert_close(calls[0][0],upgraded['pred_boxes'][0,pairs[0][0]])
    torch.testing.assert_close(calls[0][1],dt['video_target'][0]['boxes'][pairs[0][1]])
    assert calls[0][2]==(2,8) and torch.isfinite(losses['loss_proposal_mask'])
    sum(v for k,v in losses.items() if k.startswith('loss_')).backward()
    assert head.proposal_mask.query_proj.weight.grad.abs().sum()>0
    assert hc2.grad.abs().sum()>0
    del head.proposal_mask.objective
    # Force very small soft weights: all outside-context positions must be pruned.
    with torch.no_grad():
        head.proposal_mask.mask_model[-1].weight.zero_()
        head.proposal_mask.mask_model[-1].bias.fill_(-12.)
    head.eval()
    prune_boxes=torch.tensor([[[.25,.05]]]).expand(1,q,2)
    context=pm.mask_layer_context(others,upgraded['pred_logits'],inter[0],prune_boxes)
    qstates=hc2[-1,0].detach()[:,None]
    feature,keep=head.visual_context(qstates,video.detach(),prune_boxes[0],valid,
        pm.proposal_confidence(context).detach(),pm.proposal_anchors(context))
    detail=head.proposal_mask(prune_boxes[0],valid,
        pm.proposal_confidence(context).detach(),pm.proposal_anchors(context),query=qstates[:,0])
    assert torch.equal(keep,detail['hard'].bool()) and (keep.sum(-1)>=4).all()
    assert not keep[:,8:].any() and detail['gate'].min()>=.1 and detail['gate'].max()<=.85
    # Actual GPT-2 output must ignore changes to frames pruned for every caption query.
    changed=video.detach().clone();outside=~keep.any(0);assert outside[:8].any();changed[:,outside]=500.
    with torch.no_grad():
        a=head.generate(qstates,None,{},video.detach(),prune_boxes[0],valid,
            pm.proposal_confidence(context).detach(),pm.proposal_anchors(context))
        b=head.generate(qstates,None,{},changed,prune_boxes[0],valid,
            pm.proposal_confidence(context).detach(),pm.proposal_anchors(context))
    assert torch.equal(a.sequences,b.sequences)
    for x,y in zip(a.scores,b.scores):torch.testing.assert_close(x,y)
    print('PASS: upgraded matched-GT query alignment, query-projection backward, bounded gate, context/pruning, real beam generation invariance')


if __name__=='__main__':main()
