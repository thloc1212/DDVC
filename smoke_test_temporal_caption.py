"""Actual scratch GPT-2 CE/backward, beam/cache and DDVC query pairing on CPU."""
import copy
from types import SimpleNamespace
import torch
from smoke_test_proposal_mask import main as legacy_fixture


def main():
    fixture=legacy_fixture()
    head=fixture['head']
    Temporal=fixture['read']('temporal_caption').TemporalCaptionConditioning
    head.opt.learned_proposal_mask=False
    head.opt.local_visual_attention=False
    head.proposal_mask=None
    hs=fixture['hs'].detach().clone().requires_grad_()
    video=fixture['video'].detach().clone().requires_grad_()
    labels=fixture['labels'];valid=fixture['valid']
    boxes=fixture['boxes'].detach().clone().requires_grad_()
    kwargs=dict(event_boxes=boxes,video_mask=valid)
    head.eval()
    base=head(hs,labels,torch.ones_like(labels),{},labels,video,**kwargs).logits.detach()
    head.temporal_conditioner=Temporal()
    head.eval()
    conditioned=head(hs,labels,torch.ones_like(labels),{},labels,video,**kwargs)
    torch.testing.assert_close(conditioned.logits,base,atol=0,rtol=0)
    cross,mask=head.visual_context(hs,video,**kwargs)
    torch.testing.assert_close(cross,video.detach().repeat(2,1,1),atol=0,rtol=0)
    assert torch.equal(mask,valid.expand(2,-1))
    print('PASS zero-initialized residuals exactly preserve global caption logits and all valid CLIP frames')

    head.zero_grad(set_to_none=True)
    conditioned.loss.backward()
    tc=head.temporal_conditioner
    assert not torch.allclose(tc.encode(torch.tensor([[0.,.2]])),tc.encode(torch.tensor([[1.,.2]])))
    assert tc.prefix_mlp[-1].weight.grad.abs().sum()>0
    assert tc.frame_projection.weight.grad.abs().sum()>0
    assert boxes.grad is None and video.grad is None
    assert hs.grad is not None and torch.isfinite(hs.grad).all()
    torch.optim.SGD(tc.parameters(),lr=.2).step()
    assert not torch.allclose(tc.prefix_delta(boxes),tc.prefix_delta(boxes.flip(0)))
    # An optional differentiable-box variant must be deliberate, not accidental.
    detached=tc.detach_boxes;tc.detach_boxes=False
    assert torch.autograd.grad(tc.prefix_delta(boxes).square().sum(),boxes)[0].abs().sum()>0
    tc.detach_boxes=detached
    print('PASS real CE trains temporal projections; predicted boxes detached by default; interval sensitivity after update')

    with torch.no_grad():
        ordinary=head(hs,labels,torch.ones_like(labels),{},labels,video,**kwargs).logits
        extra=torch.cat((video.detach(),torch.full((1,3,768),1e5)),dim=1)
        extra_valid=torch.cat((valid,torch.zeros(1,3,dtype=torch.bool)),dim=1)
        padded=head(hs,labels,torch.ones_like(labels),{},labels,extra,event_boxes=boxes,video_mask=extra_valid).logits
        torch.testing.assert_close(ordinary,padded,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(tc.frame_delta(valid)[:,:8],tc.frame_delta(extra_valid)[:,:8])
    print('PASS padding cannot change relative timestamps, valid-frame embeddings or caption logits')

    expected_prefix=head.caption_prefix(hs,boxes).detach()
    expected_context,_=head.visual_context(hs,video,**kwargs)
    captured=[]
    def hook(_,args,kw):
        captured.append((kw.get('inputs_embeds'),kw['encoder_hidden_states'].detach().clone()))
    handle=head.model.register_forward_pre_hook(hook,with_kwargs=True)
    generated=head.generate(hs,None,{},video,**kwargs)
    handle.remove()
    assert generated.sequences.shape[0]==2 and len(captured)>1
    first=next(item for item,_ in captured if item is not None)
    torch.testing.assert_close(first[:,:1],expected_prefix.repeat_interleave(2,0))
    for _,context in captured:
        torch.testing.assert_close(context,expected_context.repeat_interleave(2,0))
    print('PASS actual beam generation and cached decoding retain conditioned prefix and frame positions')

    model=fixture['model'];model.train();head.eval()
    dt=fixture['dt'];pair_ids=fixture['pair_ids'];captions=fixture['captions']
    hs_cap=fixture['hc'];output=fixture['output'];context=fixture['context']
    observed=[]
    def pairing(_,args,kw):
        observed.append(kw['event_boxes'].detach().clone())
    handle=head.register_forward_pre_hook(pairing,with_kwargs=True)
    with torch.no_grad():
        model.opt.caption_cost_chunk_size=1
        one=model.caption_prediction_no_grad(head,dt,hs_cap[-1],None,context,'gpt2',[(pair_ids,captions)],event_boxes=output['pred_boxes'])[0]
        aligned=torch.cat(observed);observed.clear()
        torch.testing.assert_close(aligned,output['pred_boxes'][0,pair_ids])
        model.opt.caption_cost_chunk_size=4
        many=model.caption_prediction_no_grad(head,dt,hs_cap[-1],None,context,'gpt2',[(pair_ids,captions)],event_boxes=output['pred_boxes'])[0]
        torch.testing.assert_close(one,many)
    handle.remove()
    model.zero_grad(set_to_none=True)
    out,loss=model.parallel_prediction_matched(dt,fixture['Criterion'](),None,fixture['h'],hs_cap,None,
                                               fixture['init'],fixture['inter'],fixture['others'],False)
    sum(v for k,v in loss.items() if k.startswith('loss_')).backward()
    assert tc.prefix_mlp[-1].weight.grad.abs().sum()>0 and tc.frame_projection.weight.grad.abs().sum()>0
    model.eval()
    with torch.no_grad():
        predicted,_=model.parallel_prediction_full(dt,fixture['Criterion'](),None,fixture['h'],hs_cap,None,
                                                  fixture['init'],fixture['inter'],fixture['others'],False)
    assert predicted['seq'].shape[:2]==(1,3)
    saved=copy.deepcopy(head.state_dict());head.load_state_dict(saved,strict=True)
    print('PASS actual DDVC all-pair nontrivial query alignment/chunk equivalence, matched backward and full generation')

    # Exercise the actual constructor without downloading GPT-2/tokenizer weights.
    globals_=head.__class__.__init__.__globals__
    original_tokenizer,original_model=globals_['GPT2Tokenizer'],globals_['GPT2LMHeadModel']
    Config=globals_['GPT2Config']
    class Tokenizer:
        eos_token='<eos>';pad_token_id=0
        def __call__(self,*a,**kw):return {'input_ids':torch.zeros(1,1,dtype=torch.long)}
        @classmethod
        def from_pretrained(cls,*a,**kw):return cls()
    globals_['GPT2Tokenizer']=Tokenizer
    globals_['GPT2LMHeadModel']=lambda _:original_model(Config(vocab_size=32,n_embd=768,n_layer=1,n_head=12,
        n_inner=32,n_positions=32,add_cross_attention=True,bos_token_id=0,eos_token_id=1,pad_token_id=0))
    try:
        states=[];rng=[]
        for time,frame in [(False,False),(True,False),(False,True),(True,True)]:
            torch.manual_seed(777)
            constructed=head.__class__(SimpleNamespace(hidden_dim=8,add_cross_attention=True,gpt_layer=1,
                learned_proposal_mask=False,local_visual_attention=False,device='cpu',
                caption_time_conditioning=time,caption_frame_time_embedding=frame))
            states.append({k:v for k,v in constructed.state_dict().items() if not k.startswith('temporal_conditioner.')})
            rng.append(torch.get_rng_state())
        for other,state in zip(rng[1:],states[1:]):
            assert torch.equal(other,rng[0])
            for key in states[0]:torch.testing.assert_close(state[key],states[0][key],atol=0,rtol=0)
    finally:
        globals_['GPT2Tokenizer'],globals_['GPT2LMHeadModel']=original_tokenizer,original_model
    from misc.training_state import validate_resume
    try:validate_resume({'caption_time_conditioning':True},{'caption_time_conditioning':False})
    except ValueError:pass
    else:raise AssertionError('Resume enabled a different temporal architecture')
    print('PASS actual constructors preserve shared backbone weights/RNG in every ablation; resume rejects mode change')


if __name__=='__main__':main()
