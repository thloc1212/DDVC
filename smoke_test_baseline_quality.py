"""Offline regression: global caption gradients, baseline recipe and frozen ranking."""
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
os.environ['DDVC_RUN_VARIANT']='baseline_quality'
import torch
from misc.baseline_recipe import validate_method
from misc.c1_recipe import validate_training_method
import c1_pipeline


def check_recipe():
    for ds in ('yc2','anet'):
        cfg=c1_pipeline.make_config({},ROOT.parent,ds)
        validate_method(cfg);validate_training_method(cfg)
        assert cfg['learned_proposal_mask'] is False and cfg['proposal_mask_loss_coef']==0
        assert cfg['epoch']==(15 if ds=='yc2' else 6)
        assert cfg['num_queries']==(50 if ds=='yc2' else 10)
        assert not cfg.get('online_quality',False)
        assert cfg['id']==f'{ds}_baseline_e{cfg["epoch"]}_b1'
        assert cfg['lr']==5e-5 and cfg['gradient_accumulation_steps']==1
        for key,value in [('learned_proposal_mask',True),('online_quality',True),('event_concepts',True)]:
            try:validate_method(dict(cfg,**{key:value}))
            except ValueError:pass
            else:raise AssertionError('Invalid baseline accepted: '+key)
    # Reuse full-split byte-staging/drift fixtures with the baseline adapter active.
    from smoke_test_c1 import check_preparation
    check_preparation()
    import c1_rclone
    from c1_rerank import RERANK_NAMESPACE
    assert c1_rclone.RUN_VARIANT=='baseline_quality'
    assert c1_rclone.run_namespace()=='ddvc_baseline'
    assert RERANK_NAMESPACE=='ddvc_baseline_rerank'
    assert c1_rclone.run_name('anet')=='anet_baseline_e6_b1'
    print('PASS baseline recipes, full-split staging, no retired modules, independent Drive namespaces')


def check_global_caption():
    from ddvc.CaptioningHead import gpt2
    torch.set_num_threads(2)
    original_config,original_tokenizer=gpt2.GPT2Config,gpt2.GPT2Tokenizer
    class Tokenizer:
        eos_token='<eos>';pad_token_id=63
        @classmethod
        def from_pretrained(cls,*a,**kw):return cls()
        def __call__(self,*a,**kw):return {'input_ids':torch.tensor([[63]])}
        def encode(self,*a,**kw):return [13]
    def config(**kw):
        return original_config(vocab_size=64,n_embd=768,n_head=12,n_layer=1,n_positions=32,
            bos_token_id=63,eos_token_id=63,pad_token_id=63,add_cross_attention=True,
            resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.,output_attentions=True)
    gpt2.GPT2Config,gpt2.GPT2Tokenizer=config,Tokenizer
    try:
        opt=SimpleNamespace(hidden_dim=512,device='cpu',gpt_layer=1,add_cross_attention=True,
            learned_proposal_mask=False,proposal_mask_bins=64,proposal_mask_encoding_dim=128,
            max_caption_len=6,gpt_beam_size=2)
        torch.manual_seed(777);head=gpt2.GPT2CaptionModel(opt);head.eval()
        assert head.proposal_mask is None
        assert not any('proposal_mask' in k for k in head.state_dict())
        baseline_parameters={k:v.detach().clone() for k,v in head.state_dict().items()}
        # Removing the isolated-RNG mask must preserve every shared initialized weight.
        torch.manual_seed(777);masked=gpt2.GPT2CaptionModel(SimpleNamespace(**dict(vars(opt),learned_proposal_mask=True)))
        for k,v in baseline_parameters.items():assert torch.equal(v,masked.state_dict()[k]),k
        del masked
        h=torch.randn(2,1,512,requires_grad=True)
        f=torch.randn(1,7,768,requires_grad=True);valid=torch.tensor([[True]*6+[False]])
        labels=torch.tensor([[63,2,3,13],[63,5,6,13]])
        def forward(feature):return head(h,labels,torch.ones_like(labels),{},labels,feature,video_mask=valid)
        visual,padding=head.visual_context(h,f,video_mask=valid)
        assert visual.shape==(2,7,768) and padding.shape==(2,7)
        assert torch.equal(visual[0],f[0]) and torch.equal(visual[1],f[0])
        out=forward(f);assert torch.isfinite(out.loss)
        out.loss.backward();assert h.grad is not None and h.grad.abs().sum()>0
        assert head.fc1.weight.grad.abs().sum()>0 and f.grad is None
        assert head.model.lm_head.weight.grad.abs().sum()>0
        assert out.cross_attentions, 'Missing cross-attention evidence'
        for attention in out.cross_attentions:
            assert torch.isfinite(attention).all()
            assert torch.count_nonzero(attention[...,~valid[0]])==0, 'Padding received cross-attention'
        mutated=f.detach().clone();mutated[:,-1]=1000
        # Report native FP32 behaviour; large padding sentinels can change CPU GEMM
        # rounding/packing. Test semantic invariance in FP64, without changing training.
        with torch.no_grad():
            changed=forward(mutated)
            for attention in changed.cross_attentions:
                assert torch.count_nonzero(attention[...,~valid[0]])==0, 'Mutated padding received cross-attention'
            delta=float((out.logits.detach()-changed.logits).abs().max())
            print('PADDING_FP32_DIAGNOSTIC', 'torch',torch.__version__,'max_logit_delta',delta,
                  'masked_attention_mass',0.,flush=True)
            head.double();h=h.detach().double();reference=f.detach().double()
            original=forward(reference).logits
            changed=reference.clone();changed[:,-1]=1000
            padded=forward(changed).logits
            delta64=float((original-padded).abs().max())
            assert torch.allclose(original,padded,rtol=1e-7,atol=1e-9), f'FP64 padding invariance failed: {delta64}'
            changed[:,0:6]=torch.randn_like(changed[:,0:6])*3
            assert not torch.allclose(original,forward(changed).logits,rtol=1e-7,atol=1e-9), 'Valid visual frames had no effect'
            print('PADDING_FP64_INVARIANCE_PASS','max_logit_delta',delta64,flush=True)
        print('PASS actual GPT2 global forward/backward; original CLIP values, padding only, no mask parameters, shared initialization parity')
    finally:gpt2.GPT2Config,gpt2.GPT2Tokenizer=original_config,original_tokenizer


def check_ranking():
    from reranking.core import select,joint_score
    from misc.proposal_selection import select_proposals
    rows=[dict(query_id=i,timestamp=[float(i),float(i+3)],sentence='wash the pan',
               proposal_score=.7-i*.1,sentence_score=-2.,pred_event_count=2) for i in range(4)]
    before=copy.deepcopy(rows)
    for method in ('topk','softnms'):
        a=select(rows,2,[0.]*4,0.,method=method)
        b=select_proposals(rows,2,method=method,sigma=.5,alpha=1.,temperature=2.)
        assert [r['query_id'] for r in a]==[r['query_id'] for r in b]
    q=[0.,0.,0.,1.]
    chosen=select(rows,2,q,2.,method='topk')
    expected=sorted(range(4),key=lambda i:joint_score(rows[i])+2*q[i],reverse=True)[:2]
    assert {r['query_id'] for r in chosen}==set(expected)
    assert rows==before
    print('PASS quality Top-N, beta=0 selector parity, fixed counter/captions/timestamps')


if __name__=='__main__':
    check_recipe();check_global_caption();check_ranking()
