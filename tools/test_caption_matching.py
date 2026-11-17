"""Cost assignment, warm-up, RNG isolation and actual offline GPT-2 checks."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from misc.caption_matching import matching_weight, joint_reward, free_running_captions, add_joint_cost
from reranking.core import fit_idf


class MatchingTests(unittest.TestCase):
    def test_control_config_and_invalid_coefficient(self):
        sys.path.insert(0,str(ROOT/'tools'))
        import c1_pipeline
        with patch.dict('os.environ',{'DDVC_MATCHING_WEIGHT':'0'}):
            cfg=c1_pipeline.make_config({},Path('/unused/control'),'yc2')
            self.assertEqual(cfg['caption_match_weight'],0)
        for invalid in ('nan','-1','3'):
            with patch.dict('os.environ',{'DDVC_MATCHING_WEIGHT':invalid}),self.assertRaises(ValueError):
                c1_pipeline.make_config({},Path('/unused/control'),'yc2')

    def test_warmup_disabled_and_resume_epoch(self):
        opt=SimpleNamespace(caption_match_enabled=True,caption_match_weight=.5,
                            caption_match_warmup_epochs=2,caption_match_ramp_epochs=2)
        self.assertEqual([matching_weight(opt,e) for e in range(5)],[0,0,.25,.5,.5])
        self.assertEqual(matching_weight(opt,3),.5)
        opt.caption_match_weight=0
        self.assertEqual(matching_weight(opt,20),0)
        opt.caption_match_enabled=False
        self.assertEqual(matching_weight(opt,20),0)

    def test_same_gt_joint_reward_not_two_independent_maxima(self):
        idf=fit_idf({'t':{'sentences':['wash bowl','cut carrot']}},['t'])
        boxes=torch.tensor([[[.25,.4],[.75,.4]]],requires_grad=True)
        gt=boxes.detach()[0].clone()
        reward=joint_reward(boxes,gt,['cut carrot','wash bowl'],['wash bowl','cut carrot'],idf)
        self.assertFalse(reward.requires_grad)
        self.assertEqual(float(reward.max()),0.)
        aligned=joint_reward(boxes,gt,['wash bowl','cut carrot'],['wash bowl','cut carrot'],idf)
        torch.testing.assert_close(aligned[0],torch.eye(2))
        with self.assertRaises(ValueError):
            add_joint_cost(torch.zeros(2,2),torch.ones(2,2,requires_grad=True),.5)

    def test_actual_hungarian_cost_can_change_pairs_and_zero_is_exact(self):
        from ddvc.matcher import HungarianMatcher_cl
        matcher=HungarianMatcher_cl(cost_class=0,cost_bbox=1,cost_giou=0,cost_caption=0)
        boxes=torch.tensor([[[.2,.2],[.8,.2]]])
        targets=[dict(labels=torch.zeros(2,dtype=torch.long),boxes=boxes[0].clone())]
        output=dict(pred_logits=torch.zeros(1,2,1),pred_boxes=boxes,cl_match_mats=0)
        original=matcher(output,targets)[0][0]
        self.assertEqual(tuple(original[1].tolist()),(0,1))
        output.update(caption_match_reward=torch.tensor([[[0.,1.],[1.,0.]]]),
                      caption_match_weight=0.,caption_match_decode_seconds=0.)
        zero=matcher(output,targets)[0][0]
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(original,zero)))
        output['caption_match_weight']=1.
        changed=matcher(output,targets)[0][0]
        self.assertEqual(tuple(changed[1].tolist()),(1,0))
        self.assertEqual(matcher.caption_match_stats['changed_pairs'],2)
        self.assertEqual(len(set(changed[1].tolist())),2)

    def test_real_tiny_gpt2_generation_rng_modes_and_backward(self):
        from ddvc.CaptioningHead import gpt2
        original_config=gpt2.GPT2Config
        class Tokenizer:
            pad_token_id=63; eos_token='<eos>'
            @classmethod
            def from_pretrained(cls,*a,**kw):return cls()
            def __call__(self,*a,**kw):return {'input_ids':torch.tensor([[63]])}
            def encode(self,text):return [13]
            def batch_decode(self,rows,**kw):return [' '.join(map(str,r.tolist())) for r in rows]
        def tiny(**kw):
            return original_config(vocab_size=64,n_embd=768,n_head=12,n_layer=1,n_positions=16,
                bos_token_id=63,eos_token_id=63,pad_token_id=63,add_cross_attention=True,
                resid_pdrop=.1,embd_pdrop=.1,attn_pdrop=.1)
        opt=SimpleNamespace(hidden_dim=512,device='cpu',gpt_layer=1,add_cross_attention=True,
            learned_proposal_mask=True,proposal_mask_bins=64,proposal_mask_encoding_dim=128,
            proposal_mask_content_conditioned=True,proposal_mask_content_dim=128,max_caption_len=5,gpt_beam_size=2)
        torch.set_num_threads(2)
        with patch.object(gpt2,'GPT2Config',side_effect=tiny),patch.object(gpt2,'GPT2Tokenizer',Tokenizer):
            head=gpt2.GPT2CaptionModel(opt).train()
        head.proposal_mask.eval()
        q=torch.randn(1,2,512,requires_grad=True)
        boxes=torch.tensor([[[.3,.3],[.7,.3]]],requires_grad=True)
        video=torch.randn(1,5,768);valid=torch.ones(1,5,dtype=torch.bool)
        p=torch.tensor([.3,.6],requires_grad=True);anchors=boxes.detach()[0]
        modes=[m.training for m in head.modules()];rng=torch.get_rng_state().clone()
        parameters=[x.detach().clone() for x in head.parameters()]
        captions=free_running_captions(head,q,boxes,video,valid,p,anchors,chunk_size=1)
        self.assertEqual(len(captions),2)
        self.assertEqual(modes,[m.training for m in head.modules()])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(parameters,head.parameters())))
        self.assertTrue(all(x.grad is None for x in head.parameters()))
        with patch.object(head,'generate',side_effect=RuntimeError('injected failure')):
            with self.assertRaises(RuntimeError):
                free_running_captions(head,q,boxes,video,valid,p,anchors)
        self.assertEqual(modes,[m.training for m in head.modules()])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        labels=torch.tensor([[63,2,3,13],[63,5,6,13]])
        loss=head(q.permute(1,0,2),labels,torch.ones_like(labels),{},labels,video,
                  event_boxes=boxes[0],video_mask=valid,event_scores=p,anchor_boxes=anchors).loss
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(boxes.grad)
        self.assertTrue(all(bool(torch.isfinite(x.grad).all()) for x in head.parameters() if x.grad is not None))


if __name__=='__main__':unittest.main()
