"""CPU controls for online feature parity, gradient routing, label isolation and data recovery."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from torch import nn
import yaml
from reranking.core import feature_pool,targets,fit_idf
from reranking.head import JointQualityHead,objective
from reranking.online import differentiable_features,partition,initialize,augment,generation_mode,optimizer_groups
from online_quality_evaluate import select_modes,choose_beta


class Tokenizer:
    eos_token_id=0
    def encode(self,text,add_special_tokens=False):return [1+len(w)%11 for w in text.split()]
    def decode(self,tokens,skip_special_tokens=True):return 'cut the onion.'


class Caption(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Module();self.model.transformer=nn.Module()
        self.model.transformer.wte=nn.Embedding(20,768);self.tokenizer=Tokenizer();self.drop=nn.Dropout(.1)
    def generate(self,hs,*args,**kwargs):
        assert not self.training and not self.drop.training
        return dict(sequences=torch.ones(len(hs),4,dtype=torch.long),sequences_scores=torch.full((len(hs),),-.3))


def fake_model(folder,mode='joint'):
    annotation={f'v{i}':dict(timestamps=[[1.,4.]],sentences=['cut the onion.']) for i in range(10)}
    path=folder/'train.json';path.write_text(json.dumps(annotation))
    model=nn.Module();model.caption_head=nn.ModuleList([Caption()])
    model.opt=SimpleNamespace(online_quality=True,batch_size=1,quality_gradient_mode=mode,
        quality_loss_coef=.1,quality_generation_chunk=2,quality_annotation_file=str(path),lr=5e-5,quality_head_lr=.001)
    initialize(model,model.opt);return model


def actual_mask_helpers():
    name='online_mask_helpers_under_test'
    spec=importlib.util.spec_from_file_location(name,ROOT/'ddvc/CaptioningHead/proposal_mask.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class OnlineQualityTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(5);torch.set_num_threads(1)

    def test_feature_parity_including_context_and_quantization_tolerance(self):
        f=torch.randn(13,768);embedding=nn.Embedding(20,768);tokenizer=Tokenizer()
        edges=torch.tensor([[.2,.6],[.31,.32],[0.,1.]],requires_grad=True)
        confidence=torch.tensor([.2,.5,.8],requires_grad=True)
        rows=[dict(timestamp=(e.detach()*100).tolist(),sentence='cut the onion.',proposal_score=float(p.detach()),sentence_score=-.3)
              for e,p in zip(edges,confidence)]
        live=differentiable_features(f,rows,edges,confidence,embedding,tokenizer)
        cached=feature_pool(f,rows,100.,embedding,tokenizer)
        np.testing.assert_allclose(live.detach().numpy().astype(np.float16),cached,atol=1e-3,rtol=1e-3)
        self.assertEqual(live.shape,(3,2312))
        live.sum().backward()
        self.assertGreater(float(confidence.grad.abs().sum()),0)
        self.assertGreater(float(edges.grad.abs().sum()),0)
        self.assertGreater(float(embedding.weight.grad.abs().sum()),0)

    def test_joint_vs_detached_gradient_route_real_head(self):
        masks=actual_mask_helpers()
        for mode in ('joint','detached'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                model=fake_model(Path(temp),mode);vid=next(iter(model._quality_fit))
                out=dict(pred_logits=torch.randn(1,4,1,requires_grad=True),
                    pred_boxes=torch.tensor([[[.25,.2],[.4,.3],[.6,.25],[.8,.2]]],requires_grad=True),pred_count=torch.randn(1,21))
                dt=dict(video_tensor=torch.randn(1,13,768),video_mask=torch.ones(1,13,dtype=torch.bool),
                        video_length=torch.tensor([[13.,10.,13.]]),video_key=[vid])
                hs=torch.randn(1,4,512);reference=torch.rand(1,4,2);rng=torch.get_rng_state().clone()
                with patch.dict(sys.modules,{'ddvc.CaptioningHead.proposal_mask':masks}):
                    losses=augment(model,dt,out,hs,reference,{},False)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                losses['loss_quality'].backward()
                self.assertTrue(model._quality_gradient_verified)
                self.assertEqual(out['pred_logits'].grad is not None,mode=='joint')
                self.assertEqual(model.caption_head[0].model.transformer.wte.weight.grad is not None,mode=='joint')
                self.assertTrue(model.caption_head[0].training)
                self.assertTrue(model.caption_head[0].drop.training)
                groups=optimizer_groups(model,model.opt)
                self.assertEqual(len({id(p) for g in groups for p in g['params']}),len(list(model.parameters())))

    def test_head_dev_has_no_quality_label_loss_and_unknown_video_rejected(self):
        masks=actual_mask_helpers()
        with tempfile.TemporaryDirectory() as temp:
            model=fake_model(Path(temp));vid=next(iter(model._quality_dev))
            out=dict(pred_logits=torch.randn(1,2,1,requires_grad=True),pred_boxes=torch.tensor([[[.3,.2],[.6,.2]]],requires_grad=True),pred_count=torch.randn(1,21))
            dt=dict(video_tensor=torch.randn(1,10,768),video_mask=torch.ones(1,10,dtype=torch.bool),video_length=torch.tensor([[10.,10.,10.]]),video_key=[vid])
            with patch.dict(sys.modules,{'ddvc.CaptioningHead.proposal_mask':masks}):
                loss=augment(model,dt,out,torch.randn(1,2,512),torch.rand(1,2,2),{},False)
                self.assertEqual(float(loss['loss_quality'].detach()),0.)
                loss['loss_quality'].backward()
                self.assertTrue(all(p.grad is None for p in model.quality_head.parameters()))
                dt['video_key']=['VALIDATION_VIDEO']
                with self.assertRaisesRegex(ValueError,'outside TRAIN'):augment(model,dt,out,torch.randn(1,2,512),torch.rand(1,2,2),{},False)

    def test_generation_preserves_rng_and_module_modes(self):
        head=Caption();head.train();head.drop.eval();before=torch.get_rng_state().clone()
        with generation_mode(head):
            torch.rand(12);self.assertFalse(head.training)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(head.training);self.assertFalse(head.drop.training)

    def test_initialization_preserves_baseline_rng_and_parameter_count(self):
        with tempfile.TemporaryDirectory() as temp:
            model=fake_model(Path(temp));before=torch.get_rng_state().clone();initialize(model,model.opt)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertEqual(sum(p.numel() for p in model.quality_head.parameters()),280577)

    def test_quality_norm_does_not_rescale_detached_base_gradients(self):
        from misc.gradient_guard import clip_gradients_or_fail
        model=nn.Module();model.base=nn.Linear(1,1,bias=False);model.quality_head=nn.Linear(1,1,bias=False)
        model.opt=SimpleNamespace(online_quality=True)
        model.base.weight.grad=torch.tensor([[3.]])
        model.quality_head.weight.grad=torch.tensor([[10000.]])
        with tempfile.TemporaryDirectory() as temp:
            clip_gradients_or_fail(model,2.,temp,epoch=0,batch_index=0,iteration=1)
        self.assertAlmostEqual(float(model.base.weight.grad),2.,places=5)
        self.assertAlmostEqual(float(model.quality_head.weight.grad),2.,places=5)

    def test_same_gt_target_not_product_of_independent_maxima(self):
        annotation={'timestamps':[[0,1],[2,3]],'sentences':['cut onion','boil water']}
        rows=[dict(timestamp=[0,1],sentence='boil water')]
        idf=fit_idf({'v':annotation},['v']);y,_=targets(rows,annotation,idf)
        self.assertEqual(float(y[0]),0.)

    def test_selection_preserves_candidates_and_beta_zero_parity(self):
        rows=[dict(query_id=i,timestamp=[float(i),float(i+3)],sentence='cut onion',proposal_score=.7-i*.1,
                   sentence_score=-.3,pred_event_count=2,reranker_quality=.1*i) for i in range(4)]
        modes=select_modes({'v':rows},0.)
        keys=lambda m:[r['query_id'] for r in modes[m]['v']]
        self.assertEqual(keys('online_base_softnms'),keys('online_quality_softnms'))
        self.assertEqual(keys('online_base_topk'),keys('online_quality_topk'))
        self.assertTrue(all(len(m['v'])==2 for m in modes.values()))
        winner,grid=choose_beta([dict(rows=rows,N=2,quality=[0.]*4,reward=[[.3],[.2],[.1],[0.]])])
        self.assertEqual(winner,0.)

    def test_cli_imports_safe_path_and_source_syntax(self):
        env=dict(os.environ,PYTHONSAFEPATH='1',PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory() as temp:
            result=subprocess.run([sys.executable,'-P',str(ROOT/'tools/online_quality_pipeline.py'),'--help'],
                                  cwd=temp,env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        for name in ('reranking/online.py','tools/online_quality_pipeline.py','tools/online_quality_evaluate.py','train.py','opts.py','eval_utils_clip.py'):
            ast.parse((ROOT/name).read_text(encoding='utf8'))

    def test_config_preserves_base_recipe_and_separates_online_run(self):
        import online_quality_pipeline as pipeline
        for ds,nq,epochs in [('yc2',50,15),('anet',10,6)]:
            cfg=pipeline.make_config(dict(dataset=ds,bank_type=[ds],train_caption_file='/data/train.json'),Path('/work'),ds)
            self.assertEqual(cfg['num_queries'],nq);self.assertEqual(cfg['epoch'],epochs)
            self.assertEqual(cfg['proposal_mask_loss_coef'],0.);self.assertEqual(cfg['gradient_accumulation_steps'],1)
            self.assertEqual(cfg['precision'],'fp32');self.assertFalse(cfg['start_from']);self.assertIsNone(cfg['pretrain'])
            self.assertTrue(cfg['online_quality']);self.assertIn('online_quality_',cfg['id'])

    def test_actual_tiny_gpt2_generation_with_online_quality(self):
        from ddvc.CaptioningHead import gpt2
        original_config,original_tokenizer=gpt2.GPT2Config,gpt2.GPT2Tokenizer
        class OfflineTokenizer(Tokenizer):
            pad_token_id=63;eos_token='<eos>'
            @classmethod
            def from_pretrained(cls,*a,**kw):return cls()
            def __call__(self,*a,**kw):return {'input_ids':torch.tensor([[63]])}
        def tiny_config(**kw):
            return original_config(vocab_size=64,n_embd=768,n_head=12,n_layer=1,n_positions=32,
                bos_token_id=63,eos_token_id=63,pad_token_id=63,add_cross_attention=True,
                resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.)
        with tempfile.TemporaryDirectory() as temp,patch.object(gpt2,'GPT2Config',tiny_config),patch.object(gpt2,'GPT2Tokenizer',OfflineTokenizer):
            model=fake_model(Path(temp));opt=model.opt
            opt.hidden_dim=512;opt.device='cpu';opt.gpt_layer=1;opt.add_cross_attention=True
            opt.learned_proposal_mask=True;opt.proposal_mask_bins=64;opt.proposal_mask_encoding_dim=128
            opt.max_caption_len=6;opt.gpt_beam_size=2
            model.caption_head=nn.ModuleList([gpt2.GPT2CaptionModel(opt)])
            out=dict(pred_logits=torch.randn(1,2,1,requires_grad=True),
                     pred_boxes=torch.tensor([[[.3,.3],[.7,.2]]],requires_grad=True),pred_count=torch.randn(1,21))
            dt=dict(video_tensor=torch.randn(1,7,768),video_mask=torch.ones(1,7,dtype=torch.bool),
                    video_length=torch.tensor([[7.,10.,1.]]),video_key=[next(iter(model._quality_fit))])
            losses=augment(model,dt,out,torch.randn(1,2,512),torch.rand(1,2,2),{},False)
            losses['loss_quality'].backward()
            self.assertEqual(out['pred_quality'].shape,(1,2))
            self.assertTrue(model._quality_gradient_verified)
            self.assertTrue(torch.isfinite(model.caption_head[0].model.transformer.wte.weight.grad).all())

    def test_online_backup_namespace_calibration_progress_and_slot_fallback(self):
        spec=importlib.util.spec_from_file_location('online_transport_tests',ROOT/'tools/test_c1_rclone.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(os.environ,DDVC_RUN_VARIANT='online_quality',DDVC_QUALITY_MODE='joint'):
            spec.loader.exec_module(module)
        q=module.q
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);work=root/'work';work.mkdir();cloud=root/'cloud';cloud.mkdir()
            with patch.object(q,'WORK',work),patch.object(q,'revision',lambda:'FIXED_SOURCE'):
                name=q.run_name('anet');run=work/'runs'/name;run.mkdir(parents=True)
                (run/'model-last.pth').write_bytes(b'COMPLETE_MODEL_AND_OPTIMIZER')
                prep=work/'prepared'/name;prep.mkdir(parents=True);(prep/'config.yml').write_text('id: '+name)
                state=dict(dataset='anet',run_id='test_online',source_commit='FIXED_SOURCE',work=str(work),variant=q.RUN_VARIANT)
                backup=q.Recovery(module.FileDrive(cloud),state)
                self.assertIn('ddvc_online_quality/joint/anet',backup.remote.replace('\\','/'))
                self.assertTrue(backup.sync())
                folder=run/'quality_evaluation';folder.mkdir();(folder/'policy.json').write_text('{"beta":1}')
                self.assertTrue(backup.sync())  # checkpoint unchanged; calibration progress must still sync.
                pointer=json.loads((Path(backup.remote)/'recovery/LATEST.json').read_text())
                slot=Path(backup.remote)/'recovery'/('slot-'+str(pointer['slot']))
                (slot/'runs'/name/'model-last.pth').write_bytes(b'CORRUPT')
                for p in work.rglob('*'):
                    if p.is_file():
                        self.assertTrue(p.resolve().is_relative_to(work.resolve()));p.unlink()
                restored=backup.restore()
                self.assertNotEqual(restored['slot'],pointer['slot'])
                self.assertEqual((run/'model-last.pth').read_bytes(),b'COMPLETE_MODEL_AND_OPTIMIZER')

    def test_finalize_freezes_before_validation_and_resume_keeps_beta(self):
        import online_quality_pipeline as pipeline
        import online_quality_evaluate as evaluation
        from reranking.pipeline import METRICS
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);run=work/'runs'/'anet_test';pred=run/'prediction';pred.mkdir(parents=True)
            cfg=dict(id='anet_test',epoch=1,train_caption_file=str(work/'train.json'),
                     gt_file_for_eval=[],gt_file_for_para_eval=[],quality_gradient_mode='joint',start_from='')
            config=work/'config.yml';config.write_text(yaml.safe_dump(cfg));(work/'train.json').write_text(json.dumps({'train':{}}))
            (run/'info.json').write_text(json.dumps(dict(last=dict(epoch=0),best=dict(epoch=0,result_json_path='/old/num1_epoch0.json'))))
            (run/'model-best.pth').write_bytes(b'MODEL_IDENTITY')
            rows=[dict(query_id=i,timestamp=[float(i),float(i+3)],sentence='cut onion',proposal_score=.7-i*.1,
                       sentence_score=-.3,pred_event_count=2,reranker_quality=.1*i) for i in range(4)]
            (pred/'num1_epoch0.raw.json').write_text(json.dumps(dict(results={'VALIDATION':rows})))
            (pred/'num1_epoch0.json').write_text(json.dumps({k:.1 for k in METRICS}))
            dev=[dict(video_id='TRAIN_DEV',rows=rows,N=2,quality=[0.]*4,reward=[[.3],[.2],[.1],[0.]])]
            def fake_score(*a,**kw):
                self.assertTrue((run/'quality_evaluation/policy.json').is_file())
                return {k:10. for k in METRICS}
            with patch.object(pipeline,'check',lambda p:dict(cfg)),patch.object(pipeline,'config_path',lambda *a:config),\
                 patch.object(evaluation,'partition',lambda ann:(['TRAIN_FIT'],['TRAIN_DEV'])),\
                 patch.object(evaluation,'extract_dev',return_value=dev) as extract,\
                 patch.object(evaluation,'official_scores',fake_score):
                args=SimpleNamespace(work=work,dataset='anet');evaluation.finalize(args)
                frozen=(run/'quality_evaluation/policy.json').read_bytes()
                cfg['start_from']='anet_test';config.write_text(yaml.safe_dump(cfg))
                evaluation.finalize(args)
                self.assertEqual(extract.call_count,1)
                self.assertEqual(frozen,(run/'quality_evaluation/policy.json').read_bytes())
                (run/'model-best.pth').write_bytes(b'OTHER_MODEL')
                with self.assertRaisesRegex(ValueError,'identity changed'):evaluation.finalize(args)

    def test_cross_root_data_import_and_corruption_guard(self):
        from online_quality_pipeline import import_c1_data
        import c1_rclone as helpers
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'remote';source.mkdir();work=root/'new';work.mkdir()
            old='/marimo/DDVC_C1/prepared/anet_c1_base_assets'
            for n,payload in [('train.json',{'v1':{}}),('val.json',{'v2':{}}),('vocab.json',{'word_to_ix':{}})]:
                (source/n).write_text(json.dumps(payload))
            (source/'clipvitl14.pth').write_bytes(b'FAKE_FEATURE_BYTES')
            cfg=dict(dataset='anet',bank_type=['anet'],train_caption_file=old+'/train.json',val_caption_file=old+'/val.json',
                     dict_file=old+'/vocab.json',bank_path=old+'/bank',visual_feature_folder=[old+'/clipvitl14.pth'])
            cfg['data_signature']=hashlib.sha256(b''.join((source/n).read_bytes() for n in ('train.json','val.json','vocab.json'))).hexdigest()
            (source/'config.yml').write_text(yaml.safe_dump(cfg))
            meta=dict(protocol='c1-base-assets-v1',dataset='anet',work='/marimo/DDVC_C1',asset_code=helpers.data_identity(),files=helpers.inventory(source))
            class Drive:
                def path(self,n):return 'remote:'+n
                def optional_json(self,n):return {'generation':'a'*32}
                def read_json(self,n):return meta
                def copy(self,remote,target,listing):shutil.copytree(source,target)
            h=SimpleNamespace(WORK=work,base_path=lambda ds:work/'prepared'/(ds+'_c1_base_assets'),
                data_identity=helpers.data_identity,fetch_verified=helpers.fetch_verified,write_json=helpers.write_json)
            self.assertTrue(import_c1_data(Drive(),h,'anet'))
            moved=yaml.safe_load((h.base_path('anet')/'config.yml').read_text())
            self.assertEqual(moved['train_caption_file'],str(work)+'/prepared/anet_c1_base_assets/train.json')
            other=root/'corrupt';other.mkdir();h.WORK=other;h.base_path=lambda ds:other/'prepared'/(ds+'_c1_base_assets')
            (source/'clipvitl14.pth').write_bytes(b'CORRUPT')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):import_c1_data(Drive(),h,'anet')


if __name__=='__main__':unittest.main()
