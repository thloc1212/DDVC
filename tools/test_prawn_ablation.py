"""Ablation isolation, fixed-candidate selection, resume namespaces and scoring cache."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from misc.prawn_ablation import variant,configure,validate,run_name,namespace
import c1_pipeline
from prawn_ablation_control import validate_full,fetch_binary,beta_ablation
import prawn_ablation_eval as ev


class AblationTests(unittest.TestCase):
    def test_profiles_change_only_the_intended_factor(self):
        for ds in ('yc2','anet'):
            source=dict(dataset=ds,bank_type=[ds],train_caption_file='/data/train.json')
            with patch.dict(os.environ,PRAWN_ABLATION_VARIANT='full'):
                full=c1_pipeline.make_config(source,Path('/work'),ds)
            for v in ('no_content','joint'):
                with patch.dict(os.environ,PRAWN_ABLATION_VARIANT=v):
                    cfg=c1_pipeline.make_config(source,Path('/work'),ds)
                    validate(cfg)
                    self.assertIn('loo_'+v,cfg['id'])
                allowed={'id','exp_name','prawn_ablation_variant','online_quality','proposal_mask_content_conditioned',
                         'quality_gradient_mode','quality_loss_coef','quality_head_lr','quality_generation_chunk','quality_annotation_file'}
                self.assertEqual({k:x for k,x in full.items() if k not in allowed},
                                 {k:x for k,x in cfg.items() if k not in allowed})
                self.assertTrue(cfg['caption_match_enabled']);self.assertEqual(cfg['caption_match_weight'],.5)
                self.assertEqual(cfg['proposal_mask_loss_coef'],0.)
                self.assertEqual(cfg['proposal_mask_content_conditioned'],v!='no_content')

    def test_full_control_rejects_old_geometry_or_no_matching_runs(self):
        with patch.dict(os.environ,PRAWN_ABLATION_VARIANT='full'):
            cfg=c1_pipeline.make_config(dict(train_caption_file='/train'),Path('/work'),'anet')
        validate_full(cfg,'anet')
        for key,value in [('caption_match_enabled',False),('proposal_mask_content_conditioned',False),
                          ('proposal_mask_loss_coef',1.),('epoch',8),('online_quality',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):validate_full(dict(cfg,**{key:value}),'anet')

    def test_all_ablations_fail_if_matching_or_auxiliary_loss_changes(self):
        for v in ('no_content','joint'):
            with patch.dict(os.environ,PRAWN_ABLATION_VARIANT=v):
                cfg=c1_pipeline.make_config(dict(train_caption_file='/train'),Path('/work'),'yc2')
            for change in [dict(caption_match_enabled=False),dict(caption_match_weight=0.),dict(proposal_mask_loss_coef=1.)]:
                with self.assertRaises(ValueError):validate(dict(cfg,**change))

    def test_namespaces_and_names_do_not_collide_with_full(self):
        values=[]
        for v in ('full','no_content','joint'):
            with patch.dict(os.environ,PRAWN_ABLATION_VARIANT=v):
                values.append((namespace('generator'),namespace('quality'),run_name('anet')))
        self.assertEqual(len(set(values)),3)
        self.assertEqual(values[0][0],'prawn_caption_match_generator')
        self.assertIn('prawn_loo_no_content',values[1][0])
        self.assertIn('prawn_loo_joint',values[2][0])

    def test_fixed_selection_keeps_query_pool_captions_timestamps_and_counter(self):
        rows=[dict(query_id=i,timestamp=[float(i),float(i+3)],sentence='cut onion',proposal_score=.7-i*.1,
                   sentence_score=-.3,pred_event_count=2) for i in range(4)]
        raw={'v':rows};original=copy.deepcopy(raw)
        from misc.proposal_selection import select_proposals
        quality={'v':{str(i):i*.2 for i in range(4)}}
        for beta in ev.BETAS:
            selected=ev.select_fixed(raw,quality,beta)
            self.assertEqual(len(selected['v']),2)
            for r in selected['v']:
                self.assertEqual(r['timestamp'],rows[r['query_id']]['timestamp'])
                self.assertEqual(r['sentence'],rows[r['query_id']]['sentence'])
        zero=ev.select_fixed(raw,quality,0.)['v']
        expected=select_proposals(rows,2,method='softnms',sigma=.5,alpha=1.,temperature=2.)
        self.assertEqual([r['query_id'] for r in zero],[r['query_id'] for r in expected])
        self.assertEqual(raw,original)

    def test_control_download_cannot_fetch_generator_weights_or_train_cache(self):
        for name in ['runs/anet/model-best.pth','runs/anet/cache/train/shard_00000.npz']:
            with self.assertRaises(ValueError):fetch_binary(None,{},name,Path('/unused'))

    def test_beta_grid_uses_same_frozen_head_and_val_shards_without_training(self):
        import hashlib
        import numpy as np
        import torch
        from reranking.cache import write_shard
        from reranking.head import JointQualityHead
        class BinaryDrive:
            def __init__(self,files):self.files=files;self.downloads=[]
            def path(self,name):return name
            def call(self,command,source,target,**kw):
                self.downloads.append(source)
                assert command=='copyto' and '/cache/train/' not in source and not source.endswith('.pth')
                Path(target).write_bytes(self.files[source])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'cloud/runs/yc2';folder.mkdir(parents=True)
            row=dict(query_id=0,timestamp=[0.,1.],sentence='cut onion',proposal_score=.7,sentence_score=-.3,pred_event_count=1)
            sig='test-head-and-cache-signature'
            torch.save(dict(signature=sig,beta=1.,model=JointQualityHead().state_dict()),folder/'head-best.pt')
            shard=folder/'cache/val/shard_00000.npz'
            write_shard(shard,[dict(video_id='v',rows=[row],N=1)],[np.zeros((1,2312),dtype=np.float16)],sig,'val')
            files={p.relative_to(root/'cloud').as_posix():p.read_bytes() for p in folder.rglob('*') if p.is_file()}
            drive=BinaryDrive({'remote/'+n:b for n,b in files.items()})
            ready={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in files.items()}
            scores={k:10. for k in ev.METRICS}
            control=dict(qs=dict(remote='remote',ready={'files':ready}),qm={'signature':sig},store=SimpleNamespace(drive=drive),
                         g={'raw':{'v':[row]}},identity={'fixed':True},cfg={},
                         policies={m:dict(predictions={'v':[row]},scores=scores) for m in ('baseline_softnms','reranker_softnms')})
            def score(*a,**kw):return dict(method=a[1],beta=a[-1],**scores)
            with patch('prawn_ablation_control.score',side_effect=score) as scoring:
                beta_ablation(control,'yc2',root/'output/beta')
            self.assertEqual(scoring.call_count,8)
            self.assertEqual([c.args[-1] for c in scoring.call_args_list],list(ev.BETAS))
            self.assertEqual(len(drive.downloads),3)
            self.assertTrue((root/'output/beta/summary.tsv').is_file())

    def test_scoring_resume_uses_full_official_metric_signature(self):
        scores={k:10. for k in ev.METRICS};scores['F1']=10.
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);ref=folder/'ref.json';ref.write_text('{}')
            cfg=dict(gt_file_for_eval=[str(ref)],gt_file_for_para_eval=[str(ref)])
            with patch.object(ev,'official_scores',return_value=scores) as official:
                a=ev.score(folder,'beta_1',{'v':[]},cfg,{'same_head':True},1.)
                b=ev.score(folder,'beta_1',{'v':[]},cfg,{'same_head':True},1.)
                self.assertEqual(official.call_count,1);self.assertEqual(a,b)
                ref.write_text('{"changed":true}')
                ev.score(folder,'beta_1',{'v':[]},cfg,{'same_head':True},1.)
                self.assertEqual(official.call_count,2)

    def test_joint_validation_keeps_fixed_beta_and_separate_generator_metric_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'prediction/num1_epoch0.raw.json';path.parent.mkdir()
            rows=[dict(query_id=0,timestamp=[0.,1.],sentence='cut onion',proposal_score=.7,
                       sentence_score=-.3,pred_event_count=1,reranker_quality=.4)]
            path.write_text(json.dumps(dict(results={'v':rows})))
            original=path.read_bytes()
            with patch.object(ev,'score',return_value={'CIDEr':10.}) as scorer:
                ev.score_joint_epoch(path,SimpleNamespace(dataset='anet'))
                self.assertEqual(scorer.call_args.args[-1],6.)
            self.assertEqual(path.read_bytes(),original)
            self.assertTrue((path.parent.parent/'quality_epoch_history.json').is_file())

    def test_joint_smoke_uses_head_fit_video_for_both_fresh_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(work=Path(tmp),dataset='anet',checkpoint=False)
            calls=[]
            def logged(command,log):
                calls.append(command);log.parent.mkdir(parents=True,exist_ok=True)
                log.write_text('ONLINE_QUALITY_GRADIENT_PASS joint')
            with patch.object(c1_pipeline,'check',return_value={'id':'joint','online_quality':True}),\
                 patch.object(c1_pipeline,'logged',side_effect=logged),patch.object(c1_pipeline,'fingerprint',return_value={'source':'same'}):
                c1_pipeline.smoke(args)
                args.checkpoint=True;c1_pipeline.smoke(args)
            self.assertEqual(calls[1][2],'tools/prawn_joint_smoke.py')
            self.assertEqual(calls[3][2],'tools/prawn_joint_smoke.py')
            self.assertIn('--checkpoint',calls[3])

    def test_checkout_allows_only_recorded_runtime_patches(self):
        import prawn_ablation as runner
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);source=work/'train_no_content/source';source.mkdir(parents=True)
            (source/'eval_utils_clip.py').write_text('runtime patch')
            record=work/'train_no_content/provenance/runtime_sources/eval_utils_clip.py';record.parent.mkdir(parents=True)
            record.write_text('runtime patch')
            with patch.object(runner,'WORK',work),patch.object(runner,'revision',return_value='FIXED'),\
                 patch.object(runner.subprocess,'check_output',side_effect=['FIXED\n','eval_utils_clip.py\n']):
                self.assertEqual(runner.child_source('no-content'),source)
            record.write_text('different patch')
            with patch.object(runner,'WORK',work),patch.object(runner,'revision',return_value='FIXED'),\
                 patch.object(runner.subprocess,'check_output',side_effect=['FIXED\n','eval_utils_clip.py\n']):
                with self.assertRaises(ValueError):runner.child_source('no-content')

    def test_new_cli_safe_path_and_syntax(self):
        env=dict(os.environ,PYTHONSAFEPATH='1',PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-P',str(ROOT/'tools/prawn_ablation.py'),'--help'],
                                  cwd=tmp,env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for name in ['tools/prawn_ablation.py','tools/prawn_ablation_control.py','tools/prawn_ablation_train.py',
                     'tools/prawn_ablation_eval.py','misc/prawn_ablation.py','reranking/online.py','train.py',
                     'ddvc/ddvc_ret_encdec_clip_capQ.py']:
            ast.parse((ROOT/name).read_text(encoding='utf8'))


if __name__=='__main__':unittest.main()
