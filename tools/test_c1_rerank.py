"""Offline behavioural tests; the Molab runner separately gates on real C1 GPU smoke."""
import copy
import importlib.util
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
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
import torch
from reranking import core
from reranking.cache import atomic_json,sha,write_shard,read_shard
from reranking.head import JointQualityHead,objective
from reranking.pipeline import Workflow
from misc.proposal_selection import select_proposals
import c1_rerank as runner

torch.set_num_threads(2)


def rows():
    return [dict(query_id=i,timestamp=span,sentence=text,proposal_score=p,sentence_score=s,pred_event_count=2)
            for i,(span,text,p,s) in enumerate([
                ([0.,10.],'cut the onion',.8,-.3),([1.,11.],'cut the onion',.75,-.3),
                ([15.,23.],'boil the water',.65,-.3),([20.,30.],'stir',.01,-2.),
                ([0.,30.],'cook a meal',.1,-3.)])]


class CoreTests(unittest.TestCase):
    def test_beta_zero_exact_original_softnms_with_negative_scores_and_ties(self):
        rng=np.random.default_rng(777)
        cases=[rows(),rows()+[dict(rows()[0],query_id=9)]]
        for _ in range(10):
            r=rows()
            for v in r:v.update(proposal_score=float(rng.random()),sentence_score=-float(rng.random()*8))
            cases.append(r)
        for r in cases:
            before=copy.deepcopy(r)
            for n in range(len(r)+2):
                old=select_proposals(r,n);new=core.select(r,n,[.8]*len(r),0.)
                self.assertEqual([p['query_id'] for p in old],[p['query_id'] for p in new])
                self.assertEqual([p['selection_score'] for p in old],[p['selection_score'] for p in new])
                self.assertEqual([p['timestamp'] for p in old],[p['timestamp'] for p in new])
            self.assertEqual(before,r)

    def test_reranker_can_recover_distinct_candidate_without_mutation(self):
        r=rows();before=copy.deepcopy(r)
        selected=core.select(r,2,[0,0,1,0,0],2.)
        self.assertIn(2,[p['query_id'] for p in selected]);self.assertEqual(len(selected),2)
        self.assertEqual(before,r)
        for p in selected:self.assertEqual(p['sentence'],r[p['query_id']]['sentence'])

    def test_joint_target_cannot_mix_time_of_one_gt_with_text_of_another(self):
        annotation=dict(timestamps=[[0,10],[20,30]],sentences=['chop onion','boil water'])
        idf=core.fit_idf({'fit':annotation},['fit'])
        wrong=[dict(timestamp=[0,10],sentence='boil water')]
        quality,reward=core.targets(wrong,annotation,idf)
        self.assertEqual(quality.tolist(),[0.]);self.assertEqual(reward.shape,(1,2))
        self.assertEqual(core.targets([dict(timestamp=[20,30],sentence='boil water')],annotation,idf)[0][0],1.)

    def test_idf_fit_partition_and_validation_labels_rejected(self):
        ann={'fit':dict(sentences=['fit vocabulary']), 'dev':dict(sentences=['secret label'])}
        idf=core.fit_idf(ann,['fit']);self.assertNotIn('secret',idf['values'])
        with self.assertRaisesRegex(ValueError,'TRAIN labels'):
            Workflow.batch([dict(features=np.zeros((2,2312),np.float16))],'cpu')

    def test_feature_pool_no_gradient_or_absolute_time_query_id_input(self):
        class Tokenizer:
            eos_token_id=0
            def encode(self,text,**kwargs):return [1,2,3]
        torch.manual_seed(7);wte=torch.nn.Embedding(6,768);wte.requires_grad_(False)
        f=torch.randn(100,768,requires_grad=True)
        r=[dict(rows()[0],timestamp=[30,40])]
        x=core.feature_pool(f,r,100,wte,Tokenizer())
        shifted=[dict(r[0],timestamp=[40,50],query_id=987)]
        y=core.feature_pool(torch.roll(f,10,0),shifted,100,wte,Tokenizer())
        np.testing.assert_allclose(x,y,atol=1e-3)
        self.assertEqual(x.shape,(1,2312));self.assertEqual(x.dtype,np.float16)
        self.assertIsNone(f.grad);self.assertIsNone(wte.weight.grad)

    def test_head_learns_with_only_head_optimizer(self):
        torch.manual_seed(777);model=JointQualityHead(hidden=16)
        unrelated=torch.nn.Linear(8,8);frozen=copy.deepcopy(unrelated.state_dict())
        x=torch.randn(2,5,2312);target=torch.tensor([[0.,.2,.8,.9,.4]]*2);valid=torch.ones_like(target,dtype=torch.bool)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        model.eval();before=float(objective(model(x),target,valid).detach())
        for _ in range(35):
            optimizer.zero_grad();loss=objective(model(x),target,valid)
            self.assertTrue(bool(torch.isfinite(loss)));loss.backward();optimizer.step()
        after=float(objective(model(x),target,valid).detach());self.assertLess(after,before*.3)
        for k,v in unrelated.state_dict().items():self.assertTrue(torch.equal(v,frozen[k]))
        self.assertFalse(any(id(p) in {id(q) for q in unrelated.parameters()} for g in optimizer.param_groups for p in g['params']))


class CacheTests(unittest.TestCase):
    def test_roundtrip_hash_and_validation_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'train.npz';r=rows();x=np.ones((len(r),2312),np.float16)
            v=dict(video_id='fit',rows=r,N=2,fold='head_fit',gt_captions=['a','b'],reward=np.ones((len(r),2),np.float32))
            write_shard(path,[v],[x],'sig','train');loaded=read_shard(path,'sig','train')[0]
            np.testing.assert_array_equal(loaded['features'],x);self.assertEqual(loaded['reward'].shape,(5,2))
            with self.assertRaises(ValueError):read_shard(path,'changed','train')
            val=Path(temp)/'val.npz'
            with self.assertRaisesRegex(ValueError,'Validation candidate'):
                write_shard(val,[loaded],[x],'sig','val')
            write_shard(val,[dict(video_id='val',rows=r,N=2)],[x],'sig','val')
            self.assertNotIn('quality',read_shard(val,'sig','val')[0])
            with val.open('ab') as f:f.write(b'corrupt')
            with self.assertRaises(ValueError):read_shard(val,'sig','val')

    def test_full_head_fit_resume_and_frozen_beta_without_val_access(self):
        with tempfile.TemporaryDirectory() as temp:
            workflow=Workflow.__new__(Workflow);workflow.folder=Path(temp);workflow.sig='fit-only-test'
            rng=np.random.default_rng(777);videos=[]
            for i in range(10):
                v=dict(video_id=str(i),rows=rows(),N=2,fold='head_fit' if i<8 else 'head_dev',
                       features=rng.normal(size=(5,2312)).astype(np.float16),quality=np.array([.1,.2,.9,.05,.3],np.float32),
                       reward=np.array([[.1,0],[.2,0],[0,.9],[0,.05],[.3,0]],np.float32))
                videos.append(v)
            def load(split):
                self.assertEqual(split,'train','Head fitting must never load validation candidates/labels')
                return videos
            workflow.load=load
            with patch('torch.cuda.is_available',return_value=False):workflow.fit()
            cp=torch.load(workflow.folder/'head-best.pt',weights_only=False)
            self.assertIn(cp['beta'],(0.,.25,.5,1.,2.));self.assertEqual(len(cp['history']) if 'history' in cp else 20,20)
            self.assertIn('TRAIN',cp['beta_selection'])
            before=sha(workflow.folder/'head-best.pt')
            with patch('torch.cuda.is_available',return_value=False):workflow.fit()
            self.assertEqual(before,sha(workflow.folder/'head-best.pt'))

    def test_cli_imports_in_safe_path_mode_from_another_directory(self):
        env=dict(os.environ,PYTHONSAFEPATH='1')
        result=subprocess.run([sys.executable,'-P',str(ROOT/'tools/c1_rerank.py'),'--help'],cwd=tempfile.gettempdir(),env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('{yc2,anet}',result.stdout)

    def test_validation_evaluation_reuses_candidates_fixed_N_and_frozen_head(self):
        with tempfile.TemporaryDirectory() as temp:
            previous=Path.cwd()
            try:
                root=Path(temp);w=Workflow.__new__(Workflow);w.folder=root/'work';w.folder.mkdir()
                w.base=root/'base';(w.base/'source').mkdir(parents=True);w.sig='eval-signature'
                w.checkpoint=root/'model-best.pth';w.checkpoint.write_bytes(b'unchanged C1 weights')
                w.manifest={'base_checkpoint_sha256':sha(w.checkpoint)}
                from reranking.pipeline import atomic_torch,METRICS
                atomic_torch(w.folder/'head-best.pt',dict(model=JointQualityHead().state_dict(),signature=w.sig,beta=0.))
                headhash=sha(w.folder/'head-best.pt');r=rows()
                scores={key:.01*(i+1) for i,key in enumerate(METRICS)}
                w.final_path=root/'final.json'
                atomic_json(w.final_path,dict(results={'val':select_proposals(r,2)},**scores))
                ref=root/'reference.json';ref.write_text('{}');w.cfg=dict(gt_file_for_eval=[str(ref)],gt_file_for_para_eval=[str(ref)])
                def load(split):
                    self.assertEqual(split,'val');return [dict(video_id='val',rows=r,N=2,features=np.zeros((5,2312),np.float16))]
                w.load=load
                # Mirror the real DDVC default: without infer_mode it returns METEOR only.
                # Keep actual head inference, selection, score validation and cache checks.
                calls=[]
                def evaluate_metrics(*args,**kwargs):
                    calls.append(kwargs)
                    self.assertEqual(kwargs.get('dvc_eval_version'),'2018')
                    return dict(scores) if kwargs.get('infer_mode') else {'METEOR':scores['METEOR']}
                scorer=SimpleNamespace(eval_metrics=evaluate_metrics)
                with patch.dict(sys.modules,{'eval_utils_clip':scorer}),patch('torch.cuda.is_available',return_value=False):w.evaluate()
                self.assertEqual(len(calls),3);self.assertTrue(all(c['infer_mode'] for c in calls))
                metric_file=w.folder/'evaluation/baseline_topk.metrics.json'
                partial=json.loads(metric_file.read_text());partial['scores'].pop('CIDEr');atomic_json(metric_file,partial)
                # Even a matching fingerprint must not make partial metrics reusable.
                with patch.dict(sys.modules,{'eval_utils_clip':scorer}),patch('torch.cuda.is_available',return_value=False):w.evaluate()
                self.assertEqual(len(calls),4)
                self.assertIn('CIDEr',json.loads(metric_file.read_text())['scores'])
                a=json.loads((w.folder/'evaluation/baseline_softnms.json').read_text())['results']
                b=json.loads((w.folder/'evaluation/reranker_softnms.json').read_text())['results']
                self.assertEqual([p['query_id'] for p in a['val']],[p['query_id'] for p in b['val']])
                self.assertEqual(len(b['val']),2);self.assertEqual(headhash,sha(w.folder/'head-best.pt'))
                self.assertTrue((w.folder/'evaluation/summary.tsv').is_file())
            finally:os.chdir(previous)

    def test_completed_head_can_migrate_only_from_explicit_evaluation_fix_source(self):
        from reranking.pipeline import atomic_torch
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);base=folder/'base';loaded=[]
            workflow=SimpleNamespace(base=base,folder=folder,sig='same-training-signature',load=lambda split:loaded.append(split))
            atomic_torch(folder/'head-last.pt',dict(signature=workflow.sig,next_epoch=20))
            atomic_torch(folder/'head-best.pt',dict(signature=workflow.sig,beta=.5,epoch=11))
            hashes=[sha(folder/name) for name in ('head-last.pt','head-best.pt')]
            state=dict(base_work=str(base),source_commit=next(iter(runner.EVALUATION_FIX_SOURCES)))
            updated=runner.compatible_resume(state,'fixed-commit',workflow)
            self.assertEqual(updated['source_commit'],'fixed-commit');self.assertEqual(loaded,['train','val'])
            self.assertEqual(state['source_commit'],updated['source_history'][0]['source_commit'])
            self.assertEqual(hashes,[sha(folder/name) for name in ('head-last.pt','head-best.pt')])
            with self.assertRaisesRegex(ValueError,'source changed'):
                runner.compatible_resume(dict(state,source_commit='other-change'),'fixed-commit',workflow)
            atomic_torch(folder/'head-last.pt',dict(signature=workflow.sig,next_epoch=19))
            with self.assertRaisesRegex(ValueError,'already-completed'):
                runner.compatible_resume(state,'fixed-commit',workflow)


class BackupTests(unittest.TestCase):
    def test_two_slot_restore_falls_back_from_corruption_and_excludes_secrets(self):
        from test_c1_rclone import FileDrive,q
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);work=root/'work';work.mkdir();cloud=FileDrive(root/'cloud')
            run=work/'runs/yc2';run.mkdir(parents=True);(work/'logs').mkdir()
            (run/'head-best.pt').write_bytes(b'head-v1');(work/'secret.env').write_text('NEVER_UPLOAD')
            state=dict(dataset='yc2',run_id='test',base_work=str(root/'base'),source_commit='test-commit')
            atomic_json(runner.state_path(work,'yc2'),state)
            backup=runner.Backup(cloud,q,work,state)
            with patch.object(runner,'revision',return_value='test-commit'):
                backup.sync();(run/'head-best.pt').write_bytes(b'head-v2');backup.sync()
                remote=Path(backup.remote)/'recovery'
                names=cloud.read_json(remote/'slot-0/READY.json')['files'];self.assertNotIn('secret.env',names)
                (remote/'slot-1/runs/yc2/head-best.pt').write_bytes(b'corrupt')
                # Reset only this test-owned temporary workspace, preserving cloud fixture.
                import shutil
                shutil.rmtree(work);work.mkdir()
                args=SimpleNamespace(work=work,base=root/'base',dataset='yc2',run_id='test')
                with patch.object(runner,'drive',return_value=(cloud,q)):runner.restore(args)
                self.assertEqual((work/'runs/yc2/head-best.pt').read_bytes(),b'head-v1')


if __name__=='__main__':unittest.main()
