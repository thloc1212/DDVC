"""Offline integration tests; Molab additionally runs real-head CUDA forward/backward smoke."""
import copy
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
import torch
from reranking.cache import atomic_json,sha,write_shard
from reranking.pipeline import atomic_torch,METRICS
from reranking.head import JointQualityHead
from reranking.core import fit_idf,targets,select
from event_selection.core import pair_targets,geometry,normalize_features,relation_select,calibrated_count,median_count
from event_selection.heads import RelationHead,CountHead,relation_loss,count_loss
from event_selection.pipeline import Experiment
from event_selection.storage import prepare,RERANK_SOURCES
from test_c1_rclone import FileDrive,q as helper
import relation_count as runner

torch.set_num_threads(2)


def example():
    anno=dict(duration=100.,timestamps=[[0.,70.],[25.,60.]],sentences=['the person performs an activity','the person jumps'])
    spans=[[0.,70.],[1.,69.],[25.,60.],[26.,59.]]
    rows=[dict(query_id=i,timestamp=s,sentence=anno['sentences'][i//2],proposal_score=.8-i*.03,
               sentence_score=-.2,pred_event_count=2,vid_duration=100.) for i,s in enumerate(spans)]
    idf=fit_idf({'fit':anno},['fit']);quality,reward=targets(rows,anno,idf)
    return anno,rows,quality,reward


def fixture(root,queries=4,full_count=2):
    """Real NPZ/head/checksum fixture, local Drive transport, no DDVC/CLIP files."""
    drive=FileDrive(root/'cloud');base_id='base_test';rank_id='rank_test';ds='yc2'
    base_remote=Path(drive.path('runs/ddvc_c1/yc2/'+base_id+'/recovery/slot-0'))
    rank_remote=Path(drive.path('runs/ddvc_c1_rerank/yc2/'+rank_id+'/recovery/slot-0'))
    base_remote.mkdir(parents=True);rank_remote.mkdir(parents=True)
    anno,rows,quality,reward=example()
    for i in range(full_count-2):anno['timestamps'].append([75.+i,76.+i]);anno['sentences'].append('extra action '+str(i))
    rows += [dict(rows[0],query_id=i,proposal_score=.01,timestamp=[90.,95.],sentence='background') for i in range(4,queries)]
    if full_count>10:
        for r in rows:r['pred_event_count']=10
    _,reward=targets(rows,anno,fit_idf({'fit':anno},['fit']))
    train={v:copy.deepcopy(anno) for v in ['a','b','c','d']};val={v:copy.deepcopy(anno) for v in ['v','w']}
    prep='prepared/yc2_c1_e15_b1/'
    cfg=dict(data_signature='data-test',proposal_mask_loss_coef=0.,eval_selection_method='softnms',num_queries=queries,
             train_caption_file='/marimo/DDVC_C1/'+prep+'train.json',val_caption_file='/marimo/DDVC_C1/'+prep+'val.json',
             gt_file_for_eval=['/marimo/DDVC_C1/'+prep+'ref.json'],gt_file_for_para_eval=['/marimo/DDVC_C1/'+prep+'para.json'])
    import yaml
    p=base_remote/(prep+'config.yml');p.parent.mkdir(parents=True);p.write_text(yaml.safe_dump(cfg))
    for name,data in [('train.json',train),('val.json',val),('ref.json',val),('para.json',val)]:atomic_json(base_remote/(prep+name),data)
    cm=dict(input_sha256=dict(train_caption_file=sha(base_remote/(prep+'train.json')),val_caption_file=sha(base_remote/(prep+'val.json'))))
    atomic_json(base_remote/(prep+'c1_manifest.json'),cm)
    base_files=helper.inventory(base_remote);base_files['runs/yc2_c1_e15_b1/model-best.pth']=dict(bytes=999999,sha256='frozen-c1-test-hash')
    bmeta=dict(created_ns=1,dataset=ds,run_id=base_id,source_commit='original-base-commit',files=base_files)
    atomic_json(base_remote/'READY.json',bmeta)
    prefix='runs/yc2/';folder=rank_remote/prefix;folder.mkdir(parents=True)
    manifest=dict(signature='original-quality-signature',fit_ids=['a','b','c'],dev_ids=['d'],
                  base_checkpoint_sha256='frozen-c1-test-hash',base_epoch=11,data_signature='data-test',
                  base_config_sha256=sha(base_remote/(prep+'config.yml')))
    atomic_json(folder/'manifest.json',manifest);torch.manual_seed(777)
    atomic_torch(folder/'head-best.pt',dict(signature=manifest['signature'],model=JointQualityHead().state_dict(),beta=1.))
    atomic_json(folder/'gpu_smoke.json',dict(passed=True))
    scores={k:.01*(i+1) for i,k in enumerate(METRICS)}
    for mode in ('baseline_softnms','reranker_softnms'):atomic_json(folder/('evaluation/'+mode+'.metrics.json'),dict(scores=scores))
    session=dict(base_c1_drive_session=dict(dataset=ds,run_id=base_id,source_commit='original-base-commit',
                 remote='kltnbackup',root='KLTNRUNS',work='/marimo/DDVC_C1'))
    atomic_json(rank_remote/'provenance/rerank_yc2.json',session)
    rng=np.random.default_rng(777)
    for split,data in [('train',train),('val',val)]:
        videos=[];arrays=[]
        for vid in data:
            v=dict(video_id=vid,rows=copy.deepcopy(rows),N=rows[0]['pred_event_count'])
            if split=='train':v.update(fold='head_dev' if vid=='d' else 'head_fit',gt_captions=anno['sentences'],reward=reward.copy())
            f=rng.normal(size=(queries,2312)).astype(np.float16);f[:2,:128]+=2;f[2:,:128]-=2
            arrays.append(f);videos.append(v)
        write_shard(folder/('cache/'+split+'/shard_00000.npz'),videos,arrays,manifest['signature'],split)
    atomic_json(rank_remote/'READY.json',dict(created_ns=1,dataset=ds,run_id=rank_id,source_commit=next(iter(RERANK_SOURCES)),files=helper.inventory(rank_remote)))
    return drive,rank_id,rank_remote,scores


class CoreTests(unittest.TestCase):
    def test_numpy_scalar_prediction_json_preserves_values_and_candidate_order(self):
        _,rows,_,_=example();quality=np.array([.9,.8,.7,.6],dtype=np.float32)
        result=select(rows,2,quality,1.)
        # NumPy 2 can produce float32 here, while NumPy 1 promotes the same expression to float64.
        for row in result:row['selection_score']=np.float32(row['selection_score'])
        candidate_ids=[r['query_id'] for r in result]
        payload=dict(results=result,count=np.int64(2),passed=np.bool_(True))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'predictions.json';atomic_json(path,payload);restored=json.loads(path.read_text())
            self.assertEqual([r['query_id'] for r in restored['results']],candidate_ids)
            self.assertEqual(restored['count'],2);self.assertIs(restored['passed'],True)
            for a,b in zip(result,restored['results']):
                self.assertEqual(a['selection_score'].item(),b['selection_score'])
                self.assertEqual(a['timestamp'],b['timestamp']);self.assertEqual(a['sentence'],b['sentence'])
            # Arrays/features must still be rejected rather than silently dumped into a huge JSON file.
            with self.assertRaises(TypeError):atomic_json(Path(temp)/'bad.json',dict(features=np.zeros((2,3))))

    def test_same_gt_positive_distinct_overlapping_gt_negative(self):
        anno,rows,_,reward=example();labels,valid,hard=pair_targets(rows,anno,reward)
        self.assertTrue(valid[0,1]);self.assertEqual(labels[0,1],1.)
        self.assertTrue(valid[0,2]);self.assertEqual(labels[0,2],0.)
        self.assertTrue(hard[0,2]);self.assertFalse(valid[0,0]);self.assertFalse(valid[1,0])

    def test_ambiguous_assignment_ignored_and_repeated_caption_can_be_distinct(self):
        anno,rows,_,reward=example();ambiguous=np.ones_like(reward)*.4
        self.assertFalse(pair_targets(rows,anno,ambiguous)[1].any())
        anno=dict(timestamps=[[0,10],[20,30]],sentences=['stir the pot','stir the pot'])
        r=[dict(timestamp=[0,10],sentence='stir the pot'),dict(timestamp=[20,30],sentence='stir the pot')]
        _,reward=targets(r,anno,fit_idf({'fit':anno},['fit']));label,valid,_=pair_targets(r,anno,reward)
        self.assertTrue(valid[0,1]);self.assertEqual(label[0,1],0)

    def test_relation_selection_suppresses_duplicate_but_retains_nested_event(self):
        _,r,_,_=example();before=copy.deepcopy(r);q=[.9,.85,.8,.75]
        duplicate=np.zeros((4,4));duplicate[0,1]=duplicate[1,0]=1;duplicate[2,3]=duplicate[3,2]=1
        result=relation_select(r,2,q,1.,duplicate,2.)
        self.assertEqual([v['query_id'] for v in result],[0,2]);self.assertEqual(r,before)
        for v in result:self.assertEqual(v['timestamp'],r[v['query_id']]['timestamp'])

    def test_negative_score_never_improves_and_zero_penalty_is_quality_topk(self):
        _,r,_,_=example();q=[0,0,0,0];d=np.ones((4,4))
        for row in r:row.update(proposal_score=0.,sentence_score=-10.)
        a=relation_select(r,4,q,1,d,0);b=relation_select(r,4,q,1,d,2)
        byid={p['query_id']:p for p in a}
        for p in b:self.assertLessEqual(p['selection_score'],byid[p['query_id']]['selection_score'])

    def test_heads_symmetric_permutation_invariant_and_count_weights_active(self):
        torch.manual_seed(777);x=torch.randn(4,2314);g=torch.randn(4,4,9);g=(g+g.transpose(0,1))/2
        r=RelationHead().eval();c=CountHead(4).eval();order=torch.tensor([2,0,3,1])
        self.assertTrue(torch.allclose(r(x,g),r(x,g).T,atol=1e-6))
        self.assertTrue(torch.allclose(r(x[order],g[order][:,order]),r(x,g)[order][:,order],atol=1e-6))
        self.assertTrue(torch.allclose(c(x,2),c(x[order],2),atol=1e-6))
        logits=torch.zeros(5);one=torch.ones(5);two=one.clone();two[2]=2
        self.assertGreater(float(count_loss(logits,2,two)),float(count_loss(logits,2,one))*1.8)
        self.assertEqual(calibrated_count(0,3,0.,4),0);self.assertEqual(calibrated_count(2,7,1.,4),4)
        self.assertEqual(median_count([.1,.2,.4,.2,.1]),2)


class IntegrationTests(unittest.TestCase):
    def test_json_fix_migrates_completed_heads_without_updating_them_or_policy(self):
        from reranking.pipeline import signature
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);folder=work/'runs/yc2';folder.mkdir(parents=True)
            manifest=dict(schema='test',input_receipt_sha256='same-input')
            manifest['signature']=signature(manifest);atomic_json(folder/'manifest.json',manifest)
            atomic_torch(folder/'heads-last.pt',dict(signature=manifest['signature'],next_epoch=20))
            atomic_torch(folder/'relation-best.pt',dict(model=RelationHead().state_dict()))
            atomic_torch(folder/'count-best.pt',dict(model=CountHead(4).state_dict()))
            policy=dict(signature=manifest['signature'],relation_sha256=sha(folder/'relation-best.pt'),count_sha256=sha(folder/'count-best.pt'))
            atomic_json(folder/'policy.json',policy)
            files=['heads-last.pt','relation-best.pt','count-best.pt','policy.json']
            before={n:sha(folder/n) for n in files}
            state=dict(source_commit=next(iter(runner.JSON_FIX_SOURCES)),input_receipt_sha256='same-input')
            updated=runner.compatible_resume(state,'serialization-fix',work,'yc2')
            self.assertEqual(updated['source_commit'],'serialization-fix')
            self.assertEqual(updated['source_history'][0]['source_commit'],state['source_commit'])
            self.assertEqual(before,{n:sha(folder/n) for n in files})
            with self.assertRaisesRegex(ValueError,'source changed'):
                runner.compatible_resume(dict(state,source_commit='unrelated-change'),'serialization-fix',work,'yc2')
            atomic_torch(folder/'heads-last.pt',dict(signature=manifest['signature'],next_epoch=19))
            with self.assertRaisesRegex(ValueError,'completed unchanged'):
                runner.compatible_resume(state,'serialization-fix',work,'yc2')

    @unittest.skipUnless(shutil.which('java'),'Java unavailable for actual vendored scorer smoke')
    def test_actual_official_scorer_includes_full_metrics_and_paragraph_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            exp=Experiment.__new__(Experiment);exp.folder=Path(temp)
            exp.scorer_smoke()
            self.assertTrue(json.loads((exp.folder/'scorer_smoke/PASS.json').read_text())['synthetic_only'])

    def test_verified_minimal_restore_fit_resume_and_six_paired_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);drive,run,_,scores=fixture(root);work=root/'work'
            receipt=prepare(work,'yc2',drive,helper,run)
            self.assertFalse(list((work/'inputs').rglob('*.pth')))
            self.assertFalse(any(p.name.startswith('clipvitl14') for p in (work/'inputs').rglob('*')))
            exp=Experiment(work,'yc2',device='cpu');train=exp.load('train')
            self.assertEqual(train[0]['full_count'],2);self.assertEqual(train[0]['count_target'],2)
            before=sha(exp.quality_path);exp.smoke(require_cuda=False)
            original_load=exp.load
            def training_only(split):
                self.assertEqual(split,'train','Training/model selection must not load DVC validation')
                return original_load(split)
            with patch.object(exp,'load',side_effect=training_only):exp.fit()
            policy=exp.policy();self.assertEqual(policy['selection_scope'],'TRAIN head_dev only; frozen before DVC validation')
            hashes=[sha(exp.folder/n) for n in ('relation-best.pt','count-best.pt','policy.json')]
            exp.fit();self.assertEqual(hashes,[sha(exp.folder/n) for n in ('relation-best.pt','count-best.pt','policy.json')])
            with patch('event_selection.pipeline.official_scores',return_value=scores):exp.evaluate()
            table=list(csv.DictReader(io.StringIO((exp.folder/'evaluation/summary.tsv').read_text(encoding='utf-8-sig')),delimiter='\t'))
            self.assertEqual(len(table),6);self.assertEqual(before,sha(exp.quality_path))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in exp.quality.parameters()))
            outputs={r['method']:json.loads((exp.folder/('evaluation/'+r['method']+'.json')).read_text())['results'] for r in table}
            for vid in outputs['C1_softnms']:
                self.assertEqual(len(outputs['quality_relation'][vid]),len(outputs['quality_softnms'][vid]))
                self.assertEqual(len(outputs['quality_relation_count'][vid]),len(outputs['quality_softnms_count'][vid]))
            # A second preparation keeps identical input identity; no redownload/overwrite.
            self.assertEqual(receipt,prepare(work,'yc2',drive,helper,run))

    def test_corrupt_latest_input_slot_falls_back_and_wrong_base_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);drive,run,remote,_=fixture(root)
            newer=remote.with_name('slot-1');shutil.copytree(remote,newer)
            meta=json.loads((newer/'READY.json').read_text());meta['created_ns']=2;atomic_json(newer/'READY.json',meta)
            with (newer/'runs/yc2/cache/train/shard_00000.npz').open('ab') as f:f.write(b'corrupt')
            prepare(root/'work','yc2',drive,helper,run)
            self.assertTrue((root/'work/inputs/yc2/READY.json').is_file())
            bmeta=root/'cloud/runs/ddvc_c1/yc2/base_test/recovery/slot-0/READY.json'
            value=json.loads(bmeta.read_text());value['files']['runs/yc2_c1_e15_b1/model-best.pth']['sha256']='wrong'
            atomic_json(bmeta,value)
            with self.assertRaisesRegex(ValueError,'checkpoint differs'):prepare(root/'wrong','yc2',drive,helper,run)

    def test_full_train_count_is_not_sampled_caption_counter_and_Q_capacity_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            for queries,target in [(50,14),(10,10)]:
                root=Path(temp)/str(queries);drive,run,_,_=fixture(root,queries=queries,full_count=14)
                work=root/'work';prepare(work,'yc2',drive,helper,run);exp=Experiment(work,'yc2',device='cpu')
                video=exp.load('train')[0]
                self.assertEqual(video['full_count'],14);self.assertEqual(video['N'],10)
                self.assertEqual(video['count_target'],target)

    def test_safe_path_cli_and_no_model_import_dependency(self):
        result=subprocess.run([sys.executable,'-P',str(ROOT/'tools/relation_count.py'),'--help'],cwd=tempfile.gettempdir(),capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('prepare',result.stdout)

    def test_derived_head_backup_restore_fallback_preserves_new_session_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);drive,rank,_,_=fixture(root);work=root/'work';receipt=prepare(work,'yc2',drive,helper,rank)
            run=work/'runs/yc2';run.mkdir(parents=True);(work/'logs').mkdir()
            atomic_json(work/'provenance/input_receipt_yc2.json',receipt)
            atomic_json(work/'provenance/native_runtime.json',dict(session='old'))
            state=dict(dataset='yc2',run_id='derived_test',source_commit='derived-commit',work=str(work),
                       input_receipt_sha256=sha(work/'inputs/yc2/READY.json'),source_reranker_run=rank)
            atomic_json(runner.state_file(work,'yc2'),state);(run/'heads-last.pt').write_bytes(b'optimizer-epoch1')
            recovery=runner.Recovery(drive,helper,work,state)
            with patch.object(runner,'revision',return_value='derived-commit'):
                recovery.sync()
                tmp=run/'heads.tmp';tmp.write_bytes(b'optimizer-epoch2');os.replace(tmp,run/'heads-last.pt');recovery.sync()
                newest=Path(recovery.remote)/'recovery/slot-1/runs/yc2/heads-last.pt';newest.write_bytes(b'corrupt')
                # Delete only fixture-owned temporary folders, keeping immutable inputs/cloud.
                self.assertTrue(work.resolve().is_relative_to(root.resolve()))
                for name in ('runs','logs','provenance'):shutil.rmtree(work/name)
                atomic_json(work/'provenance/native_runtime.json',dict(session='new'))
                args=SimpleNamespace(work=work,dataset='yc2',run_id='derived_test')
                with patch.object(runner,'connection',return_value=(drive,helper)):runner.restore(args)
                self.assertEqual((run/'heads-last.pt').read_bytes(),b'optimizer-epoch1')
                self.assertEqual(json.loads((work/'provenance/native_runtime.json').read_text())['session'],'new')


if __name__=='__main__':unittest.main()
