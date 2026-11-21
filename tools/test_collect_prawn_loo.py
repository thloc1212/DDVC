"""Pure-stdlib offline fixtures; no credentials or network access."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import collect_prawn_loo as c
import prawn_results_viewer as helpers


class MemoryDrive:
    def __init__(self):self.files={};self.downloads=[]
    def path(self,name=''):return 'memory:'+name
    def call(self,command,source,*args,quiet=False):
        key=source.removeprefix('memory:').strip('/')
        if command=='copyto':
            assert not key.endswith(('.pt','.pth','.npz','.npy')) and '/cache/' not in key
            self.downloads.append(key);Path(args[0]).write_bytes(self.files[key]);return ''
        if command in ('lsf','lsjson'):
            prefix=key+'/' if key else ''
            tails=[n[len(prefix):] for n in self.files if n.startswith(prefix)]
            if command=='lsf':
                if '--dirs-only' in args:return '\n'.join(sorted({t.split('/')[0]+'/' for t in tails if '/' in t}))
                return '\n'.join(t for t in tails if '/' not in t)
            return json.dumps([dict(Path=t,Size=len(self.files[prefix+t]),Hashes={'md5':hashlib.md5(self.files[prefix+t]).hexdigest()})
                               for t in tails if '/' not in t])
        if command=='cat':return self.files[key].decode()
        raise AssertionError(command)
    def read_json(self,source):return json.loads(self.files[source.removeprefix('memory:')])


def encoded(value):return json.dumps(value,ensure_ascii=False,indent=2).encode()


def fixture():
    drive=MemoryDrive();dataset='yc2';full=None;fullq=None
    scores={k:10. for k in c.REQUIRED}
    reference={v:dict(duration=10.,timestamps=[[0.,4.]],sentences=['cut onion']) for v in ('a','b')}
    for profile,namespace in [('full','prawn_caption_match_generator'),('no_content','prawn_loo_no_content_generator'),('joint','prawn_loo_joint_generator')]:
        run_id='20261121_'+profile;run='yc2_'+profile;work='/marimo/'+profile
        root=f'runs/{namespace}/{dataset}/{run_id}/recovery/slot-1'
        refs=[f'prepared/{run}/reference.json',f'prepared/{run}/paragraph.json']
        cfg=dict(dataset=dataset,seed=777,batch_size=1,epoch=15,num_queries=50,caption_match_enabled=True,
                 caption_match_weight=.5,caption_match_warmup_epochs=2,caption_match_ramp_epochs=2,proposal_mask_loss_coef=0.,
                 proposal_mask_content_conditioned=profile!='no_content',online_quality=profile=='joint',data_signature='same-data',
                 gt_file_for_eval=[work+'/'+refs[0]],gt_file_for_para_eval=[work+'/'+refs[1]])
        raw={v:[dict(query_id=i,timestamp=[float(i),float(i+4)],sentence='cut onion',proposal_score=.8,
                          sentence_score=-.3,pred_event_count=1,vid_duration=10.,**({'reranker_quality':.5} if profile=='joint' else {}))
                for i in range(2)] for v in reference}
        selected={v:[copy.deepcopy(rows[0])] for v,rows in raw.items()}
        texts={f'prepared/{run}/c1_manifest.json':dict(expected_config=cfg),f'prepared/{run}/config.yml':'test-config',
               refs[0]:reference,refs[1]:{v:'cut onion' for v in reference},
               f'runs/{run}/info.json':dict(last={'epoch':14},best={'epoch':11,'result_json_path':work+f'/runs/{run}/prediction/num2_epoch11.json'}),
               f'runs/{run}/prediction/num2_epoch11.raw.json':{'results':raw},
               f'runs/{run}/prediction/num2_epoch11.json':dict(results=selected,**{k:v/100 for k,v in scores.items()})}
        files={}
        for name,value in texts.items():
            data=value.encode() if isinstance(value,str) else encoded(value)
            drive.files[root+'/'+name]=data;files[name]=dict(bytes=len(data),sha256=c.digest(data))
        checkpoint=c.digest(profile.encode());files[f'runs/{run}/model-best.pth']=dict(bytes=999,sha256=checkpoint)
        rh=[files[r]['sha256'] for r in refs]
        if profile=='joint':
            name=f'runs/{run}/quality_epochs/num2_epoch11/joint_quality_softnms'
            pred=dict(results=selected,selection={'beta':1.})
            data=encoded(pred);drive.files[root+'/'+name+'.json']=data;files[name+'.json']=dict(bytes=len(data),sha256=c.digest(data))
            metric={'signature':c.signature(dict(prediction=c.digest(data),refs=rh,protocol=c.PROTOCOL)),'scores':scores}
            data=encoded(metric);drive.files[root+'/'+name+'.metrics.json']=data;files[name+'.metrics.json']=dict(bytes=len(data),sha256=c.digest(data))
        ready=dict(dataset=dataset,run_id=run_id,work=work,files=files,created_ns=1,source_commit='fixture')
        drive.files[root+'/READY.json']=encoded(ready)
        if profile=='joint':continue
        qnamespace='prawn_caption_match_quality' if profile=='full' else 'prawn_loo_no_content_quality'
        qrun='head_'+profile;qroot=f'runs/{qnamespace}/{dataset}/{qrun}/recovery/slot-1'
        manifest=dict(base_checkpoint_sha256=checkpoint,base_config_sha256=files[f'prepared/{run}/config.yml']['sha256'],
                      raw_sha256=files[f'runs/{run}/prediction/num2_epoch11.raw.json']['sha256'],base_epoch=11,epochs=20,fixed_beta=1.)
        qfiles={};qtexts={f'runs/{dataset}/manifest.json':manifest,f'runs/{dataset}/head_history.json':[{'epoch':i} for i in range(20)]}
        for mode in ('baseline_softnms','reranker_softnms'):
            name=f'runs/{dataset}/evaluation/{mode}'
            pred=dict(results=selected,selection={'beta':0. if mode=='baseline_softnms' else 1.})
            qtexts[name+'.json']=pred
            qtexts[name+'.metrics.json']=dict(signature=c.signature(dict(file=c.digest(encoded(pred)),refs=rh,scoring_protocol=c.PROTOCOL)),scores=scores)
        for name,value in qtexts.items():
            data=encoded(value);drive.files[qroot+'/'+name]=data;qfiles[name]=dict(bytes=len(data),sha256=c.digest(data))
        headsha=c.digest(('head'+profile).encode());qfiles[f'runs/{dataset}/head-best.pt']=dict(bytes=999,sha256=headsha)
        drive.files[qroot+'/READY.json']=encoded(dict(dataset=dataset,run_id=qrun,work=work+'/frozen_quality',files=qfiles,created_ns=1))
        if profile=='full':
            full=dict(generator_run_id=run_id,quality_run_id=qrun,checkpoint_sha256=checkpoint,raw_sha256=manifest['raw_sha256'],config_sha256=manifest['base_config_sha256'])
            fullq=dict(headsha=headsha,raw=raw,selected=selected,refs=rh)
    folder=f'analysis/prawn_loo/{dataset}/full_{full["generator_run_id"]}/beta'
    drive.files[folder+'/quality_scores.json']=encoded(dict(identity=full,head_sha256=fullq['headsha'],qualities={v:{'0':.5,'1':.5} for v in reference}))
    summary=[]
    for beta in c.BETAS:
        name='beta_'+format(beta,'g').replace('.','p')
        pred=dict(results=fullq['selected'],selection={'beta':beta});data=encoded(pred)
        drive.files[folder+'/'+name+'.json']=data
        metric=dict(signature=c.signature(dict(prediction=c.digest(data),refs=fullq['refs'],protocol=c.PROTOCOL)),
                    identity=dict(full,head_sha256=fullq['headsha']),scores=scores)
        drive.files[folder+'/'+name+'.metrics.json']=encoded(metric);summary.append(dict(method=name,beta=beta,**scores))
    stream=io.StringIO();writer=csv.DictWriter(stream,list(summary[0]),delimiter='\t');writer.writeheader();writer.writerows(summary)
    drive.files[folder+'/summary.tsv']=stream.getvalue().encode()
    return drive


class ExportTests(unittest.TestCase):
    def collect(self,drive,out):
        with patch.dict(c.EXPECTED_VIDEOS,yc2=2):
            collector=c.Collector(drive,ROOT,out,'yc2',helpers)
            rows=collector.collect();target=collector.finish(rows)
            return collector,rows,target

    def test_complete_remote_collection_without_weights(self):
        drive=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            collector,rows,target=self.collect(drive,Path(tmp)/'output')
            self.assertTrue(collector.report['complete']);self.assertEqual(len(rows),12)
            self.assertTrue(target.is_file())
            with zipfile.ZipFile(target) as z:
                self.assertIn('results/yc2/ablation_summary.tsv',z.namelist())
                self.assertFalse(any(n.endswith(('.pth','.pt','.npz','.env','.conf')) for n in z.namelist()))
        self.assertFalse(any('/cache/' in n for n in drive.downloads))

    def test_missing_no_content_never_gets_complete_zip(self):
        drive=fixture();drive.files={k:v for k,v in drive.files.items() if 'prawn_loo_no_content_' not in k}
        with tempfile.TemporaryDirectory() as tmp:
            collector,rows,target=self.collect(drive,Path(tmp)/'output')
            self.assertIsNone(target);self.assertFalse(collector.report['complete'])
            self.assertTrue(any('no_content' in m for m in collector.report['missing']))

    def test_partial_beta_grid_is_rejected(self):
        drive=fixture();drive.files={k:v for k,v in drive.files.items() if not k.endswith('/beta_8.metrics.json')}
        with tempfile.TemporaryDirectory() as tmp:
            collector,rows,target=self.collect(drive,Path(tmp)/'output')
            self.assertIsNone(target);self.assertTrue(any('beta' in m for m in collector.report['missing']))

    def test_different_head_in_beta_is_rejected(self):
        drive=fixture();name=next(k for k in drive.files if k.endswith('/beta_4.metrics.json'))
        metric=json.loads(drive.files[name]);metric['identity']['head_sha256']='wrong-head';drive.files[name]=encoded(metric)
        with tempfile.TemporaryDirectory() as tmp:
            collector,rows,target=self.collect(drive,Path(tmp)/'output')
            self.assertIsNone(target);self.assertTrue(any('frozen models' in m for m in collector.report['missing']))

    def test_changed_caption_cannot_be_exported_as_same_candidates(self):
        raw={'v':[dict(query_id=0,timestamp=[0,1],sentence='cut onion',pred_event_count=1)]}
        changed={'v':[dict(query_id=0,timestamp=[0,1],sentence='boil water')]}
        with self.assertRaises(ValueError):c.validate_predictions(raw,changed)


if __name__=='__main__':unittest.main()
