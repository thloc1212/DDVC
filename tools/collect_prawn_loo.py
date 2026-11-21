"""Collect completed PRAWN LOO results from Drive. No training or neural inference.

Runs outside the immutable training checkout. Standard library only.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile

EXPECTED_VIDEOS={'yc2':436,'anet':4917}
BETAS=(0.,.25,.5,1.,2.,4.,6.,8.)
REQUIRED=('CIDEr','METEOR','Bleu_4','soda_c','Precision','Recall','MetaScore')
PROTOCOL='ddvc-2018-full-language-infer-mode-v1'


def digest(data):return hashlib.sha256(data).hexdigest()
def signature(value):return digest(json.dumps(value,sort_keys=True).encode())


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def table(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fields,delimiter='\t');writer.writeheader();writer.writerows(rows)


def complete_scores(scores):
    if not isinstance(scores,dict) or not all(isinstance(scores.get(k),(int,float)) and math.isfinite(scores[k]) for k in REQUIRED):
        raise ValueError('Incomplete/nonfinite saved official metrics')
    result=dict(scores);p,r=result['Precision'],result['Recall']
    result['F1']=2*p*r/(p+r) if p+r else 0.
    return result


def validate_config(cfg,dataset,profile):
    expected=dict(dataset=dataset,seed=777,batch_size=1,epoch=15 if dataset=='yc2' else 6,
                  num_queries=50 if dataset=='yc2' else 10,caption_match_enabled=True,caption_match_weight=.5,
                  caption_match_warmup_epochs=2,caption_match_ramp_epochs=2,proposal_mask_loss_coef=0.,
                  proposal_mask_content_conditioned=profile!='no_content')
    for key,value in expected.items():
        if cfg.get(key)!=value:raise ValueError(profile+' config mismatch: '+key)
    if bool(cfg.get('online_quality',False))!=(profile=='joint'):
        raise ValueError(profile+' optimization regime mismatch')


def validate_predictions(raw,predictions):
    if set(raw)!=set(predictions):raise ValueError('Prediction video pool differs from raw')
    for vid,selected in predictions.items():
        original={int(r['query_id']):r for r in raw[vid]}
        if not original:raise ValueError('Empty raw candidates')
        n=int(raw[vid][0]['pred_event_count'])
        if len(selected)!=min(n,len(original)):raise ValueError('Counter budget differs for '+vid)
        ids=[int(r['query_id']) for r in selected]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate selected query')
        for row in selected:
            old=original[int(row['query_id'])]
            if row['sentence']!=old['sentence'] or row['timestamp']!=old['timestamp']:
                raise ValueError('Selected timestamp/caption differs from raw')


class Collector:
    def __init__(self,drive,source,output,dataset,helpers):
        self.drive,self.source,self.output,self.ds,self.h=drive,Path(source),Path(output),dataset,helpers
        self.store=helpers.Store(drive,output,budget_mb=512)
        self.files={};self.analysis_dirs={};self.snapshots={};self.report={'dataset':dataset,'parts':{},'missing':[]}
        self.output.mkdir(parents=True,exist_ok=True)

    def remember(self,path,remote,verification):
        data=Path(path).read_bytes()
        self.files[str(Path(path).relative_to(self.output))]=dict(remote=remote,bytes=len(data),sha256=digest(data),verification=verification)

    def native(self,snapshot,name):
        value=self.store.get(snapshot,name)
        path=self.output/'artifacts'/snapshot['remote']/name
        self.remember(path,snapshot['remote']+'/'+name,'READY size + SHA256')
        return value

    def snaps(self,namespace):
        if namespace not in self.snapshots:self.snapshots[namespace]=self.store.snapshots(namespace,self.ds)
        return self.snapshots[namespace]

    def exists(self,path):
        parent=''
        for part in path.split('/'):
            if part not in self.store.dirs(parent):return False
            parent=parent+'/'+part if parent else part
        return True

    def analysis(self,path):
        self.h.safe_path(path)
        p=PurePosixPath(path)
        if p.suffix not in ('.json','.tsv') or p.parts[:2]!=('analysis','prawn_loo'):
            raise ValueError('Only analysis text artifacts may be fetched')
        parent=str(p.parent)
        if parent not in self.analysis_dirs:
            if not self.exists(parent):raise FileNotFoundError(parent)
            rows=json.loads(self.drive.call('lsjson',self.drive.path(parent),'--files-only','--hash','--max-depth','1',quiet=True))
            self.analysis_dirs[parent]={r['Path']:r for r in rows}
        meta=self.analysis_dirs[parent].get(p.name)
        if meta is None:raise FileNotFoundError(path)
        size=int(meta['Size'])
        if not 0<=size<=96*2**20 or self.store.used+size>self.store.limit:raise ValueError('Text-download size limit exceeded')
        local=self.output/'analysis_artifacts'/path
        expected_md5=meta.get('Hashes',{}).get('md5')
        valid=local.is_file() and local.stat().st_size==size
        if valid and expected_md5:valid=hashlib.md5(local.read_bytes()).hexdigest()==expected_md5
        if not valid:
            local.parent.mkdir(parents=True,exist_ok=True);temp=local.with_name(local.name+'.partial')
            self.drive.call('copyto',self.drive.path(path),str(temp),quiet=True)
            if temp.stat().st_size!=size or expected_md5 and hashlib.md5(temp.read_bytes()).hexdigest()!=expected_md5:
                raise ValueError('Analysis artifact transfer failed verification: '+path)
            os.replace(temp,local);self.store.used+=size
        self.remember(local,path,'Drive size + MD5' if expected_md5 else 'Drive size; saved metric signature checked separately')
        print('COLLECTED',path,round(size/2**20,2),'MiB',flush=True)
        return json.loads(local.read_text(encoding='utf-8-sig')) if p.suffix=='.json' else local.read_text(encoding='utf-8-sig')

    def get_generator(self,namespace,profile,run_id=None):
        for snap in self.snaps(namespace):
            if run_id and snap['ready']['run_id']!=run_id:continue
            try:
                g=self.h.generator(self.store,snap);validate_config(g['cfg'],self.ds,profile)
                if len(g['raw'])!=EXPECTED_VIDEOS[self.ds]:raise ValueError('Wrong validation video count')
                if set().union(*(set(r) for r in g['refs']))!=set(g['raw']):raise ValueError('GT reference and raw video pools differ')
                g['info']=self.native(snap,self.h.unique_file(snap['ready']['files'],'/info.json'))
                g['ref_hashes']=[]
                for key in ('gt_file_for_eval','gt_file_for_para_eval'):
                    for original in g['cfg'][key]:
                        relative=str(PurePosixPath(original).relative_to(snap['ready']['work']))
                        self.native(snap,relative);g['ref_hashes'].append(snap['ready']['files'][relative]['sha256'])
                dump(self.output/'metadata'/f'{profile}_generator_READY.json',snap['ready'])
                return g
            except (ValueError,KeyError) as error:print('SKIP_GENERATOR',profile,snap['ready']['run_id'],str(error),flush=True)
        raise FileNotFoundError('No complete '+profile+' generator snapshot in '+namespace+'/'+self.ds)

    def get_quality(self,g,namespace,run_id=None):
        for snap in self.snaps(namespace):
            if run_id and snap['ready']['run_id']!=run_id:continue
            files=snap['ready']['files'];base=f'runs/{self.ds}'
            if base+'/manifest.json' not in files:continue
            manifest=self.native(snap,base+'/manifest.json')
            if not self.h.quality_matches(g,manifest):continue
            modes=('baseline_softnms','reranker_softnms')
            if not all(base+'/evaluation/'+m+s in files for m in modes for s in ('.json','.metrics.json')):continue
            try:
                policies={}
                for mode in modes:
                    name=base+'/evaluation/'+mode
                    pred=self.native(snap,name+'.json');metric=self.native(snap,name+'.metrics.json')
                    validate_predictions(g['raw'],pred['results'])
                    phash=files[name+'.json']['sha256']
                    expected=signature(dict(file=phash,refs=g['ref_hashes'],scoring_protocol=PROTOCOL))
                    if metric.get('signature')!=expected:raise ValueError('Native scorer signature mismatch')
                    scores=complete_scores(metric['scores'])
                    policies[mode]=dict(pred=pred,scores=scores)
                history_name=base+'/head_history.json'
                if history_name in files:
                    history=self.native(snap,history_name)
                    if sorted(int(r['epoch']) for r in history)!=list(range(20)):raise ValueError('Frozen head has not finished 20 epochs')
                if manifest.get('epochs')!=20:raise ValueError('Wrong frozen-head training budget')
                if manifest.get('fixed_beta')!=(1. if self.ds=='yc2' else 6.):raise ValueError('Wrong fixed quality coefficient')
                dump(self.output/'metadata'/(namespace+'_READY.json'),snap['ready'])
                return snap,manifest,policies
            except (ValueError,KeyError) as error:print('SKIP_QUALITY',snap['ready']['run_id'],str(error),flush=True)
        raise FileNotFoundError('No complete hash-matched quality evaluation in '+namespace+'/'+self.ds)

    def beta_identity(self):
        parent=f'analysis/prawn_loo/{self.ds}'
        if not self.exists(parent):return None
        for name in reversed(self.store.dirs(parent)):
            if not name.startswith('full_'):continue
            folder=parent+'/'+name+'/beta'
            try:
                summary=self.analysis(folder+'/summary.tsv')
                rows=list(csv.DictReader(io.StringIO(summary),delimiter='\t'))
                if {float(r['beta']) for r in rows}!=set(BETAS):continue
                payload=self.analysis(folder+'/quality_scores.json')
                if payload['identity']['generator_run_id']!=name[5:]:raise ValueError('Beta folder/control run mismatch')
                return folder,payload
            except (ValueError,KeyError,FileNotFoundError) as error:print('SKIP_BETA_ROOT',folder,str(error),flush=True)
        return None

    def row(self,name,pred,scores,beta,content=True,regime='two-stage'):
        return dict(method=name,beta=beta,events=sum(map(len,pred['results'].values())),**scores,
                    dataset=self.ds,caption_matching=True,content_conditioned=content,
                    quality_enabled=beta!=0.,mask_loss=0.,regime=regime)

    def collect(self):
        rows=[];beta_root=self.beta_identity();identity=beta_root[1]['identity'] if beta_root else None
        full=self.get_generator('prawn_caption_match_generator','full',identity['generator_run_id'] if identity else None)
        qs,qm,policies=self.get_quality(full,'prawn_caption_match_quality',identity['quality_run_id'] if identity else None)
        fixed=1. if self.ds=='yc2' else 6.
        for name,mode,beta in [('Full','reranker_softnms',fixed),('w_o_quality','baseline_softnms',0.)]:
            item=policies[mode];rows.append(self.row(name,item['pred'],item['scores'],beta))
            dump(self.output/'results'/self.ds/'quality'/(name+'.json'),item['pred'])
            dump(self.output/'results'/self.ds/'quality'/(name+'.metrics.json'),{'scores':item['scores']})
        table(self.output/'results'/self.ds/'quality/summary.tsv',rows)
        self.report['parts']['quality']=dict(complete=True,generator_run_id=full['snapshot']['ready']['run_id'],quality_run_id=qs['ready']['run_id'])
        if beta_root:
            try:
                folder,payload=beta_root
                expected_id=dict(generator_run_id=full['snapshot']['ready']['run_id'],quality_run_id=qs['ready']['run_id'],
                                 checkpoint_sha256=full['checkpoint'],raw_sha256=full['raw_sha'],config_sha256=full['config_sha'])
                if any(payload['identity'].get(k)!=v for k,v in expected_id.items()):raise ValueError('Beta head/generator control mismatch')
                head_meta=qs['ready']['files'].get(f'runs/{self.ds}/head-best.pt',{})
                if payload.get('head_sha256')!=head_meta.get('sha256'):raise ValueError('Beta grid did not use the saved Full head')
                if set(payload['qualities'])!=set(full['raw']):raise ValueError('Beta quality pool differs from Full')
                beta_rows=[]
                for beta in BETAS:
                    name='beta_'+format(beta,'g').replace('.','p')
                    pred=self.analysis(folder+'/'+name+'.json');metric=self.analysis(folder+'/'+name+'.metrics.json')
                    if pred.get('selection',{}).get('beta')!=beta:raise ValueError('Wrong beta in saved policy: '+name)
                    validate_predictions(full['raw'],pred['results'])
                    record=self.files[str(Path('analysis_artifacts')/(folder+'/'+name+'.json'))]
                    expected=signature(dict(prediction=record['sha256'],refs=full['ref_hashes'],protocol=PROTOCOL))
                    if metric.get('signature')!=expected:raise ValueError('Beta scorer signature mismatch: '+name)
                    saved_id=metric.get('identity',{})
                    if saved_id.get('head_sha256')!=payload['head_sha256'] or any(saved_id.get(k)!=v for k,v in expected_id.items()):
                        raise ValueError('Beta coefficients use different frozen models')
                    scores=complete_scores(metric['scores'])
                    if beta in (0.,fixed):
                        reference=policies['baseline_softnms' if beta==0. else 'reranker_softnms']['scores']
                        if any(abs(scores[k]-reference[k])>1e-4 for k in REQUIRED):raise ValueError('Beta boundary does not reproduce saved Full')
                    beta_rows.append(self.row(name,pred,scores,beta))
                rows+=beta_rows;table(self.output/'results'/self.ds/'beta/summary.tsv',beta_rows)
                self.report['parts']['beta']=dict(complete=True,betas=list(BETAS),remote=folder)
            except (ValueError,KeyError,FileNotFoundError) as error:self.report['missing'].append('beta: '+str(error))
        else:self.report['missing'].append('beta: no complete 8-coefficient results under analysis/prawn_loo/'+self.ds+'/full_<run_id>/beta')
        for profile,namespace in [('no_content','prawn_loo_no_content_generator'),('joint','prawn_loo_joint_generator')]:
            try:
                g=self.get_generator(namespace,profile)
                if g['cfg'].get('data_signature')!=full['cfg'].get('data_signature'):raise ValueError('Ablation data signature differs from Full')
                if g['ref_hashes']!=full['ref_hashes']:raise ValueError('Ablation GT references differ from Full')
                if profile=='no_content':
                    q2,m2,p2=self.get_quality(g,'prawn_loo_no_content_quality')
                    item=p2['reranker_softnms'];pred,scores=item['pred'],item['scores']
                else:
                    stem=PurePosixPath(g['info']['best']['result_json_path']).stem
                    run=str(PurePosixPath(self.h.unique_file(g['snapshot']['ready']['files'],'/info.json')).parent)
                    base=run+'/quality_epochs/'+stem+'/joint_quality_softnms'
                    pred=self.native(g['snapshot'],base+'.json');metric=self.native(g['snapshot'],base+'.metrics.json')
                    expected=signature(dict(prediction=g['snapshot']['ready']['files'][base+'.json']['sha256'],
                                            refs=g['ref_hashes'],protocol=PROTOCOL))
                    if metric.get('signature')!=expected:raise ValueError('Joint scorer signature mismatch')
                    scores=complete_scores(metric['scores']);validate_predictions(g['raw'],pred['results'])
                    history=run+'/quality_epoch_history.json'
                    if history in g['snapshot']['ready']['files']:self.native(g['snapshot'],history)
                name='w_o_content' if profile=='no_content' else 'Joint'
                if pred.get('selection',{}).get('beta')!=fixed:raise ValueError('Ablation fixed beta differs from Full')
                row=self.row(name,pred,scores,fixed,profile!='no_content','joint' if profile=='joint' else 'two-stage')
                rows.append(row);folder=self.output/'results'/self.ds/profile
                dump(folder/(name+'.json'),pred);dump(folder/(name+'.metrics.json'),{'scores':scores})
                table(folder/'summary.tsv',[row])
                self.report['parts'][profile]=dict(complete=True,run_id=g['snapshot']['ready']['run_id'],
                    last_epoch=g['info']['last']['epoch'],best_epoch=g['epoch'],source_commit=g['snapshot']['ready'].get('source_commit'))
            except (ValueError,KeyError,FileNotFoundError) as error:self.report['missing'].append(profile+': '+str(error))
        table(self.output/'results'/self.ds/'ablation_summary.tsv',rows)
        self.report['full_control']=dict(checkpoint_sha256=full['checkpoint'],raw_sha256=full['raw_sha'],
                                        generator_run_id=full['snapshot']['ready']['run_id'])
        return rows

    def finish(self,rows):
        for record in self.store.records:
            path=self.output/'artifacts'/record['remote']
            if path.is_file():self.remember(path,record['remote'],'READY size + SHA256')
        self.report.update(complete=not self.report['missing'],training_started=False,neural_inference_started=False,
                           downloaded_bytes=self.store.used,created_utc=datetime.now(timezone.utc).isoformat())
        dump(self.output/'COLLECTION_REPORT.json',self.report)
        dump(self.output/'FILES.json',self.files)
        if self.report['missing']:
            print('INCOMPLETE_COLLECTION',json.dumps(self.report['missing'],ensure_ascii=False),flush=True)
            print('REPORT_READY',self.output/'COLLECTION_REPORT.json',flush=True)
            return None
        if {r['method'] for r in rows}!={'Full','w_o_quality','w_o_content','Joint'}|{'beta_'+format(b,'g').replace('.','p') for b in BETAS}:
            raise ValueError('Final ablation table is incomplete')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        target=self.output.parent/f'PRAWN_LOO_COMPLETE_{self.ds}_{stamp}.zip'
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name in self.files:archive.write(self.output/name,name)
            for folder in ('results','metadata'):
                for p in (self.output/folder).rglob('*'):
                    if p.is_file():archive.write(p,p.relative_to(self.output).as_posix())
            archive.write(self.output/'COLLECTION_REPORT.json','COLLECTION_REPORT.json')
            archive.write(self.output/'FILES.json','FILES.json')
            archive.write(Path(__file__),'export_source/collect_prawn_loo.py')
            critical=['train.py','opts.py','eval_utils_clip.py','ddvc/matcher.py','ddvc/ddvc_ret_encdec_clip_capQ.py',
                      'ddvc/CaptioningHead/gpt2.py','ddvc/CaptioningHead/proposal_mask.py','misc/caption_matching.py',
                      'misc/c1_recipe.py','misc/prawn_ablation.py','reranking/core.py','reranking/head.py','reranking/pipeline.py',
                      'reranking/online.py','tools/c1_pipeline.py','tools/prawn_ablation.py','tools/prawn_ablation_eval.py']
            for name in critical:
                path=self.source/name
                if path.is_file():archive.write(path,'project_source/'+name)
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None:raise ValueError('ZIP CRC validation failed')
        print('ALL_FOUR_ABLATIONS_COMPLETE',self.ds,flush=True)
        print('ZIP_READY',target,round(target.stat().st_size/2**20,1),'MiB',flush=True)
        return target


def load_helpers(source):
    if not (source/'tools/c1_rclone.py').is_file():raise FileNotFoundError('Existing PRAWN source checkout is required: '+str(source))
    sys.path[:0]=[str(source),str(source/'tools')];os.environ['PRAWN_ABLATION_VARIANT']='full'
    import c1_rclone
    import prawn_results_viewer
    return c1_rclone,prawn_results_viewer


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('dataset',choices=('yc2','anet'))
    parser.add_argument('--source',type=Path,default=Path('/marimo/PRAWN_Ablations/source'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--upload',action='store_true')
    args=parser.parse_args();source=args.source.resolve()
    auth,helpers=load_helpers(source)
    output=args.output or Path('/marimo/PRAWN_LOO_COMPLETE')/args.dataset
    output.mkdir(parents=True,exist_ok=True)
    helpers.ensure_rclone(output)
    config=Path('/marimo/secrets/rclone.conf')
    auth.restore_auth(Path('/marimo/KLTN_DRIVE_PRIVATE.env'),config,'kltnbackup')
    drive=auth.Drive(config,'kltnbackup','KLTNRUNS')
    collector=Collector(drive,source,output,args.dataset,helpers)
    rows=[]
    try:rows=collector.collect()
    except (ValueError,KeyError,FileNotFoundError) as error:collector.report['missing'].append('Full control: '+str(error))
    target=collector.finish(rows)
    if target is None:sys.exit(2)
    if args.upload:
        remote=drive.path(f'analysis/prawn_loo/{args.dataset}/exports/'+target.name)
        drive.call('copyto',str(target),remote,quiet=True)
        print('ZIP_ON_DRIVE',remote,flush=True)


if __name__=='__main__':main()
