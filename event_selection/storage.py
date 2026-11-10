"""Restore only audited frozen candidate caches + small annotation files from Drive."""
import json
from pathlib import Path,PurePosixPath
import shutil
import tempfile
import re
from reranking.cache import sha,atomic_json

RERANK_SOURCES={'d75fbe759654996d90bfac979b2a7f333bbd254f','1b26dcbea3953937c3def8405eff8de747739091'}
KNOWN_RUNS={'yc2':'20261109T092734Z_05a32edc0bd5'}


def valid_id(value):
    if not value or not re.fullmatch('[A-Za-z0-9_-]+',value):raise ValueError('Invalid Drive run ID')
    return value


def candidates(drive,remote):
    result=[]
    for slot in (0,1):
        try:
            meta=drive.read_json(remote+'/slot-'+str(slot)+'/READY.json')
            result.append((meta['created_ns'],remote+'/slot-'+str(slot),meta))
        except (RuntimeError,ValueError,KeyError,OSError):continue
    if not result:raise FileNotFoundError('No completed recovery READY.json under '+remote)
    return sorted(result,reverse=True)


def fetch(drive,helpers,remote,meta,target,names):
    if not names or any(n not in meta['files'] for n in names):raise ValueError('Snapshot lacks required files')
    subset=dict(files={n:meta['files'][n] for n in names})
    helpers.fetch_verified(drive,remote,subset,target,lambda n:n in names)


def prepare(work,dataset,drive,helpers,rank_run=None):
    work=Path(work);target=work/'inputs'/dataset
    if (target/'READY.json').is_file():
        receipt=json.loads((target/'READY.json').read_text())
        for n,v in receipt['files'].items():
            p=target/n
            if not p.is_file() or p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:raise ValueError('Existing cached input changed: '+str(p))
        if rank_run and rank_run!=receipt['source_reranker_run']:raise ValueError('Already prepared a different source run')
        print('INPUT_CACHE_ALREADY_VERIFIED',target,flush=True);return receipt
    if target.exists():raise FileExistsError('Incomplete input destination; preserve it and inspect: '+str(target))
    rank_run=rank_run or KNOWN_RUNS.get(dataset)
    if not rank_run:
        parent=drive.path('runs/ddvc_c1_rerank/'+dataset)
        try:listing=drive.call('lsf',parent,'--dirs-only','--max-depth','1',quiet=True)
        except RuntimeError:
            raise FileNotFoundError('No saved frozen-reranker cache for '+dataset+
                '. Complete/recover the matching dataset cache first; YC2 features must not be reused for ANET.') from None
        raise ValueError('Specify --rank-run for '+dataset+'; available frozen-reranker runs:\n'+listing)
    rank_run=valid_id(rank_run);parent=drive.path('runs/ddvc_c1_rerank/'+dataset+'/'+rank_run+'/recovery')
    work.mkdir(parents=True,exist_ok=True);failures=[]
    for _,remote,meta in candidates(drive,parent):
        if meta.get('dataset')!=dataset or meta.get('run_id')!=rank_run or meta.get('source_commit') not in RERANK_SOURCES:
            raise ValueError('Unexpected frozen-reranker snapshot identity')
        with tempfile.TemporaryDirectory(dir=work,prefix='.cached-input-') as temp:
            staged=Path(temp);cache=staged/'reranker';prefix='runs/'+dataset+'/'
            selected=[n for n in meta['files'] if n.startswith(prefix+'cache/') and Path(n).suffix in ('.json','.npz')]
            required=[prefix+n for n in ('manifest.json','head-best.pt','gpu_smoke.json',
                     'evaluation/baseline_softnms.metrics.json','evaluation/reranker_softnms.metrics.json')]
            required+=['provenance/rerank_'+dataset+'.json'];selected+=required
            try:fetch(drive,helpers,remote,meta,cache,selected)
            except (ValueError,RuntimeError,OSError) as error:
                failures.append(str(error));print('SOURCE_SLOT_INVALID_TRY_OLDER',str(error),flush=True);continue
            manifest=json.loads((cache/prefix/'manifest.json').read_text())
            if not json.loads((cache/prefix/'gpu_smoke.json').read_text()).get('passed'):
                raise ValueError('Original cached C1 GPU smoke did not pass')
            session=json.loads((cache/('provenance/rerank_'+dataset+'.json')).read_text())
            base=session.get('base_c1_drive_session')
            if not base or base['dataset']!=dataset or base.get('remote')!='kltnbackup' or base.get('root')!='KLTNRUNS':
                raise ValueError('Frozen run does not name a verified C1 Drive backup')
            base_id=valid_id(base['run_id']);base_parent=drive.path('runs/ddvc_c1/'+dataset+'/'+base_id+'/recovery')
            base_dest=staged/'base';base_ok=False
            for _,base_remote,bmeta in candidates(drive,base_parent):
                if bmeta.get('dataset')!=dataset or bmeta.get('run_id')!=base_id or bmeta.get('source_commit')!=base['source_commit']:
                    raise ValueError('C1 base snapshot identity changed')
                run=dataset+'_c1_e'+('15' if dataset=='yc2' else '6')+'_b1';prep='prepared/'+run+'/'
                cfg_name=prep+'config.yml';cm_name=prep+'c1_manifest.json';checkpoint='runs/'+run+'/model-best.pth'
                if bmeta['files'].get(checkpoint,{}).get('sha256')!=manifest['base_checkpoint_sha256']:
                    raise ValueError('C1 backup checkpoint differs from the frozen candidate model')
                try:
                    fetch(drive,helpers,base_remote,bmeta,base_dest,[cfg_name,cm_name])
                    import yaml
                    cfg=yaml.safe_load((base_dest/cfg_name).read_text());cm=json.loads((base_dest/cm_name).read_text())
                    if sha(base_dest/cfg_name)!=manifest['base_config_sha256'] or cfg['data_signature']!=manifest['data_signature']:
                        raise ValueError('Original C1 config/data signature changed')
                    if cfg.get('proposal_mask_loss_coef')!=0 or cfg.get('eval_selection_method')!='softnms':raise ValueError('Not the audited C1 recipe')
                    def relative(path):return PurePosixPath(path).relative_to(PurePosixPath(base['work'])).as_posix()
                    train=relative(cfg['train_caption_file']);val=relative(cfg['val_caption_file'])
                    refs=[relative(p) for p in cfg['gt_file_for_eval']];para=[relative(p) for p in cfg['gt_file_for_para_eval']]
                    names=list(dict.fromkeys([train,val,*refs,*para]))
                    if any(not n.startswith(prep) or not n.endswith('.json') for n in names):raise ValueError('Unexpected annotation path')
                    fetch(drive,helpers,base_remote,bmeta,base_dest,names)
                    for k,n in [('train_caption_file',train),('val_caption_file',val)]:
                        if sha(base_dest/n)!=cm['input_sha256'][k]:raise ValueError('Annotation hash differs from C1 manifest')
                    base_ok=True;break
                except (ValueError,RuntimeError,OSError) as error:
                    failures.append(str(error));print('BASE_SLOT_INVALID_TRY_OLDER',str(error),flush=True)
            if not base_ok:raise FileNotFoundError('No verified small annotation/reference snapshot: '+str(failures[-2:]))
            annotations=json.loads((base_dest/train).read_text());validation=json.loads((base_dest/val).read_text())
            if set(annotations)&set(validation):raise ValueError('Train/validation IDs overlap')
            if set(manifest['fit_ids'])|set(manifest['dev_ids'])!=set(annotations):raise ValueError('Original head train split changed')
            scores={}
            for original,mode in [('baseline_softnms','C1_softnms'),('reranker_softnms','quality_softnms')]:
                scores[mode]=json.loads((cache/prefix/('evaluation/'+original+'.metrics.json')).read_text())['scores']
                if not all(k in scores[mode] for k in ('CIDEr','METEOR','Bleu_4','soda_c','Precision','Recall','MetaScore')):
                    raise ValueError('Recover the completed metricsfix run, not a partial-metric snapshot')
            # Runtime status/PID is not part of immutable model/data identity.
            (cache/('provenance/rerank_'+dataset+'.json')).unlink()
            files=helpers.inventory(staged)
            receipt=dict(format='frozen-c1-relation-count-input-v1',dataset=dataset,source_reranker_run=rank_run,
                         source_base_run=base_id,source_C1_checkpoint_sha256=manifest['base_checkpoint_sha256'],
                         reranker_prefix='reranker/'+prefix.rstrip('/'),train_annotation='base/'+train,val_annotation='base/'+val,
                         references=['base/'+n for n in refs],paragraph_references=['base/'+n for n in para],
                         validation_video_metadata={v:dict(duration=a['duration']) for v,a in validation.items()},
                         queries=int(cfg['num_queries']),baseline_scores=scores,files=files)
            # Receipt contains stable file identities, not mutable backup timestamps/slot numbers.
            atomic_json(staged/'READY.json',receipt);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(staged),target)
            print('FROZEN_INPUT_RESTORED',target,'bytes',sum(v['bytes'] for v in files.values()),
                  'no_DDVC_checkpoint_or_CLIP_archive_downloaded',flush=True)
            return receipt
    raise FileNotFoundError('No hash-valid completed frozen candidate snapshot: '+str(failures[-2:]))
