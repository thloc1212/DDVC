"""Fresh C1+online-quality experiment, with independent Drive namespace and frozen recipe."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import yaml
import c1_pipeline as base
from reranking.cache import atomic_json
from reranking.online import partition

MODE=os.environ.get('DDVC_QUALITY_MODE','joint')
if MODE not in ('joint','detached'):raise ValueError('Invalid quality gradient mode')
ORIGINAL_MAKE=base.make_config;ORIGINAL_CHECK=base.check;ORIGINAL_SMOKE=base.make_smoke
ORIGINAL_FINGERPRINT=base.fingerprint


def run_name(dataset):return f'{dataset}_online_quality_{MODE}_e{base.epochs(dataset)}_b1'
def config_path(work,dataset):return Path(work)/'prepared'/run_name(dataset)/'config.yml'


def make_config(source,work,dataset):
    cfg=ORIGINAL_MAKE(source,work,dataset)
    cfg.update(online_quality=True,quality_gradient_mode=MODE,quality_loss_coef=.1,
               quality_head_lr=.001,quality_generation_chunk=8,
               quality_annotation_file=source['train_caption_file'])
    return cfg


def check(path,require_features=True):
    cfg=ORIGINAL_CHECK(path,require_features=require_features)
    expected=dict(online_quality=True,quality_gradient_mode=MODE,quality_loss_coef=.1,
                  quality_head_lr=.001,quality_generation_chunk=8)
    for k,v in expected.items():
        if cfg.get(k)!=v:raise ValueError('Online quality recipe drift: '+k)
    if base.sha(cfg['quality_annotation_file'])!=base.sha(cfg['train_caption_file']):
        raise ValueError('Quality labels must use identical full TRAIN annotation')
    print('ONLINE_QUALITY_CONFIG_OK',cfg['dataset'],MODE,'fresh C1 + head; quality loss .1; annotation TRAIN-only',flush=True)
    return cfg


def make_smoke(cfg,target):
    ORIGINAL_SMOKE(cfg,target)
    folder=target/'smoke';path=folder/'config.yml'
    tiny=yaml.safe_load(path.read_text())
    annotation=json.loads(Path(cfg['train_caption_file']).read_text())
    fit,_=partition(annotation)
    train=folder/'train_caption_file.json'
    atomic_json(train,{v:annotation[v] for v in fit[:2]})
    tiny['train_caption_file']=str(train)
    tiny['data_signature']=hashlib.sha256(b''.join(Path(tiny[k]).read_bytes() for k in
                                  ('train_caption_file','val_caption_file','dict_file'))).hexdigest()
    path.write_text(yaml.safe_dump(tiny,sort_keys=False),encoding='utf8')


def fingerprint(path):
    result=ORIGINAL_FINGERPRINT(path)
    for name in ('reranking/online.py','reranking/core.py','reranking/head.py',
                 'tools/online_quality_pipeline.py','tools/online_quality_evaluate.py','tools/online_quality_smoke.py'):
        result[str(ROOT/name)]=base.sha(ROOT/name)
    return result


def smoke(args):
    path=config_path(args.work,args.dataset);cfg=check(path)
    base.logged([sys.executable,'-u','tools/prepare_gpt2_tokenizer.py'],args.work/'logs'/(cfg['id']+'.tokenizer.log'))
    command=[sys.executable,'-u','tools/online_quality_smoke.py','--config',str(path if getattr(args,'checkpoint',False) else path.parent/'smoke/config.yml')]
    if getattr(args,'checkpoint',False):command+=['--checkpoint',str(args.work/'runs'/cfg['id']/'model-last.pth')]
    log=args.work/'logs'/(cfg['id']+'.smoke.log')
    base.logged(command,log)
    if 'ONLINE_QUALITY_GRADIENT_PASS '+MODE not in log.read_text(encoding='utf8'):
        raise RuntimeError('Real-model smoke did not verify the quality gradient route')
    base.dump(args.work/'provenance'/(cfg['id']+'_smoke.json'),fingerprint(path))


base.run_name=run_name;base.config_path=config_path;base.make_config=make_config
base.check=check;base.make_smoke=make_smoke;base.fingerprint=fingerprint;base.smoke=smoke


def import_c1_data(drive,helpers,dataset):
    """Reuse immutable old C1 assets, relocating config paths after byte verification.

    Never upload another CLIP bundle or touch the original cache pointer/weights.
    """
    parent=drive.path('data/'+dataset+'/c1-base-assets-v1')
    pointer=drive.optional_json(parent+'/LATEST.json')
    if pointer is None:return False
    generation=pointer.get('generation','')
    if not re.fullmatch('[0-9a-f]{32}',generation):raise ValueError('Invalid C1 data generation')
    remote=parent+'/generations/'+generation;meta=drive.read_json(remote+'/READY.json')
    if meta.get('protocol')!='c1-base-assets-v1' or meta.get('dataset')!=dataset or meta.get('asset_code')!=helpers.data_identity():
        raise ValueError('Original C1 data recipe does not match online experiment')
    old_work=PurePosixPath(meta['work']);old_base=old_work/'prepared'/(dataset+'_c1_base_assets')
    if not old_work.is_absolute() or '..' in old_work.parts:raise ValueError('Invalid original C1 root')
    destination=helpers.base_path(dataset)
    if destination.exists():raise FileExistsError('Partial local C1 data; preserve and inspect '+str(destination))
    with tempfile.TemporaryDirectory(dir=helpers.WORK,prefix='.online-data-') as temp:
        staged=Path(temp)/'base'
        helpers.fetch_verified(drive,remote,meta,staged,
            lambda n:Path(n).suffix in {'.json','.yml','.yaml','.npy','.pth'})
        original=yaml.safe_load((staged/'config.yml').read_text())
        if original.get('dataset')!=dataset or original.get('bank_type')!=[dataset]:raise ValueError('Dataset mismatch')
        for k in ('train_caption_file','val_caption_file','dict_file','bank_path'):
            if not PurePosixPath(original[k]).is_relative_to(old_base):raise ValueError('Data path escapes C1 base bundle')
        if len(original['visual_feature_folder'])!=1 or not PurePosixPath(original['visual_feature_folder'][0]).is_relative_to(old_base):
            raise ValueError('Feature path escapes C1 base bundle')
        actual=hashlib.sha256(b''.join((staged/PurePosixPath(original[k]).relative_to(old_base)).read_bytes()
                 for k in ('train_caption_file','val_caption_file','dict_file'))).hexdigest()
        if actual!=original['data_signature']:raise ValueError('C1 annotation signature differs')
        def relocate(x):
            if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
            if isinstance(x,list):return [relocate(v) for v in x]
            if isinstance(x,str) and (x==str(old_work) or x.startswith(str(old_work)+'/')):
                return str(helpers.WORK)+x[len(str(old_work)):]
            return x
        (staged/'config.yml').write_text(yaml.safe_dump(relocate(original),sort_keys=False),encoding='utf8')
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(staged),destination)
    helpers.write_json(helpers.WORK/'provenance'/('imported_data_'+dataset+'.json'),
                       dict(source=remote,original_manifest=meta,weights_imported=False))
    print('ONLINE_REUSED_VERIFIED_C1_ASSETS_NO_WEIGHTS',dataset,destination,flush=True)
    return True


def export(args):
    work=args.work.resolve();name=run_name(args.dataset)
    target=work/f'ONLINE_QUALITY_ANALYSIS_{args.dataset}_{MODE}_{time.strftime("%Y%m%d_%H%M%S")}.zip'
    extensions={'.json','.jsonl','.yml','.yaml','.log','.txt','.tsv','.csv'}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for folder in (work/'runs'/name,work/'prepared'/name,work/'logs',work/'provenance'):
            for p in folder.rglob('*'):
                if p.is_file() and p.suffix in extensions and 'backup' not in p.parts:z.write(p,p.relative_to(work).as_posix())
        import subprocess
        for n in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines():
            if Path(n).suffix in {'.py','.sh','.md','.txt','.json'} and (ROOT/n).is_file():z.write(ROOT/n,'source/'+n)
        z.writestr('EXPORT_MANIFEST.json',json.dumps(dict(dataset=args.dataset,mode=MODE,
            source_commit=base.helper('molab_pipeline').revision(),weights_included=False,online_quality=True)))
    print('ZIP_READY',target,flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','train','smoke','report','export','restage','finalize'):
        p=sub.add_parser(name);p.add_argument('--work',type=Path,required=True);p.add_argument('--dataset',choices=['yc2','anet'],required=True)
        if name=='prepare':p.add_argument('--source-config',type=Path)
        if name in ('prepare','restage'):p.add_argument('--features',type=Path)
        if name=='train':p.add_argument('--resume',action='store_true')
        if name=='smoke':p.add_argument('--checkpoint',action='store_true')
    args=parser.parse_args()
    if args.command=='export':export(args)
    elif args.command=='finalize':
        from online_quality_evaluate import finalize
        finalize(args)
    else:getattr(base,args.command)(args)


if __name__=='__main__':main()
