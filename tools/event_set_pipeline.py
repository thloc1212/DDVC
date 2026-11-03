"""Fresh V1 attention-counter / learned duplicate-selection configurations."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
# Molab may enable Python safe-path mode, which omits the script directory.
# Resolve this sibling by filename rather than relying on sys.path/PYTHONPATH.
_helper_spec=importlib.util.spec_from_file_location('ddvc_event_set_molab_helper', ROOT/'tools'/'molab_pipeline.py')
_helper=importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_helper)
revision=_helper.revision


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--dataset',choices=['yc2','anet'],required=True)
    p.add_argument('--variant',choices=['control','counter','duplicate','both'],default='both')
    p.add_argument('--mode',choices=['smoke','subset','full'],default='full')
    p.add_argument('--source-config',type=Path)
    p.add_argument('--epochs',type=int)
    a=p.parse_args()
    work=a.work.resolve()
    epochs=a.epochs or (2 if a.mode=='smoke' else 15 if a.dataset=='yc2' else 8)
    if epochs<1:raise ValueError('Epochs must be positive')
    run=f'{a.dataset}_{a.mode}_event_set_{a.variant}_e{epochs}_b1'
    destination=work/'prepared'/run/'config.yml'
    if destination.exists():raise FileExistsError(destination)
    source=a.source_config
    if source is None:
        base=f'{a.dataset}_{a.mode}_v1_event_set_assets'
        source=work/'prepared'/base/'config.yml'
        if not source.exists():
            if a.dataset=='yc2':
                command=[sys.executable,'-u','tools/molab_pipeline.py','prepare',
                         '--mode',a.mode,'--run-name',base,'--epochs',str(epochs),
                         '--experiment','no_dedup_same_merges','--data-root',str(work/'assets'/'yc2'),
                         '--prepared-root',str(work/'prepared'),'--output-root',str(work/'runs'),
                         '--log-root',str(work/'logs')]
            else:
                command=[sys.executable,'-u','tools/activitynet_pipeline.py','prepare',
                         '--mode',a.mode,'--run-name',base,'--work',str(work),
                         '--data-root',str(work/'assets'/'anet')]
            subprocess.run(command,cwd=ROOT,check=True)
    source=source.resolve()
    cfg=yaml.safe_load(source.read_text(encoding='utf8'))
    if cfg.get('event_concepts') or cfg.get('dedup_loss_coef',0) or not cfg.get('learned_proposal_mask'):
        raise ValueError('Source must be V1 learned mask, concepts OFF, dedup loss OFF')
    domain=cfg.get('bank_type')
    if domain not in ([a.dataset], a.dataset):raise ValueError(f'Wrong dataset bank: {domain}')
    for key in ['train_caption_file','val_caption_file','dict_file']:
        if not Path(cfg[key]).is_file():raise FileNotFoundError(cfg[key])
    counter=a.variant in ('counter','both')
    duplicate=a.variant in ('duplicate','both')
    cfg.update(id=run,exp_name=run,fixed_run_id=True,epoch=epochs,cfg_path='',
               save_dir=str(work/'runs'),start_from='',pretrain=None,
               only_ft_captioner=False,only_ft_class_head=False,
               attention_counter=counter,duplicate_head=duplicate,
               count_ordinal_coef=.2,duplicate_pair_coef=.1,
               duplicate_assign_iou=.5,duplicate_assign_margin=.1,
               duplicate_selection=duplicate,duplicate_merge_threshold=.9,
               eval_selection_method='softnms',eval_softnms_sigma=.5,
               eval_merge_duplicates=True,eval_merge_consecutive=True,
               min_epoch_when_save=0,save_checkpoint_every=1,
               experiment_recipe='v1_event_set_'+a.variant)
    cfg.pop('current_lr',None)
    if (work/'runs'/run).exists():raise FileExistsError(work/'runs'/run)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf8')
    manifest=dict(source_config=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_commit=revision(),
                  counter=counter,duplicate=duplicate,epochs=epochs,
                  note='Duplicate policy replaces legacy merges when enabled; keep-best, no interval union. Fresh weights.')
    destination.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('CONFIG_READY',destination,flush=True)


if __name__=='__main__':main()
