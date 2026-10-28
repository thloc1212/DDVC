"""CPU-only preparation/provenance regression for both datasets and all ablations."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch
import yaml

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('concept_pipeline_test',ROOT/'tools/concept_pipeline.py')
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)


def main():
    base=json.loads((ROOT/'tools/kaggle_mask_recipe.json').read_text())
    base.update(learned_proposal_mask=True,dedup_loss_coef=0.,vocab_size=3)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        # Two same-named CLIP files must never share a download cache across datasets.
        calls=[]
        def fake_prepare(command,**kwargs):
            calls.append(command)
            target=root/'cache_probe'/'prepared'/command[command.index('--run-name')+1]/'config.yml'
            target.parent.mkdir(parents=True,exist_ok=True);target.write_text('prepared: true\n')
        with patch.object(pipeline.subprocess,'run',side_effect=fake_prepare):
            for dataset in ('anet','yc2'):
                pipeline.ensure_base(root/'cache_probe',dataset,None)
        caches={call[call.index('--data-root')+1] for call in calls}
        assert caches=={str(root/'cache_probe'/'assets'/ds) for ds in ('anet','yc2')}
        for dataset in ('anet','yc2'):
            assets=root/dataset/'assets';assets.mkdir(parents=True)
            counts=(10009,4917) if dataset=='anet' else (1237,436)
            train={f'train_{i}':{'sentences':['woman washing trainingonly spoon']*6} for i in range(counts[0])}
            val={f'val_{i}':{'sentences':['heldout validationonly object']} for i in range(counts[1])}
            for name,data in [('train',train),('val',val),('para',{k:'heldout' for k in val}),
                              ('dict',{'word_to_ix':{'a':1,'b':2,'UNK':3}})]:
                (assets/f'{name}.json').write_text(json.dumps(data))
            (assets/'features.pth').write_bytes(b'not-loaded-by-metadata-test')
            bank=assets/'bank'/dataset/'clip';bank.mkdir(parents=True)
            for suffix in ('scene_sentences','clip_token_embeds','scene_videoID'):
                (bank/f'{dataset}_{suffix}.npy').write_bytes(b'not-loaded-by-metadata-test')
            cfg=copy.deepcopy(base)
            cfg.update(dataset=dataset,bank_type=[dataset],bank_folder='clip',num_queries=10 if dataset=='anet' else 50,
                train_caption_file=str(assets/'train.json'),val_caption_file=str(assets/'val.json'),
                dict_file=str(assets/'dict.json'),visual_feature_folder=[str(assets/'features.pth')],
                bank_path=str(assets/'bank'),gt_file_for_eval=[str(assets/'val.json')],
                gt_file_for_para_eval=[str(assets/'para.json')])
            cfg['data_signature']=hashlib.sha256(b''.join(Path(cfg[k]).read_bytes()
                for k in ('train_caption_file','val_caption_file','dict_file'))).hexdigest()
            source=assets/'source.yml';source.write_text(yaml.safe_dump(cfg))
            for variant in ('full','aux','control'):
                work=root/dataset/'work'
                args=SimpleNamespace(work=work,dataset=dataset,variant=variant,source_config=source)
                pipeline.prepare(args)
                path=work/'prepared'/pipeline.run_name(dataset,variant)/'config.yml'
                result=pipeline.check(path)
                assert result['epoch']==pipeline.default_epochs(dataset) and result['lr']==base['lr']
                assert result['event_concepts']==(variant!='control')
                assert result['concept_condition_prefix']==(variant=='full')
                vocab=json.loads(Path(result['concept_vocab_path']).read_text())
                assert 'trainingonly' in vocab['concepts'] and 'heldout' not in vocab['concepts']
                assert 'validationonly' not in vocab['concepts']
                assert json.loads(Path(result['train_caption_file']).read_text())==train
                pipeline.prepare(args)  # Explicitly idempotent after successful preparation.
                feature=Path(result['visual_feature_folder'][0]);feature.unlink()
                pipeline.check(path,require_features=False)  # Drive backups omit downloadable features.
                try:pipeline.check(path)
                except FileNotFoundError:pass
                else:raise AssertionError('Missing staged features were accepted for training')
                pipeline.stage_file(assets/'features.pth',feature)
                result['lr']=.123;path.write_text(yaml.safe_dump(result))
                try:pipeline.check(path)
                except ValueError:pass
                else:raise AssertionError('Unrelated LR drift was accepted')
                result['lr']=base['lr'];path.write_text(yaml.safe_dump(result))
                assert source.read_text()==yaml.safe_dump(cfg)
            if dataset=='yc2':
                old_args=SimpleNamespace(work=work,dataset=dataset,variant='full',source_config=source,epochs=8)
                pipeline.prepare(old_args)
                old_path=work/'prepared'/pipeline.run_name(dataset,'full',8)/'config.yml'
                assert pipeline.check(old_path)['epoch']==8  # Existing 8-epoch paths remain usable explicitly.
    print('PASS concept pipeline: 2 datasets x 3 variants, ANET 8 / YC2 15, separate dataset caches, '
          'legacy YC2 8 selectable, train-only vocabulary, source unchanged, recipe drift rejected')


if __name__=='__main__':main()
