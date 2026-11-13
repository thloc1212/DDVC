"""Regression checks for seed isolation, verified data relocation and stage ordering."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import yaml
import c1_pipeline
import prawn_seed2016 as chain
from reranking.pipeline import TRAINING_SEED, HEAD_SPLIT_SEED, FIXED_BETAS
from test_c1_rclone import FileDrive, q


class Seed2016Tests(unittest.TestCase):
    def test_recipe_changes_training_seed_and_preserves_method(self):
        for dataset, epochs, queries in [('yc2', 15, 50), ('anet', 6, 10)]:
            cfg = c1_pipeline.make_config({}, Path('/marimo/PRAWN_Seed2016'), dataset)
            self.assertEqual((cfg['seed'], cfg['epoch'], cfg['num_queries']), (2016, epochs, queries))
            self.assertEqual((cfg['batch_size'], cfg['gradient_accumulation_steps']), (1, 1))
            self.assertEqual((cfg['precision'], cfg['allow_tf32']), ('fp32', False))
            self.assertEqual((cfg['lr'], cfg['weight_decay']), (5e-5, 1e-4))
            self.assertTrue(cfg['learned_proposal_mask'])
            self.assertEqual(cfg['proposal_mask_loss_coef'], 0.)
            self.assertEqual(cfg['start_from'], '')
            self.assertIsNone(cfg['pretrain'])
        self.assertEqual((TRAINING_SEED, HEAD_SPLIT_SEED), (2016, 777))
        self.assertEqual(FIXED_BETAS, {'yc2': 1., 'anet': 6.})

    def test_existing_shared_data_relocates_without_weights_or_remote_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / 'work'
            work.mkdir()
            drive = FileDrive(root / 'cloud')
            old = '/marimo/DDVC_C1/prepared/yc2_c1_base_assets'
            parent = drive.root / 'data/yc2/c1-base-assets-v1'
            source = parent / 'generations' / ('a' * 32)
            source.mkdir(parents=True)
            cfg = dict(dataset='yc2', bank_type=['yc2'], bank_path=old+'/bank',
                       train_caption_file=old+'/train.json', val_caption_file=old+'/val.json',
                       dict_file=old+'/vocabulary.json', visual_feature_folder=[old+'/clipvitl14.pth'])
            for name, content in [('train.json', '{"train":{}}'), ('val.json', '{"val":{}}'),
                                  ('vocabulary.json', '{}'), ('clipvitl14.pth', 'FEATURE_BYTES')]:
                (source / name).write_text(content, encoding='utf8')
            cfg['data_signature'] = hashlib.sha256(b''.join((source / name).read_bytes() for name in
                ('train.json', 'val.json', 'vocabulary.json'))).hexdigest()
            (source / 'bank').mkdir()
            (source / 'bank/emb.npy').write_bytes(b'BANK_BYTES')
            (source / 'config.yml').write_text(yaml.safe_dump(cfg), encoding='utf8')
            q.write_json(source / 'READY.json', dict(protocol=q.DATA_PROTOCOL, dataset='yc2',
                work='/marimo/DDVC_C1', asset_code={'recipe': 'same'}, files=q.inventory(source)))
            q.write_json(parent / 'LATEST.json', dict(generation='a'*32))
            before = q.inventory(source)
            with patch.object(q, 'WORK', work), patch.object(q, 'data_identity', return_value={'recipe': 'same'}):
                self.assertTrue(q.pull_data(drive, 'yc2'))
                self.assertTrue(q.pull_data(drive, 'yc2'))
                self.assertEqual(q.base_feature('yc2').read_bytes(), b'FEATURE_BYTES')
                local = yaml.safe_load((q.base_path('yc2') / 'config.yml').read_text())
                self.assertTrue(Path(local['train_caption_file']).is_relative_to(q.base_path('yc2')))
                self.assertFalse((work / 'runs').exists())
            self.assertEqual(before, q.inventory(source))

    def test_fresh_chain_finishes_generator_backup_before_quality(self):
        for dataset in ('yc2', 'anet'):
            with tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary)
                (work / 'logs').mkdir()
                fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
                calls = []
                with patch.object(chain, 'WORK', work), patch.object(chain, 'QUALITY', work/'frozen_quality'), \
                     patch.dict(sys.modules, {'fcntl': fake_fcntl}), \
                     patch.object(chain, 'command', side_effect=lambda *args: calls.append(('command', args))), \
                     patch.object(chain, 'wait_stage', side_effect=lambda *args, **kwargs: calls.append(('wait', args, kwargs))), \
                     patch.object(chain, 'read_state', side_effect=[{}, {}]), \
                     patch('builtins.print'), patch.object(Path, 'read_text', return_value='summary'):
                    chain.worker(SimpleNamespace(dataset=dataset, resume=False))
                self.assertEqual(calls[0], ('command', ('c1_drive.sh', 'start', dataset)))
                self.assertEqual(calls[1][0], 'wait')
                self.assertEqual(calls[2][1][:3], ('c1_rerank.sh', 'start', dataset))
                self.assertEqual(calls[3], ('wait', (dataset,), {'quality': True}))


if __name__ == '__main__':
    unittest.main()
