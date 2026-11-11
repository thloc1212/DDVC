"""Verified old C1 data reuse, new paths, corrupted cache rejection; no network."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
from baseline_assets import import_c1_data
from test_c1_rclone import FileDrive,q


class AssetsTests(unittest.TestCase):
    def test_shared_c1_inputs_are_verified_and_relocated_without_weights(self):
        for ds in ('yc2','anet'):
            with self.subTest(dataset=ds),tempfile.TemporaryDirectory() as temp:
                work=Path(temp)/'new';work.mkdir();drive=FileDrive(Path(temp)/'cloud')
                old_work='/marimo/DDVC_C1';old=f'{old_work}/prepared/{ds}_c1_base_assets'
                parent=Path(drive.path(f'data/{ds}/c1-base-assets-v1'))
                generation='a'*32;source=parent/'generations'/generation;source.mkdir(parents=True)
                for name,value in [('train.json','{"train":{}}'),('val.json','{"val":{}}'),('vocab.json','{}')]:
                    (source/name).write_text(value,encoding='utf8')
                (source/'bank').mkdir();(source/'bank/memory.npy').write_bytes(b'train-caption-bank')
                (source/'clipvitl14.pth').write_bytes(b'frozen-input-features')
                cfg=dict(dataset=ds,bank_type=[ds],bank_path=old+'/bank',train_caption_file=old+'/train.json',
                         val_caption_file=old+'/val.json',dict_file=old+'/vocab.json',visual_feature_folder=[old+'/clipvitl14.pth'])
                cfg['data_signature']=hashlib.sha256(b''.join((source/n).read_bytes() for n in ('train.json','val.json','vocab.json'))).hexdigest()
                (source/'config.yml').write_text(yaml.safe_dump(cfg),encoding='utf8')
                meta=dict(protocol='c1-base-assets-v1',dataset=ds,work=old_work,asset_code=q.data_identity(),files=q.inventory(source))
                q.write_json(source/'READY.json',meta);q.write_json(parent/'LATEST.json',dict(generation=generation))
                with patch.object(q,'WORK',work):
                    self.assertTrue(import_c1_data(drive,q,ds))
                    dest=q.base_path(ds);new=yaml.safe_load((dest/'config.yml').read_text())
                    self.assertTrue(Path(new['train_caption_file']).is_relative_to(dest))
                    self.assertEqual(q.base_feature(ds).read_bytes(),b'frozen-input-features')
                    self.assertFalse((work/'runs').exists())
                    shutil.rmtree(dest);(source/'clipvitl14.pth').write_bytes(b'corrupted')
                    with self.assertRaises(ValueError):import_c1_data(drive,q,ds)
                    self.assertFalse(dest.exists())


if __name__=='__main__':unittest.main()
