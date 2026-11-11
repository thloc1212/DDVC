"""Verified old C1 data reuse, new paths, corrupted cache rejection; no network."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
from baseline_assets import import_c1_data
from test_c1_rclone import FileDrive,q


class AssetsTests(unittest.TestCase):
    def test_actual_drive_data_branch_with_python_safe_path_and_no_tools_pythonpath(self):
        script=Path(__file__).resolve().with_name('c1_rclone.py')
        tools=script.parent;root=tools.parent
        env=dict(os.environ,PYTHONSAFEPATH='1',PYTHONDONTWRITEBYTECODE='1',DDVC_RUN_VARIANT='baseline_quality')
        env['PYTHONPATH']=os.pathsep.join(p for p in env.get('PYTHONPATH','').split(os.pathsep)
                                        if p and Path(p).resolve() not in (tools,root))
        # Execute the same lazy data-import branch as a real worker. The control
        # removes only the bootstrap and must reproduce the original import error.
        for bootstrap in (False,True):
            with self.subTest(bootstrap=bootstrap),tempfile.TemporaryDirectory() as temp:
                code=f'''
import sys, types
from pathlib import Path
script=Path({str(script)!r})
source=script.read_text(encoding='utf8')
if not {bootstrap!r}:
    source=source.replace("sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]", "# old missing bootstrap")
module=types.ModuleType('worker_safe_path_probe');module.__file__=str(script)
sys.modules[module.__name__]=module
exec(compile(source,str(script),'exec'),module.__dict__)
module.WORK=Path({temp!r})
class Drive:
    def path(self,name):return str(module.WORK/'cloud'/name)
    def call(self,command,*args,**kwargs):
        if command=='mkdir':Path(args[0]).mkdir(parents=True,exist_ok=True);return ''
        if command=='lsf':return ''
        raise AssertionError(command)
    def optional_json(self,path):return None
assert module.pull_data(Drive(),'anet') is False
print('SAFE_PATH_DATA_IMPORT_PASS')
'''
                result=subprocess.run([sys.executable,'-P','-B','-c',code],cwd=temp,env=env,
                                      text=True,capture_output=True,timeout=60)
                if bootstrap:
                    self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                    self.assertIn('SAFE_PATH_DATA_IMPORT_PASS',result.stdout)
                else:
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn("No module named 'baseline_assets'",result.stderr)

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
