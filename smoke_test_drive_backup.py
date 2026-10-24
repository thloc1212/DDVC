"""Offline transport/rotation test using a local stand-in for Molab's fsspec handle."""
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path


def main():
    spec=importlib.util.spec_from_file_location('drive_backup',Path(__file__).parent/'tools/drive_backup.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class FS:
        fail=False
        def __init__(self,root):self.root=Path(root)
        def p(self,path):return self.root/path
        def exists(self,path):return self.p(path).exists()
        def makedirs(self,path,exist_ok=False):self.p(path).mkdir(parents=True,exist_ok=exist_ok)
        def rm(self,path):self.p(path).unlink()
        def open(self,path,mode):return self.p(path).open(mode,encoding='utf8')
        def put(self,source,target):
            if self.fail and target.endswith('model-last.pth'):raise IOError('simulated upload interruption')
            shutil.copy2(source,self.p(target))
        def get(self,source,target):shutil.copy2(self.p(source),target)
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp);work=base/'work';run=work/'runs/demo';run.mkdir(parents=True)
        prepared=work/'prepared/demo';prepared.mkdir(parents=True)
        (prepared/'config.yml').write_text('save_dir: '+str(work/'runs'))
        (prepared/'features.pth').write_bytes(b'exclude downloadable features')
        (prepared/'bank.npy').write_bytes(b'include bank')
        (run/'model-last.pth').write_bytes(b'first stable checkpoint')
        fs=FS(base/'remote');backup=module.DriveBackup(fs,work,'DDVC-test')
        assert backup.sync_once() and not backup.sync_once()
        (run/'model-last.pth').write_bytes(b'second stable checkpoint')
        assert backup.sync_once()
        fs.fail=True;(run/'model-last.pth').write_bytes(b'failed third upload')
        try:backup.sync_once()
        except IOError:pass
        else:raise AssertionError('Upload failure fixture did not run')
        fs.fail=False
        restored=base/'restored';restored.mkdir()
        controller=module.DriveBackup(fs,restored,'DDVC-test')
        manifest=controller.restore_latest()
        assert (restored/'runs/demo/model-last.pth').read_bytes()==b'second stable checkpoint'
        assert not (restored/'prepared/demo/features.pth').exists()
        assert (restored/'prepared/demo/bank.npy').exists()
        assert str(restored) in (restored/'prepared/demo/config.yml').read_text()
        for logger in [backup.log,controller.log]:
            for handler in logger.handlers:handler.close()
        print('PASS: backup slots, unchanged-checkpoint skip, interrupted upload preserves previous slot, restore and path remap')


if __name__=='__main__':main()
