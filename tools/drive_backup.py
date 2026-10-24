"""Use an authenticated Molab fsspec handle; no invented Drive mount or copied credentials.

Two alternating slots: READY is written last. A failed upload leaves the other slot intact.
Only owned files in the selected slot are replaced; unrelated Drive content is never deleted.
"""
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path


class DriveBackup:
    def __init__(self, fs, work, remote_root, interval=120):
        self.fs=fs;self.work=Path(work).resolve();self.remote=remote_root.rstrip('/')
        self.interval=interval;self.stop_event=threading.Event();self.thread=None
        self.signature=None;self.lock=threading.Lock()
        logdir=self.work/'logs';logdir.mkdir(exist_ok=True)
        self.log=logging.getLogger('ddvc-drive-'+str(self.work))
        self.log.setLevel(logging.INFO)
        handler=logging.FileHandler(logdir/'drive_sync.log')
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        self.log.addHandler(handler)

    def _ready(self):
        result=[]
        for slot in [0,1]:
            marker=f'{self.remote}/slot-{slot}/READY.json'
            if self.fs.exists(marker):
                try:
                    with self.fs.open(marker,'r') as stream:result.append((slot,json.load(stream)))
                except (ValueError,OSError):
                    self.log.warning('Ignore incomplete READY marker in slot %s',slot)
        return sorted(result,key=lambda p:p[1]['created_ns'])

    def sync_once(self, force=False):
        with self.lock:
            checkpoints=sorted((self.work/'runs').glob('*/model-last.pth'))
            signature=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in checkpoints)
            if not checkpoints or (not force and signature==self.signature):return False
            ready=self._ready();slot=1-ready[-1][0] if ready else 0
            destination=f'{self.remote}/slot-{slot}'
            self.fs.makedirs(destination,exist_ok=True)
            marker=f'{destination}/READY.json'
            if self.fs.exists(marker):self.fs.rm(marker)
            candidates=[]
            for folder in ['runs','prepared','logs','provenance']:
                parent=self.work/folder
                if parent.exists():
                    for p in parent.rglob('*'):
                        if not p.is_file() or 'backup' in p.parts or 'tf_summary' in p.parts:continue
                        if p.name=='drive_sync.log':continue  # controller keeps this file open during restore
                        if p.name.endswith('.tmp') or p.suffix=='.pid':continue
                        # Full video feature maps are downloadable and large. Keep banks, references and configs.
                        if p.suffix=='.pth' and p.name not in ('model-last.pth','model-best.pth'):continue
                        candidates.append(p)
            manifest={'created_ns':time.time_ns(),'source_root':str(self.work),'files':[]}
            self.log.info('BACKUP_START slot=%s files=%s',slot,len(candidates))
            with tempfile.TemporaryDirectory(prefix='ddvc-backup-',dir=self.work) as tmp:
                staged=[]
                # Pin checkpoint inodes before network operations; training uses atomic replace.
                for p in candidates:
                    rel=p.relative_to(self.work);stable=Path(tmp)/rel;stable.parent.mkdir(parents=True,exist_ok=True)
                    if p.suffix=='.pth':
                        try:os.link(p,stable)
                        except OSError:shutil.copy2(p,stable)
                    else:shutil.copy2(p,stable)
                    staged.append((rel,stable))
                for rel,stable in staged:
                    target=f'{destination}/{rel.as_posix()}'
                    self.fs.makedirs(target.rsplit('/',1)[0],exist_ok=True)
                    if self.fs.exists(target):self.fs.rm(target)
                    self.fs.put(str(stable),target)
                    manifest['files'].append({'path':rel.as_posix(),'bytes':stable.stat().st_size})
                with self.fs.open(marker,'w') as stream:json.dump(manifest,stream)
            self.signature=signature
            self.log.info('BACKUP_COMPLETE slot=%s created_ns=%s',slot,manifest['created_ns'])
            return True

    def start(self):
        if self.thread and self.thread.is_alive():return
        self.stop_event.clear()
        def loop():
            while not self.stop_event.is_set():
                try:self.sync_once()
                except Exception:self.log.exception('BACKUP_FAILED; previous completed slot remains available')
                self.stop_event.wait(self.interval)
        self.thread=threading.Thread(target=loop,name='ddvc-drive-backup',daemon=True);self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=5)

    def restore_latest(self):
        if self.thread and self.thread.is_alive():raise RuntimeError('Stop backup worker before restore')
        ready=self._ready()
        if not ready:raise FileNotFoundError('No completed Drive backup READY.json found')
        slot,manifest=ready[-1]
        for entry in manifest['files']:
            target=(self.work/entry['path']).resolve()
            if not target.is_relative_to(self.work):raise ValueError('Unsafe remote path')
            target.parent.mkdir(parents=True,exist_ok=True)
            partial=target.with_name(target.name+'.download')
            self.fs.get(f"{self.remote}/slot-{slot}/{entry['path']}",str(partial))
            if partial.stat().st_size!=entry['bytes']:raise IOError('Incomplete download: '+entry['path'])
            os.replace(partial,target)
        # Local absolute paths may change between Molab notebooks/sessions.
        old=manifest['source_root']
        for p in (self.work/'prepared').glob('*/config.yml'):
            p.write_text(p.read_text().replace(old,str(self.work)))
        self.log.info('RESTORE_COMPLETE slot=%s files=%s',slot,len(manifest['files']))
        return manifest
