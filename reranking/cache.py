"""Immutable hash-verified shards; JSON ready marker is committed last."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    os.replace(temp,path)


def write_shard(path,videos,arrays,signature,split):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    offsets=[0]
    for x in arrays:offsets.append(offsets[-1]+len(x))
    payload=dict(features=np.concatenate(arrays),offsets=np.array(offsets,dtype=np.int64))
    labelled=split=='train'
    if labelled:
        width=max(len(v['gt_captions']) for v in videos)
        reward=np.zeros((offsets[-1],width),dtype=np.float32);qualities=[]
        for i,v in enumerate(videos):
            r=np.asarray(v.pop('reward'),dtype=np.float32)
            reward[offsets[i]:offsets[i+1],:r.shape[1]]=r;qualities.append(r.max(-1))
        payload.update(quality=np.concatenate(qualities),reward=reward)
    elif any('reward' in v or 'gt_captions' in v for v in videos):
        raise ValueError('Validation candidate cache must not contain GT quality labels')
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**payload)
    os.replace(temp,path)
    atomic_json(path.with_suffix('.json'),dict(signature=signature,split=split,sha256=sha(path),videos=videos))


def read_shard(path,signature,split):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
    if meta['signature']!=signature or meta['split']!=split or sha(path)!=meta['sha256']:
        raise ValueError('Candidate shard fingerprint/hash mismatch: '+str(path))
    with np.load(path,allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    if split=='val' and any(k in arrays for k in ('quality','reward')):
        raise ValueError('Validation label leakage in candidate cache')
    rows=[]
    for i,v in enumerate(meta['videos']):
        lo,hi=arrays['offsets'][i:i+2];r=dict(v,features=arrays['features'][lo:hi])
        if split=='train':r.update(quality=arrays['quality'][lo:hi],reward=arrays['reward'][lo:hi,:len(v['gt_captions'])])
        rows.append(r)
    return rows
