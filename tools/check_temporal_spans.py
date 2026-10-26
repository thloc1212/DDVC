"""Inspect actual feature lengths for empty GT slices, without running a model."""
import argparse
import json
import math
from pathlib import Path
import torch
import yaml


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text())
    features=torch.load(cfg['visual_feature_folder'][0],map_location='cpu',mmap=True,weights_only=False)
    records=json.loads(Path(cfg['train_caption_file']).read_text());examples=[];events=0
    for vid,record in records.items():
        length=features[vid].shape[0]
        if length<1:raise ValueError('Zero feature frames: '+vid)
        for i,(s,e) in enumerate(record['timestamps']):
            events+=1
            start=max(0,min(int(s/record['duration']*length),length-1))
            end=max(0,min(int(e/record['duration']*length),length-1))
            if end<=start:examples.append(dict(video=vid,event=i,seconds=[s,e],frames=[start,end],feature_length=length))
    path=args.config.parent/'empty_temporal_spans.json'
    path.write_text(json.dumps(dict(events=events,empty_slices=len(examples),examples=examples),indent=2))
    print('EMPTY_GT_FRAME_SLICES',len(examples),'/',events,flush=True)
    print(json.dumps(examples[:10],indent=2),flush=True)
    print('REPORT',path,flush=True)


if __name__=='__main__':main()
