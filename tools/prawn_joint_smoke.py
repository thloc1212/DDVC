"""Exercise matching and Joint gradients on a TRAIN head-fit video, including resume."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]


def main():
    import train
    from reranking.online import partition
    import caption_matching_smoke
    original=train.PropSeqDataset
    def dataset(*args,**kwargs):
        ds=original(*args,**kwargs)
        opt=args[-1] if args else kwargs['opt']
        if not opt.online_quality:raise ValueError('Joint smoke requires online quality')
        annotation=json.loads(Path(opt.quality_annotation_file).read_text(encoding='utf8'))
        fit,_=partition(annotation)
        eligible=sorted(set(fit)&set(ds.keys))
        if not eligible:raise ValueError('No TRAIN head-fit video in Joint smoke dataset')
        ds.keys=eligible[:1]
        return ds
    train.PropSeqDataset=dataset
    try:caption_matching_smoke.main()
    finally:train.PropSeqDataset=original


if __name__=='__main__':main()
