"""Use the real DDVC smoke on a TRAIN head-fit video, including checkpoint resume."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--checkpoint')
    args=parser.parse_args()
    import train
    from reranking.online import partition
    from molab_pipeline import smoke
    original=train.PropSeqDataset
    def head_fit_video(*a,**kw):
        ds=original(*a,**kw);opt=a[-1] if a else kw['opt']
        annotation=json.loads(Path(opt.quality_annotation_file).read_text())
        fit,_=partition(annotation);eligible=set(fit)&set(ds.keys)
        if not eligible:raise ValueError('Smoke dataset has no TRAIN head-fit example')
        ds.keys=[sorted(eligible)[0]]
        return ds
    train.PropSeqDataset=head_fit_video
    try:
        smoke(argparse.Namespace(config=args.config,checkpoint=args.checkpoint,benchmark_steps=1))
    finally:train.PropSeqDataset=original
    from online_quality_evaluate import scorer_smoke
    scorer_smoke(Path(args.config).parent/'official_scorer_smoke')


if __name__=='__main__':main()
