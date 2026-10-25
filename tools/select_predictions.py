"""Re-select saved raw candidates without loading a model or training."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from misc.proposal_selection import select_proposals


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--method',choices=['topk','softnms'],default='softnms')
    p.add_argument('--sigma',type=float,default=.5);p.add_argument('--alpha',type=float,default=1.)
    p.add_argument('--merge',action='store_true');p.add_argument('--merge-iou',type=float,default=.7)
    p.add_argument('--merge-text',type=float,default=.9)
    p.add_argument('--merge-consecutive',action='store_true')
    p.add_argument('--consecutive-gap',type=float,default=2.)
    p.add_argument('--consecutive-max-span',type=float,default=60.)
    p.add_argument('--consecutive-max-events',type=int,default=3)
    args=p.parse_args();src=Path(args.input);dst=Path(args.output)
    if src.resolve()==dst.resolve() or dst.exists():raise FileExistsError('Use a new output file; preserve raw candidates/results')
    raw=json.loads(src.read_text());out={k:v for k,v in raw.items() if k in ['version','external_data']}
    out['results']={vid:select_proposals(rows,int(rows[0]['pred_event_count']) if rows else 0,
        args.method,args.sigma,args.alpha,2.,args.merge,args.merge_iou,args.merge_text,
        args.merge_consecutive,args.consecutive_gap,args.consecutive_max_span,args.consecutive_max_events) for vid,rows in raw['results'].items()}
    out['selection']=dict(method=args.method,sigma=args.sigma,merge=args.merge,
        merge_iou=args.merge_iou,merge_text_similarity=args.merge_text,
        merge_consecutive=args.merge_consecutive,consecutive_gap=args.consecutive_gap,
        consecutive_max_span=args.consecutive_max_span,consecutive_max_events=args.consecutive_max_events)
    dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(json.dumps(out),encoding='utf8')
    print('SELECTED_PREDICTIONS_SAVED',dst,'videos',len(out['results']),'; metrics must be recomputed')


if __name__=='__main__':main()
