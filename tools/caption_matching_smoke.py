"""Run the existing actual GPU smoke with the matching schedule active."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--benchmark-steps',type=int,default=1)
    args=parser.parse_args()
    import train
    from molab_pipeline import smoke
    original_build=train.build

    def build(opt):
        model,criterion,contrastive,postprocessors=original_build(opt)
        original_forward=model.forward
        def forward(dt,*positional,**kwargs):
            # Only the smoke input's epoch changes. Checkpoint config is untouched.
            dt['matching_epoch']=opt.caption_match_warmup_epochs+opt.caption_match_ramp_epochs-1
            result=original_forward(dt,*positional,**kwargs)
            if model.training and opt.caption_match_enabled and opt.caption_match_weight>0:
                stats=getattr(criterion.matcher,'caption_match_stats',None)
                if not stats or stats['weight']!=opt.caption_match_weight or 'caption_match_reward' not in result[0]:
                    raise RuntimeError('Active matching not exercised by actual GPU forward')
                print('SMOKE_CAPTION_MATCH_STATS',stats,flush=True)
            elif not model.training and 'caption_match_reward' in result[0]:
                raise RuntimeError('Training matching must not run at inference')
            return result
        model.forward=forward
        return model,criterion,contrastive,postprocessors
    train.build=build
    try:
        smoke(args)
    finally:
        train.build=original_build


if __name__=='__main__':main()
