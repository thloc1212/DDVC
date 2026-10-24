"""Validate the controlled global/time/frame caption experiment before launch."""
import json
from pathlib import Path


def validate(config, mode=None, full=False):
    expected_mode=mode or config['caption_conditioning_mode']
    if expected_mode not in ('global','frame_only','time_only','time_frame'):
        raise ValueError('Unknown caption conditioning mode')
    expected=dict(experiment_recipe='temporal_caption_prefix',caption_conditioning_mode=expected_mode,
                  batch_size=1,batch_size_for_eval=1,gradient_accumulation_steps=1,
                  precision='fp32',allow_tf32=False,caption_cost_chunk_size=16,nthreads=0,
                  seed=777,num_queries=50,gpt_layer=2,lr=5e-5,feature_dim=768,hidden_dim=512,
                  learned_proposal_mask=False,local_visual_attention=False,proposal_mask_loss_coef=0.,
                  mask_query_conditioning=False,mask_balanced_bce=False,corrected_query_contrastive=False,
                  caption_time_conditioning=expected_mode in ('time_only','time_frame'),
                  caption_frame_time_embedding=expected_mode in ('frame_only','time_frame'),
                  caption_time_bands=8,caption_time_hidden=128,caption_time_scale=1.,
                  caption_frame_time_scale=1.,caption_time_detach_boxes=True)
    for key,value in expected.items():
        if config.get(key)!=value:raise ValueError(f'Unexpected {key}: {config.get(key)!r}; expected {value!r}')
    if config.get('mask_prune_threshold',0)>=0:
        raise ValueError('Attention pruning must remain disabled')
    if full:
        counts=tuple(len(json.loads(Path(config[k]).read_text())) for k in ('train_caption_file','val_caption_file'))
        if counts!=(1237,436) or config['epoch']!=25:
            raise ValueError('Full recipe requires 1237/436 videos and 25 epochs; found '+str(counts))
    return expected_mode


if __name__=='__main__':
    import argparse,yaml
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--mode',required=True);p.add_argument('--full',action='store_true')
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    print('TEMPORAL_CAPTION_CONFIG_OK',validate(cfg,args.mode,args.full))
