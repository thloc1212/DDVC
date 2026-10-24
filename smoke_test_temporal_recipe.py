"""Offline checks for controlled caption ablations; no Torch/GPU imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace


def main():
    root=Path(__file__).parent
    def load(rel,name):
        spec=importlib.util.spec_from_file_location(name,root/rel)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
    cli=load('tools/molab_pipeline.py','temporal_cli_test')
    recipe=load('tools/temporal_recipe.py','temporal_recipe_test')
    profile=cli.resolve_profile(SimpleNamespace())
    assert (profile.accumulation,profile.precision,profile.caption_chunk,profile.workers)==(1,'fp32',16,0)
    with tempfile.TemporaryDirectory() as tmp:
        train,val=Path(tmp)/'train.json',Path(tmp)/'val.json'
        train.write_text(json.dumps({str(i):{} for i in range(1237)}))
        val.write_text(json.dumps({str(i):{} for i in range(436)}))
        for mode in ['global','frame_only','time_only','time_frame']:
            config=dict(experiment_recipe='temporal_caption_prefix',batch_size=1,batch_size_for_eval=1,
                gradient_accumulation_steps=1,precision='fp32',allow_tf32=False,caption_cost_chunk_size=16,
                nthreads=0,seed=777,num_queries=50,gpt_layer=2,lr=5e-5,feature_dim=768,hidden_dim=512,learned_proposal_mask=False,
                local_visual_attention=False,proposal_mask_loss_coef=0.,mask_query_conditioning=False,
                mask_balanced_bce=False,corrected_query_contrastive=False,mask_prune_threshold=-1.,
                epoch=25,train_caption_file=str(train),val_caption_file=str(val),**cli.conditioning_options(mode))
            assert recipe.validate(config,mode,full=True)==mode
            wrong=config.copy();wrong['mask_prune_threshold']=.05
            try:recipe.validate(wrong,mode)
            except ValueError:pass
            else:raise AssertionError('Pruning silently enabled')
            wrong=config.copy();wrong['caption_time_conditioning']=not config['caption_time_conditioning']
            try:recipe.validate(wrong,mode)
            except ValueError:pass
            else:raise AssertionError('Wrong ablation flags accepted')
    print('PASS all four controlled modes, full recipe checks and accidental pruning/mode changes rejected')


if __name__=='__main__':main()
