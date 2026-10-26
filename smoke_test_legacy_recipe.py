"""Verify archived model identity and reject silent Action-Aware/runtime recipe changes."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

EXPECTED = {'ddvc/CaptioningHead/gpt2.py': '3ebf69a0bccaa226b4be326aa899cbc65c886df683d80796d368b44283cfaf26', 'ddvc/CaptioningHead/proposal_mask.py': 'ce4a5bb9aaf4031d9fffb7ef844c88f5d790989b45d92c1b050a1d59bcfdffec', 'ddvc/ddvc_ret_encdec_clip_capQ.py': 'c038cbf887a6169469bda3fe601fcc2cc6d4d72e4efac208d7a42114e11bf39b'}


def main():
    root=Path(__file__).parent
    for rel,expected in EXPECTED.items():
        data=(root/rel).read_bytes().replace(b'\r\n',b'\n')
        if rel.endswith('ddvc_ret_encdec_clip_capQ.py'):
            data=data.replace(b"    if getattr(args, 'dedup_loss_coef', 0.) > 0:\n        weight_dict['loss_dedup'] = args.dedup_loss_coef\n",b'')
            # Explicit numeric repair: empty quantized GT slices now use one
            # valid frame. Apart from this exact call, retain legacy identity.
            data=data.replace(b'from .temporal_pooling import pool_temporal_segments\n',b'')
            data=data.replace(b'        taget_feature=pool_temporal_segments(dt["video_tensor"].squeeze(0), [dt["gt_featstamps"][gt_id] for gt_id in indices[0][1].tolist()])\n',
                b'        taget_feature=torch.vstack([torch.mean(dt["video_tensor"].squeeze(0)[dt["gt_featstamps"][gt_id][0]:dt["gt_featstamps"][gt_id][1]],dim=0) for gt_id in indices[0][1].tolist()])\n')
        data=data.replace(b"GPT2Tokenizer.from_pretrained('/DATA_DISK/plm_cache/gpt2')",b"GPT2Tokenizer.from_pretrained('gpt2')")
        assert hashlib.sha256(data).hexdigest()==expected, 'Legacy model source changed: '+rel
    spec=importlib.util.spec_from_file_location('legacy_cli',root/'tools/molab_pipeline.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    profile=cli.resolve_profile(SimpleNamespace(hardware_profile='legacy-cvpr2018'))
    assert (profile.accumulation,profile.precision,profile.caption_chunk,profile.workers,profile.bank_batch)==(1,'fp32',16,0,64)
    assert not profile.allow_tf32 and not profile.pin_memory
    for override in [dict(accumulation=8),dict(precision='bf16'),dict(allow_tf32=True),dict(caption_chunk=64)]:
        try:cli.resolve_profile(SimpleNamespace(hardware_profile='legacy-cvpr2018',**override))
        except ValueError:pass
        else:raise AssertionError('Silent legacy profile change: '+str(override))
    print('PASS caption/mask and baseline backbone match original ZIP except explicit dedup registration and empty-GT pooling repair; legacy profile guards')

if __name__=='__main__':main()
