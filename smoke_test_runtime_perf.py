"""Precision boundary regression; --cuda validates the actual compiled extension."""
import argparse,importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cuda',action='store_true');args=parser.parse_args()
    root=Path(__file__).parent
    spec=importlib.util.spec_from_file_location('precision_op',root/'ddvc/ops/functions/ms_deform_attn_func.py')
    op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
    class MockExtension:
        @staticmethod
        def ms_deform_attn_forward(v,shapes,starts,loc,w,step):
            assert v.dtype==loc.dtype==w.dtype==torch.float32
            return v.sum().reshape(1)+loc.sum()+w.sum()
        @staticmethod
        def ms_deform_attn_backward(v,shapes,starts,loc,w,grad,step):
            return torch.ones_like(v)*grad,torch.ones_like(loc)*grad,torch.ones_like(w)*grad
    if not args.cuda:op.MSDA=MockExtension
    device='cuda' if args.cuda else 'cpu'
    v=torch.randn(1,4,2,4,device=device,dtype=torch.bfloat16,requires_grad=True)
    loc=torch.full((1,3,2,1,2,2),.5,device=device,dtype=torch.bfloat16,requires_grad=True)
    w=torch.full((1,3,2,1,2),.5,device=device,dtype=torch.bfloat16,requires_grad=True)
    shapes=torch.tensor([[2,2]],device=device);starts=torch.tensor([0],device=device)
    with torch.autocast(device_type=device,dtype=torch.bfloat16):out=op.ms_deform_attn_safe(v,shapes,starts,loc,w,1)
    assert out.dtype==torch.float32 and torch.isfinite(out).all()
    out.sum().backward()
    for x in (v,loc,w):assert x.grad.dtype==torch.bfloat16 and torch.isfinite(x.grad).all()
    if args.cuda:
        expected=op.ms_deform_attn_safe(v.detach().float(),shapes,starts,loc.detach().float(),w.detach().float(),1)
        torch.testing.assert_close(out,expected)
        torch.cuda.synchronize()
    spec=importlib.util.spec_from_file_location('profile_cli',root/'tools/molab_pipeline.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    profile=cli.resolve_profile(SimpleNamespace(hardware_profile='legacy-cvpr2018'))
    assert (profile.accumulation,profile.precision,profile.caption_chunk,profile.workers)==(1,'fp32',16,0)
    try:cli.resolve_profile(SimpleNamespace(hardware_profile='legacy-cvpr2018',precision='bf16'))
    except ValueError:pass
    else:raise AssertionError('BF16 silently changed legacy reproduction')
    print('PASS:',device,'FP32 deformable-attention boundary, BF16 gradient propagation, profile defaults/overrides')

if __name__=='__main__':main()
