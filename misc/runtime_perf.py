"""Explicit precision controls and measured per-epoch GPU performance records."""
from contextlib import nullcontext
import json
from pathlib import Path
import torch


def amp_context(opt):
    if getattr(opt,'precision','fp32') == 'bf16':
        return torch.autocast(device_type=opt.device,dtype=torch.bfloat16)
    return nullcontext()


def configure_runtime(opt):
    precision=getattr(opt,'precision','fp32')
    if precision not in ('fp32','bf16'):raise ValueError('Only fp32 or bf16 is supported')
    tf32=getattr(opt,'allow_tf32',False)
    torch.backends.cuda.matmul.allow_tf32=tf32
    torch.backends.cudnn.allow_tf32=tf32
    if precision=='bf16':
        if opt.device!='cuda' or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Requested BF16 needs native CUDA BF16 support; select fp32 explicitly otherwise')
        # An available device is not proof that this Torch binary contains compatible kernels.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad(),amp_context(opt):
            x=torch.randn(32,32,device='cuda');y=x@x
            if not torch.isfinite(y).all():raise RuntimeError('CUDA BF16 matmul probe is non-finite')
        torch.cuda.synchronize()
    return {'precision':precision,'allow_tf32':tf32,'cuda':torch.version.cuda,
            'torch':torch.__version__}


def vram_metrics(device):
    if device!='cuda' or not torch.cuda.is_available():return {'peak_allocated_gib':0.,'peak_reserved_gib':0.}
    return {'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}


def append_performance(folder, row):
    with (Path(folder)/'performance.jsonl').open('a',encoding='utf8') as stream:
        stream.write(json.dumps(row)+'\n');stream.flush()
