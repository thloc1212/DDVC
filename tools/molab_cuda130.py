"""Stage a private CUDA 13.0 build toolchain; never replace native Molab Torch.

The five NVIDIA wheels are installed with --target and --no-deps. This directory
is a compiler/header asset, not a Python environment, and never joins sys.path.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

PACKAGES = ['nvidia-cuda-nvcc==13.0.88', 'nvidia-cuda-crt==13.0.88',
            'nvidia-nvvm==13.0.88', 'nvidia-cuda-runtime==13.0.96',
            'nvidia-cuda-cccl==13.0.85']


def runtime_header_version(path):
    match = re.search(r'^\s*#\s*define\s+CUDART_VERSION\s+(\d+)',
                      Path(path).read_text(), re.MULTILINE)
    if not match:
        raise RuntimeError('Cannot read CUDART_VERSION from '+str(path))
    return int(match.group(1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', required=True)
    args = p.parse_args()
    import torch
    from torch.utils.cpp_extension import CUDA_HOME
    if sys.platform != 'linux' or os.uname().machine != 'x86_64':
        raise RuntimeError('This recipe is for the observed Linux x86_64 Molab runtime')
    if torch.version.cuda != '13.0':
        raise RuntimeError('This recipe expects native Torch CUDA 13.0, found '+str(torch.version.cuda))
    if not CUDA_HOME:
        raise RuntimeError('Cannot locate the original CUDA headers')
    original = Path(CUDA_HOME).resolve()
    work = Path(args.work).expanduser().resolve()
    target = work/'toolchains/cuda130-pinned'
    home = target/'nvidia/cu13'
    marker = target/'DDVC_TOOLCHAIN_READY.json'
    if marker.exists():
        original = Path(json.loads(marker.read_text())['original_cuda'])
    uv = shutil.which('uv')
    if not uv:
        raise RuntimeError('uv is required; do not replace the Molab interpreter')
    if not marker.exists():
        if target.exists():
            raise FileExistsError('Incomplete toolchain directory; inspect it before retry: '+str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        print('Downloading ~105 MB of pinned NVIDIA compiler/header wheels to',target,flush=True)
        subprocess.run([uv,'pip','install','--python',sys.executable,'--target',str(target),
                        '--no-deps','--index-url','https://pypi.org/simple',*PACKAGES],check=True)
    if runtime_header_version(home/'include/cuda_runtime_api.h') != 13000:
        raise RuntimeError('Private runtime headers are not CUDA 13.0')
    nvcc = home/'bin/nvcc'
    cccl = home/'include/cccl'
    if not (cccl/'nv/target').is_file():
        raise FileNotFoundError(cccl/'nv/target')
    version = subprocess.check_output([str(nvcc),'--version'],text=True)
    if not re.search(r'release\s+13\.0\b',version):
        raise RuntimeError('Private compiler is not CUDA 13.0: '+version)
    print(version,flush=True)
    # torch CUDAContext also needs cuBLAS/cuSPARSE headers from native CUDA 13.0.
    extra = original/'include'
    if not (extra/'cusparse.h').is_file():
        raise FileNotFoundError('Original CUDA cuSPARSE headers missing: '+str(extra))
    lib = home/'lib'
    if not (lib/'libcudart.so').exists():
        shared = lib/'libcudart.so.13'
        if not shared.exists():
            raise FileNotFoundError(shared)
        (lib/'libcudart.so').symlink_to(shared.name)
    env = os.environ.copy()
    for key in ('CPATH','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS'):
        env.pop(key,None)
    env['CPATH'] = str(extra)
    probe = target/'header_probe.cu'
    probe.write_text('#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n'
                     '#include <thrust/complex.h>\n#include <cusparse.h>\n'
                     '#if CUDART_VERSION != 13000\n#error Wrong runtime headers\n#endif\n'
                     'int main(){ return 0; }\n')
    subprocess.run([str(nvcc),'-std=c++17','-I'+str(cccl),'-c',str(probe),
                    '-o',str(target/'header_probe.o')],env=env,check=True)
    manifest = dict(packages=PACKAGES,torch=torch.__version__,torch_cuda=torch.version.cuda,
                    original_cuda=str(original),cuda_home=str(home),native_interpreter=sys.executable)
    marker.write_text(json.dumps(manifest,indent=2))
    envfile = work/'cuda130_env.sh'
    envfile.write_text(
        'unset CPLUS_INCLUDE_PATH C_INCLUDE_PATH PYTORCH_NVCC NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS\n'
        f'export CUDA_HOME={shlex.quote(str(home))}\n'
        f'export CUDA_PATH={shlex.quote(str(home))}\n'
        f'export DDVC_CCCL_INCLUDE={shlex.quote(str(cccl))}\n'
        f'export CPATH={shlex.quote(str(extra))}\n'
        f'export PATH={shlex.quote(str(home/"bin"))}:"$PATH"\n')
    print('HEADER_COMPILER_PROBE_PASS',flush=True)
    print('Next: source '+shlex.quote(str(envfile)),flush=True)


if __name__ == '__main__':
    main()
