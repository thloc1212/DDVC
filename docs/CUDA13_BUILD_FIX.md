# Molab CUDA 13 build repair

Observed: native Torch cu130, nvcc 13.3, CCCL compatibility failure. Installing
CCCL alone repairs `nv/target` discovery but does not align the runtime headers
with nvcc. NVIDIA describes the same version-skew failure at
https://docs.dynamo.nvidia.com/dynamo/v1.3.0/getting-started/building-from-source.

Use `tools/molab_cuda130.py --work "$WORK"` with the existing Molab interpreter.
It stages five pinned NVIDIA wheels (about 105 MB download) under
`$WORK/toolchains/cuda130-pinned`. This is a private compiler/header directory,
not a venv. No global package or native Torch is replaced and no runtime
`LD_LIBRARY_PATH` override is introduced.

Then `source "$WORK/cuda130_env.sh"` before the existing build command.
This chooses compiler/runtime headers/CCCL 13.0. Original CUDA headers supply
cuSPARSE/cuBLAS declarations. The script first compiles a small header probe;
only a successful probe writes the environment file. The setup script selects
one CCCL tree rather than mixing fallback trees from different minor versions.

Re-run the import test and CUDA forward/backward smoke before full training.
No model, objective, optimizer, data selection or metric logic changes here.

Validation locally: checked pinned Linux wheel layouts and runtime-header
version parsing, tested preference for private CCCL over an alternate native
tree. No native Linux nvcc execution or Molab GPU build claimed locally.
Use the actual header/compiler probe and GPU smoke on Molab.

On a new Molab runtime, stage the compiler again as needed. Toolchain assets
are downloadable and excluded from Drive checkpoint snapshots.
