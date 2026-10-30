# AKT26 Homework 4 - CUDA: From Zero to Hero

[![Validation](https://github.com/atta-hilali/CUDA/actions/workflows/validate.yml/badge.svg)](https://github.com/atta-hilali/CUDA/actions/workflows/validate.yml)

CUDA C++ implementations of a matrix rank-1 update and small batched matrix
multiplication for Team HaT, AKT26 Session 4.

The assignment files stay in `3-ger/ger.cu` and `4-gemm/gemm.cu`, with their
original `main` functions. The examples in `0-hello-world`, `1-axpy`, and `2-dot`
are unchanged.

| Assignment | Operation | Implementation |
| --- | --- | --- |
| Matrix rank-1 update | `A = A + alpha * x * y^T` | 32 x 8 threads per block, contiguous rows per warp |
| Batched GEMM | `C[b] = A[b] * B[b]`, `1 <= n <= 32` | One 128-thread block per matrix, shared input matrices |

Both assignments use FP32 and column-major storage, check against the supplied
CPU references, and compare with cuBLAS. CUDA events measure device execution
after warm-up. Allocation and CPU/GPU transfers are outside the timed region.

## Build and run

```bash
git clone https://github.com/atta-hilali/CUDA.git
cd CUDA
```

Requirements: Linux, an NVIDIA GPU, a compatible driver, CUDA Toolkit 11 or
newer with cuBLAS, `make`, and Python 3 for the test/benchmark scripts.

```bash
make
./3-ger/ger 4000 2000 100
./4-gemm/gemm 16 1000 100
```

The optional arguments are:

```text
ger  [rows] [columns] [repeats]
gemm [matrix size: 1..32] [batch size] [repeats]
```

Defaults are `4000 2000 100` for GER and `16 1000 100` for GEMM. Each executable
prints GPU information, average time in microseconds, GFLOP/s, and numerical
error for the custom implementation and cuBLAS. It returns a nonzero exit code
on an API failure, invalid arguments, or failed accuracy check.

The default target is `sm_75` (T4), with PTX for forward compatibility. For a
specific GPU, set its architecture, for example:

```bash
make clean
make sm=80                     # A100
make sm=89                     # L4
make CUDADIR=/path/to/cuda sm=75
```

If CUDA's compiler is on `PATH` at another location, set `NVCC=nvcc`. The two
original per-directory makefiles also work independently.

For Google Colab, open the [GPU notebook](https://colab.research.google.com/github/atta-hilali/CUDA/blob/main/CUDA_HW4_Colab.ipynb), select a GPU
runtime, and run the cells. The notebook accepts the supplied source ZIP or a
GitHub repository URL, detects the GPU architecture, builds, tests, benchmarks,
and downloads the measured results.

## Correctness checks

```bash
make test
```

The GPU kernel tests include signed random inputs, irregular GER dimensions,
zero vectors, `alpha = 0`, all GEMM sizes from 1 to 32, identity/zero/ones cases,
and output guard values. The tests include the submitted `.cu` files directly.
The drivers additionally exercise both cuBLAS calls and reject invalid inputs.

With Compute Sanitizer installed, check device memory accesses and shared-memory
synchronization as well:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 ./tests/kernel_validation
compute-sanitizer --tool racecheck --error-exitcode 1 ./tests/kernel_validation
compute-sanitizer --tool synccheck --error-exitcode 1 ./tests/kernel_validation
```

## Performance measurements

```bash
make benchmark
```

This records 18 problem configurations (36 implementation measurements) under
`results/`, including raw program output, a CSV, GPU/driver details, compiler
version, commit, and working-tree status. Commit the files from an actual GPU
run explicitly:

```bash
git add -f results/*.csv results/*.json results/*.txt
```

The reported speedup is `cuBLAS time / custom time`: above 1 means the custom
kernel was faster on that case. Small kernels can be limited by launch and
submission overhead. No universal speedup is claimed.

## Automated checks

[GitHub Actions](https://github.com/atta-hilali/CUDA/actions) runs the host
kernel checks, validates Python/notebook syntax, and compiles both assignments
and the GPU test executable in NVIDIA's CUDA 12.6 development container. It
also verifies the programs reject invalid arguments. Standard hosted runners
have no NVIDIA GPU; these jobs do not run cuBLAS accuracy or GPU benchmarks.

## Validation status

- Both assignment executables and the GPU test executable compiled and linked
  with NVIDIA CUDA 12.6.85 and cuBLAS 12.6.4.1, targeting `sm_75` plus PTX.
- The actual kernel bodies passed 40 CPU checks, including every supported GEMM
  size. GEMM's shared-memory phase was modeled with host threads and a barrier.
- Nine invalid-argument checks passed on the compiled executables.
- **GPU execution, cuBLAS numerical results, sanitizer checks, and measured GPU
  performance are pending.** The development environment had no NVIDIA device
  or CUDA driver. CPU checks are not evidence of a successful GPU run.

See [results/README.md](results/README.md) for the exact verification boundary,
[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for indexing and timing details,
and [docs/DEMO_NOTES.md](docs/DEMO_NOTES.md) for a short code walkthrough.

## Sources

The starter files and CPU references are from
[Ahmad Abdelfattah's AKT26 repository](https://github.com/abdelfattah83/akt26_gpu_dev),
base commit `7110cb16d012b71682df318653710c4d67de7e6f`.
Assignment requirements are on slides 29, 31, and 33 of the Session 4 slides,
*A Brief Introduction to GPU Kernel Development using CUDA C++* (24 October 2026).

API references: [NVIDIA cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/)
and [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
