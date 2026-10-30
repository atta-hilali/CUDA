# Validation and measured results

Completed on 30 October 2026 in a CPU-only Linux development environment:

| Check | Status |
| --- | --- |
| Compile/link GER and GEMM with NVIDIA nvcc 12.6.85 | Passed |
| Compile/link the GPU kernel-validation executable | Passed |
| Actual kernel-body checks on CPU, all GEMM sizes 1..32 | 40 cases passed |
| Invalid arguments on compiled assignment executables | 9 cases passed |
| CUDA execution on an NVIDIA device | Pending |
| cuBLAS numerical comparisons on a GPU | Pending |
| Compute Sanitizer memcheck/racecheck/synccheck | Pending |
| Measured GPU time, GFLOP/s, and speedup | Pending |

The compiler linked against cuBLAS 12.6.4.1. Build targets contain `sm_75`
code and `compute_75` PTX. Compiler output is in `build_validation.txt`;
the host test output is in `host_validation.txt`.

A runtime probe with a 1x1 problem correctly failed at the first checked CUDA
allocation because the environment had no CUDA-capable driver/device. It did
not produce a valid result or timing. No GPU pass or speedup is inferred.

Run `make test` and `make benchmark` on a GPU, or use the Colab notebook, to
produce the outstanding evidence. Generated benchmark files are named with a
UTC timestamp and include raw outputs, a measurement CSV, and hardware,
compiler, and source metadata. Generated files are ignored by default until
you explicitly choose to commit them.
