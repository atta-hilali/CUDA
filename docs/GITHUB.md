# Repository and submission

Public repository: https://github.com/atta-hilali/CUDA

Repository description:

> AKT26 Homework 4 (Team HaT): FP32 CUDA rank-1 update and small batched GEMM, with cuBLAS comparisons, correctness tests, and a Colab benchmark workflow.

The source is stored directly in `3-ger/ger.cu` and `4-gemm/gemm.cu`.
Their original main functions and the starter example directories are retained.
The original source and base commit are credited in the main README.

GitHub Actions checks compilation, CPU kernel-body behavior, source structure,
and invalid command-line arguments. Review its actual status in the Actions
tab; this does not establish GPU correctness or GPU performance.

Before submission:

1. Open the Colab notebook linked in the README and choose a GPU runtime.
2. Run all required cells, including correctness tests and benchmarks.
3. Download the results archive and add the measured results to this repository.
4. Review the raw test outputs and error values before submitting.
5. Submit `https://github.com/atta-hilali/CUDA` in the homework form.

No GPU timing is reported until it has actually been measured.
