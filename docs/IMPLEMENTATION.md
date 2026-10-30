# Implementation notes

## 1. Rank-1 update

For each output entry,

```text
A[row, col] = A[row, col] + alpha * x[row] * y[col]
index = col * m + row
```

The launch uses `block = (32, 8)` and a grid covering the rows and columns.
CUDA's x index varies fastest, so the 32 lanes in a warp handle consecutive
rows in the same column. Matrix reads and writes are contiguous; the warp
also reads consecutive `x` values and one shared `y` value.

Bounds checks handle partial blocks. The grid's y dimension is capped at
65,535 blocks, and a column stride covers any remaining columns. `size_t`
protects linear offsets from 32-bit integer overflow. Each matrix entry has
exactly one writer, so the operation needs neither atomics nor a barrier.

The kernel uses `fmaf(alpha * x[row], y[col], A[index])`. Fused multiplication
and addition can round differently from the FP32 CPU expression, so the
comparison uses the assignment's norm and `1e-6` tolerance rather than bitwise
equality.

GER performs approximately two operations per matrix entry under the starter's
BLAS-style operation count. It reads and writes the full matrix while doing
little arithmetic, so memory bandwidth is a likely limit. This is an
expectation from the algorithm, not a measured result.

## 2. Batched GEMM

Each matrix occupies `n*n` consecutive elements; batch `b` starts at
`base = b*n*n`. The product is

```text
C[base + col*n + row] = sum over k of
    A[base + k*n + row] * B[base + col*n + k]
```

The launch uses one block per matrix and `block = (32, 4)`, or four warps.
All 128 threads cooperate to load both matrices into shared memory. The
arrays are indexed as `[column][row]` and padded to `[32][33]`; together they
use 8,448 bytes. Padding changes the shared-memory column stride from 32 to
33 words. With this mapping, the compute phase already accesses consecutive
rows of A and broadcasts B within each warp.

A block-wide barrier follows the loads. Every thread reaches the barrier,
including lanes with `row >= n`. Only afterwards are the inactive rows
excluded. This avoids divergent barrier execution for sizes below 32.

Thread `(row, lane_group)` computes columns `lane_group`, `lane_group+4`,
and so on. A warp therefore writes contiguous rows in one output column.
The inputs are reused from shared memory for each multiply-accumulate.
The design supports every `n` from 1 to 32 without a power-of-two assumption.
For tiny matrices it leaves some threads idle; this keeps the kernel simple
and is a possible subject for later measurement and tuning.

## 3. cuBLAS

GER calls `cublasSger` with vector increments of 1 and leading dimension `m`.
The initial matrix is restored before each implementation's correctness run.

GEMM calls `cublasSgemmStridedBatched` with:

- no transposition for A or B;
- `m = n`, `n = n`, and `k = n`;
- leading dimensions `lda = ldb = ldc = n`;
- strides of `n*n` **elements**, not bytes;
- `alpha = 1`, `beta = 0`, and the requested batch count.

`beta = 0` means C is overwritten; its original contents need not be copied
to the device. Pedantic math mode keeps the library comparison in FP32 rather
than allowing reduced-precision multiplication. The handle uses host scalar
pointers and the default stream, as do the custom launches and CUDA events.

## 4. Timing and accuracy

Each implementation has an untimed warm-up and a device synchronization.
Events then enclose repeated launches/calls on the default stream. The stop
event is synchronized before reading elapsed time.

```text
average_us = total_event_ms * 1000 / repeats
GFLOP/s    = operation_count / (average_us * 1000)
GER count  = 2*m*n
GEMM count = 2*n*n*n*batch_size
speedup    = cuBLAS_average_us / custom_average_us
```

Allocation, input transfers, output transfers, CPU references, and warm-up
are excluded. Host submission overhead can still leave gaps between tiny
kernels inside an event interval, so these are repeated-launch timings, not
a claim about ideal isolated-kernel latency.

GER's timed repetitions accumulate into A, as in the starter. Correctness
uses a separate reset followed by exactly one update. GEMM overwrites C on
every repetition. The two assignments therefore have different state behavior.

The supplied error definitions are retained:

- GER: `||A_gpu - A_cpu||_F / ||A_cpu||_F`.
- GEMM: maximum over batches of `||C_gpu - C_cpu||_F / (n * ||C_cpu||_F)`.

A zero reference norm returns zero only for an exactly zero difference;
otherwise it returns infinity. Non-finite data cannot silently pass through
an error reduction. Both custom and cuBLAS errors must be finite and at most
`1e-6`. The additional kernel tests use double-precision expected values and
an element-wise absolute/relative tolerance for signed random data and
analytical identity, zero, and ones cases.

## 5. Reproducibility and limitations

Input initialization in both assignment drivers is unchanged. The additional
GPU tests use fixed random seeds and guarded output allocations. The benchmark
script saves raw output alongside the measured CSV and environment metadata.
There are no hard-coded performance or accuracy results.

`tests/host_check.py` extracts the submitted kernel bodies and runs them on the
CPU, modeling GEMM's shared-memory barrier with C++20 threads. It checks
indexing and arithmetic, but cannot validate CUDA scheduling, device memory,
code generation, cuBLAS, or performance. The full GPU test and Compute
Sanitizer commands remain necessary on a CUDA-enabled system.
