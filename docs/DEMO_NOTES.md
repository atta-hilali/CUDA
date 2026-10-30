# Code walkthrough

1. **What does column-major mean?**
   The row changes fastest in memory. Entry `(row, col)` is stored at
   `col * leading_dimension + row`, not `row * columns + col`.

2. **What does rank-1 mean?**
   `x*y^T` is an outer product. Every output entry receives
   `alpha*x[row]*y[col]`.

3. **Why 32 threads along x?**
   A warp has 32 lanes. Assigning those lanes to consecutive rows gives
   contiguous matrix accesses in column-major storage.

4. **Why are there bounds checks?**
   Rounded-up grids include threads beyond the actual dimensions. They must
   not read or write matrix data.

5. **Why one block per GEMM matrix?**
   `n <= 32`, so both complete inputs fit in shared memory. Blocks can work
   independently on different batch entries.

6. **Why shared memory?**
   Each A or B value is used in several dot products. Loading the input once
   into shared memory avoids repeatedly fetching it from global memory.

7. **Why is the barrier before the row check?**
   All threads cooperate in loading inputs. They must all finish before any
   thread uses another thread's data. Moving the barrier inside `row < n`
   would make it unsafe for matrices smaller than 32 rows.

8. **Why is the GEMM stride `n*n`?**
   cuBLAS strides count elements. One contiguous square matrix contains
   `n*n` floats. A byte stride would be four times too large.

9. **Why is `beta` zero?**
   The required operation is `C=A*B`, with no previous C term. Both paths
   overwrite the result on every call.

10. **Why use CUDA events instead of a CPU timer?**
    Kernel launches return asynchronously. Events in the same stream measure
    device progress, and synchronizing the stop event waits for completion.

11. **Why can the results differ slightly?**
    Fused operations and different accumulation orders round FP32 values
    differently. A small error tolerance is appropriate; exact equality is
    not required.

12. **Can we claim the custom kernel beats cuBLAS?**
    Only for configurations where a real GPU measurement shows it. Compile
    success and CPU checks cannot establish a GPU speedup.

Before a demo, run `make test`, inspect the measured error and timing tables,
and explain the indexing in both kernel loops in your own words.
