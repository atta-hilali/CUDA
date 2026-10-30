
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <limits>

#include "../common/checks.h"

////////////////////////////////////////////////////////////////////////////////
// Custom rank-1 update kernel
// A <- A + alpha * x * y^T
// Column-major storage: A(i,j) = A[j*m + i]
__global__ void ger_kernel(int m, int n, float alpha, const float* x, const float* y, float* A)
{
    const size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t first_col = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= (size_t)m) return;

    // A warp visits consecutive rows of one column.
    // The column loop also handles grids larger than the y-dimension limit.
    for (size_t col = first_col; col < (size_t)n;
         col += (size_t)gridDim.y * blockDim.y) {
        const size_t index = col * m + row;
        A[index] = fmaf(alpha * x[row], y[col], A[index]);
    }
}

////////////////////////////////////////////////////////////////////////////////
// CPU reference implementation, FP32
void ger_cpu(int m, int n, float alpha, const float* x, const float* y, float* A)
{
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            A[(size_t)j * m + i] += alpha * x[i] * y[j];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// compute error norm
double compute_error_norm(size_t size,
                          const float* result,
                          const float* reference)
{
    double error = 0.0;
    double norm  = 0.0;

    for (size_t i = 0; i < size; ++i) {
        const double diff  = std::fabs((double)result[i] - (double)reference[i]);
        const double denom = std::fabs((double)reference[i]);

        if (!std::isfinite(diff) || !std::isfinite(denom)) return INFINITY;
        error += diff * diff;
        norm  += denom * denom;
    }

    if (norm == 0.0) return error == 0.0 ? 0.0 : INFINITY;
    return std::sqrt(error) / std::sqrt(norm);
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    const int m = (argc > 1) ? positive_int(argv[1]) : 4000;
    const int n = (argc > 2) ? positive_int(argv[2]) : 2000;
    const int repeats = (argc > 3) ? positive_int(argv[3]) : 100;
    const float alpha = 1.5f;

    if (argc > 4 || m <= 0 || n <= 0 || repeats <= 0) {
        std::fprintf(stderr, "Usage: %s [M > 0] [N > 0] [repeats > 0]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const size_t matrix_size = (size_t)m * n;
    if (matrix_size > std::numeric_limits<size_t>::max() / sizeof(float)) {
        std::fprintf(stderr, "Matrix allocation size overflows size_t\n");
        return EXIT_FAILURE;
    }
    const size_t matrix_bytes = matrix_size * sizeof(float);

    // --------------------------------------------------------
    // Initialize vectors and matrix on the CPU
    // --------------------------------------------------------
    std::vector<float> h_x(m), h_y(n);
    std::vector<float> h_A0(matrix_size);
    std::vector<float> h_ref(matrix_size);
    std::vector<float> h_custom(matrix_size);
    std::vector<float> h_cublas(matrix_size);

    for (int i = 0; i < m; ++i) {
        h_x[i] = std::sin(0.1f * (i + 1));
    }

    for (int j = 0; j < n; ++j) {
        h_y[j] = std::cos(0.2f * (j + 1));
    }

    for (size_t i = 0; i < matrix_size; ++i) {
        h_A0[i] = 0.1f * std::sin(0.1f * (float)(i + 1));
    }

    h_ref = h_A0;
    ger_cpu(m, n, alpha, h_x.data(), h_y.data(), h_ref.data());

    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------
    float *d_x = nullptr, *d_y = nullptr, *d_A = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, (size_t)m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));

    // copy input vectors from CPU -> GPU
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)m * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

    std::printf("Matrix: %d x %d, alpha = %.2f, repeats = %d\n",
                m, n, alpha, repeats);
    std::printf("Matrix storage: column-major\n\n");

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("GPU: %s (compute capability %d.%d)\n\n",
                prop.name, prop.major, prop.minor);

    const dim3 block(32, 8, 1);
    const dim3 grid((m - 1) / block.x + 1,
                    std::min((n - 1) / block.y + 1, 65535u), 1);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --------------------------------------------------------
    // Custom kernel: warm-up and timing
    // --------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice));

    ger_kernel<<<grid, block>>>(m, n, alpha, d_x, d_y, d_A);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Restore the initial matrix before the timed repetitions.
    CUDA_CHECK(cudaMemcpy(d_A, h_A0.data(), matrix_bytes,
               cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {
        ger_kernel<<<grid, block>>>(m, n, alpha, d_x, d_y, d_A);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float custom_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&custom_ms, start, stop));
    const double custom_us = custom_ms * 1000.0 / repeats;

    // Correctness run: restore initial A and update once.
    CUDA_CHECK(cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice));
    ger_kernel<<<grid, block>>>(m, n, alpha, d_x, d_y, d_A);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_custom.data(), d_A, matrix_bytes, cudaMemcpyDeviceToHost));

    // --------------------------------------------------------
    // cuBLAS SGER: warm-up and timing
    // --------------------------------------------------------
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    CUDA_CHECK(cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasSger(handle, m, n, &alpha, d_x, 1, d_y, 1, d_A, m));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Restore the initial matrix before the timed repetitions.
    CUDA_CHECK(cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {
        CUBLAS_CHECK(cublasSger(handle, m, n, &alpha, d_x, 1, d_y, 1, d_A, m));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
    const double cublas_us = cublas_ms * 1000.0 / repeats;

    // Correctness run: restore initial A and update once.
    CUDA_CHECK(cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice));
    CUBLAS_CHECK(cublasSger(handle, m, n, &alpha, d_x, 1, d_y, 1, d_A, m));
    CUDA_CHECK(cudaMemcpy(h_cublas.data(), d_A, matrix_bytes, cudaMemcpyDeviceToHost));

    // --------------------------------------------------------
    // Performance and error norm
    // Approximately 2 FLOPs per matrix element.
    // --------------------------------------------------------
    const double flops = 2.0 * m * n;

    const double custom_gflops = gflops_from_us(flops, custom_us);
    const double cublas_gflops = gflops_from_us(flops, cublas_us);

    const double custom_error = compute_error_norm( matrix_size, h_custom.data(), h_ref.data());
    const double cublas_error = compute_error_norm( matrix_size, h_cublas.data(), h_ref.data());

    std::printf("%-20s %12s %12s %20s\n",
                "Implementation", "Time (us)", "GFLOP/s",
                "Error Norm");
    std::printf("--------------------------------------------------------------------\n");

    std::printf("%-20s %12.3f %12.3f %20.6e\n",
                "Custom kernel", custom_us, custom_gflops,
                custom_error);
    std::printf("%-20s %12.3f %12.3f %20.6e\n",
                "cuBLAS SGER", cublas_us, cublas_gflops,
                cublas_error);

    const double tol = 1e-6;
    const bool passed = std::isfinite(custom_error) && custom_error <= tol &&
                        std::isfinite(cublas_error) && cublas_error <= tol;
    if (passed) {
        std::printf("\nTEST PASS\n");
        if (custom_us > 0.0)
            std::printf("Speedup (cuBLAS time / custom time): %.3fx\n", cublas_us / custom_us);
    }
    else {
        std::printf("\nTEST FAIL\n");
        std::printf("Speedup is not calculated\n");
    }

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_A));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
