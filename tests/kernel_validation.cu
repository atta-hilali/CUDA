// Include the submitted kernels directly so the tests exercise the same code.
#define main ger_example_main
#include "../3-ger/ger.cu"
#undef main
#define main gemm_example_main
#include "../4-gemm/gemm.cu"
#undef main

#include <random>

namespace {
const int guard = 32;
const float sentinel = -12345.0f;

bool close_value(float actual, double expected)
{
    return std::isfinite(actual) &&
           std::fabs(actual - expected) <= 2e-5 * (1.0 + std::fabs(expected));
}

bool intact_guards(const std::vector<float>& data, size_t count)
{
    for (int i = 0; i < guard; ++i) {
        if (data[i] != sentinel || data[guard + count + i] != sentinel)
            return false;
    }
    return true;
}

bool test_ger(int m, int n, float alpha, bool zero_vectors = false)
{
    const size_t count = (size_t)m * n;
    std::mt19937 rng(19);
    std::uniform_real_distribution<float> random(-1.0f, 1.0f);
    std::vector<float> x(m), y(n), initial(count);
    for (float& value : x) value = zero_vectors ? 0.0f : random(rng);
    for (float& value : y) value = zero_vectors ? 0.0f : random(rng);
    for (float& value : initial) value = random(rng);
    std::vector<float> result(count + 2 * guard, sentinel);
    std::copy(initial.begin(), initial.end(), result.begin() + guard);

    float *dx, *dy, *da;
    CUDA_CHECK(cudaMalloc(&dx, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&da, result.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), m * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(da, result.data(), result.size() * sizeof(float), cudaMemcpyHostToDevice));
    const dim3 block(32, 8);
    const dim3 grid((m - 1) / 32 + 1, std::min((n - 1) / 8 + 1, 65535));
    ger_kernel<<<grid, block>>>(m, n, alpha, dx, dy, da + guard);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result.data(), da, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    bool passed = intact_guards(result, count);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            const size_t index = (size_t)col * m + row;
            const double expected = initial[index] + (double)alpha * x[row] * y[col];
            passed = close_value(result[guard + index], expected) && passed;
        }
    }
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(da));
    std::printf("GER m=%d n=%d alpha=%g zero_vectors=%d: %s\n",
                m, n, alpha, zero_vectors, passed ? "PASS" : "FAIL");
    return passed;
}

// mode: 0=random signed inputs, 1=right identity, 2=left zero, 3=ones.
bool test_gemm(int n, int batch, int mode)
{
    const size_t matrix_size = (size_t)n * n;
    const size_t count = matrix_size * batch;
    std::mt19937 rng(71);
    std::uniform_real_distribution<float> random(-1.0f, 1.0f);
    std::vector<float> a(count), b(count);
    for (float& value : a) value = random(rng);
    for (float& value : b) value = random(rng);
    if (mode == 1) {
        std::fill(b.begin(), b.end(), 0.0f);
        for (int batch_id = 0; batch_id < batch; ++batch_id)
            for (int i = 0; i < n; ++i)
                b[batch_id * matrix_size + (size_t)i * n + i] = 1.0f;
    }
    if (mode == 2) std::fill(a.begin(), a.end(), 0.0f);
    if (mode == 3) {
        std::fill(a.begin(), a.end(), 1.0f);
        std::fill(b.begin(), b.end(), 1.0f);
    }
    std::vector<float> result(count + 2 * guard, sentinel);
    std::fill(result.begin() + guard, result.end() - guard, NAN);
    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, result.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, a.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, result.data(), result.size() * sizeof(float), cudaMemcpyHostToDevice));
    batch_gemm_kernel<<<batch, dim3(32, 4)>>>(n, batch, da, db, dc + guard);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result.data(), dc, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    bool passed = intact_guards(result, count);
    for (int batch_id = 0; batch_id < batch; ++batch_id) {
        const size_t base = batch_id * matrix_size;
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                double expected = 0.0;
                for (int k = 0; k < n; ++k)
                    expected += (double)a[base + (size_t)k * n + row] *
                                b[base + (size_t)col * n + k];
                if (mode == 1) expected = a[base + (size_t)col * n + row];
                if (mode == 2) expected = 0.0;
                if (mode == 3) expected = n;
                passed = close_value(result[guard + base + (size_t)col * n + row], expected) && passed;
            }
        }
    }
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    std::printf("GEMM n=%d batch=%d mode=%d: %s\n",
                n, batch, mode, passed ? "PASS" : "FAIL");
    return passed;
}
} // namespace

int main()
{
    bool passed = true;
    for (const auto& shape : std::vector<std::pair<int, int>>{
             {1, 1}, {1, 65}, {65, 1}, {31, 7}, {32, 8}, {33, 9}, {67, 35}})
        passed = test_ger(shape.first, shape.second, -1.5f) && passed;
    passed = test_ger(33, 9, 0.0f) && passed;
    passed = test_ger(33, 9, 1.5f, true) && passed;
    // Exercise the capped grid and its second column iteration.
    passed = test_ger(1, 524281, 1.5f) && passed;
    for (int n = 1; n <= 32; ++n)
        passed = test_gemm(n, 3, 0) && passed;
    passed = test_gemm(17, 1, 1) && passed;
    passed = test_gemm(31, 7, 1) && passed;
    passed = test_gemm(32, 5, 2) && passed;
    passed = test_gemm(32, 5, 3) && passed;
    std::printf("Kernel validation: %s\n", passed ? "PASS" : "FAIL");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
