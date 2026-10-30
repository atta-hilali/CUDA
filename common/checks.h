#pragma once

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    const cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, \
                     #call, cudaGetErrorString(status)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    const cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "%s:%d: %s: cuBLAS status %d\n", \
                     __FILE__, __LINE__, #call, static_cast<int>(status)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

inline int positive_int(const char* text)
{
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    return (errno == 0 && end != text && *end == '\0' &&
            value > 0 && value <= INT_MAX) ? static_cast<int>(value) : 0;
}

inline double gflops_from_us(double flops, double time_us)
{
    return time_us > 0.0 ? flops / (time_us * 1.0e3) : 0.0;
}
