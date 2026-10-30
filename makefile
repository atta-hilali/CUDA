CUDADIR ?= /usr/local/cuda
NVCC ?= $(CUDADIR)/bin/nvcc
sm ?= 75
NVCCFLAGS ?= -O3 -std=c++14 -lineinfo
GENCODE = -gencode arch=compute_$(sm),code=sm_$(sm) \
          -gencode arch=compute_$(sm),code=compute_$(sm)

.PHONY: all test benchmark clean
all:
	$(MAKE) -C 3-ger CUDADIR=$(CUDADIR) NVCC=$(NVCC) sm=$(sm)
	$(MAKE) -C 4-gemm CUDADIR=$(CUDADIR) NVCC=$(NVCC) sm=$(sm)

tests/kernel_validation: tests/kernel_validation.cu 3-ger/ger.cu 4-gemm/gemm.cu common/checks.h
	$(NVCC) $(NVCCFLAGS) $(GENCODE) $< -o $@ -L$(CUDADIR)/lib64 -lcublas

test: all tests/kernel_validation
	python3 scripts/run_tests.py

benchmark: all
	python3 scripts/benchmark.py

clean:
	$(MAKE) -C 3-ger clean
	$(MAKE) -C 4-gemm clean
	rm -f tests/kernel_validation
