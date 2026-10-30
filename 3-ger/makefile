CUDADIR ?= /usr/local/cuda
NVCC ?= $(CUDADIR)/bin/nvcc
sm ?= 75
NVCCFLAGS ?= -O3 -std=c++14 -lineinfo
GENCODE = -gencode arch=compute_$(sm),code=sm_$(sm) \
          -gencode arch=compute_$(sm),code=compute_$(sm)
LDLIBS = -L$(CUDADIR)/lib64 -lcublas

.PHONY: all clean
all: ger

ger: ger.cu ../common/checks.h
	$(NVCC) $(NVCCFLAGS) $(GENCODE) $< -o $@ $(LDLIBS)

clean:
	rm -f ger
