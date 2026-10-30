#!/usr/bin/env python3
"""Check the actual kernel bodies on CPU. This does not validate CUDA execution.

GER threads execute serially. GEMM uses 128 host threads and a C++20 barrier
to model its shared-memory load/synchronize/compute phases, one block at a time.
The generated C++ and executable live in a temporary directory.
"""
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def kernel(path, name):
    source = path.read_text()
    start = source.index("__global__ void " + name)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end].replace("__global__ ", "").replace("__shared__", "static")


PREAMBLE = r'''
#include <algorithm>
#include <barrier>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>
struct dim3 {
    unsigned x, y, z;
    dim3(unsigned a=1, unsigned b=1, unsigned c=1):x(a),y(b),z(c){}
};
thread_local dim3 threadIdx, blockIdx, blockDim, gridDim;
std::barrier<>* block_barrier;
#define __syncthreads() block_barrier->arrive_and_wait()
'''

HARNESS = r'''
bool close(float actual, double expected) {
    return std::isfinite(actual) &&
           std::fabs(actual-expected) <= 2e-5 * (1.0+std::fabs(expected));
}
int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> random(-1.0f,1.0f);
    int cases = 0;
    for (auto shape : std::vector<std::pair<int,int>>{
            {1,1},{1,65},{65,1},{31,7},{32,8},{33,9},{67,35},{1,524281}}) {
        int m=shape.first, n=shape.second;
        std::vector<float> x(m), y(n), a((size_t)m*n+2,-12345.0f);
        for (auto& v:x) v=random(rng);
        for (auto& v:y) v=random(rng);
        for (size_t i=1;i+1<a.size();++i) a[i]=random(rng);
        auto original=a;
        blockDim=dim3(32,8);
        gridDim=dim3((m-1)/32+1,std::min((n-1)/8+1,65535));
        for(unsigned gy=0;gy<gridDim.y;++gy)
            for(unsigned gx=0;gx<gridDim.x;++gx)
                for(unsigned ty=0;ty<blockDim.y;++ty)
                    for(unsigned tx=0;tx<blockDim.x;++tx) {
                        blockIdx=dim3(gx,gy,0); threadIdx=dim3(tx,ty,0);
                        ger_kernel(m,n,-1.5f,x.data(),y.data(),a.data()+1);
                    }
        if(a.front()!=-12345.0f || a.back()!=-12345.0f) return 1;
        for(int col=0;col<n;++col) for(int row=0;row<m;++row) {
            size_t i=(size_t)col*m+row;
            if(!close(a[i+1],original[i+1]-1.5*(double)x[row]*y[col])) return 2;
        }
        ++cases;
    }
    for(int n=1;n<=32;++n) {
        const int batch=3;
        const size_t matrix_size=(size_t)n*n, total=matrix_size*batch;
        std::vector<float> a(total),b(total),c(total+2,-12345.0f);
        for(auto& v:a) v=random(rng);
        for(auto& v:b) v=random(rng);
        std::fill(c.begin()+1,c.end()-1,NAN);
        std::barrier sync(128);
        block_barrier=&sync;
        std::vector<std::thread> workers;
        for(unsigned tid=0;tid<128;++tid) workers.emplace_back([&,tid] {
            blockDim=dim3(32,4); gridDim=dim3(batch,1);
            threadIdx=dim3(tid%32,tid/32,0);
            for(int batch_id=0;batch_id<batch;++batch_id) {
                blockIdx=dim3(batch_id,0,0);
                batch_gemm_kernel(n,batch,a.data(),b.data(),c.data()+1);
                sync.arrive_and_wait();
            }
        });
        for(auto& worker:workers) worker.join();
        if(c.front()!=-12345.0f || c.back()!=-12345.0f) return 3;
        for(int batch_id=0;batch_id<batch;++batch_id)
            for(int row=0;row<n;++row) for(int col=0;col<n;++col) {
                size_t base=batch_id*matrix_size;
                double expected=0;
                for(int k=0;k<n;++k)
                    expected+=(double)a[base+(size_t)k*n+row]*b[base+(size_t)col*n+k];
                if(!close(c[1+base+(size_t)col*n+row],expected)) return 4;
            }
        ++cases;
    }
    std::printf("Host kernel-body checks: %d cases PASS (not a GPU run)\n",cases);
    return 0;
}
'''


def main():
    source = PREAMBLE + kernel(ROOT / "3-ger/ger.cu", "ger_kernel")
    source += kernel(ROOT / "4-gemm/gemm.cu", "batch_gemm_kernel") + HARNESS
    with tempfile.TemporaryDirectory(prefix="akt-host-check-") as directory:
        cpp = Path(directory) / "check.cpp"
        executable = Path(directory) / "check"
        cpp.write_text(source)
        subprocess.run([os.environ.get("CXX", "g++"), "-std=c++20", "-O2",
                        "-pthread", "-Wall", "-Wextra", str(cpp), "-o", str(executable)], check=True)
        subprocess.run([str(executable)], check=True, timeout=180)


if __name__ == "__main__":
    main()
