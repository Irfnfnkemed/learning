#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <library_types.h>
#include <math.h>
#include <mma.h>
#include <stdlib.h>

#ifndef WMMA
#define WMMA

namespace wmma {

const int M = 8192;
const int N = 8192;
const int K = 4096;
const int BM = 128;
const int BN = 64;
const int BK = 32;
const int WM = 32;
const int WN = 32;
const int WK = 16;
const dim3 dimGrid(M / BM, N / BN);
const dim3 dimBlock(32, BM / WM, BN / WN);
const int smem_size =
    std::max(sizeof(half) * BK * (BM + BN), sizeof(float) * BM * BN);

__global__ void gpuWmmaSgemm(int m, int n, int k, const half *A, const half *B,
                             half *O, const float alpha, const float beta);
} // namespace wmma

#endif