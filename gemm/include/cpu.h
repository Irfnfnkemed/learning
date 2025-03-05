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

#ifndef CPU
#define CPU

// compute O = alpha * A @ B^T + beta * O

namespace cpu {

__host__ void cpuSgemm(int m, int n, int k, const half *A, const half *B,
                       half *O, const float alpha, const float beta);
} // namespace cpu

#endif