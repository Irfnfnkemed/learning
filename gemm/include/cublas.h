#include "cublas_v2.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <library_types.h>
#include <math.h>
#include <mma.h>
#include <stdlib.h>

#ifndef CUBLAS
#define CUBLAS

namespace blas {

__host__ cublasStatus_t gpuBlasSgemm(int m, int n, int k, const half *A,
                                     const half *B, half *O, const float alpha,
                                     const float beta, cublasHandle_t handle);
} // namespace blas

#endif