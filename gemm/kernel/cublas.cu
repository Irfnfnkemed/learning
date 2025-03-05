#include "../include/cublas.h"

__host__ cublasStatus_t blas::gpuBlasSgemm(int m, int n, int k, const half *A,
                                           const half *B, half *O,
                                           const float alpha, const float beta,
                                           cublasHandle_t handle) {
  // use clublas to compute
  return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, A,
                      CUDA_R_16F, k, B, CUDA_R_16F, k, &beta, O, CUDA_R_16F, n,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
