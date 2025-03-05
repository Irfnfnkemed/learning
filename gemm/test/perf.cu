#include "../include/cpu.h"
#include "../include/cublas.h"
#include "../include/noconflict_wmma.h"
#include "../include/prefetch_noconflict_wmma.h"
#include "../include/prefetch_wmma.h"
#include "../include/wmma.h"
#include "cublas_v2.h"
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
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

const float rand_max = -0.5;
const float rand_min = 0.5;

const bool run_cpu = false;
const bool run_blas = true;
const bool run_wmma = false;
const bool run_prefetch_wmma = false;
const bool run_noconflict_wmma = false;
const bool run_prefetch_noconflict_wmma = false;

const int M = 8192;
const int N = 8192;
const int K = 4096;

const int warmup = 10;
const int test = 20;

int main() {
  srand(42);

  // prepare Matrix
  half *matrix_A = (half *)malloc(sizeof(half) * M * K);
  half *matrix_B = (half *)malloc(sizeof(half) * K * N);
  half *matrix_O = (half *)malloc(sizeof(half) * M * N);
  half *A_gpu;
  half *B_gpu;
  half *O_gpu;

  // generate data
  float rand_num = 0.0;
  for (int i = 0; i < M * K; ++i) {
    rand_num = (float)rand() / RAND_MAX;
    matrix_A[i] = (half)(rand_min + rand_num * (rand_max - rand_min));
  }
  for (int i = 0; i < K * N; ++i) {
    rand_num = (float)rand() / RAND_MAX;
    matrix_B[i] = (half)(rand_min + rand_num * (rand_max - rand_min));
  }
  memset(matrix_O, 0, sizeof(half) * M * N);
  HANDLE_ERROR(cudaMalloc(&A_gpu, sizeof(half) * M * K));
  HANDLE_ERROR(cudaMalloc(&B_gpu, sizeof(half) * K * N));
  HANDLE_ERROR(cudaMalloc(&O_gpu, sizeof(half) * M * N));
  HANDLE_ERROR(cudaMemcpy(A_gpu, matrix_A, sizeof(half) * M * K,
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(B_gpu, matrix_B, sizeof(half) * K * N,
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(O_gpu, matrix_O, sizeof(half) * M * N,
                          cudaMemcpyHostToDevice));

  float alpha = 1.5, beta = 0.0;

  float duration;

  // record cpu execution time
  if (run_cpu) {
    clock_t start, stop;
    start = clock();
    cpu::cpuSgemm(M, N, K, matrix_A, matrix_B, matrix_O, alpha, beta);
    stop = clock();
    duration = (double)(stop - start) / CLOCKS_PER_SEC * 1000;
    printf("cpu time:%fms\n\n", duration);
  }

  // record gpu with cublas execution time
  if (run_blas) {
    cudaEvent_t start_gpu_blas, stop_gpu_blas;
    cublasHandle_t handle;
    cublasCreate(&handle);
    HANDLE_ERROR(cudaEventCreate(&start_gpu_blas));
    HANDLE_ERROR(cudaEventCreate(&stop_gpu_blas));
    duration = 0.0;
    for (int i = 0; i < warmup; i += 1) {
      blas::gpuBlasSgemm(M, N, K, B_gpu, A_gpu, O_gpu, alpha, beta, handle);
    }
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaEventRecord(start_gpu_blas, 0));
    for (int i = 0; i < test; i += 1) {
      blas::gpuBlasSgemm(M, N, K, B_gpu, A_gpu, O_gpu, alpha, beta, handle);
    }
    HANDLE_ERROR(cudaEventRecord(stop_gpu_blas, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop_gpu_blas));
    HANDLE_ERROR(
        cudaEventElapsedTime(&duration, start_gpu_blas, stop_gpu_blas));
    printf("gpu with blas time:%fms\n", duration / test);
    printf("TFLOPS: %f\n\n",
           (float)M * N * K * 2 * test / double(duration) * 1e3 / 1e12);
    HANDLE_ERROR(cudaEventDestroy(start_gpu_blas));
    HANDLE_ERROR(cudaEventDestroy(stop_gpu_blas));
  }

  // record gpu with WMMA execution time
  if (run_wmma) {
    cudaEvent_t start_gpu_wmma, stop_gpu_wmma;
    HANDLE_ERROR(cudaEventCreate(&start_gpu_wmma));
    HANDLE_ERROR(cudaEventCreate(&stop_gpu_wmma));
    duration = 0.0;
    for (int i = 0; i < warmup; i += 1) {
      wmma::gpuWmmaSgemm<<<wmma::dimGrid, wmma::dimBlock, wmma::smem_size,
                           nullptr>>>(M, N, K, A_gpu, B_gpu, O_gpu, alpha,
                                      beta);
    }
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaEventRecord(start_gpu_wmma, 0));
    for (int i = 0; i < test; i += 1) {
      wmma::gpuWmmaSgemm<<<wmma::dimGrid, wmma::dimBlock, wmma::smem_size,
                           nullptr>>>(M, N, K, A_gpu, B_gpu, O_gpu, alpha,
                                      beta);
    }
    HANDLE_ERROR(cudaEventRecord(stop_gpu_wmma, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop_gpu_wmma));
    HANDLE_ERROR(
        cudaEventElapsedTime(&duration, start_gpu_wmma, stop_gpu_wmma));
    printf("gpu with wmma time:%fms\n", duration / test);
    printf("TFLOPS: %f\n\n",
           (float)M * N * K * 2 * test / double(duration) * 1e3 / 1e12);
    HANDLE_ERROR(cudaEventDestroy(start_gpu_wmma));
    HANDLE_ERROR(cudaEventDestroy(stop_gpu_wmma));
  }

  // record gpu with prefetch WMMA execution time
  if (run_prefetch_wmma) {
    cudaEvent_t start_gpu_prefetch_wmma, stop_gpu_prefetch_wmma;
    HANDLE_ERROR(cudaEventCreate(&start_gpu_prefetch_wmma));
    HANDLE_ERROR(cudaEventCreate(&stop_gpu_prefetch_wmma));
    duration = 0.0;
    for (int i = 0; i < warmup; i += 1) {
      pre_wmma::gpuWmmaPrefetchSgemm<<<pre_wmma::dimGrid, pre_wmma::dimBlock,
                                       pre_wmma::smem_size, nullptr>>>(
          M, N, K, A_gpu, B_gpu, O_gpu, alpha, beta);
    }
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaEventRecord(start_gpu_prefetch_wmma, 0));
    for (int i = 0; i < test; i += 1) {
      pre_wmma::gpuWmmaPrefetchSgemm<<<pre_wmma::dimGrid, pre_wmma::dimBlock,
                                       pre_wmma::smem_size, nullptr>>>(
          M, N, K, A_gpu, B_gpu, O_gpu, alpha, beta);
    }
    HANDLE_ERROR(cudaEventRecord(stop_gpu_prefetch_wmma, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop_gpu_prefetch_wmma));
    HANDLE_ERROR(cudaEventElapsedTime(&duration, start_gpu_prefetch_wmma,
                                      stop_gpu_prefetch_wmma));
    printf("gpu with prefetch wmma time:%fms\n", duration / test);
    printf("TFLOPS: %f\n\n",
           (float)M * N * K * 2 * test / double(duration) * 1e3 / 1e12);
    HANDLE_ERROR(cudaEventDestroy(start_gpu_prefetch_wmma));
    HANDLE_ERROR(cudaEventDestroy(stop_gpu_prefetch_wmma));
  }

  // record gpu with noconflict WMMA execution time
  if (run_noconflict_wmma) {
    cudaEvent_t start_gpu_noconflict_wmma, stop_gpu_noconflict_wmma;
    HANDLE_ERROR(cudaEventCreate(&start_gpu_noconflict_wmma));
    HANDLE_ERROR(cudaEventCreate(&stop_gpu_noconflict_wmma));
    duration = 0.0;
    for (int i = 0; i < warmup; i += 1) {
      noconflict_wmma::gpuNoConflictWmmaSgemm<<<
          noconflict_wmma::dimGrid, noconflict_wmma::dimBlock,
          noconflict_wmma::smem_size, nullptr>>>(M, N, K, A_gpu, B_gpu, O_gpu,
                                                 alpha, beta);
    }
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaEventRecord(start_gpu_noconflict_wmma, 0));
    for (int i = 0; i < test; i += 1) {
      noconflict_wmma::gpuNoConflictWmmaSgemm<<<
          noconflict_wmma::dimGrid, noconflict_wmma::dimBlock,
          noconflict_wmma::smem_size, nullptr>>>(M, N, K, A_gpu, B_gpu, O_gpu,
                                                 alpha, beta);
    }
    HANDLE_ERROR(cudaEventRecord(stop_gpu_noconflict_wmma, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop_gpu_noconflict_wmma));
    HANDLE_ERROR(cudaEventElapsedTime(&duration, start_gpu_noconflict_wmma,
                                      stop_gpu_noconflict_wmma));
    printf("gpu with noconflict wmma time:%fms\n", duration / test);
    printf("TFLOPS: %f\n\n",
           (float)M * N * K * 2 * test / double(duration) * 1e3 / 1e12);
    HANDLE_ERROR(cudaEventDestroy(start_gpu_noconflict_wmma));
    HANDLE_ERROR(cudaEventDestroy(stop_gpu_noconflict_wmma));
  }

  // record gpu with prefetch-noconflict WMMA execution time
  if (run_prefetch_noconflict_wmma) {
    cudaEvent_t start_gpu_prefetch_noconflict_wmma,
        stop_gpu_prefetch_noconflict_wmma;
    HANDLE_ERROR(cudaEventCreate(&start_gpu_prefetch_noconflict_wmma));
    HANDLE_ERROR(cudaEventCreate(&stop_gpu_prefetch_noconflict_wmma));
    duration = 0.0;
    for (int i = 0; i < warmup; i += 1) {
      pre_noconflict_wmma::gpuPrefetchNoConflictWmmaSgemm<<<
          pre_noconflict_wmma::dimGrid, pre_noconflict_wmma::dimBlock,
          pre_noconflict_wmma::smem_size, nullptr>>>(M, N, K, A_gpu, B_gpu,
                                                     O_gpu, alpha, beta);
    }
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaEventRecord(start_gpu_prefetch_noconflict_wmma, 0));
    for (int i = 0; i < test; i += 1) {
      pre_noconflict_wmma::gpuPrefetchNoConflictWmmaSgemm<<<
          pre_noconflict_wmma::dimGrid, pre_noconflict_wmma::dimBlock,
          pre_noconflict_wmma::smem_size, nullptr>>>(M, N, K, A_gpu, B_gpu,
                                                     O_gpu, alpha, beta);
    }
    HANDLE_ERROR(cudaEventRecord(stop_gpu_prefetch_noconflict_wmma, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop_gpu_prefetch_noconflict_wmma));
    HANDLE_ERROR(cudaEventElapsedTime(&duration,
                                      start_gpu_prefetch_noconflict_wmma,
                                      stop_gpu_prefetch_noconflict_wmma));
    printf("gpu with prefetch-noconflict wmma time:%fms\n", duration / test);
    printf("TFLOPS: %f\n\n",
           (float)M * N * K * 2 * test / double(duration) * 1e3 / 1e12);
    HANDLE_ERROR(cudaEventDestroy(start_gpu_prefetch_noconflict_wmma));
    HANDLE_ERROR(cudaEventDestroy(stop_gpu_prefetch_noconflict_wmma));
  }

  // release memory on device and host
  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(O_gpu);
  free(matrix_A);
  free(matrix_B);
  free(matrix_O);

  return EXIT_SUCCESS;
}