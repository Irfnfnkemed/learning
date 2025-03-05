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
const bool run_wmma = true;
const bool run_prefetch_wmma = true;
const bool run_noconflict_wmma = true;
const bool run_prefetch_noconflict_wmma = true;

const int M = 8192;
const int N = 8192;
const int K = 4096;

const float EPSILON = 0.01;

int main() {
  srand(42);

  // prepare Matrix
  half *matrix_A = (half *)malloc(sizeof(half) * M * K);
  half *matrix_B = (half *)malloc(sizeof(half) * K * N);
  half *A_gpu;
  half *B_gpu;
  half *O_gpu;
  half *matrix_O;
  half *matrix_O_blas;
  half *matrix_O_wmma;
  half *matrix_O_prefetch_wmma;
  half *matrix_O_noconflict_wmma;
  half *matrix_O_prefetch_noconflict_wmma;

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

  float alpha = 1.0, beta = 0.0;

  if (run_cpu) {
    matrix_O = (half *)malloc(sizeof(half) * M * N);
    memset(matrix_O, 0, sizeof(half) * M * N);
    cpu::cpuSgemm(M, N, K, matrix_A, matrix_B, matrix_O, alpha, beta);
  }

  if (run_blas) {
    cudaDeviceSynchronize();
    cublasHandle_t handle;
    cublasCreate(&handle);
    matrix_O_blas = (half *)malloc(sizeof(half) * M * N);
    memset(matrix_O_blas, 0, sizeof(half) * M * N);
    HANDLE_ERROR(cudaMalloc(&A_gpu, sizeof(half) * M * K));
    HANDLE_ERROR(cudaMalloc(&B_gpu, sizeof(half) * K * N));
    HANDLE_ERROR(cudaMalloc(&O_gpu, sizeof(half) * M * N));
    HANDLE_ERROR(cudaMemcpy(A_gpu, matrix_A, sizeof(half) * M * K,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(B_gpu, matrix_B, sizeof(half) * K * N,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(O_gpu, matrix_O_blas, sizeof(half) * M * N,
                            cudaMemcpyHostToDevice));
    blas::gpuBlasSgemm(M, N, K, B_gpu, A_gpu, O_gpu, alpha, beta, handle);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMemcpy(matrix_O_blas, O_gpu, sizeof(half) * M * N,
                            cudaMemcpyDeviceToHost));
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(O_gpu);
  }

  if (run_wmma) {
    matrix_O_wmma = (half *)malloc(sizeof(half) * M * N);
    memset(matrix_O_wmma, 0, sizeof(half) * M * N);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMalloc(&A_gpu, sizeof(half) * M * K));
    HANDLE_ERROR(cudaMalloc(&B_gpu, sizeof(half) * K * N));
    HANDLE_ERROR(cudaMalloc(&O_gpu, sizeof(half) * M * N));
    HANDLE_ERROR(cudaMemcpy(A_gpu, matrix_A, sizeof(half) * M * K,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(B_gpu, matrix_B, sizeof(half) * K * N,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(O_gpu, matrix_O_wmma, sizeof(half) * M * N,
                            cudaMemcpyHostToDevice));
    wmma::gpuWmmaSgemm<<<wmma::dimGrid, wmma::dimBlock, wmma::smem_size,
                         nullptr>>>(M, N, K, A_gpu, B_gpu, O_gpu, alpha, beta);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMemcpy(matrix_O_wmma, O_gpu, sizeof(half) * M * N,
                            cudaMemcpyDeviceToHost));
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(O_gpu);
  }

  if (run_prefetch_wmma) {
    matrix_O_prefetch_wmma = (half *)malloc(sizeof(half) * M * N);
    memset(matrix_O_prefetch_wmma, 0, sizeof(half) * M * N);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMalloc(&A_gpu, sizeof(half) * M * K));
    HANDLE_ERROR(cudaMalloc(&B_gpu, sizeof(half) * K * N));
    HANDLE_ERROR(cudaMalloc(&O_gpu, sizeof(half) * M * N));
    HANDLE_ERROR(cudaMemcpy(A_gpu, matrix_A, sizeof(half) * M * K,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(B_gpu, matrix_B, sizeof(half) * K * N,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(O_gpu, matrix_O_prefetch_wmma, sizeof(half) * M * N,
                            cudaMemcpyHostToDevice));
    pre_wmma::gpuWmmaPrefetchSgemm<<<pre_wmma::dimGrid, pre_wmma::dimBlock,
                                     pre_wmma::smem_size, nullptr>>>(
        M, N, K, A_gpu, B_gpu, O_gpu, alpha, beta);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMemcpy(matrix_O_prefetch_wmma, O_gpu, sizeof(half) * M * N,
                            cudaMemcpyDeviceToHost));
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(O_gpu);
  }

  if (run_noconflict_wmma) {
    matrix_O_noconflict_wmma = (half *)malloc(sizeof(half) * M * N);
    memset(matrix_O_noconflict_wmma, 0, sizeof(half) * M * N);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMalloc(&A_gpu, sizeof(half) * M * K));
    HANDLE_ERROR(cudaMalloc(&B_gpu, sizeof(half) * K * N));
    HANDLE_ERROR(cudaMalloc(&O_gpu, sizeof(half) * M * N));
    HANDLE_ERROR(cudaMemcpy(A_gpu, matrix_A, sizeof(half) * M * K,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(B_gpu, matrix_B, sizeof(half) * K * N,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(O_gpu, matrix_O_prefetch_wmma, sizeof(half) * M * N,
                            cudaMemcpyHostToDevice));
    noconflict_wmma::gpuNoConflictWmmaSgemm<<<
        noconflict_wmma::dimGrid, noconflict_wmma::dimBlock,
        noconflict_wmma::smem_size, nullptr>>>(M, N, K, A_gpu, B_gpu, O_gpu,
                                               alpha, beta);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMemcpy(matrix_O_noconflict_wmma, O_gpu,
                            sizeof(half) * M * N, cudaMemcpyDeviceToHost));
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(O_gpu);
  }

  if (run_prefetch_noconflict_wmma) {
    matrix_O_prefetch_noconflict_wmma = (half *)malloc(sizeof(half) * M * N);
    memset(matrix_O_prefetch_noconflict_wmma, 0, sizeof(half) * M * N);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMalloc(&A_gpu, sizeof(half) * M * K));
    HANDLE_ERROR(cudaMalloc(&B_gpu, sizeof(half) * K * N));
    HANDLE_ERROR(cudaMalloc(&O_gpu, sizeof(half) * M * N));
    HANDLE_ERROR(cudaMemcpy(A_gpu, matrix_A, sizeof(half) * M * K,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(B_gpu, matrix_B, sizeof(half) * K * N,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(O_gpu, matrix_O_prefetch_wmma, sizeof(half) * M * N,
                            cudaMemcpyHostToDevice));
    pre_noconflict_wmma::gpuPrefetchNoConflictWmmaSgemm<<<
        pre_noconflict_wmma::dimGrid, pre_noconflict_wmma::dimBlock,
        pre_noconflict_wmma::smem_size, nullptr>>>(M, N, K, A_gpu, B_gpu, O_gpu,
                                                   alpha, beta);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMemcpy(matrix_O_prefetch_noconflict_wmma, O_gpu,
                            sizeof(half) * M * N, cudaMemcpyDeviceToHost));
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(O_gpu);
  }

  // check result
  half *baseline;
  if (run_cpu) {
    baseline = matrix_O;
  } else if (run_blas) {
    baseline = matrix_O_blas;
  } else {
    printf("Must run cpu or cuBlas as baseline.");
    return EXIT_FAILURE;
  }

  if (run_blas and run_cpu) {
    printf("Check cuBlas:\n");
    int cnt = 0;
    for (int i = 0; i < M * N; ++i) {
      float error =
          ((float)matrix_O_blas[i] - (float)baseline[i]) / (float)baseline[i];
      if (error < -EPSILON || error > EPSILON)
        cnt += 1;
    }
    if (cnt == 0) {
      printf("Right.\n\n");
    } else {
      printf("Wrong. Wrong rate %f\n\n", ((float)cnt) / (M * N));
    }
  }

  if (run_wmma) {
    printf("Check wmma:\n");
    int cnt = 0;
    for (int i = 0; i < M * N; ++i) {
      float error =
          ((float)matrix_O_wmma[i] - (float)baseline[i]) / (float)baseline[i];
      if (error < -EPSILON || error > EPSILON)
        cnt += 1;
    }
    if (cnt == 0) {
      printf("Right.\n\n");
    } else {
      printf("Wrong. Wrong rate %f\n\n", ((float)cnt) / (M * N));
    }
  }

  if (run_prefetch_wmma) {
    printf("Check prefetch wmma:\n");
    int cnt = 0;
    for (int i = 0; i < M * N; ++i) {
      float error = ((float)matrix_O_prefetch_wmma[i] - (float)baseline[i]) /
                    (float)baseline[i];
      if (error < -EPSILON || error > EPSILON)
        cnt += 1;
    }
    if (cnt == 0) {
      printf("Right.\n\n");
    } else {
      printf("Wrong. Wrong rate %f\n\n", ((float)cnt) / (M * N));
    }
  }

  if (run_noconflict_wmma) {
    printf("Check noconflict wmma:\n");
    int cnt = 0;
    for (int i = 0; i < M * N; ++i) {
      float error = ((float)matrix_O_noconflict_wmma[i] - (float)baseline[i]) /
                    (float)baseline[i];
      if (error < -EPSILON || error > EPSILON)
        cnt += 1;
    }
    if (cnt == 0) {
      printf("Right.\n\n");
    } else {
      printf("Wrong. Wrong rate %f\n\n", ((float)cnt) / (M * N));
    }
  }

  if (run_prefetch_noconflict_wmma) {
    printf("Check prefetch-noconflict wmma:\n");
    int cnt = 0;
    for (int i = 0; i < M * N; ++i) {
      float error =
          ((float)matrix_O_prefetch_noconflict_wmma[i] - (float)baseline[i]) /
          (float)baseline[i];
      if (error < -EPSILON || error > EPSILON)
        cnt += 1;
    }
    if (cnt == 0) {
      printf("Right.\n\n");
    } else {
      printf("Wrong. Wrong rate %f\n\n", ((float)cnt) / (M * N));
    }
  }

  // release memory on device
  free(matrix_A);
  free(matrix_B);
  if (run_cpu) {
    free(matrix_O);
  }
  if (run_blas) {
    free(matrix_O_blas);
  }
  if (run_wmma) {
    free(matrix_O_wmma);
  }
  if (run_prefetch_wmma) {
    free(matrix_O_prefetch_wmma);
  }
  if (run_noconflict_wmma) {
    free(matrix_O_noconflict_wmma);
  }
  if (run_prefetch_noconflict_wmma) {
    free(matrix_O_prefetch_noconflict_wmma);
  }
  return EXIT_SUCCESS;
}