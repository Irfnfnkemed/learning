#include "../include/cpu.h"

// compute O = alpha * A @ B^T + beta * O

__host__ void cpu::cpuSgemm(int m, int n, int k, const half *A, const half *B,
                            half *O, const float alpha, const float beta) {
  for (int idx_m = 0; idx_m < m; ++idx_m) {
    for (int idx_n = 0; idx_n < n; ++idx_n) {
      float sum = 0.0;
      for (int idx_k = 0; idx_k < k; ++idx_k) {
        sum += (float)A[idx_m * k + idx_k] * (float)B[idx_n * k + idx_k];
      }
      O[idx_m * n + idx_n] =
          (half)(alpha * sum + beta * (float)O[idx_m * n + idx_n]);
    }
  }
}
