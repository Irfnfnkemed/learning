#include "../include/wmma.h"
#include <cstdint>

__global__ void wmma::gpuWmmaSgemm(int m, int n, int k, const half *A,
                                   const half *B, half *O, const float alpha,
                                   const float beta) {
  extern __shared__ uint8_t shared_storage[];
  half *SA = reinterpret_cast<half *>(shared_storage);
  half *SB = reinterpret_cast<half *>(shared_storage + BM * BK * sizeof(half));
  float *SC = reinterpret_cast<float *>(shared_storage);
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int tid = tz * 32 * BM / WM + ty * 32 + tx;
  int tnum = 32 * BM / WM * BN / WN;
  assert(WK == 16);
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                         nvcuda::wmma::row_major>
      FragA[WM / 16];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
                         nvcuda::wmma::col_major>
      FragB[WN / 16];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
      Accum[WM / 16 * WN / 16];

  for (int mii = 0; mii < WM / 16; mii += 1) {
    for (int nii = 0; nii < WN / 16; nii += 1) {
      nvcuda::wmma::fill_fragment(Accum[mii * (WN / 16) + nii], 0.0);
    }
  }
  for (int ko = 0; ko < k / BK; ko += 1) {
    // load A, B to SA, SB [global memory -> shared memory]
    int elePerThreadA = BM * BK / 32 / (BM / WM) / (BN / WN);
    for (int i = 0; i < elePerThreadA; i += 1) {
      int row = (i * tnum + tid) / BK;
      int column = (i * tnum + tid) % BK;
      // SA layout: [row_out, col_out, row_in, col_in] = [BM/16, BK/16, 16, 16]
      SA[(row / 16 * (BK / 16 * 16 * 16)) + (column / 16 * (16 * 16)) +
         ((row % 16) * 16) + (column % 16)] =
          A[(row + bx * BM) * k + (column + ko * BK)];
    }
    int elePerThreadB = BN * BK / 32 / (BM / WM) / (BN / WN);
    for (int i = 0; i < elePerThreadB; i += 1) {
      int row = (i * tnum + tid) / BK;
      int column = (i * tnum + tid) % BK;
      // SB layout: [row_out, col_out, row_in, col_in] = [BN/16, BK/16, 16, 16]
      SB[(row / 16 * (BK / 16 * 16 * 16)) + (column / 16 * (16 * 16)) +
         ((row % 16) * 16) + (column % 16)] =
          B[(row + by * BN) * k + (column + ko * BK)];
    }
    __syncthreads();
    for (int kii = 0; kii < BK / WK; kii += 1) {
      // load SA, SB to tensor core (shared memory -> tensor core)
      for (int i = 0; i < WM / 16; i += 1) {
        int row = ty * (WM / 16) + i;
        int col = kii;
        nvcuda::wmma::load_matrix_sync(
            FragA[i], SA + row * (BK / 16 * 16 * 16) + col * (16 * 16), 16);
      }
      for (int i = 0; i < WN / 16; i += 1) {
        int row = tz * (WN / 16) + i;
        int col = kii;
        nvcuda::wmma::load_matrix_sync(
            FragB[i], SB + row * (BK / 16 * 16 * 16) + col * (16 * 16), 16);
      }
      // compute 16*16*16 wmma blocks
      for (int mii = 0; mii < WM / 16; mii += 1) {
        for (int nii = 0; nii < WN / 16; nii += 1) {
          // 16x16x16 for each wmma
          nvcuda::wmma::mma_sync(Accum[mii * (WN / 16) + nii], FragA[mii],
                                 FragB[nii], Accum[mii * (WN / 16) + nii]);
        }
      }
      __syncthreads(); // critical!
    }
  }
  // store Accum to SC (tensor core -> shared memory)
  for (int mii = 0; mii < WM / 16; mii += 1) {
    for (int nii = 0; nii < WN / 16; nii += 1) {
      // SC layout: [row_out, col_out, row_in, col_in] = [BM/16, BN/16, 16, 16]
      int row = ty * (WM / 16) + mii;
      int col = tz * (WN / 16) + nii;
      nvcuda::wmma::store_matrix_sync(
          SC + row * (BN / 16 * 16 * 16) + col * (16 * 16),
          Accum[mii * (WN / 16) + nii], 16, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  // store SC to O (shared memory -> global memory)
  int elePerThreadO = BM * BN / 32 / (BM / WM) / (BN / WN);
  for (int i = 0; i < elePerThreadO; i += 1) {
    int row = (i * tnum + tid) / BN;
    int column = (i * tnum + tid) % BN;
    // SC layout: [row_out, col_out, row_in, col_in] = [BM/16, BN/16, 16, 16]
    O[(row + bx * BM) * n + (column + by * BN)] =
        (half)(alpha * SC[(row / 16 * (BN / 16 * 16 * 16)) +
                          (column / 16 * (16 * 16)) + ((row % 16) * 16) +
                          column % 16] +
               beta * (float)O[(row + bx * BM) * n + (column + by * BN)]);
  }
}
