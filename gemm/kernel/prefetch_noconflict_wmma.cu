#include "../include/prefetch_noconflict_wmma.h"
#include <cassert>
#include <cstdint>

__global__ void pre_noconflict_wmma::gpuPrefetchNoConflictWmmaSgemm(
    int m, int n, int k, const half *A, const half *B, half *O,
    const float alpha, const float beta) {
  extern __shared__ uint8_t shared_storage[];
  half *SA_now = reinterpret_cast<half *>(shared_storage);
  half *SA_next =
      reinterpret_cast<half *>(shared_storage + BM * BK * sizeof(half));
  half *SB_now =
      reinterpret_cast<half *>(shared_storage + 2 * BM * BK * sizeof(half));
  half *SB_next = reinterpret_cast<half *>(
      shared_storage + (2 * BM * BK + BN * BK) * sizeof(half));
  float *SC = reinterpret_cast<float *>(shared_storage);
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int tid = tz * 32 * BM / WM + ty * 32 + tx;
  int tnum = 32 * BM / WM * BN / WN;
  const int elePerThreadA = BM * BK / 32 / (BM / WM) / (BN / WN);
  const int elePerThreadB = BN * BK / 32 / (BM / WM) / (BN / WN);
  const int elePerThreadO = BM * BN / 32 / (BM / WM) / (BN / WN);
  assert(elePerThreadA % 2 == 0);
  assert(elePerThreadB % 2 == 0);
  assert(elePerThreadO % 2 == 0);
  half regA[elePerThreadA];
  half regB[elePerThreadO];

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

  // load global A, B to regA, regB for initial stage
  for (int i = 0; i < elePerThreadA / 2; i += 1) {
    int id = i * (2 * tnum) + (2 * tid);
    int row = id / BK;
    int column = id % BK;
    regA[2 * i] = A[(row + bx * BM) * k + column];
    regA[2 * i + 1] = A[(row + bx * BM) * k + column + 1];
  }
  for (int i = 0; i < elePerThreadB / 2; i += 1) {
    int id = i * (2 * tnum) + (2 * tid);
    int row = id / BK;
    int column = id % BK;
    regB[2 * i] = B[(row + by * BN) * k + column];
    regB[2 * i + 1] = B[(row + by * BN) * k + column + 1];
  }
  // load regA, regB to SA_now, SB_now for initial stage
  for (int i = 0; i < elePerThreadA / 2; i += 1) {
    int id = i * (2 * tnum) + (2 * tid);
    int row = id / BK;
    int column = id % BK;
    // SA layout: [row_out, col_out, row_in, col_in] = [BM/16, BK/16, 16, 16]
    SA_now[(row / 16 * (BK / 16 * 16 * 16)) + (column / 16 * (16 * 16)) +
           ((row % 16) * 16) + (column % 16)] = regA[2 * i];
    SA_now[(row / 16 * (BK / 16 * 16 * 16)) + (column / 16 * (16 * 16)) +
           ((row % 16) * 16) + (column % 16) + 1] = regA[2 * i + 1];
  }
  for (int i = 0; i < elePerThreadB / 2; i += 1) {
    int id = i * (2 * tnum) + (2 * tid);
    int row = id / BK;
    int column = id % BK;
    // SB layout: [row_out, col_out, row_in, col_in] = [BN/16, BK/16, 16, 16]
    SB_now[(row / 16 * (BK / 16 * 16 * 16)) + (column / 16 * (16 * 16)) +
           ((row % 16) * 16) + (column % 16)] = regB[2 * i];
    SB_now[(row / 16 * (BK / 16 * 16 * 16)) + (column / 16 * (16 * 16)) +
           ((row % 16) * 16) + (column % 16) + 1] = regB[2 * i + 1];
  }
  __syncthreads();

  for (int ko = 1; ko < k / BK; ko += 1) {
    // load global A, B to regA, regB
    // the loading process can be overlapped with computing process
    for (int i = 0; i < elePerThreadA / 2; i += 1) {
      int id = i * (2 * tnum) + (2 * tid);
      int row = id / BK;
      int column = id % BK;
      regA[2 * i] = A[(row + bx * BM) * k + (column + ko * BK)];
      regA[2 * i + 1] = A[(row + bx * BM) * k + (column + ko * BK) + 1];
    }
    for (int i = 0; i < elePerThreadB / 2; i += 1) {
      int id = i * (2 * tnum) + (2 * tid);
      int row = id / BK;
      int column = id % BK;
      regB[2 * i] = B[(row + by * BN) * k + (column + ko * BK)];
      regB[2 * i + 1] = B[(row + by * BN) * k + (column + ko * BK) + 1];
    }
    // compute mma of SA_now and SB_now
    for (int kii = 0; kii < BK / WK; kii += 1) {
      // load SA, SB to tensor core (shared memory -> tensor core)
      for (int i = 0; i < WM / 16; i += 1) {
        int row = ty * (WM / 16) + i;
        int col = kii;
        nvcuda::wmma::load_matrix_sync(
            FragA[i], SA_now + row * (BK / 16 * 16 * 16) + col * (16 * 16), 16);
      }
      for (int i = 0; i < WN / 16; i += 1) {
        int row = tz * (WN / 16) + i;
        int col = kii;
        nvcuda::wmma::load_matrix_sync(
            FragB[i], SB_now + row * (BK / 16 * 16 * 16) + col * (16 * 16), 16);
      }
      // compute 16*16*16 wmma blocks
      for (int mii = 0; mii < WM / 16; mii += 1) {
        for (int nii = 0; nii < WN / 16; nii += 1) {
          // 16x16x16 for each wmma
          nvcuda::wmma::mma_sync(Accum[mii * (WN / 16) + nii], FragA[mii],
                                 FragB[nii], Accum[mii * (WN / 16) + nii]);
        }
      }
    }
    // load regA, regB to SA_next, SB_next
    for (int i = 0; i < elePerThreadA / 2; i += 1) {
      int id = i * (2 * tnum) + (2 * tid);
      int row = id / BK;
      int column = id % BK;
      // SA layout: [row_out, col_out, row_in, col_in] = [BM/16, BK/16, 16, 16]
      SA_next[(row / 16 * (BK / 16 * 16 * 16)) + (column / 16 * (16 * 16)) +
              ((row % 16) * 16) + (column % 16)] = regA[2 * i];
      SA_next[(row / 16 * (BK / 16 * 16 * 16)) + (column / 16 * (16 * 16)) +
              ((row % 16) * 16) + (column % 16) + 1] = regA[2 * i + 1];
    }
    for (int i = 0; i < elePerThreadB / 2; i += 1) {
      int id = i * (2 * tnum) + (2 * tid);
      int row = id / BK;
      int column = id % BK;
      // SB layout: [row_out, col_out, row_in, col_in] = [BN/16, BK/16, 16, 16]
      SB_next[(row / 16 * (BK / 16 * 16 * 16)) + (column / 16 * (16 * 16)) +
              ((row % 16) * 16) + (column % 16)] = regB[2 * i];
      SB_next[(row / 16 * (BK / 16 * 16 * 16)) + (column / 16 * (16 * 16)) +
              ((row % 16) * 16) + (column % 16) + 1] = regB[2 * i + 1];
    }
    __syncthreads();
    // switch buffer
    half *tmp = SA_now;
    SA_now = SA_next;
    SA_next = tmp;
    tmp = SB_now;
    SB_now = SB_next;
    SB_next = tmp;
  }
  // compute the last mma of SA_now and SB_now
  for (int kii = 0; kii < BK / WK; kii += 1) {
    // load SA, SB to tensor core (shared memory -> tensor core)
    for (int i = 0; i < WM / 16; i += 1) {
      int row = ty * (WM / 16) + i;
      int col = kii;
      nvcuda::wmma::load_matrix_sync(
          FragA[i], SA_now + row * (BK / 16 * 16 * 16) + col * (16 * 16), 16);
    }
    for (int i = 0; i < WN / 16; i += 1) {
      int row = tz * (WN / 16) + i;
      int col = kii;
      nvcuda::wmma::load_matrix_sync(
          FragB[i], SB_now + row * (BK / 16 * 16 * 16) + col * (16 * 16), 16);
    }
    // compute 16*16*16 wmma blocks
    for (int mii = 0; mii < WM / 16; mii += 1) {
      for (int nii = 0; nii < WN / 16; nii += 1) {
        // 16x16x16 for each wmma
        nvcuda::wmma::mma_sync(Accum[mii * (WN / 16) + nii], FragA[mii],
                               FragB[nii], Accum[mii * (WN / 16) + nii]);
      }
    }
  }
  __syncthreads();

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
  for (int i = 0; i < elePerThreadO / 2; i += 1) {
    int id = i * (2 * tnum) + (2 * tid);
    int row = id / BN;
    int column = id % BN;
    // SC layout: [row_out, col_out, row_in, col_in] = [BM/16, BN/16, 16, 16]
    O[(row + bx * BM) * n + (column + by * BN)] =
        (half)(alpha * SC[(row / 16 * (BN / 16 * 16 * 16)) +
                          (column / 16 * (16 * 16)) + ((row % 16) * 16) +
                          column % 16] +
               beta * (float)O[(row + bx * BM) * n + (column + by * BN)]);
    O[(row + bx * BM) * n + (column + by * BN) + 1] =
        (half)(alpha * SC[(row / 16 * (BN / 16 * 16 * 16)) +
                          (column / 16 * (16 * 16)) + ((row % 16) * 16) +
                          column % 16 + 1] +
               beta * (float)O[(row + bx * BM) * n + (column + by * BN) + 1]);
  }
}
