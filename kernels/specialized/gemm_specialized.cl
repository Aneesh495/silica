#include "common/math.clh"
#include "common/memory.clh"

// ============================================================================
// Specialized Register-Tiled Matrix Multiplications (SGEMM, CGEMM, Solvers)
// ============================================================================

// Register-Tiled GEMM: Tile (16x16x8), RegTile (2x2), WorkGroup (8x8)
__kernel void kernel_gemm_tile_16x16x8_reg_2x2(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[16][9];
    __local float l_B[8][17];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 16 + ty * 2;
    uint col_base = gx * 16 + tx * 2;
    float reg_c[2][2];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        for (uint i = tid; i < 128; i += 64) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 16 + r; uint gc = t * 8 + c;
            l_A[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 128; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = t * 8 + r; uint gc = gx * 16 + c;
            l_B[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[2]; float b_val[2];
            a_val[0] = l_A[ty * 2 + 0][k];
            a_val[1] = l_A[ty * 2 + 1][k];
            b_val[0] = l_B[k][tx * 2 + 0];
            b_val[1] = l_B[k][tx * 2 + 1];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
        }
        sync_local();
    }
    for (int i = 0; i < 2; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 2; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled GEMM: Tile (32x32x8), RegTile (4x4), WorkGroup (8x8)
__kernel void kernel_gemm_tile_32x32x8_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[32][9];
    __local float l_B[8][33];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 32 + ty * 4;
    uint col_base = gx * 32 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 32 + r; uint gc = t * 8 + c;
            l_A[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = t * 8 + r; uint gc = gx * 32 + c;
            l_B[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[ty * 4 + 0][k];
            a_val[1] = l_A[ty * 4 + 1][k];
            a_val[2] = l_A[ty * 4 + 2][k];
            a_val[3] = l_A[ty * 4 + 3][k];
            b_val[0] = l_B[k][tx * 4 + 0];
            b_val[1] = l_B[k][tx * 4 + 1];
            b_val[2] = l_B[k][tx * 4 + 2];
            b_val[3] = l_B[k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled GEMM: Tile (32x32x16), RegTile (4x4), WorkGroup (8x8)
__kernel void kernel_gemm_tile_32x32x16_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[32][17];
    __local float l_B[16][33];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 32 + ty * 4;
    uint col_base = gx * 32 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        for (uint i = tid; i < 512; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 32 + r; uint gc = t * 16 + c;
            l_A[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 512; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = t * 16 + r; uint gc = gx * 32 + c;
            l_B[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[ty * 4 + 0][k];
            a_val[1] = l_A[ty * 4 + 1][k];
            a_val[2] = l_A[ty * 4 + 2][k];
            a_val[3] = l_A[ty * 4 + 3][k];
            b_val[0] = l_B[k][tx * 4 + 0];
            b_val[1] = l_B[k][tx * 4 + 1];
            b_val[2] = l_B[k][tx * 4 + 2];
            b_val[3] = l_B[k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled GEMM: Tile (64x64x16), RegTile (4x4), WorkGroup (16x16)
__kernel void kernel_gemm_tile_64x64x16_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[64][17];
    __local float l_B[16][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4;
    uint col_base = gx * 64 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 64 + r; uint gc = t * 16 + c;
            l_A[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 16 + r; uint gc = gx * 64 + c;
            l_B[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[ty * 4 + 0][k];
            a_val[1] = l_A[ty * 4 + 1][k];
            a_val[2] = l_A[ty * 4 + 2][k];
            a_val[3] = l_A[ty * 4 + 3][k];
            b_val[0] = l_B[k][tx * 4 + 0];
            b_val[1] = l_B[k][tx * 4 + 1];
            b_val[2] = l_B[k][tx * 4 + 2];
            b_val[3] = l_B[k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled GEMM: Tile (64x64x8), RegTile (4x4), WorkGroup (16x16)
__kernel void kernel_gemm_tile_64x64x8_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[64][9];
    __local float l_B[8][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4;
    uint col_base = gx * 64 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        for (uint i = tid; i < 512; i += 256) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 64 + r; uint gc = t * 8 + c;
            l_A[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 512; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 8 + r; uint gc = gx * 64 + c;
            l_B[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[ty * 4 + 0][k];
            a_val[1] = l_A[ty * 4 + 1][k];
            a_val[2] = l_A[ty * 4 + 2][k];
            a_val[3] = l_A[ty * 4 + 3][k];
            b_val[0] = l_B[k][tx * 4 + 0];
            b_val[1] = l_B[k][tx * 4 + 1];
            b_val[2] = l_B[k][tx * 4 + 2];
            b_val[3] = l_B[k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled GEMM: Tile (128x64x8), RegTile (8x4), WorkGroup (16x16)
__kernel void kernel_gemm_tile_128x64x8_reg_8x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[128][9];
    __local float l_B[8][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 128 + ty * 8;
    uint col_base = gx * 64 + tx * 4;
    float reg_c[8][4];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 128 + r; uint gc = t * 8 + c;
            l_A[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 512; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 8 + r; uint gc = gx * 64 + c;
            l_B[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[8]; float b_val[4];
            a_val[0] = l_A[ty * 8 + 0][k];
            a_val[1] = l_A[ty * 8 + 1][k];
            a_val[2] = l_A[ty * 8 + 2][k];
            a_val[3] = l_A[ty * 8 + 3][k];
            a_val[4] = l_A[ty * 8 + 4][k];
            a_val[5] = l_A[ty * 8 + 5][k];
            a_val[6] = l_A[ty * 8 + 6][k];
            a_val[7] = l_A[ty * 8 + 7][k];
            b_val[0] = l_B[k][tx * 4 + 0];
            b_val[1] = l_B[k][tx * 4 + 1];
            b_val[2] = l_B[k][tx * 4 + 2];
            b_val[3] = l_B[k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
            reg_c[4][0] = fma(a_val[4], b_val[0], reg_c[4][0]);
            reg_c[4][1] = fma(a_val[4], b_val[1], reg_c[4][1]);
            reg_c[4][2] = fma(a_val[4], b_val[2], reg_c[4][2]);
            reg_c[4][3] = fma(a_val[4], b_val[3], reg_c[4][3]);
            reg_c[5][0] = fma(a_val[5], b_val[0], reg_c[5][0]);
            reg_c[5][1] = fma(a_val[5], b_val[1], reg_c[5][1]);
            reg_c[5][2] = fma(a_val[5], b_val[2], reg_c[5][2]);
            reg_c[5][3] = fma(a_val[5], b_val[3], reg_c[5][3]);
            reg_c[6][0] = fma(a_val[6], b_val[0], reg_c[6][0]);
            reg_c[6][1] = fma(a_val[6], b_val[1], reg_c[6][1]);
            reg_c[6][2] = fma(a_val[6], b_val[2], reg_c[6][2]);
            reg_c[6][3] = fma(a_val[6], b_val[3], reg_c[6][3]);
            reg_c[7][0] = fma(a_val[7], b_val[0], reg_c[7][0]);
            reg_c[7][1] = fma(a_val[7], b_val[1], reg_c[7][1]);
            reg_c[7][2] = fma(a_val[7], b_val[2], reg_c[7][2]);
            reg_c[7][3] = fma(a_val[7], b_val[3], reg_c[7][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 8; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Standard Tiled SGEMM: Mode NN
__kernel void kernel_sgemm_nn(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[16][16]; __local float l_B[16][16];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint row = get_group_id(1) * 16 + ty; uint col = get_group_id(0) * 16 + tx;
    float acc = 0.0f; uint num_tiles = (K + 15) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        l_A[ty][tx] = (row < M && (t * 16 + tx) < K) ? A[row * K + t * 16 + tx] : 0.0f;
        l_B[ty][tx] = ((t * 16 + ty) < K && col < N) ? B[(t * 16 + ty) * N + col] : 0.0f;
        sync_local();
        #pragma unroll
        for (int k = 0; k < 16; ++k) acc = fma(l_A[ty][k], l_B[k][tx], acc);
        sync_local();
    }
    if (row < M && col < N) {
        uint idx = row * N + col; float orig = (beta != 0.0f) ? C[idx] : 0.0f;
        C[idx] = alpha * acc + beta * orig;
    }
}

// Standard Tiled SGEMM: Mode NT
__kernel void kernel_sgemm_nt(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[16][16]; __local float l_B[16][16];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint row = get_group_id(1) * 16 + ty; uint col = get_group_id(0) * 16 + tx;
    float acc = 0.0f; uint num_tiles = (K + 15) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        l_A[ty][tx] = (row < M && (t * 16 + tx) < K) ? A[row * K + t * 16 + tx] : 0.0f;
        l_B[ty][tx] = ((t * 16 + ty) < K && col < N) ? B[col * K + t * 16 + ty] : 0.0f;
        sync_local();
        #pragma unroll
        for (int k = 0; k < 16; ++k) acc = fma(l_A[ty][k], l_B[k][tx], acc);
        sync_local();
    }
    if (row < M && col < N) {
        uint idx = row * N + col; float orig = (beta != 0.0f) ? C[idx] : 0.0f;
        C[idx] = alpha * acc + beta * orig;
    }
}

// Standard Tiled SGEMM: Mode TN
__kernel void kernel_sgemm_tn(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[16][16]; __local float l_B[16][16];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint row = get_group_id(1) * 16 + ty; uint col = get_group_id(0) * 16 + tx;
    float acc = 0.0f; uint num_tiles = (K + 15) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        l_A[ty][tx] = (row < M && (t * 16 + tx) < K) ? A[(t * 16 + tx) * M + row] : 0.0f;
        l_B[ty][tx] = ((t * 16 + ty) < K && col < N) ? B[(t * 16 + ty) * N + col] : 0.0f;
        sync_local();
        #pragma unroll
        for (int k = 0; k < 16; ++k) acc = fma(l_A[ty][k], l_B[k][tx], acc);
        sync_local();
    }
    if (row < M && col < N) {
        uint idx = row * N + col; float orig = (beta != 0.0f) ? C[idx] : 0.0f;
        C[idx] = alpha * acc + beta * orig;
    }
}

// Standard Tiled SGEMM: Mode TT
__kernel void kernel_sgemm_tt(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[16][16]; __local float l_B[16][16];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint row = get_group_id(1) * 16 + ty; uint col = get_group_id(0) * 16 + tx;
    float acc = 0.0f; uint num_tiles = (K + 15) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        l_A[ty][tx] = (row < M && (t * 16 + tx) < K) ? A[(t * 16 + tx) * M + row] : 0.0f;
        l_B[ty][tx] = ((t * 16 + ty) < K && col < N) ? B[col * K + t * 16 + ty] : 0.0f;
        sync_local();
        #pragma unroll
        for (int k = 0; k < 16; ++k) acc = fma(l_A[ty][k], l_B[k][tx], acc);
        sync_local();
    }
    if (row < M && col < N) {
        uint idx = row * N + col; float orig = (beta != 0.0f) ? C[idx] : 0.0f;
        C[idx] = alpha * acc + beta * orig;
    }
}

// Complex Tiled CGEMM: Mode NN
__kernel void kernel_cgemm_nn(
    __global const float2* A, __global const float2* B, __global float2* C,
    const uint M, const uint N, const uint K, const float2 alpha, const float2 beta) {
    uint row = get_global_id(1); uint col = get_global_id(0);
    if (row >= M || col >= N) return;
    float2 acc = (float2)(0.0f, 0.0f);
    for (uint k = 0; k < K; ++k) {
        float2 a_val = A[row * K + k]; float2 b_val = B[k * N + col];
        acc.x += a_val.x * b_val.x - a_val.y * b_val.y;
        acc.y += a_val.x * b_val.y + a_val.y * b_val.x;
    }
    uint idx = row * N + col;
    float2 orig = (beta.x != 0.0f || beta.y != 0.0f) ? C[idx] : (float2)(0.0f, 0.0f);
    float2 res;
    res.x = (alpha.x * acc.x - alpha.y * acc.y) + (beta.x * orig.x - beta.y * orig.y);
    res.y = (alpha.x * acc.y + alpha.y * acc.x) + (beta.x * orig.y + beta.y * orig.x);
    C[idx] = res;
}

// Complex Tiled CGEMM: Mode NT
__kernel void kernel_cgemm_nt(
    __global const float2* A, __global const float2* B, __global float2* C,
    const uint M, const uint N, const uint K, const float2 alpha, const float2 beta) {
    uint row = get_global_id(1); uint col = get_global_id(0);
    if (row >= M || col >= N) return;
    float2 acc = (float2)(0.0f, 0.0f);
    for (uint k = 0; k < K; ++k) {
        float2 a_val = A[row * K + k]; float2 b_val = B[col * K + k];
        acc.x += a_val.x * b_val.x - a_val.y * b_val.y;
        acc.y += a_val.x * b_val.y + a_val.y * b_val.x;
    }
    uint idx = row * N + col;
    float2 orig = (beta.x != 0.0f || beta.y != 0.0f) ? C[idx] : (float2)(0.0f, 0.0f);
    float2 res;
    res.x = (alpha.x * acc.x - alpha.y * acc.y) + (beta.x * orig.x - beta.y * orig.y);
    res.y = (alpha.x * acc.y + alpha.y * acc.x) + (beta.x * orig.y + beta.y * orig.x);
    C[idx] = res;
}

// Complex Tiled CGEMM: Mode TN
__kernel void kernel_cgemm_tn(
    __global const float2* A, __global const float2* B, __global float2* C,
    const uint M, const uint N, const uint K, const float2 alpha, const float2 beta) {
    uint row = get_global_id(1); uint col = get_global_id(0);
    if (row >= M || col >= N) return;
    float2 acc = (float2)(0.0f, 0.0f);
    for (uint k = 0; k < K; ++k) {
        float2 a_val = A[k * M + row]; float2 b_val = B[k * N + col];
        acc.x += a_val.x * b_val.x - a_val.y * b_val.y;
        acc.y += a_val.x * b_val.y + a_val.y * b_val.x;
    }
    uint idx = row * N + col;
    float2 orig = (beta.x != 0.0f || beta.y != 0.0f) ? C[idx] : (float2)(0.0f, 0.0f);
    float2 res;
    res.x = (alpha.x * acc.x - alpha.y * acc.y) + (beta.x * orig.x - beta.y * orig.y);
    res.y = (alpha.x * acc.y + alpha.y * acc.x) + (beta.x * orig.y + beta.y * orig.x);
    C[idx] = res;
}

// Complex Tiled CGEMM: Mode TT
__kernel void kernel_cgemm_tt(
    __global const float2* A, __global const float2* B, __global float2* C,
    const uint M, const uint N, const uint K, const float2 alpha, const float2 beta) {
    uint row = get_global_id(1); uint col = get_global_id(0);
    if (row >= M || col >= N) return;
    float2 acc = (float2)(0.0f, 0.0f);
    for (uint k = 0; k < K; ++k) {
        float2 a_val = A[k * M + row]; float2 b_val = B[col * K + k];
        acc.x += a_val.x * b_val.x - a_val.y * b_val.y;
        acc.y += a_val.x * b_val.y + a_val.y * b_val.x;
    }
    uint idx = row * N + col;
    float2 orig = (beta.x != 0.0f || beta.y != 0.0f) ? C[idx] : (float2)(0.0f, 0.0f);
    float2 res;
    res.x = (alpha.x * acc.x - alpha.y * acc.y) + (beta.x * orig.x - beta.y * orig.y);
    res.y = (alpha.x * acc.y + alpha.y * acc.x) + (beta.x * orig.y + beta.y * orig.x);
    C[idx] = res;
}

// Batched 2x2 Matrix Inversion with Partial Pivoting
__kernel void kernel_batched_inv_2x2(
    __global const float* in_mats, __global float* out_invs, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 4;
    float A[2][2]; float Inv[2][2];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) { A[i][j] = in_mats[base + i * 2 + j]; Inv[i][j] = (i == j) ? 1.0f : 0.0f; }
    for (int i = 0; i < 2; ++i) {
        int pivot = i; float max_v = fabs(A[i][i]);
        for (int k = i + 1; k < 2; ++k) { float v = fabs(A[k][i]); if (v > max_v) { max_v = v; pivot = k; } }
        if (pivot != i) {
            for (int j = 0; j < 2; ++j) {
                float ta = A[i][j]; A[i][j] = A[pivot][j]; A[pivot][j] = ta;
                float ti = Inv[i][j]; Inv[i][j] = Inv[pivot][j]; Inv[pivot][j] = ti;
            }
        }
        float diag = A[i][i]; float inv_d = (fabs(diag) > 1.0e-9f) ? (1.0f / diag) : 0.0f;
        for (int j = 0; j < 2; ++j) { A[i][j] *= inv_d; Inv[i][j] *= inv_d; }
        for (int k = 0; k < 2; ++k) {
            if (k != i) {
                float f = A[k][i];
                for (int j = 0; j < 2; ++j) { A[k][j] -= f * A[i][j]; Inv[k][j] -= f * Inv[i][j]; }
            }
        }
    }
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) out_invs[base + i * 2 + j] = Inv[i][j];
}

// Batched 3x3 Matrix Inversion with Partial Pivoting
__kernel void kernel_batched_inv_3x3(
    __global const float* in_mats, __global float* out_invs, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 9;
    float A[3][3]; float Inv[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { A[i][j] = in_mats[base + i * 3 + j]; Inv[i][j] = (i == j) ? 1.0f : 0.0f; }
    for (int i = 0; i < 3; ++i) {
        int pivot = i; float max_v = fabs(A[i][i]);
        for (int k = i + 1; k < 3; ++k) { float v = fabs(A[k][i]); if (v > max_v) { max_v = v; pivot = k; } }
        if (pivot != i) {
            for (int j = 0; j < 3; ++j) {
                float ta = A[i][j]; A[i][j] = A[pivot][j]; A[pivot][j] = ta;
                float ti = Inv[i][j]; Inv[i][j] = Inv[pivot][j]; Inv[pivot][j] = ti;
            }
        }
        float diag = A[i][i]; float inv_d = (fabs(diag) > 1.0e-9f) ? (1.0f / diag) : 0.0f;
        for (int j = 0; j < 3; ++j) { A[i][j] *= inv_d; Inv[i][j] *= inv_d; }
        for (int k = 0; k < 3; ++k) {
            if (k != i) {
                float f = A[k][i];
                for (int j = 0; j < 3; ++j) { A[k][j] -= f * A[i][j]; Inv[k][j] -= f * Inv[i][j]; }
            }
        }
    }
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) out_invs[base + i * 3 + j] = Inv[i][j];
}

// Batched 4x4 Matrix Inversion with Partial Pivoting
__kernel void kernel_batched_inv_4x4(
    __global const float* in_mats, __global float* out_invs, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 16;
    float A[4][4]; float Inv[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) { A[i][j] = in_mats[base + i * 4 + j]; Inv[i][j] = (i == j) ? 1.0f : 0.0f; }
    for (int i = 0; i < 4; ++i) {
        int pivot = i; float max_v = fabs(A[i][i]);
        for (int k = i + 1; k < 4; ++k) { float v = fabs(A[k][i]); if (v > max_v) { max_v = v; pivot = k; } }
        if (pivot != i) {
            for (int j = 0; j < 4; ++j) {
                float ta = A[i][j]; A[i][j] = A[pivot][j]; A[pivot][j] = ta;
                float ti = Inv[i][j]; Inv[i][j] = Inv[pivot][j]; Inv[pivot][j] = ti;
            }
        }
        float diag = A[i][i]; float inv_d = (fabs(diag) > 1.0e-9f) ? (1.0f / diag) : 0.0f;
        for (int j = 0; j < 4; ++j) { A[i][j] *= inv_d; Inv[i][j] *= inv_d; }
        for (int k = 0; k < 4; ++k) {
            if (k != i) {
                float f = A[k][i];
                for (int j = 0; j < 4; ++j) { A[k][j] -= f * A[i][j]; Inv[k][j] -= f * Inv[i][j]; }
            }
        }
    }
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) out_invs[base + i * 4 + j] = Inv[i][j];
}

// Batched 5x5 Matrix Inversion with Partial Pivoting
__kernel void kernel_batched_inv_5x5(
    __global const float* in_mats, __global float* out_invs, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 25;
    float A[5][5]; float Inv[5][5];
    for (int i = 0; i < 5; ++i) for (int j = 0; j < 5; ++j) { A[i][j] = in_mats[base + i * 5 + j]; Inv[i][j] = (i == j) ? 1.0f : 0.0f; }
    for (int i = 0; i < 5; ++i) {
        int pivot = i; float max_v = fabs(A[i][i]);
        for (int k = i + 1; k < 5; ++k) { float v = fabs(A[k][i]); if (v > max_v) { max_v = v; pivot = k; } }
        if (pivot != i) {
            for (int j = 0; j < 5; ++j) {
                float ta = A[i][j]; A[i][j] = A[pivot][j]; A[pivot][j] = ta;
                float ti = Inv[i][j]; Inv[i][j] = Inv[pivot][j]; Inv[pivot][j] = ti;
            }
        }
        float diag = A[i][i]; float inv_d = (fabs(diag) > 1.0e-9f) ? (1.0f / diag) : 0.0f;
        for (int j = 0; j < 5; ++j) { A[i][j] *= inv_d; Inv[i][j] *= inv_d; }
        for (int k = 0; k < 5; ++k) {
            if (k != i) {
                float f = A[k][i];
                for (int j = 0; j < 5; ++j) { A[k][j] -= f * A[i][j]; Inv[k][j] -= f * Inv[i][j]; }
            }
        }
    }
    for (int i = 0; i < 5; ++i) for (int j = 0; j < 5; ++j) out_invs[base + i * 5 + j] = Inv[i][j];
}

// Batched 6x6 Matrix Inversion with Partial Pivoting
__kernel void kernel_batched_inv_6x6(
    __global const float* in_mats, __global float* out_invs, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 36;
    float A[6][6]; float Inv[6][6];
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) { A[i][j] = in_mats[base + i * 6 + j]; Inv[i][j] = (i == j) ? 1.0f : 0.0f; }
    for (int i = 0; i < 6; ++i) {
        int pivot = i; float max_v = fabs(A[i][i]);
        for (int k = i + 1; k < 6; ++k) { float v = fabs(A[k][i]); if (v > max_v) { max_v = v; pivot = k; } }
        if (pivot != i) {
            for (int j = 0; j < 6; ++j) {
                float ta = A[i][j]; A[i][j] = A[pivot][j]; A[pivot][j] = ta;
                float ti = Inv[i][j]; Inv[i][j] = Inv[pivot][j]; Inv[pivot][j] = ti;
            }
        }
        float diag = A[i][i]; float inv_d = (fabs(diag) > 1.0e-9f) ? (1.0f / diag) : 0.0f;
        for (int j = 0; j < 6; ++j) { A[i][j] *= inv_d; Inv[i][j] *= inv_d; }
        for (int k = 0; k < 6; ++k) {
            if (k != i) {
                float f = A[k][i];
                for (int j = 0; j < 6; ++j) { A[k][j] -= f * A[i][j]; Inv[k][j] -= f * Inv[i][j]; }
            }
        }
    }
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) out_invs[base + i * 6 + j] = Inv[i][j];
}

// Batched 8x8 Matrix Inversion with Partial Pivoting
__kernel void kernel_batched_inv_8x8(
    __global const float* in_mats, __global float* out_invs, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 64;
    float A[8][8]; float Inv[8][8];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) { A[i][j] = in_mats[base + i * 8 + j]; Inv[i][j] = (i == j) ? 1.0f : 0.0f; }
    for (int i = 0; i < 8; ++i) {
        int pivot = i; float max_v = fabs(A[i][i]);
        for (int k = i + 1; k < 8; ++k) { float v = fabs(A[k][i]); if (v > max_v) { max_v = v; pivot = k; } }
        if (pivot != i) {
            for (int j = 0; j < 8; ++j) {
                float ta = A[i][j]; A[i][j] = A[pivot][j]; A[pivot][j] = ta;
                float ti = Inv[i][j]; Inv[i][j] = Inv[pivot][j]; Inv[pivot][j] = ti;
            }
        }
        float diag = A[i][i]; float inv_d = (fabs(diag) > 1.0e-9f) ? (1.0f / diag) : 0.0f;
        for (int j = 0; j < 8; ++j) { A[i][j] *= inv_d; Inv[i][j] *= inv_d; }
        for (int k = 0; k < 8; ++k) {
            if (k != i) {
                float f = A[k][i];
                for (int j = 0; j < 8; ++j) { A[k][j] -= f * A[i][j]; Inv[k][j] -= f * Inv[i][j]; }
            }
        }
    }
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) out_invs[base + i * 8 + j] = Inv[i][j];
}

// Batched 10x10 Matrix Inversion with Partial Pivoting
__kernel void kernel_batched_inv_10x10(
    __global const float* in_mats, __global float* out_invs, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 100;
    float A[10][10]; float Inv[10][10];
    for (int i = 0; i < 10; ++i) for (int j = 0; j < 10; ++j) { A[i][j] = in_mats[base + i * 10 + j]; Inv[i][j] = (i == j) ? 1.0f : 0.0f; }
    for (int i = 0; i < 10; ++i) {
        int pivot = i; float max_v = fabs(A[i][i]);
        for (int k = i + 1; k < 10; ++k) { float v = fabs(A[k][i]); if (v > max_v) { max_v = v; pivot = k; } }
        if (pivot != i) {
            for (int j = 0; j < 10; ++j) {
                float ta = A[i][j]; A[i][j] = A[pivot][j]; A[pivot][j] = ta;
                float ti = Inv[i][j]; Inv[i][j] = Inv[pivot][j]; Inv[pivot][j] = ti;
            }
        }
        float diag = A[i][i]; float inv_d = (fabs(diag) > 1.0e-9f) ? (1.0f / diag) : 0.0f;
        for (int j = 0; j < 10; ++j) { A[i][j] *= inv_d; Inv[i][j] *= inv_d; }
        for (int k = 0; k < 10; ++k) {
            if (k != i) {
                float f = A[k][i];
                for (int j = 0; j < 10; ++j) { A[k][j] -= f * A[i][j]; Inv[k][j] -= f * Inv[i][j]; }
            }
        }
    }
    for (int i = 0; i < 10; ++i) for (int j = 0; j < 10; ++j) out_invs[base + i * 10 + j] = Inv[i][j];
}

// Batched 12x12 Matrix Inversion with Partial Pivoting
__kernel void kernel_batched_inv_12x12(
    __global const float* in_mats, __global float* out_invs, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 144;
    float A[12][12]; float Inv[12][12];
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) { A[i][j] = in_mats[base + i * 12 + j]; Inv[i][j] = (i == j) ? 1.0f : 0.0f; }
    for (int i = 0; i < 12; ++i) {
        int pivot = i; float max_v = fabs(A[i][i]);
        for (int k = i + 1; k < 12; ++k) { float v = fabs(A[k][i]); if (v > max_v) { max_v = v; pivot = k; } }
        if (pivot != i) {
            for (int j = 0; j < 12; ++j) {
                float ta = A[i][j]; A[i][j] = A[pivot][j]; A[pivot][j] = ta;
                float ti = Inv[i][j]; Inv[i][j] = Inv[pivot][j]; Inv[pivot][j] = ti;
            }
        }
        float diag = A[i][i]; float inv_d = (fabs(diag) > 1.0e-9f) ? (1.0f / diag) : 0.0f;
        for (int j = 0; j < 12; ++j) { A[i][j] *= inv_d; Inv[i][j] *= inv_d; }
        for (int k = 0; k < 12; ++k) {
            if (k != i) {
                float f = A[k][i];
                for (int j = 0; j < 12; ++j) { A[k][j] -= f * A[i][j]; Inv[k][j] -= f * Inv[i][j]; }
            }
        }
    }
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) out_invs[base + i * 12 + j] = Inv[i][j];
}

// Batched 3x3 Cholesky Factorization (A = L * L^T)
__kernel void kernel_batched_cholesky_3x3(
    __global const float* in_mats, __global float* out_L, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 9;
    float L[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) L[i][j] = 0.0f;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < j; ++k) sum += L[i][k] * L[j][k];
            if (i == j) {
                float val = in_mats[base + i * 3 + i] - sum;
                L[i][j] = (val > 1.0e-9f) ? sqrt(val) : 0.0f;
            } else {
                float diag = L[j][j];
                L[i][j] = (diag > 1.0e-9f) ? ((in_mats[base + i * 3 + j] - sum) / diag) : 0.0f;
            }
        }
    }
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) out_L[base + i * 3 + j] = L[i][j];
}

// Batched 4x4 Cholesky Factorization (A = L * L^T)
__kernel void kernel_batched_cholesky_4x4(
    __global const float* in_mats, __global float* out_L, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 16;
    float L[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) L[i][j] = 0.0f;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < j; ++k) sum += L[i][k] * L[j][k];
            if (i == j) {
                float val = in_mats[base + i * 4 + i] - sum;
                L[i][j] = (val > 1.0e-9f) ? sqrt(val) : 0.0f;
            } else {
                float diag = L[j][j];
                L[i][j] = (diag > 1.0e-9f) ? ((in_mats[base + i * 4 + j] - sum) / diag) : 0.0f;
            }
        }
    }
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) out_L[base + i * 4 + j] = L[i][j];
}

// Batched 5x5 Cholesky Factorization (A = L * L^T)
__kernel void kernel_batched_cholesky_5x5(
    __global const float* in_mats, __global float* out_L, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 25;
    float L[5][5];
    for (int i = 0; i < 5; ++i) for (int j = 0; j < 5; ++j) L[i][j] = 0.0f;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < j; ++k) sum += L[i][k] * L[j][k];
            if (i == j) {
                float val = in_mats[base + i * 5 + i] - sum;
                L[i][j] = (val > 1.0e-9f) ? sqrt(val) : 0.0f;
            } else {
                float diag = L[j][j];
                L[i][j] = (diag > 1.0e-9f) ? ((in_mats[base + i * 5 + j] - sum) / diag) : 0.0f;
            }
        }
    }
    for (int i = 0; i < 5; ++i) for (int j = 0; j < 5; ++j) out_L[base + i * 5 + j] = L[i][j];
}

// Batched 6x6 Cholesky Factorization (A = L * L^T)
__kernel void kernel_batched_cholesky_6x6(
    __global const float* in_mats, __global float* out_L, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 36;
    float L[6][6];
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) L[i][j] = 0.0f;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < j; ++k) sum += L[i][k] * L[j][k];
            if (i == j) {
                float val = in_mats[base + i * 6 + i] - sum;
                L[i][j] = (val > 1.0e-9f) ? sqrt(val) : 0.0f;
            } else {
                float diag = L[j][j];
                L[i][j] = (diag > 1.0e-9f) ? ((in_mats[base + i * 6 + j] - sum) / diag) : 0.0f;
            }
        }
    }
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) out_L[base + i * 6 + j] = L[i][j];
}

// Batched 8x8 Cholesky Factorization (A = L * L^T)
__kernel void kernel_batched_cholesky_8x8(
    __global const float* in_mats, __global float* out_L, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 64;
    float L[8][8];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) L[i][j] = 0.0f;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < j; ++k) sum += L[i][k] * L[j][k];
            if (i == j) {
                float val = in_mats[base + i * 8 + i] - sum;
                L[i][j] = (val > 1.0e-9f) ? sqrt(val) : 0.0f;
            } else {
                float diag = L[j][j];
                L[i][j] = (diag > 1.0e-9f) ? ((in_mats[base + i * 8 + j] - sum) / diag) : 0.0f;
            }
        }
    }
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) out_L[base + i * 8 + j] = L[i][j];
}

// Batched 10x10 Cholesky Factorization (A = L * L^T)
__kernel void kernel_batched_cholesky_10x10(
    __global const float* in_mats, __global float* out_L, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 100;
    float L[10][10];
    for (int i = 0; i < 10; ++i) for (int j = 0; j < 10; ++j) L[i][j] = 0.0f;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < j; ++k) sum += L[i][k] * L[j][k];
            if (i == j) {
                float val = in_mats[base + i * 10 + i] - sum;
                L[i][j] = (val > 1.0e-9f) ? sqrt(val) : 0.0f;
            } else {
                float diag = L[j][j];
                L[i][j] = (diag > 1.0e-9f) ? ((in_mats[base + i * 10 + j] - sum) / diag) : 0.0f;
            }
        }
    }
    for (int i = 0; i < 10; ++i) for (int j = 0; j < 10; ++j) out_L[base + i * 10 + j] = L[i][j];
}


// ============================================================================
// High-Performance Double-Buffered Register-Tiled GEMM (SGEMM & CGEMM)
// ============================================================================

// Register-Tiled Double-Buffered SGEMM: Mode NN, Tile (16x16x8), RegTile (2x2)
__kernel void kernel_gemm_dbuf_nn_tile_16x16x8_reg_2x2(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][16][9];
    __local float l_B[2][8][17];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 16 + ty * 2; uint col_base = gx * 16 + tx * 2;
    float reg_c[2][2];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 128; i += 64) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 16 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 128; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = t * 8 + r; uint gc = gx * 16 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[2]; float b_val[2];
            a_val[0] = l_A[buf][ty * 2 + 0][k];
            a_val[1] = l_A[buf][ty * 2 + 1][k];
            b_val[0] = l_B[buf][k][tx * 2 + 0];
            b_val[1] = l_B[buf][k][tx * 2 + 1];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
        }
        sync_local();
    }
    for (int i = 0; i < 2; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 2; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NN, Tile (16x16x16), RegTile (2x2)
__kernel void kernel_gemm_dbuf_nn_tile_16x16x16_reg_2x2(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][16][17];
    __local float l_B[2][16][17];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 16 + ty * 2; uint col_base = gx * 16 + tx * 2;
    float reg_c[2][2];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 16 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = t * 16 + r; uint gc = gx * 16 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[2]; float b_val[2];
            a_val[0] = l_A[buf][ty * 2 + 0][k];
            a_val[1] = l_A[buf][ty * 2 + 1][k];
            b_val[0] = l_B[buf][k][tx * 2 + 0];
            b_val[1] = l_B[buf][k][tx * 2 + 1];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
        }
        sync_local();
    }
    for (int i = 0; i < 2; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 2; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NN, Tile (32x32x8), RegTile (4x4)
__kernel void kernel_gemm_dbuf_nn_tile_32x32x8_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][32][9];
    __local float l_B[2][8][33];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 32 + ty * 4; uint col_base = gx * 32 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 32 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = t * 8 + r; uint gc = gx * 32 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NN, Tile (32x32x16), RegTile (4x4)
__kernel void kernel_gemm_dbuf_nn_tile_32x32x16_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][32][17];
    __local float l_B[2][16][33];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 32 + ty * 4; uint col_base = gx * 32 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 512; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 32 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 512; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = t * 16 + r; uint gc = gx * 32 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NN, Tile (32x32x32), RegTile (4x4)
__kernel void kernel_gemm_dbuf_nn_tile_32x32x32_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][32][33];
    __local float l_B[2][32][33];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 32 + ty * 4; uint col_base = gx * 32 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 32 - 1) / 32;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = gy * 32 + r; uint gc = t * 32 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 1024; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = t * 32 + r; uint gc = gx * 32 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 32; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NN, Tile (64x64x8), RegTile (4x4)
__kernel void kernel_gemm_dbuf_nn_tile_64x64x8_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][9];
    __local float l_B[2][8][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 64 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 512; i += 256) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 64 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 512; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 8 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NN, Tile (64x64x16), RegTile (4x4)
__kernel void kernel_gemm_dbuf_nn_tile_64x64x16_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][17];
    __local float l_B[2][16][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 64 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 64 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 16 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NN, Tile (64x64x32), RegTile (4x4)
__kernel void kernel_gemm_dbuf_nn_tile_64x64x32_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][33];
    __local float l_B[2][32][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 64 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 32 - 1) / 32;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 32; uint c = i % 32;
            uint gr = gy * 64 + r; uint gc = t * 32 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 32 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 32; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NN, Tile (128x64x8), RegTile (8x4)
__kernel void kernel_gemm_dbuf_nn_tile_128x64x8_reg_8x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][128][9];
    __local float l_B[2][8][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 128 + ty * 8; uint col_base = gx * 64 + tx * 4;
    float reg_c[8][4];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 128 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 512; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 8 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[8]; float b_val[4];
            a_val[0] = l_A[buf][ty * 8 + 0][k];
            a_val[1] = l_A[buf][ty * 8 + 1][k];
            a_val[2] = l_A[buf][ty * 8 + 2][k];
            a_val[3] = l_A[buf][ty * 8 + 3][k];
            a_val[4] = l_A[buf][ty * 8 + 4][k];
            a_val[5] = l_A[buf][ty * 8 + 5][k];
            a_val[6] = l_A[buf][ty * 8 + 6][k];
            a_val[7] = l_A[buf][ty * 8 + 7][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
            reg_c[4][0] = fma(a_val[4], b_val[0], reg_c[4][0]);
            reg_c[4][1] = fma(a_val[4], b_val[1], reg_c[4][1]);
            reg_c[4][2] = fma(a_val[4], b_val[2], reg_c[4][2]);
            reg_c[4][3] = fma(a_val[4], b_val[3], reg_c[4][3]);
            reg_c[5][0] = fma(a_val[5], b_val[0], reg_c[5][0]);
            reg_c[5][1] = fma(a_val[5], b_val[1], reg_c[5][1]);
            reg_c[5][2] = fma(a_val[5], b_val[2], reg_c[5][2]);
            reg_c[5][3] = fma(a_val[5], b_val[3], reg_c[5][3]);
            reg_c[6][0] = fma(a_val[6], b_val[0], reg_c[6][0]);
            reg_c[6][1] = fma(a_val[6], b_val[1], reg_c[6][1]);
            reg_c[6][2] = fma(a_val[6], b_val[2], reg_c[6][2]);
            reg_c[6][3] = fma(a_val[6], b_val[3], reg_c[6][3]);
            reg_c[7][0] = fma(a_val[7], b_val[0], reg_c[7][0]);
            reg_c[7][1] = fma(a_val[7], b_val[1], reg_c[7][1]);
            reg_c[7][2] = fma(a_val[7], b_val[2], reg_c[7][2]);
            reg_c[7][3] = fma(a_val[7], b_val[3], reg_c[7][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 8; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NN, Tile (128x64x16), RegTile (8x4)
__kernel void kernel_gemm_dbuf_nn_tile_128x64x16_reg_8x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][128][17];
    __local float l_B[2][16][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 128 + ty * 8; uint col_base = gx * 64 + tx * 4;
    float reg_c[8][4];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 128 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 16 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[8]; float b_val[4];
            a_val[0] = l_A[buf][ty * 8 + 0][k];
            a_val[1] = l_A[buf][ty * 8 + 1][k];
            a_val[2] = l_A[buf][ty * 8 + 2][k];
            a_val[3] = l_A[buf][ty * 8 + 3][k];
            a_val[4] = l_A[buf][ty * 8 + 4][k];
            a_val[5] = l_A[buf][ty * 8 + 5][k];
            a_val[6] = l_A[buf][ty * 8 + 6][k];
            a_val[7] = l_A[buf][ty * 8 + 7][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
            reg_c[4][0] = fma(a_val[4], b_val[0], reg_c[4][0]);
            reg_c[4][1] = fma(a_val[4], b_val[1], reg_c[4][1]);
            reg_c[4][2] = fma(a_val[4], b_val[2], reg_c[4][2]);
            reg_c[4][3] = fma(a_val[4], b_val[3], reg_c[4][3]);
            reg_c[5][0] = fma(a_val[5], b_val[0], reg_c[5][0]);
            reg_c[5][1] = fma(a_val[5], b_val[1], reg_c[5][1]);
            reg_c[5][2] = fma(a_val[5], b_val[2], reg_c[5][2]);
            reg_c[5][3] = fma(a_val[5], b_val[3], reg_c[5][3]);
            reg_c[6][0] = fma(a_val[6], b_val[0], reg_c[6][0]);
            reg_c[6][1] = fma(a_val[6], b_val[1], reg_c[6][1]);
            reg_c[6][2] = fma(a_val[6], b_val[2], reg_c[6][2]);
            reg_c[6][3] = fma(a_val[6], b_val[3], reg_c[6][3]);
            reg_c[7][0] = fma(a_val[7], b_val[0], reg_c[7][0]);
            reg_c[7][1] = fma(a_val[7], b_val[1], reg_c[7][1]);
            reg_c[7][2] = fma(a_val[7], b_val[2], reg_c[7][2]);
            reg_c[7][3] = fma(a_val[7], b_val[3], reg_c[7][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 8; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NN, Tile (64x128x16), RegTile (4x8)
__kernel void kernel_gemm_dbuf_nn_tile_64x128x16_reg_4x8(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][17];
    __local float l_B[2][16][129];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 128 + tx * 8;
    float reg_c[4][8];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 8; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 64 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 128; uint c = i % 128;
            uint gr = t * 16 + r; uint gc = gx * 128 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[4]; float b_val[8];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 8 + 0];
            b_val[1] = l_B[buf][k][tx * 8 + 1];
            b_val[2] = l_B[buf][k][tx * 8 + 2];
            b_val[3] = l_B[buf][k][tx * 8 + 3];
            b_val[4] = l_B[buf][k][tx * 8 + 4];
            b_val[5] = l_B[buf][k][tx * 8 + 5];
            b_val[6] = l_B[buf][k][tx * 8 + 6];
            b_val[7] = l_B[buf][k][tx * 8 + 7];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[0][4] = fma(a_val[0], b_val[4], reg_c[0][4]);
            reg_c[0][5] = fma(a_val[0], b_val[5], reg_c[0][5]);
            reg_c[0][6] = fma(a_val[0], b_val[6], reg_c[0][6]);
            reg_c[0][7] = fma(a_val[0], b_val[7], reg_c[0][7]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[1][4] = fma(a_val[1], b_val[4], reg_c[1][4]);
            reg_c[1][5] = fma(a_val[1], b_val[5], reg_c[1][5]);
            reg_c[1][6] = fma(a_val[1], b_val[6], reg_c[1][6]);
            reg_c[1][7] = fma(a_val[1], b_val[7], reg_c[1][7]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[2][4] = fma(a_val[2], b_val[4], reg_c[2][4]);
            reg_c[2][5] = fma(a_val[2], b_val[5], reg_c[2][5]);
            reg_c[2][6] = fma(a_val[2], b_val[6], reg_c[2][6]);
            reg_c[2][7] = fma(a_val[2], b_val[7], reg_c[2][7]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
            reg_c[3][4] = fma(a_val[3], b_val[4], reg_c[3][4]);
            reg_c[3][5] = fma(a_val[3], b_val[5], reg_c[3][5]);
            reg_c[3][6] = fma(a_val[3], b_val[6], reg_c[3][6]);
            reg_c[3][7] = fma(a_val[3], b_val[7], reg_c[3][7]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 8; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NT, Tile (16x16x8), RegTile (2x2)
__kernel void kernel_gemm_dbuf_nt_tile_16x16x8_reg_2x2(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][16][9];
    __local float l_B[2][8][17];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 16 + ty * 2; uint col_base = gx * 16 + tx * 2;
    float reg_c[2][2];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 128; i += 64) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 16 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 128; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = t * 8 + r; uint gc = gx * 16 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[2]; float b_val[2];
            a_val[0] = l_A[buf][ty * 2 + 0][k];
            a_val[1] = l_A[buf][ty * 2 + 1][k];
            b_val[0] = l_B[buf][k][tx * 2 + 0];
            b_val[1] = l_B[buf][k][tx * 2 + 1];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
        }
        sync_local();
    }
    for (int i = 0; i < 2; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 2; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NT, Tile (16x16x16), RegTile (2x2)
__kernel void kernel_gemm_dbuf_nt_tile_16x16x16_reg_2x2(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][16][17];
    __local float l_B[2][16][17];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 16 + ty * 2; uint col_base = gx * 16 + tx * 2;
    float reg_c[2][2];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 16 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = t * 16 + r; uint gc = gx * 16 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[2]; float b_val[2];
            a_val[0] = l_A[buf][ty * 2 + 0][k];
            a_val[1] = l_A[buf][ty * 2 + 1][k];
            b_val[0] = l_B[buf][k][tx * 2 + 0];
            b_val[1] = l_B[buf][k][tx * 2 + 1];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
        }
        sync_local();
    }
    for (int i = 0; i < 2; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 2; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NT, Tile (32x32x8), RegTile (4x4)
__kernel void kernel_gemm_dbuf_nt_tile_32x32x8_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][32][9];
    __local float l_B[2][8][33];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 32 + ty * 4; uint col_base = gx * 32 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 32 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = t * 8 + r; uint gc = gx * 32 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NT, Tile (32x32x16), RegTile (4x4)
__kernel void kernel_gemm_dbuf_nt_tile_32x32x16_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][32][17];
    __local float l_B[2][16][33];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 32 + ty * 4; uint col_base = gx * 32 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 512; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 32 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 512; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = t * 16 + r; uint gc = gx * 32 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NT, Tile (32x32x32), RegTile (4x4)
__kernel void kernel_gemm_dbuf_nt_tile_32x32x32_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][32][33];
    __local float l_B[2][32][33];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 32 + ty * 4; uint col_base = gx * 32 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 32 - 1) / 32;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = gy * 32 + r; uint gc = t * 32 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 1024; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = t * 32 + r; uint gc = gx * 32 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 32; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NT, Tile (64x64x8), RegTile (4x4)
__kernel void kernel_gemm_dbuf_nt_tile_64x64x8_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][9];
    __local float l_B[2][8][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 64 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 512; i += 256) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 64 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 512; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 8 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NT, Tile (64x64x16), RegTile (4x4)
__kernel void kernel_gemm_dbuf_nt_tile_64x64x16_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][17];
    __local float l_B[2][16][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 64 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 64 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 16 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NT, Tile (64x64x32), RegTile (4x4)
__kernel void kernel_gemm_dbuf_nt_tile_64x64x32_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][33];
    __local float l_B[2][32][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 64 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 32 - 1) / 32;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 32; uint c = i % 32;
            uint gr = gy * 64 + r; uint gc = t * 32 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 32 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 32; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NT, Tile (128x64x8), RegTile (8x4)
__kernel void kernel_gemm_dbuf_nt_tile_128x64x8_reg_8x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][128][9];
    __local float l_B[2][8][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 128 + ty * 8; uint col_base = gx * 64 + tx * 4;
    float reg_c[8][4];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 128 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 512; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 8 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[8]; float b_val[4];
            a_val[0] = l_A[buf][ty * 8 + 0][k];
            a_val[1] = l_A[buf][ty * 8 + 1][k];
            a_val[2] = l_A[buf][ty * 8 + 2][k];
            a_val[3] = l_A[buf][ty * 8 + 3][k];
            a_val[4] = l_A[buf][ty * 8 + 4][k];
            a_val[5] = l_A[buf][ty * 8 + 5][k];
            a_val[6] = l_A[buf][ty * 8 + 6][k];
            a_val[7] = l_A[buf][ty * 8 + 7][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
            reg_c[4][0] = fma(a_val[4], b_val[0], reg_c[4][0]);
            reg_c[4][1] = fma(a_val[4], b_val[1], reg_c[4][1]);
            reg_c[4][2] = fma(a_val[4], b_val[2], reg_c[4][2]);
            reg_c[4][3] = fma(a_val[4], b_val[3], reg_c[4][3]);
            reg_c[5][0] = fma(a_val[5], b_val[0], reg_c[5][0]);
            reg_c[5][1] = fma(a_val[5], b_val[1], reg_c[5][1]);
            reg_c[5][2] = fma(a_val[5], b_val[2], reg_c[5][2]);
            reg_c[5][3] = fma(a_val[5], b_val[3], reg_c[5][3]);
            reg_c[6][0] = fma(a_val[6], b_val[0], reg_c[6][0]);
            reg_c[6][1] = fma(a_val[6], b_val[1], reg_c[6][1]);
            reg_c[6][2] = fma(a_val[6], b_val[2], reg_c[6][2]);
            reg_c[6][3] = fma(a_val[6], b_val[3], reg_c[6][3]);
            reg_c[7][0] = fma(a_val[7], b_val[0], reg_c[7][0]);
            reg_c[7][1] = fma(a_val[7], b_val[1], reg_c[7][1]);
            reg_c[7][2] = fma(a_val[7], b_val[2], reg_c[7][2]);
            reg_c[7][3] = fma(a_val[7], b_val[3], reg_c[7][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 8; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NT, Tile (128x64x16), RegTile (8x4)
__kernel void kernel_gemm_dbuf_nt_tile_128x64x16_reg_8x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][128][17];
    __local float l_B[2][16][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 128 + ty * 8; uint col_base = gx * 64 + tx * 4;
    float reg_c[8][4];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 128 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 16 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[8]; float b_val[4];
            a_val[0] = l_A[buf][ty * 8 + 0][k];
            a_val[1] = l_A[buf][ty * 8 + 1][k];
            a_val[2] = l_A[buf][ty * 8 + 2][k];
            a_val[3] = l_A[buf][ty * 8 + 3][k];
            a_val[4] = l_A[buf][ty * 8 + 4][k];
            a_val[5] = l_A[buf][ty * 8 + 5][k];
            a_val[6] = l_A[buf][ty * 8 + 6][k];
            a_val[7] = l_A[buf][ty * 8 + 7][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
            reg_c[4][0] = fma(a_val[4], b_val[0], reg_c[4][0]);
            reg_c[4][1] = fma(a_val[4], b_val[1], reg_c[4][1]);
            reg_c[4][2] = fma(a_val[4], b_val[2], reg_c[4][2]);
            reg_c[4][3] = fma(a_val[4], b_val[3], reg_c[4][3]);
            reg_c[5][0] = fma(a_val[5], b_val[0], reg_c[5][0]);
            reg_c[5][1] = fma(a_val[5], b_val[1], reg_c[5][1]);
            reg_c[5][2] = fma(a_val[5], b_val[2], reg_c[5][2]);
            reg_c[5][3] = fma(a_val[5], b_val[3], reg_c[5][3]);
            reg_c[6][0] = fma(a_val[6], b_val[0], reg_c[6][0]);
            reg_c[6][1] = fma(a_val[6], b_val[1], reg_c[6][1]);
            reg_c[6][2] = fma(a_val[6], b_val[2], reg_c[6][2]);
            reg_c[6][3] = fma(a_val[6], b_val[3], reg_c[6][3]);
            reg_c[7][0] = fma(a_val[7], b_val[0], reg_c[7][0]);
            reg_c[7][1] = fma(a_val[7], b_val[1], reg_c[7][1]);
            reg_c[7][2] = fma(a_val[7], b_val[2], reg_c[7][2]);
            reg_c[7][3] = fma(a_val[7], b_val[3], reg_c[7][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 8; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode NT, Tile (64x128x16), RegTile (4x8)
__kernel void kernel_gemm_dbuf_nt_tile_64x128x16_reg_4x8(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][17];
    __local float l_B[2][16][129];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 128 + tx * 8;
    float reg_c[4][8];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 8; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 64 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 128; uint c = i % 128;
            uint gr = t * 16 + r; uint gc = gx * 128 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[4]; float b_val[8];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 8 + 0];
            b_val[1] = l_B[buf][k][tx * 8 + 1];
            b_val[2] = l_B[buf][k][tx * 8 + 2];
            b_val[3] = l_B[buf][k][tx * 8 + 3];
            b_val[4] = l_B[buf][k][tx * 8 + 4];
            b_val[5] = l_B[buf][k][tx * 8 + 5];
            b_val[6] = l_B[buf][k][tx * 8 + 6];
            b_val[7] = l_B[buf][k][tx * 8 + 7];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[0][4] = fma(a_val[0], b_val[4], reg_c[0][4]);
            reg_c[0][5] = fma(a_val[0], b_val[5], reg_c[0][5]);
            reg_c[0][6] = fma(a_val[0], b_val[6], reg_c[0][6]);
            reg_c[0][7] = fma(a_val[0], b_val[7], reg_c[0][7]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[1][4] = fma(a_val[1], b_val[4], reg_c[1][4]);
            reg_c[1][5] = fma(a_val[1], b_val[5], reg_c[1][5]);
            reg_c[1][6] = fma(a_val[1], b_val[6], reg_c[1][6]);
            reg_c[1][7] = fma(a_val[1], b_val[7], reg_c[1][7]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[2][4] = fma(a_val[2], b_val[4], reg_c[2][4]);
            reg_c[2][5] = fma(a_val[2], b_val[5], reg_c[2][5]);
            reg_c[2][6] = fma(a_val[2], b_val[6], reg_c[2][6]);
            reg_c[2][7] = fma(a_val[2], b_val[7], reg_c[2][7]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
            reg_c[3][4] = fma(a_val[3], b_val[4], reg_c[3][4]);
            reg_c[3][5] = fma(a_val[3], b_val[5], reg_c[3][5]);
            reg_c[3][6] = fma(a_val[3], b_val[6], reg_c[3][6]);
            reg_c[3][7] = fma(a_val[3], b_val[7], reg_c[3][7]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 8; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TN, Tile (16x16x8), RegTile (2x2)
__kernel void kernel_gemm_dbuf_tn_tile_16x16x8_reg_2x2(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][16][9];
    __local float l_B[2][8][17];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 16 + ty * 2; uint col_base = gx * 16 + tx * 2;
    float reg_c[2][2];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 128; i += 64) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 16 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 128; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = t * 8 + r; uint gc = gx * 16 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[2]; float b_val[2];
            a_val[0] = l_A[buf][ty * 2 + 0][k];
            a_val[1] = l_A[buf][ty * 2 + 1][k];
            b_val[0] = l_B[buf][k][tx * 2 + 0];
            b_val[1] = l_B[buf][k][tx * 2 + 1];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
        }
        sync_local();
    }
    for (int i = 0; i < 2; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 2; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TN, Tile (16x16x16), RegTile (2x2)
__kernel void kernel_gemm_dbuf_tn_tile_16x16x16_reg_2x2(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][16][17];
    __local float l_B[2][16][17];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 16 + ty * 2; uint col_base = gx * 16 + tx * 2;
    float reg_c[2][2];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 16 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = t * 16 + r; uint gc = gx * 16 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[2]; float b_val[2];
            a_val[0] = l_A[buf][ty * 2 + 0][k];
            a_val[1] = l_A[buf][ty * 2 + 1][k];
            b_val[0] = l_B[buf][k][tx * 2 + 0];
            b_val[1] = l_B[buf][k][tx * 2 + 1];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
        }
        sync_local();
    }
    for (int i = 0; i < 2; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 2; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TN, Tile (32x32x8), RegTile (4x4)
__kernel void kernel_gemm_dbuf_tn_tile_32x32x8_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][32][9];
    __local float l_B[2][8][33];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 32 + ty * 4; uint col_base = gx * 32 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 32 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = t * 8 + r; uint gc = gx * 32 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TN, Tile (32x32x16), RegTile (4x4)
__kernel void kernel_gemm_dbuf_tn_tile_32x32x16_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][32][17];
    __local float l_B[2][16][33];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 32 + ty * 4; uint col_base = gx * 32 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 512; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 32 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 512; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = t * 16 + r; uint gc = gx * 32 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TN, Tile (32x32x32), RegTile (4x4)
__kernel void kernel_gemm_dbuf_tn_tile_32x32x32_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][32][33];
    __local float l_B[2][32][33];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 32 + ty * 4; uint col_base = gx * 32 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 32 - 1) / 32;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = gy * 32 + r; uint gc = t * 32 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 1024; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = t * 32 + r; uint gc = gx * 32 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 32; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TN, Tile (64x64x8), RegTile (4x4)
__kernel void kernel_gemm_dbuf_tn_tile_64x64x8_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][9];
    __local float l_B[2][8][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 64 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 512; i += 256) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 64 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 512; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 8 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TN, Tile (64x64x16), RegTile (4x4)
__kernel void kernel_gemm_dbuf_tn_tile_64x64x16_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][17];
    __local float l_B[2][16][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 64 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 64 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 16 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TN, Tile (64x64x32), RegTile (4x4)
__kernel void kernel_gemm_dbuf_tn_tile_64x64x32_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][33];
    __local float l_B[2][32][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 64 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 32 - 1) / 32;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 32; uint c = i % 32;
            uint gr = gy * 64 + r; uint gc = t * 32 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 32 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 32; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TN, Tile (128x64x8), RegTile (8x4)
__kernel void kernel_gemm_dbuf_tn_tile_128x64x8_reg_8x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][128][9];
    __local float l_B[2][8][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 128 + ty * 8; uint col_base = gx * 64 + tx * 4;
    float reg_c[8][4];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 128 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 512; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 8 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[8]; float b_val[4];
            a_val[0] = l_A[buf][ty * 8 + 0][k];
            a_val[1] = l_A[buf][ty * 8 + 1][k];
            a_val[2] = l_A[buf][ty * 8 + 2][k];
            a_val[3] = l_A[buf][ty * 8 + 3][k];
            a_val[4] = l_A[buf][ty * 8 + 4][k];
            a_val[5] = l_A[buf][ty * 8 + 5][k];
            a_val[6] = l_A[buf][ty * 8 + 6][k];
            a_val[7] = l_A[buf][ty * 8 + 7][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
            reg_c[4][0] = fma(a_val[4], b_val[0], reg_c[4][0]);
            reg_c[4][1] = fma(a_val[4], b_val[1], reg_c[4][1]);
            reg_c[4][2] = fma(a_val[4], b_val[2], reg_c[4][2]);
            reg_c[4][3] = fma(a_val[4], b_val[3], reg_c[4][3]);
            reg_c[5][0] = fma(a_val[5], b_val[0], reg_c[5][0]);
            reg_c[5][1] = fma(a_val[5], b_val[1], reg_c[5][1]);
            reg_c[5][2] = fma(a_val[5], b_val[2], reg_c[5][2]);
            reg_c[5][3] = fma(a_val[5], b_val[3], reg_c[5][3]);
            reg_c[6][0] = fma(a_val[6], b_val[0], reg_c[6][0]);
            reg_c[6][1] = fma(a_val[6], b_val[1], reg_c[6][1]);
            reg_c[6][2] = fma(a_val[6], b_val[2], reg_c[6][2]);
            reg_c[6][3] = fma(a_val[6], b_val[3], reg_c[6][3]);
            reg_c[7][0] = fma(a_val[7], b_val[0], reg_c[7][0]);
            reg_c[7][1] = fma(a_val[7], b_val[1], reg_c[7][1]);
            reg_c[7][2] = fma(a_val[7], b_val[2], reg_c[7][2]);
            reg_c[7][3] = fma(a_val[7], b_val[3], reg_c[7][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 8; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TN, Tile (128x64x16), RegTile (8x4)
__kernel void kernel_gemm_dbuf_tn_tile_128x64x16_reg_8x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][128][17];
    __local float l_B[2][16][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 128 + ty * 8; uint col_base = gx * 64 + tx * 4;
    float reg_c[8][4];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 128 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 16 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[8]; float b_val[4];
            a_val[0] = l_A[buf][ty * 8 + 0][k];
            a_val[1] = l_A[buf][ty * 8 + 1][k];
            a_val[2] = l_A[buf][ty * 8 + 2][k];
            a_val[3] = l_A[buf][ty * 8 + 3][k];
            a_val[4] = l_A[buf][ty * 8 + 4][k];
            a_val[5] = l_A[buf][ty * 8 + 5][k];
            a_val[6] = l_A[buf][ty * 8 + 6][k];
            a_val[7] = l_A[buf][ty * 8 + 7][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
            reg_c[4][0] = fma(a_val[4], b_val[0], reg_c[4][0]);
            reg_c[4][1] = fma(a_val[4], b_val[1], reg_c[4][1]);
            reg_c[4][2] = fma(a_val[4], b_val[2], reg_c[4][2]);
            reg_c[4][3] = fma(a_val[4], b_val[3], reg_c[4][3]);
            reg_c[5][0] = fma(a_val[5], b_val[0], reg_c[5][0]);
            reg_c[5][1] = fma(a_val[5], b_val[1], reg_c[5][1]);
            reg_c[5][2] = fma(a_val[5], b_val[2], reg_c[5][2]);
            reg_c[5][3] = fma(a_val[5], b_val[3], reg_c[5][3]);
            reg_c[6][0] = fma(a_val[6], b_val[0], reg_c[6][0]);
            reg_c[6][1] = fma(a_val[6], b_val[1], reg_c[6][1]);
            reg_c[6][2] = fma(a_val[6], b_val[2], reg_c[6][2]);
            reg_c[6][3] = fma(a_val[6], b_val[3], reg_c[6][3]);
            reg_c[7][0] = fma(a_val[7], b_val[0], reg_c[7][0]);
            reg_c[7][1] = fma(a_val[7], b_val[1], reg_c[7][1]);
            reg_c[7][2] = fma(a_val[7], b_val[2], reg_c[7][2]);
            reg_c[7][3] = fma(a_val[7], b_val[3], reg_c[7][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 8; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TN, Tile (64x128x16), RegTile (4x8)
__kernel void kernel_gemm_dbuf_tn_tile_64x128x16_reg_4x8(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][17];
    __local float l_B[2][16][129];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 128 + tx * 8;
    float reg_c[4][8];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 8; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 64 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 128; uint c = i % 128;
            uint gr = t * 16 + r; uint gc = gx * 128 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[4]; float b_val[8];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 8 + 0];
            b_val[1] = l_B[buf][k][tx * 8 + 1];
            b_val[2] = l_B[buf][k][tx * 8 + 2];
            b_val[3] = l_B[buf][k][tx * 8 + 3];
            b_val[4] = l_B[buf][k][tx * 8 + 4];
            b_val[5] = l_B[buf][k][tx * 8 + 5];
            b_val[6] = l_B[buf][k][tx * 8 + 6];
            b_val[7] = l_B[buf][k][tx * 8 + 7];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[0][4] = fma(a_val[0], b_val[4], reg_c[0][4]);
            reg_c[0][5] = fma(a_val[0], b_val[5], reg_c[0][5]);
            reg_c[0][6] = fma(a_val[0], b_val[6], reg_c[0][6]);
            reg_c[0][7] = fma(a_val[0], b_val[7], reg_c[0][7]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[1][4] = fma(a_val[1], b_val[4], reg_c[1][4]);
            reg_c[1][5] = fma(a_val[1], b_val[5], reg_c[1][5]);
            reg_c[1][6] = fma(a_val[1], b_val[6], reg_c[1][6]);
            reg_c[1][7] = fma(a_val[1], b_val[7], reg_c[1][7]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[2][4] = fma(a_val[2], b_val[4], reg_c[2][4]);
            reg_c[2][5] = fma(a_val[2], b_val[5], reg_c[2][5]);
            reg_c[2][6] = fma(a_val[2], b_val[6], reg_c[2][6]);
            reg_c[2][7] = fma(a_val[2], b_val[7], reg_c[2][7]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
            reg_c[3][4] = fma(a_val[3], b_val[4], reg_c[3][4]);
            reg_c[3][5] = fma(a_val[3], b_val[5], reg_c[3][5]);
            reg_c[3][6] = fma(a_val[3], b_val[6], reg_c[3][6]);
            reg_c[3][7] = fma(a_val[3], b_val[7], reg_c[3][7]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 8; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TT, Tile (16x16x8), RegTile (2x2)
__kernel void kernel_gemm_dbuf_tt_tile_16x16x8_reg_2x2(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][16][9];
    __local float l_B[2][8][17];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 16 + ty * 2; uint col_base = gx * 16 + tx * 2;
    float reg_c[2][2];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 128; i += 64) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 16 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 128; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = t * 8 + r; uint gc = gx * 16 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[2]; float b_val[2];
            a_val[0] = l_A[buf][ty * 2 + 0][k];
            a_val[1] = l_A[buf][ty * 2 + 1][k];
            b_val[0] = l_B[buf][k][tx * 2 + 0];
            b_val[1] = l_B[buf][k][tx * 2 + 1];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
        }
        sync_local();
    }
    for (int i = 0; i < 2; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 2; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TT, Tile (16x16x16), RegTile (2x2)
__kernel void kernel_gemm_dbuf_tt_tile_16x16x16_reg_2x2(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][16][17];
    __local float l_B[2][16][17];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 16 + ty * 2; uint col_base = gx * 16 + tx * 2;
    float reg_c[2][2];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 16 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = t * 16 + r; uint gc = gx * 16 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[2]; float b_val[2];
            a_val[0] = l_A[buf][ty * 2 + 0][k];
            a_val[1] = l_A[buf][ty * 2 + 1][k];
            b_val[0] = l_B[buf][k][tx * 2 + 0];
            b_val[1] = l_B[buf][k][tx * 2 + 1];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
        }
        sync_local();
    }
    for (int i = 0; i < 2; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 2; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TT, Tile (32x32x8), RegTile (4x4)
__kernel void kernel_gemm_dbuf_tt_tile_32x32x8_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][32][9];
    __local float l_B[2][8][33];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 32 + ty * 4; uint col_base = gx * 32 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 32 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 256; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = t * 8 + r; uint gc = gx * 32 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TT, Tile (32x32x16), RegTile (4x4)
__kernel void kernel_gemm_dbuf_tt_tile_32x32x16_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][32][17];
    __local float l_B[2][16][33];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 32 + ty * 4; uint col_base = gx * 32 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 512; i += 64) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 32 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 512; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = t * 16 + r; uint gc = gx * 32 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TT, Tile (32x32x32), RegTile (4x4)
__kernel void kernel_gemm_dbuf_tt_tile_32x32x32_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][32][33];
    __local float l_B[2][32][33];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 8 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 32 + ty * 4; uint col_base = gx * 32 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 32 - 1) / 32;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = gy * 32 + r; uint gc = t * 32 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 1024; i += 64) {
            uint r = i / 32; uint c = i % 32;
            uint gr = t * 32 + r; uint gc = gx * 32 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 32; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TT, Tile (64x64x8), RegTile (4x4)
__kernel void kernel_gemm_dbuf_tt_tile_64x64x8_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][9];
    __local float l_B[2][8][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 64 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 512; i += 256) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 64 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 512; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 8 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TT, Tile (64x64x16), RegTile (4x4)
__kernel void kernel_gemm_dbuf_tt_tile_64x64x16_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][17];
    __local float l_B[2][16][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 64 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 64 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 16 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TT, Tile (64x64x32), RegTile (4x4)
__kernel void kernel_gemm_dbuf_tt_tile_64x64x32_reg_4x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][33];
    __local float l_B[2][32][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 64 + tx * 4;
    float reg_c[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 32 - 1) / 32;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 32; uint c = i % 32;
            uint gr = gy * 64 + r; uint gc = t * 32 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 32 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 32; ++k) {
            float a_val[4]; float b_val[4];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TT, Tile (128x64x8), RegTile (8x4)
__kernel void kernel_gemm_dbuf_tt_tile_128x64x8_reg_8x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][128][9];
    __local float l_B[2][8][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 128 + ty * 8; uint col_base = gx * 64 + tx * 4;
    float reg_c[8][4];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 8 - 1) / 8;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 8; uint c = i % 8;
            uint gr = gy * 128 + r; uint gc = t * 8 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 512; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 8 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            float a_val[8]; float b_val[4];
            a_val[0] = l_A[buf][ty * 8 + 0][k];
            a_val[1] = l_A[buf][ty * 8 + 1][k];
            a_val[2] = l_A[buf][ty * 8 + 2][k];
            a_val[3] = l_A[buf][ty * 8 + 3][k];
            a_val[4] = l_A[buf][ty * 8 + 4][k];
            a_val[5] = l_A[buf][ty * 8 + 5][k];
            a_val[6] = l_A[buf][ty * 8 + 6][k];
            a_val[7] = l_A[buf][ty * 8 + 7][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
            reg_c[4][0] = fma(a_val[4], b_val[0], reg_c[4][0]);
            reg_c[4][1] = fma(a_val[4], b_val[1], reg_c[4][1]);
            reg_c[4][2] = fma(a_val[4], b_val[2], reg_c[4][2]);
            reg_c[4][3] = fma(a_val[4], b_val[3], reg_c[4][3]);
            reg_c[5][0] = fma(a_val[5], b_val[0], reg_c[5][0]);
            reg_c[5][1] = fma(a_val[5], b_val[1], reg_c[5][1]);
            reg_c[5][2] = fma(a_val[5], b_val[2], reg_c[5][2]);
            reg_c[5][3] = fma(a_val[5], b_val[3], reg_c[5][3]);
            reg_c[6][0] = fma(a_val[6], b_val[0], reg_c[6][0]);
            reg_c[6][1] = fma(a_val[6], b_val[1], reg_c[6][1]);
            reg_c[6][2] = fma(a_val[6], b_val[2], reg_c[6][2]);
            reg_c[6][3] = fma(a_val[6], b_val[3], reg_c[6][3]);
            reg_c[7][0] = fma(a_val[7], b_val[0], reg_c[7][0]);
            reg_c[7][1] = fma(a_val[7], b_val[1], reg_c[7][1]);
            reg_c[7][2] = fma(a_val[7], b_val[2], reg_c[7][2]);
            reg_c[7][3] = fma(a_val[7], b_val[3], reg_c[7][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 8; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TT, Tile (128x64x16), RegTile (8x4)
__kernel void kernel_gemm_dbuf_tt_tile_128x64x16_reg_8x4(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][128][17];
    __local float l_B[2][16][65];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 128 + ty * 8; uint col_base = gx * 64 + tx * 4;
    float reg_c[8][4];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 128 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 64; uint c = i % 64;
            uint gr = t * 16 + r; uint gc = gx * 64 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[8]; float b_val[4];
            a_val[0] = l_A[buf][ty * 8 + 0][k];
            a_val[1] = l_A[buf][ty * 8 + 1][k];
            a_val[2] = l_A[buf][ty * 8 + 2][k];
            a_val[3] = l_A[buf][ty * 8 + 3][k];
            a_val[4] = l_A[buf][ty * 8 + 4][k];
            a_val[5] = l_A[buf][ty * 8 + 5][k];
            a_val[6] = l_A[buf][ty * 8 + 6][k];
            a_val[7] = l_A[buf][ty * 8 + 7][k];
            b_val[0] = l_B[buf][k][tx * 4 + 0];
            b_val[1] = l_B[buf][k][tx * 4 + 1];
            b_val[2] = l_B[buf][k][tx * 4 + 2];
            b_val[3] = l_B[buf][k][tx * 4 + 3];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
            reg_c[4][0] = fma(a_val[4], b_val[0], reg_c[4][0]);
            reg_c[4][1] = fma(a_val[4], b_val[1], reg_c[4][1]);
            reg_c[4][2] = fma(a_val[4], b_val[2], reg_c[4][2]);
            reg_c[4][3] = fma(a_val[4], b_val[3], reg_c[4][3]);
            reg_c[5][0] = fma(a_val[5], b_val[0], reg_c[5][0]);
            reg_c[5][1] = fma(a_val[5], b_val[1], reg_c[5][1]);
            reg_c[5][2] = fma(a_val[5], b_val[2], reg_c[5][2]);
            reg_c[5][3] = fma(a_val[5], b_val[3], reg_c[5][3]);
            reg_c[6][0] = fma(a_val[6], b_val[0], reg_c[6][0]);
            reg_c[6][1] = fma(a_val[6], b_val[1], reg_c[6][1]);
            reg_c[6][2] = fma(a_val[6], b_val[2], reg_c[6][2]);
            reg_c[6][3] = fma(a_val[6], b_val[3], reg_c[6][3]);
            reg_c[7][0] = fma(a_val[7], b_val[0], reg_c[7][0]);
            reg_c[7][1] = fma(a_val[7], b_val[1], reg_c[7][1]);
            reg_c[7][2] = fma(a_val[7], b_val[2], reg_c[7][2]);
            reg_c[7][3] = fma(a_val[7], b_val[3], reg_c[7][3]);
        }
        sync_local();
    }
    for (int i = 0; i < 8; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 4; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}

// Register-Tiled Double-Buffered SGEMM: Mode TT, Tile (64x128x16), RegTile (4x8)
__kernel void kernel_gemm_dbuf_tt_tile_64x128x16_reg_4x8(
    __global const float* A, __global const float* B, __global float* C,
    const uint M, const uint N, const uint K, const float alpha, const float beta) {
    __local float l_A[2][64][17];
    __local float l_B[2][16][129];
    uint tx = get_local_id(0); uint ty = get_local_id(1);
    uint tid = ty * 16 + tx;
    uint gx = get_group_id(0); uint gy = get_group_id(1);
    uint row_base = gy * 64 + ty * 4; uint col_base = gx * 128 + tx * 8;
    float reg_c[4][8];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 8; ++j) reg_c[i][j] = 0.0f;
    uint num_tiles = (K + 16 - 1) / 16;
    for (uint t = 0; t < num_tiles; ++t) {
        uint buf = t & 1;
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 16; uint c = i % 16;
            uint gr = gy * 64 + r; uint gc = t * 16 + c;
            l_A[buf][r][c] = (gr < M && gc < K) ? A[gc * M + gr] : 0.0f;
        }
        for (uint i = tid; i < 2048; i += 256) {
            uint r = i / 128; uint c = i % 128;
            uint gr = t * 16 + r; uint gc = gx * 128 + c;
            l_B[buf][r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }
        sync_local();
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            float a_val[4]; float b_val[8];
            a_val[0] = l_A[buf][ty * 4 + 0][k];
            a_val[1] = l_A[buf][ty * 4 + 1][k];
            a_val[2] = l_A[buf][ty * 4 + 2][k];
            a_val[3] = l_A[buf][ty * 4 + 3][k];
            b_val[0] = l_B[buf][k][tx * 8 + 0];
            b_val[1] = l_B[buf][k][tx * 8 + 1];
            b_val[2] = l_B[buf][k][tx * 8 + 2];
            b_val[3] = l_B[buf][k][tx * 8 + 3];
            b_val[4] = l_B[buf][k][tx * 8 + 4];
            b_val[5] = l_B[buf][k][tx * 8 + 5];
            b_val[6] = l_B[buf][k][tx * 8 + 6];
            b_val[7] = l_B[buf][k][tx * 8 + 7];
            reg_c[0][0] = fma(a_val[0], b_val[0], reg_c[0][0]);
            reg_c[0][1] = fma(a_val[0], b_val[1], reg_c[0][1]);
            reg_c[0][2] = fma(a_val[0], b_val[2], reg_c[0][2]);
            reg_c[0][3] = fma(a_val[0], b_val[3], reg_c[0][3]);
            reg_c[0][4] = fma(a_val[0], b_val[4], reg_c[0][4]);
            reg_c[0][5] = fma(a_val[0], b_val[5], reg_c[0][5]);
            reg_c[0][6] = fma(a_val[0], b_val[6], reg_c[0][6]);
            reg_c[0][7] = fma(a_val[0], b_val[7], reg_c[0][7]);
            reg_c[1][0] = fma(a_val[1], b_val[0], reg_c[1][0]);
            reg_c[1][1] = fma(a_val[1], b_val[1], reg_c[1][1]);
            reg_c[1][2] = fma(a_val[1], b_val[2], reg_c[1][2]);
            reg_c[1][3] = fma(a_val[1], b_val[3], reg_c[1][3]);
            reg_c[1][4] = fma(a_val[1], b_val[4], reg_c[1][4]);
            reg_c[1][5] = fma(a_val[1], b_val[5], reg_c[1][5]);
            reg_c[1][6] = fma(a_val[1], b_val[6], reg_c[1][6]);
            reg_c[1][7] = fma(a_val[1], b_val[7], reg_c[1][7]);
            reg_c[2][0] = fma(a_val[2], b_val[0], reg_c[2][0]);
            reg_c[2][1] = fma(a_val[2], b_val[1], reg_c[2][1]);
            reg_c[2][2] = fma(a_val[2], b_val[2], reg_c[2][2]);
            reg_c[2][3] = fma(a_val[2], b_val[3], reg_c[2][3]);
            reg_c[2][4] = fma(a_val[2], b_val[4], reg_c[2][4]);
            reg_c[2][5] = fma(a_val[2], b_val[5], reg_c[2][5]);
            reg_c[2][6] = fma(a_val[2], b_val[6], reg_c[2][6]);
            reg_c[2][7] = fma(a_val[2], b_val[7], reg_c[2][7]);
            reg_c[3][0] = fma(a_val[3], b_val[0], reg_c[3][0]);
            reg_c[3][1] = fma(a_val[3], b_val[1], reg_c[3][1]);
            reg_c[3][2] = fma(a_val[3], b_val[2], reg_c[3][2]);
            reg_c[3][3] = fma(a_val[3], b_val[3], reg_c[3][3]);
            reg_c[3][4] = fma(a_val[3], b_val[4], reg_c[3][4]);
            reg_c[3][5] = fma(a_val[3], b_val[5], reg_c[3][5]);
            reg_c[3][6] = fma(a_val[3], b_val[6], reg_c[3][6]);
            reg_c[3][7] = fma(a_val[3], b_val[7], reg_c[3][7]);
        }
        sync_local();
    }
    for (int i = 0; i < 4; ++i) {
        uint r = row_base + i;
        if (r < M) {
            for (int j = 0; j < 8; ++j) {
                uint c = col_base + j;
                if (c < N) {
                    uint idx = r * N + c;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * reg_c[i][j] + beta * orig;
                }
            }
        }
    }
}


// ============================================================================
// Batched Triangular Forward/Backward Substitution Linear Solvers
// ============================================================================

// Batched 3x3 Lower Triangular Forward Substitution (L * x = b)
__kernel void kernel_batched_trsv_lower_3x3(
    __global const float* in_L, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 9; size_t vec_base = (size_t)bid * 3;
    float x[3];
    for (int i = 0; i < 3; ++i) {
        float sum = in_b[vec_base + i];
        for (int j = 0; j < i; ++j) sum -= in_L[mat_base + i * 3 + j] * x[j];
        float diag = in_L[mat_base + i * 3 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 3; ++i) out_x[vec_base + i] = x[i];
}

// Batched 3x3 Upper Triangular Backward Substitution (U * x = b)
__kernel void kernel_batched_trsv_upper_3x3(
    __global const float* in_U, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 9; size_t vec_base = (size_t)bid * 3;
    float x[3];
    for (int i = 2; i >= 0; --i) {
        float sum = in_b[vec_base + i];
        for (int j = i + 1; j < 3; ++j) sum -= in_U[mat_base + i * 3 + j] * x[j];
        float diag = in_U[mat_base + i * 3 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 3; ++i) out_x[vec_base + i] = x[i];
}

// Batched 4x4 Lower Triangular Forward Substitution (L * x = b)
__kernel void kernel_batched_trsv_lower_4x4(
    __global const float* in_L, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 16; size_t vec_base = (size_t)bid * 4;
    float x[4];
    for (int i = 0; i < 4; ++i) {
        float sum = in_b[vec_base + i];
        for (int j = 0; j < i; ++j) sum -= in_L[mat_base + i * 4 + j] * x[j];
        float diag = in_L[mat_base + i * 4 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 4; ++i) out_x[vec_base + i] = x[i];
}

// Batched 4x4 Upper Triangular Backward Substitution (U * x = b)
__kernel void kernel_batched_trsv_upper_4x4(
    __global const float* in_U, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 16; size_t vec_base = (size_t)bid * 4;
    float x[4];
    for (int i = 3; i >= 0; --i) {
        float sum = in_b[vec_base + i];
        for (int j = i + 1; j < 4; ++j) sum -= in_U[mat_base + i * 4 + j] * x[j];
        float diag = in_U[mat_base + i * 4 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 4; ++i) out_x[vec_base + i] = x[i];
}

// Batched 5x5 Lower Triangular Forward Substitution (L * x = b)
__kernel void kernel_batched_trsv_lower_5x5(
    __global const float* in_L, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 25; size_t vec_base = (size_t)bid * 5;
    float x[5];
    for (int i = 0; i < 5; ++i) {
        float sum = in_b[vec_base + i];
        for (int j = 0; j < i; ++j) sum -= in_L[mat_base + i * 5 + j] * x[j];
        float diag = in_L[mat_base + i * 5 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 5; ++i) out_x[vec_base + i] = x[i];
}

// Batched 5x5 Upper Triangular Backward Substitution (U * x = b)
__kernel void kernel_batched_trsv_upper_5x5(
    __global const float* in_U, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 25; size_t vec_base = (size_t)bid * 5;
    float x[5];
    for (int i = 4; i >= 0; --i) {
        float sum = in_b[vec_base + i];
        for (int j = i + 1; j < 5; ++j) sum -= in_U[mat_base + i * 5 + j] * x[j];
        float diag = in_U[mat_base + i * 5 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 5; ++i) out_x[vec_base + i] = x[i];
}

// Batched 6x6 Lower Triangular Forward Substitution (L * x = b)
__kernel void kernel_batched_trsv_lower_6x6(
    __global const float* in_L, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 36; size_t vec_base = (size_t)bid * 6;
    float x[6];
    for (int i = 0; i < 6; ++i) {
        float sum = in_b[vec_base + i];
        for (int j = 0; j < i; ++j) sum -= in_L[mat_base + i * 6 + j] * x[j];
        float diag = in_L[mat_base + i * 6 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 6; ++i) out_x[vec_base + i] = x[i];
}

// Batched 6x6 Upper Triangular Backward Substitution (U * x = b)
__kernel void kernel_batched_trsv_upper_6x6(
    __global const float* in_U, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 36; size_t vec_base = (size_t)bid * 6;
    float x[6];
    for (int i = 5; i >= 0; --i) {
        float sum = in_b[vec_base + i];
        for (int j = i + 1; j < 6; ++j) sum -= in_U[mat_base + i * 6 + j] * x[j];
        float diag = in_U[mat_base + i * 6 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 6; ++i) out_x[vec_base + i] = x[i];
}

// Batched 8x8 Lower Triangular Forward Substitution (L * x = b)
__kernel void kernel_batched_trsv_lower_8x8(
    __global const float* in_L, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 64; size_t vec_base = (size_t)bid * 8;
    float x[8];
    for (int i = 0; i < 8; ++i) {
        float sum = in_b[vec_base + i];
        for (int j = 0; j < i; ++j) sum -= in_L[mat_base + i * 8 + j] * x[j];
        float diag = in_L[mat_base + i * 8 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 8; ++i) out_x[vec_base + i] = x[i];
}

// Batched 8x8 Upper Triangular Backward Substitution (U * x = b)
__kernel void kernel_batched_trsv_upper_8x8(
    __global const float* in_U, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 64; size_t vec_base = (size_t)bid * 8;
    float x[8];
    for (int i = 7; i >= 0; --i) {
        float sum = in_b[vec_base + i];
        for (int j = i + 1; j < 8; ++j) sum -= in_U[mat_base + i * 8 + j] * x[j];
        float diag = in_U[mat_base + i * 8 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 8; ++i) out_x[vec_base + i] = x[i];
}

// Batched 10x10 Lower Triangular Forward Substitution (L * x = b)
__kernel void kernel_batched_trsv_lower_10x10(
    __global const float* in_L, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 100; size_t vec_base = (size_t)bid * 10;
    float x[10];
    for (int i = 0; i < 10; ++i) {
        float sum = in_b[vec_base + i];
        for (int j = 0; j < i; ++j) sum -= in_L[mat_base + i * 10 + j] * x[j];
        float diag = in_L[mat_base + i * 10 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 10; ++i) out_x[vec_base + i] = x[i];
}

// Batched 10x10 Upper Triangular Backward Substitution (U * x = b)
__kernel void kernel_batched_trsv_upper_10x10(
    __global const float* in_U, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 100; size_t vec_base = (size_t)bid * 10;
    float x[10];
    for (int i = 9; i >= 0; --i) {
        float sum = in_b[vec_base + i];
        for (int j = i + 1; j < 10; ++j) sum -= in_U[mat_base + i * 10 + j] * x[j];
        float diag = in_U[mat_base + i * 10 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 10; ++i) out_x[vec_base + i] = x[i];
}

// Batched 12x12 Lower Triangular Forward Substitution (L * x = b)
__kernel void kernel_batched_trsv_lower_12x12(
    __global const float* in_L, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 144; size_t vec_base = (size_t)bid * 12;
    float x[12];
    for (int i = 0; i < 12; ++i) {
        float sum = in_b[vec_base + i];
        for (int j = 0; j < i; ++j) sum -= in_L[mat_base + i * 12 + j] * x[j];
        float diag = in_L[mat_base + i * 12 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 12; ++i) out_x[vec_base + i] = x[i];
}

// Batched 12x12 Upper Triangular Backward Substitution (U * x = b)
__kernel void kernel_batched_trsv_upper_12x12(
    __global const float* in_U, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 144; size_t vec_base = (size_t)bid * 12;
    float x[12];
    for (int i = 11; i >= 0; --i) {
        float sum = in_b[vec_base + i];
        for (int j = i + 1; j < 12; ++j) sum -= in_U[mat_base + i * 12 + j] * x[j];
        float diag = in_U[mat_base + i * 12 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 12; ++i) out_x[vec_base + i] = x[i];
}

// Batched 16x16 Lower Triangular Forward Substitution (L * x = b)
__kernel void kernel_batched_trsv_lower_16x16(
    __global const float* in_L, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 256; size_t vec_base = (size_t)bid * 16;
    float x[16];
    for (int i = 0; i < 16; ++i) {
        float sum = in_b[vec_base + i];
        for (int j = 0; j < i; ++j) sum -= in_L[mat_base + i * 16 + j] * x[j];
        float diag = in_L[mat_base + i * 16 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 16; ++i) out_x[vec_base + i] = x[i];
}

// Batched 16x16 Upper Triangular Backward Substitution (U * x = b)
__kernel void kernel_batched_trsv_upper_16x16(
    __global const float* in_U, __global const float* in_b, __global float* out_x, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t mat_base = (size_t)bid * 256; size_t vec_base = (size_t)bid * 16;
    float x[16];
    for (int i = 15; i >= 0; --i) {
        float sum = in_b[vec_base + i];
        for (int j = i + 1; j < 16; ++j) sum -= in_U[mat_base + i * 16 + j] * x[j];
        float diag = in_U[mat_base + i * 16 + i];
        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;
    }
    for (int i = 0; i < 16; ++i) out_x[vec_base + i] = x[i];
}
