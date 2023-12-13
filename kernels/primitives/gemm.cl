#include "common/math.clh"
#include "common/memory.clh"

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16

#define REG_M 4
#define REG_N 4

#define WG_X (TILE_N / REG_N) // 16
#define WG_Y (TILE_M / REG_M) // 16

// 2D Register-tiled General Matrix Multiply (GEMM): C = alpha * (A * B) + beta * C
// A is M x K, B is K x N, C is M x N. All row-major.
__kernel void kernel_gemm_tiled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint M,
    const uint N,
    const uint K,
    const float alpha,
    const float beta) {
    
    // Tiled shared local memory with padding to prevent 32-bank conflicts
    __local float l_A[TILE_M][TILE_K + 1];
    __local float l_B[TILE_K][TILE_N + 1];

    uint tx = get_local_id(0); // 0..15
    uint ty = get_local_id(1); // 0..15
    uint tid = ty * WG_X + tx; // 0..255

    uint group_x = get_group_id(0);
    uint group_y = get_group_id(1);

    uint c_row_base = group_y * TILE_M + ty * REG_M;
    uint c_col_base = group_x * TILE_N + tx * REG_N;

    // Thread-private registers for accumulating 4x4 sub-tile
    float accum[REG_M][REG_N];
    for (int i = 0; i < REG_M; ++i) {
        for (int j = 0; j < REG_N; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        // Cooperative load of A tile into local memory (64 x 16 elements = 1024 floats)
        // 256 threads load 4 elements each
        for (uint i = 0; i < 4; ++i) {
            uint load_idx = tid * 4 + i;
            uint a_row = load_idx / TILE_K;
            uint a_col = load_idx % TILE_K;

            uint global_a_row = group_y * TILE_M + a_row;
            uint global_a_col = kt * TILE_K + a_col;

            float val_a = 0.0f;
            if (global_a_row < M && global_a_col < K) {
                val_a = A[global_a_row * K + global_a_col];
            }
            l_A[a_row][a_col] = val_a;
        }

        // Cooperative load of B tile into local memory (16 x 64 elements = 1024 floats)
        for (uint i = 0; i < 4; ++i) {
            uint load_idx = tid * 4 + i;
            uint b_row = load_idx / TILE_N;
            uint b_col = load_idx % TILE_N;

            uint global_b_row = kt * TILE_K + b_row;
            uint global_b_col = group_x * TILE_N + b_col;

            float val_b = 0.0f;
            if (global_b_row < K && global_b_col < N) {
                val_b = B[global_b_row * N + global_b_col];
            }
            l_B[b_row][b_col] = val_b;
        }

        sync_local();

        // Compute 4x4 outer-product accumulations from local memory tiles
        #pragma unroll
        for (uint k = 0; k < TILE_K; ++k) {
            float reg_a[REG_M];
            float reg_b[REG_N];

            #pragma unroll
            for (int i = 0; i < REG_M; ++i) {
                reg_a[i] = l_A[ty * REG_M + i][k];
            }

            #pragma unroll
            for (int j = 0; j < REG_N; ++j) {
                reg_b[j] = l_B[k][tx * REG_N + j];
            }

            #pragma unroll
            for (int i = 0; i < REG_M; ++i) {
                #pragma unroll
                for (int j = 0; j < REG_N; ++j) {
                    accum[i][j] = fma(reg_a[i], reg_b[j], accum[i][j]);
                }
            }
        }

        sync_local();
    }

    // Write final 4x4 results to C in global memory
    for (int i = 0; i < REG_M; ++i) {
        uint row = c_row_base + i;
        if (row < M) {
            for (int j = 0; j < REG_N; ++j) {
                uint col = c_col_base + j;
                if (col < N) {
                    uint idx = row * N + col;
                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * accum[i][j] + beta * orig;
                }
            }
        }
    }
}

// Batched Matrix-Vector Multiply for Small Linear Systems: y = A * x
// Each work-item solves one matrix-vector product of dimension Dim <= 16
__kernel void kernel_batched_gemv_small(
    __global const float* A_batch,
    __global const float* x_batch,
    __global float* y_batch,
    const uint dim,
    const uint total_systems) {
    
    uint gid = get_global_id(0);
    if (gid >= total_systems) return;

    uint a_offset = gid * dim * dim;
    uint x_offset = gid * dim;
    uint y_offset = gid * dim;

    for (uint i = 0; i < dim; ++i) {
        float sum = 0.0f;
        for (uint j = 0; j < dim; ++j) {
            sum += A_batch[a_offset + i * dim + j] * x_batch[x_offset + j];
        }
        y_batch[y_offset + i] = sum;
    }
}
