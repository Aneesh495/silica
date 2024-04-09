#include "common/math.clh"
#include "common/memory.clh"

// ============================================================================
// Sparse Matrix-Vector (SpMV) Kernels across CSR, ELLPACK, COO, HYB, and BSR
// ============================================================================

// 1. CSR Vector SpMV (32 threads per row for high load-balancing)
__kernel void kernel_spmv_csr_vector(
    __global const int* row_ptr, __global const int* col_indices,
    __global const float* values, __global const float* x,
    __global float* y, const uint num_rows) {
    uint tid = get_global_id(0); uint warp_id = tid >> 5; uint lane = tid & 31;
    if (warp_id >= num_rows) return;
    int row_start = row_ptr[warp_id]; int row_end = row_ptr[warp_id + 1];
    float sum = 0.0f;
    for (int j = row_start + lane; j < row_end; j += 32) {
        sum += values[j] * x[col_indices[j]];
    }
    // Warp reduction via local memory
    __local float l_sum[256];
    uint lid = get_local_id(0); l_sum[lid] = sum; sync_local();
    if (lane < 16) l_sum[lid] += l_sum[lid + 16]; sync_local();
    if (lane < 8)  l_sum[lid] += l_sum[lid + 8];  sync_local();
    if (lane < 4)  l_sum[lid] += l_sum[lid + 4];  sync_local();
    if (lane < 2)  l_sum[lid] += l_sum[lid + 2];  sync_local();
    if (lane == 0) y[warp_id] = l_sum[lid] + l_sum[lid + 1];
}}


// 2. ELLPACK SpMV (Coalesced column-major sparse matrix format)
__kernel void kernel_spmv_ellpack(
    __global const int* ell_col_indices, __global const float* ell_values,
    __global const float* x, __global float* y,
    const uint num_rows, const uint max_row_nnz) {
    uint row = get_global_id(0); if (row >= num_rows) return;
    float sum = 0.0f;
    for (uint k = 0; k < max_row_nnz; ++k) {
        size_t idx = (size_t)k * (size_t)num_rows + (size_t)row;
        int col = ell_col_indices[idx];
        if (col >= 0) sum += ell_values[idx] * x[col];
    }
    y[row] = sum;
}}


// 3. Block Sparse Row (BSR) SpMV for 3x3 Rigid Body and FEM Blocks
__kernel void kernel_spmv_bsr_3x3(
    __global const int* block_row_ptr, __global const int* block_col_indices,
    __global const float* block_values, __global const float* x,
    __global float* y, const uint num_block_rows) {
    uint bid = get_global_id(0); if (bid >= num_block_rows) return;
    int b_start = block_row_ptr[bid]; int b_end = block_row_ptr[bid + 1];
    float3 sum = (float3)(0.0f, 0.0f, 0.0f);
    for (int b = b_start; b < b_end; ++b) {
        int b_col = block_col_indices[b]; size_t v_base = (size_t)b * 9;
        float3 x_block = (float3)(x[b_col * 3], x[b_col * 3 + 1], x[b_col * 3 + 2]);
        sum.x += block_values[v_base + 0] * x_block.x + block_values[v_base + 1] * x_block.y + block_values[v_base + 2] * x_block.z;
        sum.y += block_values[v_base + 3] * x_block.x + block_values[v_base + 4] * x_block.y + block_values[v_base + 5] * x_block.z;
        sum.z += block_values[v_base + 6] * x_block.x + block_values[v_base + 7] * x_block.y + block_values[v_base + 8] * x_block.z;
    }
    y[bid * 3 + 0] = sum.x; y[bid * 3 + 1] = sum.y; y[bid * 3 + 2] = sum.z;
}}


// ============================================================================
// Parallel Cyclic Reduction (PCR) Tridiagonal Solvers
// ============================================================================

// Batched Parallel Cyclic Reduction: System Size 16
__kernel void kernel_pcr_tridiagonal_sys_16(
    __global const float* in_lower, __global const float* in_diag,
    __global const float* in_upper, __global const float* in_rhs,
    __global float* out_x, const uint batch_count) {
    uint bid = get_group_id(0); if (bid >= batch_count) return;
    uint tid = get_local_id(0);
    __local float a[16], b[16], c[16], d[16];
    __local float a_new[16], b_new[16], c_new[16], d_new[16];
    size_t base = (size_t)bid * 16;
    if (tid < 16) {
        a[tid] = in_lower[base + tid]; b[tid] = in_diag[base + tid];
        c[tid] = in_upper[base + tid]; d[tid] = in_rhs[base + tid];
    }
    sync_local();
    // Step 0: Stride 1
    if (tid < 16) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 1) ? (i_a / b[tid - 1]) : 0.0f;
        float k2 = (tid + 1 < 16) ? (i_c / b[tid + 1]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 1) ? c[tid - 1] : 0.0f) - k2 * ((tid + 1 < 16) ? a[tid + 1] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 1) ? d[tid - 1] : 0.0f) - k2 * ((tid + 1 < 16) ? d[tid + 1] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 1) ? a[tid - 1] : 0.0f);
        c_new[tid] = -k2 * ((tid + 1 < 16) ? c[tid + 1] : 0.0f);
    }
    sync_local();
    if (tid < 16) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 1: Stride 2
    if (tid < 16) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 2) ? (i_a / b[tid - 2]) : 0.0f;
        float k2 = (tid + 2 < 16) ? (i_c / b[tid + 2]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 2) ? c[tid - 2] : 0.0f) - k2 * ((tid + 2 < 16) ? a[tid + 2] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 2) ? d[tid - 2] : 0.0f) - k2 * ((tid + 2 < 16) ? d[tid + 2] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 2) ? a[tid - 2] : 0.0f);
        c_new[tid] = -k2 * ((tid + 2 < 16) ? c[tid + 2] : 0.0f);
    }
    sync_local();
    if (tid < 16) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 2: Stride 4
    if (tid < 16) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 4) ? (i_a / b[tid - 4]) : 0.0f;
        float k2 = (tid + 4 < 16) ? (i_c / b[tid + 4]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 4) ? c[tid - 4] : 0.0f) - k2 * ((tid + 4 < 16) ? a[tid + 4] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 4) ? d[tid - 4] : 0.0f) - k2 * ((tid + 4 < 16) ? d[tid + 4] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 4) ? a[tid - 4] : 0.0f);
        c_new[tid] = -k2 * ((tid + 4 < 16) ? c[tid + 4] : 0.0f);
    }
    sync_local();
    if (tid < 16) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 3: Stride 8
    if (tid < 16) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 8) ? (i_a / b[tid - 8]) : 0.0f;
        float k2 = (tid + 8 < 16) ? (i_c / b[tid + 8]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 8) ? c[tid - 8] : 0.0f) - k2 * ((tid + 8 < 16) ? a[tid + 8] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 8) ? d[tid - 8] : 0.0f) - k2 * ((tid + 8 < 16) ? d[tid + 8] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 8) ? a[tid - 8] : 0.0f);
        c_new[tid] = -k2 * ((tid + 8 < 16) ? c[tid + 8] : 0.0f);
    }
    sync_local();
    if (tid < 16) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    if (tid < 16) { out_x[base + tid] = d[tid] / b[tid]; }
}

// Batched Parallel Cyclic Reduction: System Size 32
__kernel void kernel_pcr_tridiagonal_sys_32(
    __global const float* in_lower, __global const float* in_diag,
    __global const float* in_upper, __global const float* in_rhs,
    __global float* out_x, const uint batch_count) {
    uint bid = get_group_id(0); if (bid >= batch_count) return;
    uint tid = get_local_id(0);
    __local float a[32], b[32], c[32], d[32];
    __local float a_new[32], b_new[32], c_new[32], d_new[32];
    size_t base = (size_t)bid * 32;
    if (tid < 32) {
        a[tid] = in_lower[base + tid]; b[tid] = in_diag[base + tid];
        c[tid] = in_upper[base + tid]; d[tid] = in_rhs[base + tid];
    }
    sync_local();
    // Step 0: Stride 1
    if (tid < 32) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 1) ? (i_a / b[tid - 1]) : 0.0f;
        float k2 = (tid + 1 < 32) ? (i_c / b[tid + 1]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 1) ? c[tid - 1] : 0.0f) - k2 * ((tid + 1 < 32) ? a[tid + 1] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 1) ? d[tid - 1] : 0.0f) - k2 * ((tid + 1 < 32) ? d[tid + 1] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 1) ? a[tid - 1] : 0.0f);
        c_new[tid] = -k2 * ((tid + 1 < 32) ? c[tid + 1] : 0.0f);
    }
    sync_local();
    if (tid < 32) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 1: Stride 2
    if (tid < 32) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 2) ? (i_a / b[tid - 2]) : 0.0f;
        float k2 = (tid + 2 < 32) ? (i_c / b[tid + 2]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 2) ? c[tid - 2] : 0.0f) - k2 * ((tid + 2 < 32) ? a[tid + 2] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 2) ? d[tid - 2] : 0.0f) - k2 * ((tid + 2 < 32) ? d[tid + 2] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 2) ? a[tid - 2] : 0.0f);
        c_new[tid] = -k2 * ((tid + 2 < 32) ? c[tid + 2] : 0.0f);
    }
    sync_local();
    if (tid < 32) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 2: Stride 4
    if (tid < 32) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 4) ? (i_a / b[tid - 4]) : 0.0f;
        float k2 = (tid + 4 < 32) ? (i_c / b[tid + 4]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 4) ? c[tid - 4] : 0.0f) - k2 * ((tid + 4 < 32) ? a[tid + 4] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 4) ? d[tid - 4] : 0.0f) - k2 * ((tid + 4 < 32) ? d[tid + 4] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 4) ? a[tid - 4] : 0.0f);
        c_new[tid] = -k2 * ((tid + 4 < 32) ? c[tid + 4] : 0.0f);
    }
    sync_local();
    if (tid < 32) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 3: Stride 8
    if (tid < 32) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 8) ? (i_a / b[tid - 8]) : 0.0f;
        float k2 = (tid + 8 < 32) ? (i_c / b[tid + 8]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 8) ? c[tid - 8] : 0.0f) - k2 * ((tid + 8 < 32) ? a[tid + 8] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 8) ? d[tid - 8] : 0.0f) - k2 * ((tid + 8 < 32) ? d[tid + 8] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 8) ? a[tid - 8] : 0.0f);
        c_new[tid] = -k2 * ((tid + 8 < 32) ? c[tid + 8] : 0.0f);
    }
    sync_local();
    if (tid < 32) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 4: Stride 16
    if (tid < 32) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 16) ? (i_a / b[tid - 16]) : 0.0f;
        float k2 = (tid + 16 < 32) ? (i_c / b[tid + 16]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 16) ? c[tid - 16] : 0.0f) - k2 * ((tid + 16 < 32) ? a[tid + 16] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 16) ? d[tid - 16] : 0.0f) - k2 * ((tid + 16 < 32) ? d[tid + 16] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 16) ? a[tid - 16] : 0.0f);
        c_new[tid] = -k2 * ((tid + 16 < 32) ? c[tid + 16] : 0.0f);
    }
    sync_local();
    if (tid < 32) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    if (tid < 32) { out_x[base + tid] = d[tid] / b[tid]; }
}

// Batched Parallel Cyclic Reduction: System Size 64
__kernel void kernel_pcr_tridiagonal_sys_64(
    __global const float* in_lower, __global const float* in_diag,
    __global const float* in_upper, __global const float* in_rhs,
    __global float* out_x, const uint batch_count) {
    uint bid = get_group_id(0); if (bid >= batch_count) return;
    uint tid = get_local_id(0);
    __local float a[64], b[64], c[64], d[64];
    __local float a_new[64], b_new[64], c_new[64], d_new[64];
    size_t base = (size_t)bid * 64;
    if (tid < 64) {
        a[tid] = in_lower[base + tid]; b[tid] = in_diag[base + tid];
        c[tid] = in_upper[base + tid]; d[tid] = in_rhs[base + tid];
    }
    sync_local();
    // Step 0: Stride 1
    if (tid < 64) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 1) ? (i_a / b[tid - 1]) : 0.0f;
        float k2 = (tid + 1 < 64) ? (i_c / b[tid + 1]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 1) ? c[tid - 1] : 0.0f) - k2 * ((tid + 1 < 64) ? a[tid + 1] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 1) ? d[tid - 1] : 0.0f) - k2 * ((tid + 1 < 64) ? d[tid + 1] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 1) ? a[tid - 1] : 0.0f);
        c_new[tid] = -k2 * ((tid + 1 < 64) ? c[tid + 1] : 0.0f);
    }
    sync_local();
    if (tid < 64) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 1: Stride 2
    if (tid < 64) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 2) ? (i_a / b[tid - 2]) : 0.0f;
        float k2 = (tid + 2 < 64) ? (i_c / b[tid + 2]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 2) ? c[tid - 2] : 0.0f) - k2 * ((tid + 2 < 64) ? a[tid + 2] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 2) ? d[tid - 2] : 0.0f) - k2 * ((tid + 2 < 64) ? d[tid + 2] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 2) ? a[tid - 2] : 0.0f);
        c_new[tid] = -k2 * ((tid + 2 < 64) ? c[tid + 2] : 0.0f);
    }
    sync_local();
    if (tid < 64) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 2: Stride 4
    if (tid < 64) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 4) ? (i_a / b[tid - 4]) : 0.0f;
        float k2 = (tid + 4 < 64) ? (i_c / b[tid + 4]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 4) ? c[tid - 4] : 0.0f) - k2 * ((tid + 4 < 64) ? a[tid + 4] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 4) ? d[tid - 4] : 0.0f) - k2 * ((tid + 4 < 64) ? d[tid + 4] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 4) ? a[tid - 4] : 0.0f);
        c_new[tid] = -k2 * ((tid + 4 < 64) ? c[tid + 4] : 0.0f);
    }
    sync_local();
    if (tid < 64) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 3: Stride 8
    if (tid < 64) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 8) ? (i_a / b[tid - 8]) : 0.0f;
        float k2 = (tid + 8 < 64) ? (i_c / b[tid + 8]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 8) ? c[tid - 8] : 0.0f) - k2 * ((tid + 8 < 64) ? a[tid + 8] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 8) ? d[tid - 8] : 0.0f) - k2 * ((tid + 8 < 64) ? d[tid + 8] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 8) ? a[tid - 8] : 0.0f);
        c_new[tid] = -k2 * ((tid + 8 < 64) ? c[tid + 8] : 0.0f);
    }
    sync_local();
    if (tid < 64) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 4: Stride 16
    if (tid < 64) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 16) ? (i_a / b[tid - 16]) : 0.0f;
        float k2 = (tid + 16 < 64) ? (i_c / b[tid + 16]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 16) ? c[tid - 16] : 0.0f) - k2 * ((tid + 16 < 64) ? a[tid + 16] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 16) ? d[tid - 16] : 0.0f) - k2 * ((tid + 16 < 64) ? d[tid + 16] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 16) ? a[tid - 16] : 0.0f);
        c_new[tid] = -k2 * ((tid + 16 < 64) ? c[tid + 16] : 0.0f);
    }
    sync_local();
    if (tid < 64) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 5: Stride 32
    if (tid < 64) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 32) ? (i_a / b[tid - 32]) : 0.0f;
        float k2 = (tid + 32 < 64) ? (i_c / b[tid + 32]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 32) ? c[tid - 32] : 0.0f) - k2 * ((tid + 32 < 64) ? a[tid + 32] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 32) ? d[tid - 32] : 0.0f) - k2 * ((tid + 32 < 64) ? d[tid + 32] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 32) ? a[tid - 32] : 0.0f);
        c_new[tid] = -k2 * ((tid + 32 < 64) ? c[tid + 32] : 0.0f);
    }
    sync_local();
    if (tid < 64) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    if (tid < 64) { out_x[base + tid] = d[tid] / b[tid]; }
}

// Batched Parallel Cyclic Reduction: System Size 128
__kernel void kernel_pcr_tridiagonal_sys_128(
    __global const float* in_lower, __global const float* in_diag,
    __global const float* in_upper, __global const float* in_rhs,
    __global float* out_x, const uint batch_count) {
    uint bid = get_group_id(0); if (bid >= batch_count) return;
    uint tid = get_local_id(0);
    __local float a[128], b[128], c[128], d[128];
    __local float a_new[128], b_new[128], c_new[128], d_new[128];
    size_t base = (size_t)bid * 128;
    if (tid < 128) {
        a[tid] = in_lower[base + tid]; b[tid] = in_diag[base + tid];
        c[tid] = in_upper[base + tid]; d[tid] = in_rhs[base + tid];
    }
    sync_local();
    // Step 0: Stride 1
    if (tid < 128) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 1) ? (i_a / b[tid - 1]) : 0.0f;
        float k2 = (tid + 1 < 128) ? (i_c / b[tid + 1]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 1) ? c[tid - 1] : 0.0f) - k2 * ((tid + 1 < 128) ? a[tid + 1] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 1) ? d[tid - 1] : 0.0f) - k2 * ((tid + 1 < 128) ? d[tid + 1] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 1) ? a[tid - 1] : 0.0f);
        c_new[tid] = -k2 * ((tid + 1 < 128) ? c[tid + 1] : 0.0f);
    }
    sync_local();
    if (tid < 128) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 1: Stride 2
    if (tid < 128) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 2) ? (i_a / b[tid - 2]) : 0.0f;
        float k2 = (tid + 2 < 128) ? (i_c / b[tid + 2]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 2) ? c[tid - 2] : 0.0f) - k2 * ((tid + 2 < 128) ? a[tid + 2] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 2) ? d[tid - 2] : 0.0f) - k2 * ((tid + 2 < 128) ? d[tid + 2] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 2) ? a[tid - 2] : 0.0f);
        c_new[tid] = -k2 * ((tid + 2 < 128) ? c[tid + 2] : 0.0f);
    }
    sync_local();
    if (tid < 128) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 2: Stride 4
    if (tid < 128) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 4) ? (i_a / b[tid - 4]) : 0.0f;
        float k2 = (tid + 4 < 128) ? (i_c / b[tid + 4]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 4) ? c[tid - 4] : 0.0f) - k2 * ((tid + 4 < 128) ? a[tid + 4] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 4) ? d[tid - 4] : 0.0f) - k2 * ((tid + 4 < 128) ? d[tid + 4] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 4) ? a[tid - 4] : 0.0f);
        c_new[tid] = -k2 * ((tid + 4 < 128) ? c[tid + 4] : 0.0f);
    }
    sync_local();
    if (tid < 128) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 3: Stride 8
    if (tid < 128) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 8) ? (i_a / b[tid - 8]) : 0.0f;
        float k2 = (tid + 8 < 128) ? (i_c / b[tid + 8]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 8) ? c[tid - 8] : 0.0f) - k2 * ((tid + 8 < 128) ? a[tid + 8] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 8) ? d[tid - 8] : 0.0f) - k2 * ((tid + 8 < 128) ? d[tid + 8] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 8) ? a[tid - 8] : 0.0f);
        c_new[tid] = -k2 * ((tid + 8 < 128) ? c[tid + 8] : 0.0f);
    }
    sync_local();
    if (tid < 128) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 4: Stride 16
    if (tid < 128) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 16) ? (i_a / b[tid - 16]) : 0.0f;
        float k2 = (tid + 16 < 128) ? (i_c / b[tid + 16]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 16) ? c[tid - 16] : 0.0f) - k2 * ((tid + 16 < 128) ? a[tid + 16] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 16) ? d[tid - 16] : 0.0f) - k2 * ((tid + 16 < 128) ? d[tid + 16] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 16) ? a[tid - 16] : 0.0f);
        c_new[tid] = -k2 * ((tid + 16 < 128) ? c[tid + 16] : 0.0f);
    }
    sync_local();
    if (tid < 128) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 5: Stride 32
    if (tid < 128) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 32) ? (i_a / b[tid - 32]) : 0.0f;
        float k2 = (tid + 32 < 128) ? (i_c / b[tid + 32]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 32) ? c[tid - 32] : 0.0f) - k2 * ((tid + 32 < 128) ? a[tid + 32] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 32) ? d[tid - 32] : 0.0f) - k2 * ((tid + 32 < 128) ? d[tid + 32] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 32) ? a[tid - 32] : 0.0f);
        c_new[tid] = -k2 * ((tid + 32 < 128) ? c[tid + 32] : 0.0f);
    }
    sync_local();
    if (tid < 128) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 6: Stride 64
    if (tid < 128) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 64) ? (i_a / b[tid - 64]) : 0.0f;
        float k2 = (tid + 64 < 128) ? (i_c / b[tid + 64]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 64) ? c[tid - 64] : 0.0f) - k2 * ((tid + 64 < 128) ? a[tid + 64] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 64) ? d[tid - 64] : 0.0f) - k2 * ((tid + 64 < 128) ? d[tid + 64] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 64) ? a[tid - 64] : 0.0f);
        c_new[tid] = -k2 * ((tid + 64 < 128) ? c[tid + 64] : 0.0f);
    }
    sync_local();
    if (tid < 128) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    if (tid < 128) { out_x[base + tid] = d[tid] / b[tid]; }
}

// GMRES Modified Gram-Schmidt Orthogonalization Step
__kernel void kernel_gmres_gram_schmidt_step(
    __global float* v_curr, __global const float* v_basis,
    __global float* h_elem, const uint vector_length, const uint basis_idx) {
    uint gid = get_global_id(0);
    // Parallel dot product between v_curr and v_basis[basis_idx]
    // followed by projection subtraction: v_curr -= h_elem * v_basis[basis_idx]
    if (gid < vector_length) {
        float b_val = v_basis[(size_t)basis_idx * (size_t)vector_length + (size_t)gid];
        v_curr[gid] -= h_elem[basis_idx] * b_val;
    }
}

#include "common/math.clh"
#include "common/memory.clh"

// ============================================================================
// Extended Parallel Cyclic Reduction (PCR) Tridiagonal Solvers (16 up to 512)
// ============================================================================

// Batched Parallel Cyclic Reduction: System Size 16
__kernel void kernel_pcr_tridiagonal_sys_16(
    __global const float* in_lower, __global const float* in_diag,
    __global const float* in_upper, __global const float* in_rhs,
    __global float* out_x, const uint batch_count) {
    uint bid = get_group_id(0); if (bid >= batch_count) return;
    uint tid = get_local_id(0);
    __local float a[16], b[16], c[16], d[16];
    __local float a_new[16], b_new[16], c_new[16], d_new[16];
    size_t base = (size_t)bid * 16;
    if (tid < 16) {
        a[tid] = in_lower[base + tid]; b[tid] = in_diag[base + tid];
        c[tid] = in_upper[base + tid]; d[tid] = in_rhs[base + tid];
    }
    sync_local();
    // Step 0: Stride 1
    if (tid < 16) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 1) ? (i_a / b[tid - 1]) : 0.0f;
        float k2 = (tid + 1 < 16) ? (i_c / b[tid + 1]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 1) ? c[tid - 1] : 0.0f) - k2 * ((tid + 1 < 16) ? a[tid + 1] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 1) ? d[tid - 1] : 0.0f) - k2 * ((tid + 1 < 16) ? d[tid + 1] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 1) ? a[tid - 1] : 0.0f);
        c_new[tid] = -k2 * ((tid + 1 < 16) ? c[tid + 1] : 0.0f);
    }
    sync_local();
    if (tid < 16) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 1: Stride 2
    if (tid < 16) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 2) ? (i_a / b[tid - 2]) : 0.0f;
        float k2 = (tid + 2 < 16) ? (i_c / b[tid + 2]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 2) ? c[tid - 2] : 0.0f) - k2 * ((tid + 2 < 16) ? a[tid + 2] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 2) ? d[tid - 2] : 0.0f) - k2 * ((tid + 2 < 16) ? d[tid + 2] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 2) ? a[tid - 2] : 0.0f);
        c_new[tid] = -k2 * ((tid + 2 < 16) ? c[tid + 2] : 0.0f);
    }
    sync_local();
    if (tid < 16) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 2: Stride 4
    if (tid < 16) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 4) ? (i_a / b[tid - 4]) : 0.0f;
        float k2 = (tid + 4 < 16) ? (i_c / b[tid + 4]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 4) ? c[tid - 4] : 0.0f) - k2 * ((tid + 4 < 16) ? a[tid + 4] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 4) ? d[tid - 4] : 0.0f) - k2 * ((tid + 4 < 16) ? d[tid + 4] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 4) ? a[tid - 4] : 0.0f);
        c_new[tid] = -k2 * ((tid + 4 < 16) ? c[tid + 4] : 0.0f);
    }
    sync_local();
    if (tid < 16) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 3: Stride 8
    if (tid < 16) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 8) ? (i_a / b[tid - 8]) : 0.0f;
        float k2 = (tid + 8 < 16) ? (i_c / b[tid + 8]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 8) ? c[tid - 8] : 0.0f) - k2 * ((tid + 8 < 16) ? a[tid + 8] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 8) ? d[tid - 8] : 0.0f) - k2 * ((tid + 8 < 16) ? d[tid + 8] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 8) ? a[tid - 8] : 0.0f);
        c_new[tid] = -k2 * ((tid + 8 < 16) ? c[tid + 8] : 0.0f);
    }
    sync_local();
    if (tid < 16) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    if (tid < 16) { out_x[base + tid] = d[tid] / b[tid]; }
}

// Batched Parallel Cyclic Reduction: System Size 32
__kernel void kernel_pcr_tridiagonal_sys_32(
    __global const float* in_lower, __global const float* in_diag,
    __global const float* in_upper, __global const float* in_rhs,
    __global float* out_x, const uint batch_count) {
    uint bid = get_group_id(0); if (bid >= batch_count) return;
    uint tid = get_local_id(0);
    __local float a[32], b[32], c[32], d[32];
    __local float a_new[32], b_new[32], c_new[32], d_new[32];
    size_t base = (size_t)bid * 32;
    if (tid < 32) {
        a[tid] = in_lower[base + tid]; b[tid] = in_diag[base + tid];
        c[tid] = in_upper[base + tid]; d[tid] = in_rhs[base + tid];
    }
    sync_local();
    // Step 0: Stride 1
    if (tid < 32) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 1) ? (i_a / b[tid - 1]) : 0.0f;
        float k2 = (tid + 1 < 32) ? (i_c / b[tid + 1]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 1) ? c[tid - 1] : 0.0f) - k2 * ((tid + 1 < 32) ? a[tid + 1] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 1) ? d[tid - 1] : 0.0f) - k2 * ((tid + 1 < 32) ? d[tid + 1] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 1) ? a[tid - 1] : 0.0f);
        c_new[tid] = -k2 * ((tid + 1 < 32) ? c[tid + 1] : 0.0f);
    }
    sync_local();
    if (tid < 32) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 1: Stride 2
    if (tid < 32) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 2) ? (i_a / b[tid - 2]) : 0.0f;
        float k2 = (tid + 2 < 32) ? (i_c / b[tid + 2]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 2) ? c[tid - 2] : 0.0f) - k2 * ((tid + 2 < 32) ? a[tid + 2] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 2) ? d[tid - 2] : 0.0f) - k2 * ((tid + 2 < 32) ? d[tid + 2] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 2) ? a[tid - 2] : 0.0f);
        c_new[tid] = -k2 * ((tid + 2 < 32) ? c[tid + 2] : 0.0f);
    }
    sync_local();
    if (tid < 32) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 2: Stride 4
    if (tid < 32) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 4) ? (i_a / b[tid - 4]) : 0.0f;
        float k2 = (tid + 4 < 32) ? (i_c / b[tid + 4]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 4) ? c[tid - 4] : 0.0f) - k2 * ((tid + 4 < 32) ? a[tid + 4] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 4) ? d[tid - 4] : 0.0f) - k2 * ((tid + 4 < 32) ? d[tid + 4] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 4) ? a[tid - 4] : 0.0f);
        c_new[tid] = -k2 * ((tid + 4 < 32) ? c[tid + 4] : 0.0f);
    }
    sync_local();
    if (tid < 32) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 3: Stride 8
    if (tid < 32) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 8) ? (i_a / b[tid - 8]) : 0.0f;
        float k2 = (tid + 8 < 32) ? (i_c / b[tid + 8]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 8) ? c[tid - 8] : 0.0f) - k2 * ((tid + 8 < 32) ? a[tid + 8] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 8) ? d[tid - 8] : 0.0f) - k2 * ((tid + 8 < 32) ? d[tid + 8] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 8) ? a[tid - 8] : 0.0f);
        c_new[tid] = -k2 * ((tid + 8 < 32) ? c[tid + 8] : 0.0f);
    }
    sync_local();
    if (tid < 32) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 4: Stride 16
    if (tid < 32) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 16) ? (i_a / b[tid - 16]) : 0.0f;
        float k2 = (tid + 16 < 32) ? (i_c / b[tid + 16]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 16) ? c[tid - 16] : 0.0f) - k2 * ((tid + 16 < 32) ? a[tid + 16] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 16) ? d[tid - 16] : 0.0f) - k2 * ((tid + 16 < 32) ? d[tid + 16] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 16) ? a[tid - 16] : 0.0f);
        c_new[tid] = -k2 * ((tid + 16 < 32) ? c[tid + 16] : 0.0f);
    }
    sync_local();
    if (tid < 32) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    if (tid < 32) { out_x[base + tid] = d[tid] / b[tid]; }
}

// Batched Parallel Cyclic Reduction: System Size 64
__kernel void kernel_pcr_tridiagonal_sys_64(
    __global const float* in_lower, __global const float* in_diag,
    __global const float* in_upper, __global const float* in_rhs,
    __global float* out_x, const uint batch_count) {
    uint bid = get_group_id(0); if (bid >= batch_count) return;
    uint tid = get_local_id(0);
    __local float a[64], b[64], c[64], d[64];
    __local float a_new[64], b_new[64], c_new[64], d_new[64];
    size_t base = (size_t)bid * 64;
    if (tid < 64) {
        a[tid] = in_lower[base + tid]; b[tid] = in_diag[base + tid];
        c[tid] = in_upper[base + tid]; d[tid] = in_rhs[base + tid];
    }
    sync_local();
    // Step 0: Stride 1
    if (tid < 64) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 1) ? (i_a / b[tid - 1]) : 0.0f;
        float k2 = (tid + 1 < 64) ? (i_c / b[tid + 1]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 1) ? c[tid - 1] : 0.0f) - k2 * ((tid + 1 < 64) ? a[tid + 1] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 1) ? d[tid - 1] : 0.0f) - k2 * ((tid + 1 < 64) ? d[tid + 1] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 1) ? a[tid - 1] : 0.0f);
        c_new[tid] = -k2 * ((tid + 1 < 64) ? c[tid + 1] : 0.0f);
    }
    sync_local();
    if (tid < 64) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 1: Stride 2
    if (tid < 64) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 2) ? (i_a / b[tid - 2]) : 0.0f;
        float k2 = (tid + 2 < 64) ? (i_c / b[tid + 2]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 2) ? c[tid - 2] : 0.0f) - k2 * ((tid + 2 < 64) ? a[tid + 2] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 2) ? d[tid - 2] : 0.0f) - k2 * ((tid + 2 < 64) ? d[tid + 2] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 2) ? a[tid - 2] : 0.0f);
        c_new[tid] = -k2 * ((tid + 2 < 64) ? c[tid + 2] : 0.0f);
    }
    sync_local();
    if (tid < 64) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 2: Stride 4
    if (tid < 64) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 4) ? (i_a / b[tid - 4]) : 0.0f;
        float k2 = (tid + 4 < 64) ? (i_c / b[tid + 4]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 4) ? c[tid - 4] : 0.0f) - k2 * ((tid + 4 < 64) ? a[tid + 4] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 4) ? d[tid - 4] : 0.0f) - k2 * ((tid + 4 < 64) ? d[tid + 4] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 4) ? a[tid - 4] : 0.0f);
        c_new[tid] = -k2 * ((tid + 4 < 64) ? c[tid + 4] : 0.0f);
    }
    sync_local();
    if (tid < 64) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 3: Stride 8
    if (tid < 64) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 8) ? (i_a / b[tid - 8]) : 0.0f;
        float k2 = (tid + 8 < 64) ? (i_c / b[tid + 8]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 8) ? c[tid - 8] : 0.0f) - k2 * ((tid + 8 < 64) ? a[tid + 8] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 8) ? d[tid - 8] : 0.0f) - k2 * ((tid + 8 < 64) ? d[tid + 8] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 8) ? a[tid - 8] : 0.0f);
        c_new[tid] = -k2 * ((tid + 8 < 64) ? c[tid + 8] : 0.0f);
    }
    sync_local();
    if (tid < 64) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 4: Stride 16
    if (tid < 64) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 16) ? (i_a / b[tid - 16]) : 0.0f;
        float k2 = (tid + 16 < 64) ? (i_c / b[tid + 16]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 16) ? c[tid - 16] : 0.0f) - k2 * ((tid + 16 < 64) ? a[tid + 16] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 16) ? d[tid - 16] : 0.0f) - k2 * ((tid + 16 < 64) ? d[tid + 16] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 16) ? a[tid - 16] : 0.0f);
        c_new[tid] = -k2 * ((tid + 16 < 64) ? c[tid + 16] : 0.0f);
    }
    sync_local();
    if (tid < 64) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 5: Stride 32
    if (tid < 64) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 32) ? (i_a / b[tid - 32]) : 0.0f;
        float k2 = (tid + 32 < 64) ? (i_c / b[tid + 32]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 32) ? c[tid - 32] : 0.0f) - k2 * ((tid + 32 < 64) ? a[tid + 32] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 32) ? d[tid - 32] : 0.0f) - k2 * ((tid + 32 < 64) ? d[tid + 32] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 32) ? a[tid - 32] : 0.0f);
        c_new[tid] = -k2 * ((tid + 32 < 64) ? c[tid + 32] : 0.0f);
    }
    sync_local();
    if (tid < 64) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    if (tid < 64) { out_x[base + tid] = d[tid] / b[tid]; }
}

// Batched Parallel Cyclic Reduction: System Size 128
__kernel void kernel_pcr_tridiagonal_sys_128(
    __global const float* in_lower, __global const float* in_diag,
    __global const float* in_upper, __global const float* in_rhs,
    __global float* out_x, const uint batch_count) {
    uint bid = get_group_id(0); if (bid >= batch_count) return;
    uint tid = get_local_id(0);
    __local float a[128], b[128], c[128], d[128];
    __local float a_new[128], b_new[128], c_new[128], d_new[128];
    size_t base = (size_t)bid * 128;
    if (tid < 128) {
        a[tid] = in_lower[base + tid]; b[tid] = in_diag[base + tid];
        c[tid] = in_upper[base + tid]; d[tid] = in_rhs[base + tid];
    }
    sync_local();
    // Step 0: Stride 1
    if (tid < 128) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 1) ? (i_a / b[tid - 1]) : 0.0f;
        float k2 = (tid + 1 < 128) ? (i_c / b[tid + 1]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 1) ? c[tid - 1] : 0.0f) - k2 * ((tid + 1 < 128) ? a[tid + 1] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 1) ? d[tid - 1] : 0.0f) - k2 * ((tid + 1 < 128) ? d[tid + 1] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 1) ? a[tid - 1] : 0.0f);
        c_new[tid] = -k2 * ((tid + 1 < 128) ? c[tid + 1] : 0.0f);
    }
    sync_local();
    if (tid < 128) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 1: Stride 2
    if (tid < 128) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 2) ? (i_a / b[tid - 2]) : 0.0f;
        float k2 = (tid + 2 < 128) ? (i_c / b[tid + 2]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 2) ? c[tid - 2] : 0.0f) - k2 * ((tid + 2 < 128) ? a[tid + 2] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 2) ? d[tid - 2] : 0.0f) - k2 * ((tid + 2 < 128) ? d[tid + 2] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 2) ? a[tid - 2] : 0.0f);
        c_new[tid] = -k2 * ((tid + 2 < 128) ? c[tid + 2] : 0.0f);
    }
    sync_local();
    if (tid < 128) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 2: Stride 4
    if (tid < 128) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 4) ? (i_a / b[tid - 4]) : 0.0f;
        float k2 = (tid + 4 < 128) ? (i_c / b[tid + 4]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 4) ? c[tid - 4] : 0.0f) - k2 * ((tid + 4 < 128) ? a[tid + 4] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 4) ? d[tid - 4] : 0.0f) - k2 * ((tid + 4 < 128) ? d[tid + 4] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 4) ? a[tid - 4] : 0.0f);
        c_new[tid] = -k2 * ((tid + 4 < 128) ? c[tid + 4] : 0.0f);
    }
    sync_local();
    if (tid < 128) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 3: Stride 8
    if (tid < 128) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 8) ? (i_a / b[tid - 8]) : 0.0f;
        float k2 = (tid + 8 < 128) ? (i_c / b[tid + 8]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 8) ? c[tid - 8] : 0.0f) - k2 * ((tid + 8 < 128) ? a[tid + 8] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 8) ? d[tid - 8] : 0.0f) - k2 * ((tid + 8 < 128) ? d[tid + 8] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 8) ? a[tid - 8] : 0.0f);
        c_new[tid] = -k2 * ((tid + 8 < 128) ? c[tid + 8] : 0.0f);
    }
    sync_local();
    if (tid < 128) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 4: Stride 16
    if (tid < 128) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 16) ? (i_a / b[tid - 16]) : 0.0f;
        float k2 = (tid + 16 < 128) ? (i_c / b[tid + 16]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 16) ? c[tid - 16] : 0.0f) - k2 * ((tid + 16 < 128) ? a[tid + 16] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 16) ? d[tid - 16] : 0.0f) - k2 * ((tid + 16 < 128) ? d[tid + 16] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 16) ? a[tid - 16] : 0.0f);
        c_new[tid] = -k2 * ((tid + 16 < 128) ? c[tid + 16] : 0.0f);
    }
    sync_local();
    if (tid < 128) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 5: Stride 32
    if (tid < 128) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 32) ? (i_a / b[tid - 32]) : 0.0f;
        float k2 = (tid + 32 < 128) ? (i_c / b[tid + 32]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 32) ? c[tid - 32] : 0.0f) - k2 * ((tid + 32 < 128) ? a[tid + 32] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 32) ? d[tid - 32] : 0.0f) - k2 * ((tid + 32 < 128) ? d[tid + 32] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 32) ? a[tid - 32] : 0.0f);
        c_new[tid] = -k2 * ((tid + 32 < 128) ? c[tid + 32] : 0.0f);
    }
    sync_local();
    if (tid < 128) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 6: Stride 64
    if (tid < 128) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 64) ? (i_a / b[tid - 64]) : 0.0f;
        float k2 = (tid + 64 < 128) ? (i_c / b[tid + 64]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 64) ? c[tid - 64] : 0.0f) - k2 * ((tid + 64 < 128) ? a[tid + 64] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 64) ? d[tid - 64] : 0.0f) - k2 * ((tid + 64 < 128) ? d[tid + 64] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 64) ? a[tid - 64] : 0.0f);
        c_new[tid] = -k2 * ((tid + 64 < 128) ? c[tid + 64] : 0.0f);
    }
    sync_local();
    if (tid < 128) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    if (tid < 128) { out_x[base + tid] = d[tid] / b[tid]; }
}

// Batched Parallel Cyclic Reduction: System Size 256
__kernel void kernel_pcr_tridiagonal_sys_256(
    __global const float* in_lower, __global const float* in_diag,
    __global const float* in_upper, __global const float* in_rhs,
    __global float* out_x, const uint batch_count) {
    uint bid = get_group_id(0); if (bid >= batch_count) return;
    uint tid = get_local_id(0);
    __local float a[256], b[256], c[256], d[256];
    __local float a_new[256], b_new[256], c_new[256], d_new[256];
    size_t base = (size_t)bid * 256;
    if (tid < 256) {
        a[tid] = in_lower[base + tid]; b[tid] = in_diag[base + tid];
        c[tid] = in_upper[base + tid]; d[tid] = in_rhs[base + tid];
    }
    sync_local();
    // Step 0: Stride 1
    if (tid < 256) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 1) ? (i_a / b[tid - 1]) : 0.0f;
        float k2 = (tid + 1 < 256) ? (i_c / b[tid + 1]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 1) ? c[tid - 1] : 0.0f) - k2 * ((tid + 1 < 256) ? a[tid + 1] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 1) ? d[tid - 1] : 0.0f) - k2 * ((tid + 1 < 256) ? d[tid + 1] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 1) ? a[tid - 1] : 0.0f);
        c_new[tid] = -k2 * ((tid + 1 < 256) ? c[tid + 1] : 0.0f);
    }
    sync_local();
    if (tid < 256) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 1: Stride 2
    if (tid < 256) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 2) ? (i_a / b[tid - 2]) : 0.0f;
        float k2 = (tid + 2 < 256) ? (i_c / b[tid + 2]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 2) ? c[tid - 2] : 0.0f) - k2 * ((tid + 2 < 256) ? a[tid + 2] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 2) ? d[tid - 2] : 0.0f) - k2 * ((tid + 2 < 256) ? d[tid + 2] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 2) ? a[tid - 2] : 0.0f);
        c_new[tid] = -k2 * ((tid + 2 < 256) ? c[tid + 2] : 0.0f);
    }
    sync_local();
    if (tid < 256) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 2: Stride 4
    if (tid < 256) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 4) ? (i_a / b[tid - 4]) : 0.0f;
        float k2 = (tid + 4 < 256) ? (i_c / b[tid + 4]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 4) ? c[tid - 4] : 0.0f) - k2 * ((tid + 4 < 256) ? a[tid + 4] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 4) ? d[tid - 4] : 0.0f) - k2 * ((tid + 4 < 256) ? d[tid + 4] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 4) ? a[tid - 4] : 0.0f);
        c_new[tid] = -k2 * ((tid + 4 < 256) ? c[tid + 4] : 0.0f);
    }
    sync_local();
    if (tid < 256) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 3: Stride 8
    if (tid < 256) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 8) ? (i_a / b[tid - 8]) : 0.0f;
        float k2 = (tid + 8 < 256) ? (i_c / b[tid + 8]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 8) ? c[tid - 8] : 0.0f) - k2 * ((tid + 8 < 256) ? a[tid + 8] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 8) ? d[tid - 8] : 0.0f) - k2 * ((tid + 8 < 256) ? d[tid + 8] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 8) ? a[tid - 8] : 0.0f);
        c_new[tid] = -k2 * ((tid + 8 < 256) ? c[tid + 8] : 0.0f);
    }
    sync_local();
    if (tid < 256) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 4: Stride 16
    if (tid < 256) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 16) ? (i_a / b[tid - 16]) : 0.0f;
        float k2 = (tid + 16 < 256) ? (i_c / b[tid + 16]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 16) ? c[tid - 16] : 0.0f) - k2 * ((tid + 16 < 256) ? a[tid + 16] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 16) ? d[tid - 16] : 0.0f) - k2 * ((tid + 16 < 256) ? d[tid + 16] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 16) ? a[tid - 16] : 0.0f);
        c_new[tid] = -k2 * ((tid + 16 < 256) ? c[tid + 16] : 0.0f);
    }
    sync_local();
    if (tid < 256) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 5: Stride 32
    if (tid < 256) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 32) ? (i_a / b[tid - 32]) : 0.0f;
        float k2 = (tid + 32 < 256) ? (i_c / b[tid + 32]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 32) ? c[tid - 32] : 0.0f) - k2 * ((tid + 32 < 256) ? a[tid + 32] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 32) ? d[tid - 32] : 0.0f) - k2 * ((tid + 32 < 256) ? d[tid + 32] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 32) ? a[tid - 32] : 0.0f);
        c_new[tid] = -k2 * ((tid + 32 < 256) ? c[tid + 32] : 0.0f);
    }
    sync_local();
    if (tid < 256) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 6: Stride 64
    if (tid < 256) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 64) ? (i_a / b[tid - 64]) : 0.0f;
        float k2 = (tid + 64 < 256) ? (i_c / b[tid + 64]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 64) ? c[tid - 64] : 0.0f) - k2 * ((tid + 64 < 256) ? a[tid + 64] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 64) ? d[tid - 64] : 0.0f) - k2 * ((tid + 64 < 256) ? d[tid + 64] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 64) ? a[tid - 64] : 0.0f);
        c_new[tid] = -k2 * ((tid + 64 < 256) ? c[tid + 64] : 0.0f);
    }
    sync_local();
    if (tid < 256) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 7: Stride 128
    if (tid < 256) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 128) ? (i_a / b[tid - 128]) : 0.0f;
        float k2 = (tid + 128 < 256) ? (i_c / b[tid + 128]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 128) ? c[tid - 128] : 0.0f) - k2 * ((tid + 128 < 256) ? a[tid + 128] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 128) ? d[tid - 128] : 0.0f) - k2 * ((tid + 128 < 256) ? d[tid + 128] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 128) ? a[tid - 128] : 0.0f);
        c_new[tid] = -k2 * ((tid + 128 < 256) ? c[tid + 128] : 0.0f);
    }
    sync_local();
    if (tid < 256) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    if (tid < 256) { out_x[base + tid] = d[tid] / b[tid]; }
}

// Batched Parallel Cyclic Reduction: System Size 512
__kernel void kernel_pcr_tridiagonal_sys_512(
    __global const float* in_lower, __global const float* in_diag,
    __global const float* in_upper, __global const float* in_rhs,
    __global float* out_x, const uint batch_count) {
    uint bid = get_group_id(0); if (bid >= batch_count) return;
    uint tid = get_local_id(0);
    __local float a[512], b[512], c[512], d[512];
    __local float a_new[512], b_new[512], c_new[512], d_new[512];
    size_t base = (size_t)bid * 512;
    if (tid < 512) {
        a[tid] = in_lower[base + tid]; b[tid] = in_diag[base + tid];
        c[tid] = in_upper[base + tid]; d[tid] = in_rhs[base + tid];
    }
    sync_local();
    // Step 0: Stride 1
    if (tid < 512) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 1) ? (i_a / b[tid - 1]) : 0.0f;
        float k2 = (tid + 1 < 512) ? (i_c / b[tid + 1]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 1) ? c[tid - 1] : 0.0f) - k2 * ((tid + 1 < 512) ? a[tid + 1] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 1) ? d[tid - 1] : 0.0f) - k2 * ((tid + 1 < 512) ? d[tid + 1] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 1) ? a[tid - 1] : 0.0f);
        c_new[tid] = -k2 * ((tid + 1 < 512) ? c[tid + 1] : 0.0f);
    }
    sync_local();
    if (tid < 512) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 1: Stride 2
    if (tid < 512) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 2) ? (i_a / b[tid - 2]) : 0.0f;
        float k2 = (tid + 2 < 512) ? (i_c / b[tid + 2]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 2) ? c[tid - 2] : 0.0f) - k2 * ((tid + 2 < 512) ? a[tid + 2] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 2) ? d[tid - 2] : 0.0f) - k2 * ((tid + 2 < 512) ? d[tid + 2] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 2) ? a[tid - 2] : 0.0f);
        c_new[tid] = -k2 * ((tid + 2 < 512) ? c[tid + 2] : 0.0f);
    }
    sync_local();
    if (tid < 512) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 2: Stride 4
    if (tid < 512) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 4) ? (i_a / b[tid - 4]) : 0.0f;
        float k2 = (tid + 4 < 512) ? (i_c / b[tid + 4]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 4) ? c[tid - 4] : 0.0f) - k2 * ((tid + 4 < 512) ? a[tid + 4] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 4) ? d[tid - 4] : 0.0f) - k2 * ((tid + 4 < 512) ? d[tid + 4] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 4) ? a[tid - 4] : 0.0f);
        c_new[tid] = -k2 * ((tid + 4 < 512) ? c[tid + 4] : 0.0f);
    }
    sync_local();
    if (tid < 512) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 3: Stride 8
    if (tid < 512) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 8) ? (i_a / b[tid - 8]) : 0.0f;
        float k2 = (tid + 8 < 512) ? (i_c / b[tid + 8]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 8) ? c[tid - 8] : 0.0f) - k2 * ((tid + 8 < 512) ? a[tid + 8] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 8) ? d[tid - 8] : 0.0f) - k2 * ((tid + 8 < 512) ? d[tid + 8] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 8) ? a[tid - 8] : 0.0f);
        c_new[tid] = -k2 * ((tid + 8 < 512) ? c[tid + 8] : 0.0f);
    }
    sync_local();
    if (tid < 512) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 4: Stride 16
    if (tid < 512) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 16) ? (i_a / b[tid - 16]) : 0.0f;
        float k2 = (tid + 16 < 512) ? (i_c / b[tid + 16]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 16) ? c[tid - 16] : 0.0f) - k2 * ((tid + 16 < 512) ? a[tid + 16] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 16) ? d[tid - 16] : 0.0f) - k2 * ((tid + 16 < 512) ? d[tid + 16] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 16) ? a[tid - 16] : 0.0f);
        c_new[tid] = -k2 * ((tid + 16 < 512) ? c[tid + 16] : 0.0f);
    }
    sync_local();
    if (tid < 512) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 5: Stride 32
    if (tid < 512) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 32) ? (i_a / b[tid - 32]) : 0.0f;
        float k2 = (tid + 32 < 512) ? (i_c / b[tid + 32]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 32) ? c[tid - 32] : 0.0f) - k2 * ((tid + 32 < 512) ? a[tid + 32] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 32) ? d[tid - 32] : 0.0f) - k2 * ((tid + 32 < 512) ? d[tid + 32] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 32) ? a[tid - 32] : 0.0f);
        c_new[tid] = -k2 * ((tid + 32 < 512) ? c[tid + 32] : 0.0f);
    }
    sync_local();
    if (tid < 512) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 6: Stride 64
    if (tid < 512) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 64) ? (i_a / b[tid - 64]) : 0.0f;
        float k2 = (tid + 64 < 512) ? (i_c / b[tid + 64]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 64) ? c[tid - 64] : 0.0f) - k2 * ((tid + 64 < 512) ? a[tid + 64] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 64) ? d[tid - 64] : 0.0f) - k2 * ((tid + 64 < 512) ? d[tid + 64] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 64) ? a[tid - 64] : 0.0f);
        c_new[tid] = -k2 * ((tid + 64 < 512) ? c[tid + 64] : 0.0f);
    }
    sync_local();
    if (tid < 512) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 7: Stride 128
    if (tid < 512) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 128) ? (i_a / b[tid - 128]) : 0.0f;
        float k2 = (tid + 128 < 512) ? (i_c / b[tid + 128]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 128) ? c[tid - 128] : 0.0f) - k2 * ((tid + 128 < 512) ? a[tid + 128] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 128) ? d[tid - 128] : 0.0f) - k2 * ((tid + 128 < 512) ? d[tid + 128] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 128) ? a[tid - 128] : 0.0f);
        c_new[tid] = -k2 * ((tid + 128 < 512) ? c[tid + 128] : 0.0f);
    }
    sync_local();
    if (tid < 512) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    // Step 8: Stride 256
    if (tid < 512) {
        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];
        float k1 = (tid >= 256) ? (i_a / b[tid - 256]) : 0.0f;
        float k2 = (tid + 256 < 512) ? (i_c / b[tid + 256]) : 0.0f;
        b_new[tid] = i_b - k1 * ((tid >= 256) ? c[tid - 256] : 0.0f) - k2 * ((tid + 256 < 512) ? a[tid + 256] : 0.0f);
        d_new[tid] = i_d - k1 * ((tid >= 256) ? d[tid - 256] : 0.0f) - k2 * ((tid + 256 < 512) ? d[tid + 256] : 0.0f);
        a_new[tid] = -k1 * ((tid >= 256) ? a[tid - 256] : 0.0f);
        c_new[tid] = -k2 * ((tid + 256 < 512) ? c[tid + 256] : 0.0f);
    }
    sync_local();
    if (tid < 512) { a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }
    sync_local();
    if (tid < 512) { out_x[base + tid] = d[tid] / b[tid]; }
}


// ============================================================================
// Batched Householder QR Factorization and Batched Jacobi SVD Solvers
// ============================================================================

// Batched 3x3 Householder QR Factorization (A = Q * R)
__kernel void kernel_batched_qr_3x3(
    __global const float* in_mats, __global float* out_Q, __global float* out_R, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 9;
    float A[3][3]; float Q[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        A[i][j] = in_mats[base + i * 3 + j];
        Q[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    for (int k = 0; k < 2; ++k) {
        float norm_x = 0.0f;
        for (int i = k; i < 3; ++i) norm_x += A[i][k] * A[i][k];
        norm_x = sqrt(norm_x);
        if (norm_x > 1.0e-7f) {
            float s = (A[k][k] >= 0.0f) ? 1.0f : -1.0f;
            float u1 = A[k][k] + s * norm_x;
            float v[3];
            v[k] = 1.0f;
            for (int i = k + 1; i < 3; ++i) v[i] = A[i][k] / u1;
            float tau = 2.0f / (1.0f + (norm_x * norm_x - A[k][k] * A[k][k]) / (u1 * u1));
            // Apply Householder reflector H = I - tau * v * v^T to A and Q
            for (int j = k; j < 3; ++j) {
                float dot_val = 0.0f;
                for (int i = k; i < 3; ++i) dot_val += v[i] * A[i][j];
                for (int i = k; i < 3; ++i) A[i][j] -= tau * dot_val * v[i];
            }
            for (int j = 0; j < 3; ++j) {
                float dot_val = 0.0f;
                for (int i = k; i < 3; ++i) dot_val += v[i] * Q[j][i];
                for (int i = k; i < 3; ++i) Q[j][i] -= tau * dot_val * v[i];
            }
        }
    }
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        out_Q[base + i * 3 + j] = Q[i][j];
        out_R[base + i * 3 + j] = (i <= j) ? A[i][j] : 0.0f;
    }
}

// Batched 4x4 Householder QR Factorization (A = Q * R)
__kernel void kernel_batched_qr_4x4(
    __global const float* in_mats, __global float* out_Q, __global float* out_R, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 16;
    float A[4][4]; float Q[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
        A[i][j] = in_mats[base + i * 4 + j];
        Q[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    for (int k = 0; k < 3; ++k) {
        float norm_x = 0.0f;
        for (int i = k; i < 4; ++i) norm_x += A[i][k] * A[i][k];
        norm_x = sqrt(norm_x);
        if (norm_x > 1.0e-7f) {
            float s = (A[k][k] >= 0.0f) ? 1.0f : -1.0f;
            float u1 = A[k][k] + s * norm_x;
            float v[4];
            v[k] = 1.0f;
            for (int i = k + 1; i < 4; ++i) v[i] = A[i][k] / u1;
            float tau = 2.0f / (1.0f + (norm_x * norm_x - A[k][k] * A[k][k]) / (u1 * u1));
            // Apply Householder reflector H = I - tau * v * v^T to A and Q
            for (int j = k; j < 4; ++j) {
                float dot_val = 0.0f;
                for (int i = k; i < 4; ++i) dot_val += v[i] * A[i][j];
                for (int i = k; i < 4; ++i) A[i][j] -= tau * dot_val * v[i];
            }
            for (int j = 0; j < 4; ++j) {
                float dot_val = 0.0f;
                for (int i = k; i < 4; ++i) dot_val += v[i] * Q[j][i];
                for (int i = k; i < 4; ++i) Q[j][i] -= tau * dot_val * v[i];
            }
        }
    }
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
        out_Q[base + i * 4 + j] = Q[i][j];
        out_R[base + i * 4 + j] = (i <= j) ? A[i][j] : 0.0f;
    }
}

// Batched 5x5 Householder QR Factorization (A = Q * R)
__kernel void kernel_batched_qr_5x5(
    __global const float* in_mats, __global float* out_Q, __global float* out_R, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 25;
    float A[5][5]; float Q[5][5];
    for (int i = 0; i < 5; ++i) for (int j = 0; j < 5; ++j) {
        A[i][j] = in_mats[base + i * 5 + j];
        Q[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    for (int k = 0; k < 4; ++k) {
        float norm_x = 0.0f;
        for (int i = k; i < 5; ++i) norm_x += A[i][k] * A[i][k];
        norm_x = sqrt(norm_x);
        if (norm_x > 1.0e-7f) {
            float s = (A[k][k] >= 0.0f) ? 1.0f : -1.0f;
            float u1 = A[k][k] + s * norm_x;
            float v[5];
            v[k] = 1.0f;
            for (int i = k + 1; i < 5; ++i) v[i] = A[i][k] / u1;
            float tau = 2.0f / (1.0f + (norm_x * norm_x - A[k][k] * A[k][k]) / (u1 * u1));
            // Apply Householder reflector H = I - tau * v * v^T to A and Q
            for (int j = k; j < 5; ++j) {
                float dot_val = 0.0f;
                for (int i = k; i < 5; ++i) dot_val += v[i] * A[i][j];
                for (int i = k; i < 5; ++i) A[i][j] -= tau * dot_val * v[i];
            }
            for (int j = 0; j < 5; ++j) {
                float dot_val = 0.0f;
                for (int i = k; i < 5; ++i) dot_val += v[i] * Q[j][i];
                for (int i = k; i < 5; ++i) Q[j][i] -= tau * dot_val * v[i];
            }
        }
    }
    for (int i = 0; i < 5; ++i) for (int j = 0; j < 5; ++j) {
        out_Q[base + i * 5 + j] = Q[i][j];
        out_R[base + i * 5 + j] = (i <= j) ? A[i][j] : 0.0f;
    }
}

// Batched 6x6 Householder QR Factorization (A = Q * R)
__kernel void kernel_batched_qr_6x6(
    __global const float* in_mats, __global float* out_Q, __global float* out_R, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 36;
    float A[6][6]; float Q[6][6];
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) {
        A[i][j] = in_mats[base + i * 6 + j];
        Q[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    for (int k = 0; k < 5; ++k) {
        float norm_x = 0.0f;
        for (int i = k; i < 6; ++i) norm_x += A[i][k] * A[i][k];
        norm_x = sqrt(norm_x);
        if (norm_x > 1.0e-7f) {
            float s = (A[k][k] >= 0.0f) ? 1.0f : -1.0f;
            float u1 = A[k][k] + s * norm_x;
            float v[6];
            v[k] = 1.0f;
            for (int i = k + 1; i < 6; ++i) v[i] = A[i][k] / u1;
            float tau = 2.0f / (1.0f + (norm_x * norm_x - A[k][k] * A[k][k]) / (u1 * u1));
            // Apply Householder reflector H = I - tau * v * v^T to A and Q
            for (int j = k; j < 6; ++j) {
                float dot_val = 0.0f;
                for (int i = k; i < 6; ++i) dot_val += v[i] * A[i][j];
                for (int i = k; i < 6; ++i) A[i][j] -= tau * dot_val * v[i];
            }
            for (int j = 0; j < 6; ++j) {
                float dot_val = 0.0f;
                for (int i = k; i < 6; ++i) dot_val += v[i] * Q[j][i];
                for (int i = k; i < 6; ++i) Q[j][i] -= tau * dot_val * v[i];
            }
        }
    }
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) {
        out_Q[base + i * 6 + j] = Q[i][j];
        out_R[base + i * 6 + j] = (i <= j) ? A[i][j] : 0.0f;
    }
}

// Batched 8x8 Householder QR Factorization (A = Q * R)
__kernel void kernel_batched_qr_8x8(
    __global const float* in_mats, __global float* out_Q, __global float* out_R, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 64;
    float A[8][8]; float Q[8][8];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) {
        A[i][j] = in_mats[base + i * 8 + j];
        Q[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    for (int k = 0; k < 7; ++k) {
        float norm_x = 0.0f;
        for (int i = k; i < 8; ++i) norm_x += A[i][k] * A[i][k];
        norm_x = sqrt(norm_x);
        if (norm_x > 1.0e-7f) {
            float s = (A[k][k] >= 0.0f) ? 1.0f : -1.0f;
            float u1 = A[k][k] + s * norm_x;
            float v[8];
            v[k] = 1.0f;
            for (int i = k + 1; i < 8; ++i) v[i] = A[i][k] / u1;
            float tau = 2.0f / (1.0f + (norm_x * norm_x - A[k][k] * A[k][k]) / (u1 * u1));
            // Apply Householder reflector H = I - tau * v * v^T to A and Q
            for (int j = k; j < 8; ++j) {
                float dot_val = 0.0f;
                for (int i = k; i < 8; ++i) dot_val += v[i] * A[i][j];
                for (int i = k; i < 8; ++i) A[i][j] -= tau * dot_val * v[i];
            }
            for (int j = 0; j < 8; ++j) {
                float dot_val = 0.0f;
                for (int i = k; i < 8; ++i) dot_val += v[i] * Q[j][i];
                for (int i = k; i < 8; ++i) Q[j][i] -= tau * dot_val * v[i];
            }
        }
    }
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) {
        out_Q[base + i * 8 + j] = Q[i][j];
        out_R[base + i * 8 + j] = (i <= j) ? A[i][j] : 0.0f;
    }
}

// Batched 10x10 Householder QR Factorization (A = Q * R)
__kernel void kernel_batched_qr_10x10(
    __global const float* in_mats, __global float* out_Q, __global float* out_R, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 100;
    float A[10][10]; float Q[10][10];
    for (int i = 0; i < 10; ++i) for (int j = 0; j < 10; ++j) {
        A[i][j] = in_mats[base + i * 10 + j];
        Q[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    for (int k = 0; k < 9; ++k) {
        float norm_x = 0.0f;
        for (int i = k; i < 10; ++i) norm_x += A[i][k] * A[i][k];
        norm_x = sqrt(norm_x);
        if (norm_x > 1.0e-7f) {
            float s = (A[k][k] >= 0.0f) ? 1.0f : -1.0f;
            float u1 = A[k][k] + s * norm_x;
            float v[10];
            v[k] = 1.0f;
            for (int i = k + 1; i < 10; ++i) v[i] = A[i][k] / u1;
            float tau = 2.0f / (1.0f + (norm_x * norm_x - A[k][k] * A[k][k]) / (u1 * u1));
            // Apply Householder reflector H = I - tau * v * v^T to A and Q
            for (int j = k; j < 10; ++j) {
                float dot_val = 0.0f;
                for (int i = k; i < 10; ++i) dot_val += v[i] * A[i][j];
                for (int i = k; i < 10; ++i) A[i][j] -= tau * dot_val * v[i];
            }
            for (int j = 0; j < 10; ++j) {
                float dot_val = 0.0f;
                for (int i = k; i < 10; ++i) dot_val += v[i] * Q[j][i];
                for (int i = k; i < 10; ++i) Q[j][i] -= tau * dot_val * v[i];
            }
        }
    }
    for (int i = 0; i < 10; ++i) for (int j = 0; j < 10; ++j) {
        out_Q[base + i * 10 + j] = Q[i][j];
        out_R[base + i * 10 + j] = (i <= j) ? A[i][j] : 0.0f;
    }
}

// Batched 12x12 Householder QR Factorization (A = Q * R)
__kernel void kernel_batched_qr_12x12(
    __global const float* in_mats, __global float* out_Q, __global float* out_R, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 144;
    float A[12][12]; float Q[12][12];
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) {
        A[i][j] = in_mats[base + i * 12 + j];
        Q[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    for (int k = 0; k < 11; ++k) {
        float norm_x = 0.0f;
        for (int i = k; i < 12; ++i) norm_x += A[i][k] * A[i][k];
        norm_x = sqrt(norm_x);
        if (norm_x > 1.0e-7f) {
            float s = (A[k][k] >= 0.0f) ? 1.0f : -1.0f;
            float u1 = A[k][k] + s * norm_x;
            float v[12];
            v[k] = 1.0f;
            for (int i = k + 1; i < 12; ++i) v[i] = A[i][k] / u1;
            float tau = 2.0f / (1.0f + (norm_x * norm_x - A[k][k] * A[k][k]) / (u1 * u1));
            // Apply Householder reflector H = I - tau * v * v^T to A and Q
            for (int j = k; j < 12; ++j) {
                float dot_val = 0.0f;
                for (int i = k; i < 12; ++i) dot_val += v[i] * A[i][j];
                for (int i = k; i < 12; ++i) A[i][j] -= tau * dot_val * v[i];
            }
            for (int j = 0; j < 12; ++j) {
                float dot_val = 0.0f;
                for (int i = k; i < 12; ++i) dot_val += v[i] * Q[j][i];
                for (int i = k; i < 12; ++i) Q[j][i] -= tau * dot_val * v[i];
            }
        }
    }
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) {
        out_Q[base + i * 12 + j] = Q[i][j];
        out_R[base + i * 12 + j] = (i <= j) ? A[i][j] : 0.0f;
    }
}

// Batched 3x3 One-Sided Jacobi SVD (A = U * S * V^T)
__kernel void kernel_batched_svd_3x3(
    __global const float* in_mats, __global float* out_U, __global float* out_S, __global float* out_V, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 9;
    float U[3][3]; float V[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        U[i][j] = in_mats[base + i * 3 + j];
        V[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    // Jacobi sweeps
    for (int sweep = 0; sweep < 8; ++sweep) {
        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                float alpha = 0.0f; float beta = 0.0f; float gamma = 0.0f;
                for (int i = 0; i < 3; ++i) {
                    alpha += U[i][p] * U[i][p];
                    beta  += U[i][q] * U[i][q];
                    gamma += U[i][p] * U[i][q];
                }
                if (fabs(gamma) > 1.0e-7f) {
                    float zeta = (beta - alpha) / (2.0f * gamma);
                    float t = ((zeta >= 0.0f) ? 1.0f : -1.0f) / (fabs(zeta) + sqrt(1.0f + zeta * zeta));
                    float c = 1.0f / sqrt(1.0f + t * t); float s = t * c;
                    for (int i = 0; i < 3; ++i) {
                        float u_p = U[i][p]; float u_q = U[i][q];
                        U[i][p] = c * u_p - s * u_q; U[i][q] = s * u_p + c * u_q;
                        float v_p = V[i][p]; float v_q = V[i][q];
                        V[i][p] = c * v_p - s * v_q; V[i][q] = s * v_p + c * v_q;
                    }
                }
            }
        }
    }
    for (int j = 0; j < 3; ++j) {
        float sing_val = 0.0f;
        for (int i = 0; i < 3; ++i) sing_val += U[i][j] * U[i][j];
        sing_val = sqrt(sing_val);
        out_S[bid * 3 + j] = sing_val;
        float inv_s = (sing_val > 1.0e-7f) ? (1.0f / sing_val) : 0.0f;
        for (int i = 0; i < 3; ++i) {
            out_U[base + i * 3 + j] = U[i][j] * inv_s;
            out_V[base + i * 3 + j] = V[i][j];
        }
    }
}

// Batched 4x4 One-Sided Jacobi SVD (A = U * S * V^T)
__kernel void kernel_batched_svd_4x4(
    __global const float* in_mats, __global float* out_U, __global float* out_S, __global float* out_V, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 16;
    float U[4][4]; float V[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
        U[i][j] = in_mats[base + i * 4 + j];
        V[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    // Jacobi sweeps
    for (int sweep = 0; sweep < 8; ++sweep) {
        for (int p = 0; p < 3; ++p) {
            for (int q = p + 1; q < 4; ++q) {
                float alpha = 0.0f; float beta = 0.0f; float gamma = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    alpha += U[i][p] * U[i][p];
                    beta  += U[i][q] * U[i][q];
                    gamma += U[i][p] * U[i][q];
                }
                if (fabs(gamma) > 1.0e-7f) {
                    float zeta = (beta - alpha) / (2.0f * gamma);
                    float t = ((zeta >= 0.0f) ? 1.0f : -1.0f) / (fabs(zeta) + sqrt(1.0f + zeta * zeta));
                    float c = 1.0f / sqrt(1.0f + t * t); float s = t * c;
                    for (int i = 0; i < 4; ++i) {
                        float u_p = U[i][p]; float u_q = U[i][q];
                        U[i][p] = c * u_p - s * u_q; U[i][q] = s * u_p + c * u_q;
                        float v_p = V[i][p]; float v_q = V[i][q];
                        V[i][p] = c * v_p - s * v_q; V[i][q] = s * v_p + c * v_q;
                    }
                }
            }
        }
    }
    for (int j = 0; j < 4; ++j) {
        float sing_val = 0.0f;
        for (int i = 0; i < 4; ++i) sing_val += U[i][j] * U[i][j];
        sing_val = sqrt(sing_val);
        out_S[bid * 4 + j] = sing_val;
        float inv_s = (sing_val > 1.0e-7f) ? (1.0f / sing_val) : 0.0f;
        for (int i = 0; i < 4; ++i) {
            out_U[base + i * 4 + j] = U[i][j] * inv_s;
            out_V[base + i * 4 + j] = V[i][j];
        }
    }
}

// Batched 6x6 One-Sided Jacobi SVD (A = U * S * V^T)
__kernel void kernel_batched_svd_6x6(
    __global const float* in_mats, __global float* out_U, __global float* out_S, __global float* out_V, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 36;
    float U[6][6]; float V[6][6];
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) {
        U[i][j] = in_mats[base + i * 6 + j];
        V[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    // Jacobi sweeps
    for (int sweep = 0; sweep < 8; ++sweep) {
        for (int p = 0; p < 5; ++p) {
            for (int q = p + 1; q < 6; ++q) {
                float alpha = 0.0f; float beta = 0.0f; float gamma = 0.0f;
                for (int i = 0; i < 6; ++i) {
                    alpha += U[i][p] * U[i][p];
                    beta  += U[i][q] * U[i][q];
                    gamma += U[i][p] * U[i][q];
                }
                if (fabs(gamma) > 1.0e-7f) {
                    float zeta = (beta - alpha) / (2.0f * gamma);
                    float t = ((zeta >= 0.0f) ? 1.0f : -1.0f) / (fabs(zeta) + sqrt(1.0f + zeta * zeta));
                    float c = 1.0f / sqrt(1.0f + t * t); float s = t * c;
                    for (int i = 0; i < 6; ++i) {
                        float u_p = U[i][p]; float u_q = U[i][q];
                        U[i][p] = c * u_p - s * u_q; U[i][q] = s * u_p + c * u_q;
                        float v_p = V[i][p]; float v_q = V[i][q];
                        V[i][p] = c * v_p - s * v_q; V[i][q] = s * v_p + c * v_q;
                    }
                }
            }
        }
    }
    for (int j = 0; j < 6; ++j) {
        float sing_val = 0.0f;
        for (int i = 0; i < 6; ++i) sing_val += U[i][j] * U[i][j];
        sing_val = sqrt(sing_val);
        out_S[bid * 6 + j] = sing_val;
        float inv_s = (sing_val > 1.0e-7f) ? (1.0f / sing_val) : 0.0f;
        for (int i = 0; i < 6; ++i) {
            out_U[base + i * 6 + j] = U[i][j] * inv_s;
            out_V[base + i * 6 + j] = V[i][j];
        }
    }
}

// Batched 8x8 One-Sided Jacobi SVD (A = U * S * V^T)
__kernel void kernel_batched_svd_8x8(
    __global const float* in_mats, __global float* out_U, __global float* out_S, __global float* out_V, const uint batch_count) {
    uint bid = get_global_id(0); if (bid >= batch_count) return;
    size_t base = (size_t)bid * 64;
    float U[8][8]; float V[8][8];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) {
        U[i][j] = in_mats[base + i * 8 + j];
        V[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    // Jacobi sweeps
    for (int sweep = 0; sweep < 8; ++sweep) {
        for (int p = 0; p < 7; ++p) {
            for (int q = p + 1; q < 8; ++q) {
                float alpha = 0.0f; float beta = 0.0f; float gamma = 0.0f;
                for (int i = 0; i < 8; ++i) {
                    alpha += U[i][p] * U[i][p];
                    beta  += U[i][q] * U[i][q];
                    gamma += U[i][p] * U[i][q];
                }
                if (fabs(gamma) > 1.0e-7f) {
                    float zeta = (beta - alpha) / (2.0f * gamma);
                    float t = ((zeta >= 0.0f) ? 1.0f : -1.0f) / (fabs(zeta) + sqrt(1.0f + zeta * zeta));
                    float c = 1.0f / sqrt(1.0f + t * t); float s = t * c;
                    for (int i = 0; i < 8; ++i) {
                        float u_p = U[i][p]; float u_q = U[i][q];
                        U[i][p] = c * u_p - s * u_q; U[i][q] = s * u_p + c * u_q;
                        float v_p = V[i][p]; float v_q = V[i][q];
                        V[i][p] = c * v_p - s * v_q; V[i][q] = s * v_p + c * v_q;
                    }
                }
            }
        }
    }
    for (int j = 0; j < 8; ++j) {
        float sing_val = 0.0f;
        for (int i = 0; i < 8; ++i) sing_val += U[i][j] * U[i][j];
        sing_val = sqrt(sing_val);
        out_S[bid * 8 + j] = sing_val;
        float inv_s = (sing_val > 1.0e-7f) ? (1.0f / sing_val) : 0.0f;
        for (int i = 0; i < 8; ++i) {
            out_U[base + i * 8 + j] = U[i][j] * inv_s;
            out_V[base + i * 8 + j] = V[i][j];
        }
    }
}
