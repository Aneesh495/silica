#include "common/math.clh"
#include "common/memory.clh"

#define CSR_WARP_SIZE 32

// Scalar Sparse Matrix-Vector Multiplication (SpMV) in Compressed Sparse Row (CSR) Format
// Each work-item computes 1 output row y[row] = sum(A[row, c] * x[c])
__kernel void kernel_spmv_csr_scalar(
    __global const int* row_ptr,      // [num_rows + 1]
    __global const int* col_indices,  // [nnz]
    __global const float* values,     // [nnz]
    __global const float* x_vec,      // [num_cols]
    __global float* y_out,            // [num_rows]
    const uint num_rows) {
    
    uint row = get_global_id(0);
    if (row >= num_rows) return;

    int start = row_ptr[row];
    int end   = row_ptr[row + 1];

    float sum = 0.0f;
    for (int k = start; k < end; ++k) {
        int col = col_indices[k];
        float val = values[k];
        sum += val * x_vec[col];
    }

    y_out[row] = sum;
}

// Vectorized Warp-level SpMV for High-Degree Dense Rows
// 32 threads cooperatively process each row to maximize memory bandwidth and avoid divergence.
__kernel void kernel_spmv_csr_vector(
    __global const int* row_ptr,
    __global const int* col_indices,
    __global const float* values,
    __global const float* x_vec,
    __global float* y_out,
    const uint num_rows) {
    
    __local float l_sums[256];

    uint tid = get_local_id(0);
    uint warp_id = tid / CSR_WARP_SIZE;
    uint lane_id = tid % CSR_WARP_SIZE;
    uint global_warp_id = (get_group_id(0) * (get_local_size(0) / CSR_WARP_SIZE)) + warp_id;

    if (global_warp_id >= num_rows) return;

    int row_start = row_ptr[global_warp_id];
    int row_end   = row_ptr[global_warp_id + 1];

    float sum = 0.0f;
    for (int k = row_start + lane_id; k < row_end; k += CSR_WARP_SIZE) {
        sum += values[k] * x_vec[col_indices[k]];
    }

    // Warp shuffle / tree reduction in local memory
    l_sums[tid] = sum;
    sync_local();

    #pragma unroll
    for (int offset = CSR_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        if (lane_id < offset) {
            l_sums[tid] += l_sums[tid + offset];
        }
        sync_local();
    }

    if (lane_id == 0) {
        y_out[global_warp_id] = l_sums[tid];
    }
}
