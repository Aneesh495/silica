#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

#define EPIPOLAR_WG_SIZE 256

// Evaluates Sampson Epipolar Distance for Matched Correspondences
inline float compute_sampson_error(
    float2 p1,
    float2 p2,
    const float E[9]) {
    
    // Homogeneous coordinates x1 = [p1.x, p1.y, 1], x2 = [p2.x, p2.y, 1]
    // Ex = E * x1
    float Ex_0 = E[0] * p1.x + E[1] * p1.y + E[2];
    float Ex_1 = E[3] * p1.x + E[4] * p1.y + E[5];
    float Ex_2 = E[6] * p1.x + E[7] * p1.y + E[8];

    // Et_x2 = E^T * x2
    float Et_x2_0 = E[0] * p2.x + E[3] * p2.y + E[6];
    float Et_x2_1 = E[1] * p2.x + E[4] * p2.y + E[7];

    // Epipolar constraint: x2^T * E * x1
    float epipolar_residual = p2.x * Ex_0 + p2.y * Ex_1 + Ex_2;

    float denom = Ex_0 * Ex_0 + Ex_1 * Ex_1 + Et_x2_0 * Et_x2_0 + Et_x2_1 * Et_x2_1;
    return (denom > 1.0e-7f) ? ((epipolar_residual * epipolar_residual) / denom) : LARGE_FLOAT_F;
}

// Parallel Epipolar Verification Kernel
__kernel void kernel_verify_essential_matrix_hypotheses(
    __global const float2* pts_view1,
    __global const float2* pts_view2,
    __global const float* candidate_E_matrices, // [num_hyp x 9 floats]
    __global uint* out_inlier_counts,
    const uint total_matches,
    const float sampson_threshold_sq) {
    
    __local float l_E[9];
    __local uint l_inliers[EPIPOLAR_WG_SIZE];

    uint hyp_id = get_group_id(0);
    uint tid = get_local_id(0);

    if (tid < 9) {
        l_E[tid] = candidate_E_matrices[hyp_id * 9 + tid];
    }
    sync_local();

    float E[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) E[i] = l_E[i];

    uint local_count = 0;
    for (uint i = tid; i < total_matches; i += EPIPOLAR_WG_SIZE) {
        float2 p1 = pts_view1[i];
        float2 p2 = pts_view2[i];

        float err_sq = compute_sampson_error(p1, p2, E);
        if (err_sq <= sampson_threshold_sq) {
            local_count++;
        }
    }

    l_inliers[tid] = local_count;
    sync_local();

    for (uint s = EPIPOLAR_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            l_inliers[tid] += l_inliers[tid + s];
        }
        sync_local();
    }

    if (tid == 0) {
        out_inlier_counts[hyp_id] = l_inliers[0];
    }
}
