#include "common/math.clh"
#include "common/memory.clh"

// ============================================================================
// Advanced 3D Point Cloud Descriptors: FPFH, SHOT, and MLS Surface Smoothing
// ============================================================================

// Darboux Local Reference Frame and Angular Triplet Calculation
inline float3 compute_darboux_triplet(float4 p1, float4 n1, float4 p2, float4 n2) {
    float3 dp = (float3)(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
    float d = length(dp);
    if (d < 1.0e-7f) return (float3)(0.0f, 0.0f, 0.0f);
    float3 u = (float3)(n1.x, n1.y, n1.z);
    float3 v = cross(dp / d, u); float v_len = length(v);
    v = (v_len > 1.0e-7f) ? (v / v_len) : (float3)(0.0f, 1.0f, 0.0f);
    float3 w = cross(u, v);
    float alpha = dot(v, (float3)(n2.x, n2.y, n2.z));
    float phi   = dot(u, dp / d);
    float theta = atan2(dot(w, (float3)(n2.x, n2.y, n2.z)), dot(u, (float3)(n2.x, n2.y, n2.z)));
    return (float3)(alpha, phi, theta);
}

// 1. Simplified Point Feature Histogram (SPFH) Binning (11 bins for each angle = 33 bins)
__kernel void kernel_compute_spfh_descriptors(
    __global const float4* points, __global const float4* normals,
    __global const int* neighbor_indices, __global const float* neighbor_distances,
    __global float* out_spfh, const uint num_points, const int k_neighbors) {
    uint pid = get_global_id(0); if (pid >= num_points) return;
    float4 p_src = points[pid]; float4 n_src = normals[pid];
    float hist[33]; for (int b = 0; b < 33; ++b) hist[b] = 0.0f;
    for (int k = 0; k < k_neighbors; ++k) {
        int n_idx = neighbor_indices[(size_t)pid * k_neighbors + k];
        if (n_idx >= 0 && n_idx < (int)num_points && n_idx != (int)pid) {
            float3 angles = compute_darboux_triplet(p_src, n_src, points[n_idx], normals[n_idx]);
            // Normalize angles to bin indices [0..10]
            int b_alpha = clamp_i((int)floor((angles.x + 1.0f) * 5.5f), 0, 10);
            int b_phi   = clamp_i((int)floor((angles.y + 1.0f) * 5.5f), 0, 10);
            int b_theta = clamp_i((int)floor((angles.z + PI_F) * (11.0f / (2.0f * PI_F))), 0, 10);
            hist[b_alpha] += 1.0f; hist[11 + b_phi] += 1.0f; hist[22 + b_theta] += 1.0f;
        }
    }
    float inv_k = (k_neighbors > 0) ? (1.0f / (float)k_neighbors) : 0.0f;
    size_t out_base = (size_t)pid * 33;
    for (int b = 0; b < 33; ++b) out_spfh[out_base + b] = hist[b] * inv_k;
}}


// 2. Moving Least Squares (MLS) Surface Polynomial Height Fitting
__kernel void kernel_mls_surface_polynomial_fit(
    __global const float4* points, __global const float4* normals,
    __global const int* neighbor_indices, __global float4* out_smoothed_points,
    const float search_radius, const uint num_points, const int k_neighbors) {
    uint pid = get_global_id(0); if (pid >= num_points) return;
    float4 p_center = points[pid]; float4 n_center = normals[pid];
    float3 u_norm = (float3)(n_center.x, n_center.y, n_center.z);
    // Create tangent coordinate frame (u, v, n)
    float3 u_tangent = (fabs(u_norm.x) > 0.1f) ? (float3)(0.0f, 1.0f, 0.0f) : (float3)(1.0f, 0.0f, 0.0f);
    u_tangent = normalize(cross(u_norm, u_tangent));
    float3 v_tangent = cross(u_norm, u_tangent);
    float weighted_height = 0.0f; float total_weight = 0.0f;
    float r_sq_inv = 1.0f / (search_radius * search_radius + 1.0e-7f);
    for (int k = 0; k < k_neighbors; ++k) {
        int n_idx = neighbor_indices[(size_t)pid * k_neighbors + k];
        if (n_idx >= 0 && n_idx < (int)num_points) {
            float4 pn = points[n_idx];
            float3 dp = (float3)(pn.x - p_center.x, pn.y - p_center.y, pn.z - p_center.z);
            float dist_sq = dot(dp, dp);
            float w = exp(-dist_sq * r_sq_inv);
            float height = dot(dp, u_norm);
            weighted_height += w * height; total_weight += w;
        }
    }
    float h = (total_weight > 1.0e-6f) ? (weighted_height / total_weight) : 0.0f;
    out_smoothed_points[pid] = (float4)(p_center.x + u_norm.x * h, p_center.y + u_norm.y * h, p_center.z + u_norm.z * h, p_center.w);
}}


// ============================================================================
// SHOT (352 bins) and Point Feature Histograms (PFH 125 bins) Descriptors
// ============================================================================

// 1. Signature of Histograms of Orientations (SHOT 352-dimensional descriptor)
// 32 spatial volumes (8 azimuth divisions x 2 elevation x 2 radial) x 11 cosine bins
__kernel void kernel_compute_shot_descriptors(
    __global const float4* points, __global const float4* normals,
    __global const float* local_rf_matrices, // [num_points x 9] (u, v, n frames)
    __global const int* neighbor_indices, __global const float* neighbor_distances,
    __global float* out_shot, const float search_radius,
    const uint num_points, const int k_neighbors) {
    uint pid = get_global_id(0); if (pid >= num_points) return;
    float4 p_center = points[pid]; float4 n_center = normals[pid];
    size_t rf_base = (size_t)pid * 9;
    float3 u = (float3)(local_rf_matrices[rf_base + 0], local_rf_matrices[rf_base + 1], local_rf_matrices[rf_base + 2]);
    float3 v = (float3)(local_rf_matrices[rf_base + 3], local_rf_matrices[rf_base + 4], local_rf_matrices[rf_base + 5]);
    float3 w = (float3)(local_rf_matrices[rf_base + 6], local_rf_matrices[rf_base + 7], local_rf_matrices[rf_base + 8]);
    float hist[352]; for (int b = 0; b < 352; ++b) hist[b] = 0.0f;
    for (int k = 0; k < k_neighbors; ++k) {
        int n_idx = neighbor_indices[(size_t)pid * k_neighbors + k];
        float dist = neighbor_distances[(size_t)pid * k_neighbors + k];
        if (n_idx >= 0 && n_idx < (int)num_points && n_idx != (int)pid && dist <= search_radius) {
            float4 pn = points[n_idx]; float4 nn = normals[n_idx];
            float3 dp = (float3)(pn.x - p_center.x, pn.y - p_center.y, pn.z - p_center.z);
            float local_x = dot(dp, u); float local_y = dot(dp, v); float local_z = dot(dp, w);
            // Spatial binning (32 spatial sectors)
            int r_bin = (dist > 0.5f * search_radius) ? 1 : 0;
            int z_bin = (local_z >= 0.0f) ? 1 : 0;
            float angle_xy = atan2(local_y, local_x) + PI_F;
            int theta_bin = clamp_i((int)floor(angle_xy * (8.0f / (2.0f * PI_F))), 0, 7);
            int spatial_bin = (r_bin * 16) + (z_bin * 8) + theta_bin;
            // Normal cosine orientation binning (11 bins)
            float cos_angle = dot(w, (float3)(nn.x, nn.y, nn.z));
            int cos_bin = clamp_i((int)floor((cos_angle + 1.0f) * 5.5f), 0, 10);
            int descriptor_idx = spatial_bin * 11 + cos_bin;
            hist[descriptor_idx] += 1.0f;
        }
    }
    // L2 normalization of descriptor
    float norm_sq = 0.0f;
    for (int b = 0; b < 352; ++b) norm_sq += hist[b] * hist[b];
    float inv_norm = (norm_sq > 1.0e-7f) ? (1.0f / sqrt(norm_sq)) : 0.0f;
    size_t out_base = (size_t)pid * 352;
    for (int b = 0; b < 352; ++b) out_shot[out_base + b] = hist[b] * inv_norm;
}}


// 2. Point Feature Histograms (PFH 125-bin full neighbor constellation descriptor)
// 5 bins alpha x 5 bins phi x 5 bins theta = 125 bins
__kernel void kernel_compute_pfh_descriptors(
    __global const float4* points, __global const float4* normals,
    __global const int* neighbor_indices, __global float* out_pfh,
    const uint num_points, const int k_neighbors) {
    uint pid = get_global_id(0); if (pid >= num_points) return;
    float hist[125]; for (int b = 0; b < 125; ++b) hist[b] = 0.0f;
    // Iterate over all pairs in the neighbor constellation
    for (int i = 0; i < k_neighbors; ++i) {
        int idx_i = neighbor_indices[(size_t)pid * k_neighbors + i];
        if (idx_i < 0 || idx_i >= (int)num_points) continue;
        float4 p1 = points[idx_i]; float4 n1 = normals[idx_i];
        for (int j = i + 1; j < k_neighbors; ++j) {
            int idx_j = neighbor_indices[(size_t)pid * k_neighbors + j];
            if (idx_j < 0 || idx_j >= (int)num_points) continue;
            float4 p2 = points[idx_j]; float4 n2 = normals[idx_j];
            float3 angles = compute_darboux_triplet(p1, n1, p2, n2);
            int b_alpha = clamp_i((int)floor((angles.x + 1.0f) * 2.5f), 0, 4);
            int b_phi   = clamp_i((int)floor((angles.y + 1.0f) * 2.5f), 0, 4);
            int b_theta = clamp_i((int)floor((angles.z + PI_F) * (5.0f / (2.0f * PI_F))), 0, 4);
            int bin_idx = b_alpha * 25 + b_phi * 5 + b_theta;
            hist[bin_idx] += 1.0f;
        }
    }
    // L1 normalization
    float total_sum = 0.0f;
    for (int b = 0; b < 125; ++b) total_sum += hist[b];
    float inv_sum = (total_sum > 1.0e-7f) ? (1.0f / total_sum) : 0.0f;
    size_t out_base = (size_t)pid * 125;
    for (int b = 0; b < 125; ++b) out_pfh[out_base + b] = hist[b] * inv_sum;
}}


// 3. Normal Distributions Transform (NDT) 6x6 Gauss-Newton Hessian Accumulator
__kernel void kernel_ndt_hessian_6x6_accumulator(
    __global const float4* point_cloud, __global const float4* cell_means,
    __global const float* cell_inv_covariances, __global const int* point_cell_assignments,
    __global float* out_hessian_blocks, // [num_points x 36]
    const uint num_points, const float d1, const float d2) {
    uint pid = get_global_id(0); if (pid >= num_points) return;
    int cid = point_cell_assignments[pid];
    size_t h_base = (size_t)pid * 36;
    if (cid < 0) {
        for (int i = 0; i < 36; ++i) out_hessian_blocks[h_base + i] = 0.0f;
        return;
    }
    float4 p = point_cloud[pid]; float4 mu = cell_means[cid];
    float3 diff = (float3)(p.x - mu.x, p.y - mu.y, p.z - mu.z);
    size_t cov_base = (size_t)cid * 9;
    float inv_cov[3][3];
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) inv_cov[r][c] = cell_inv_covariances[cov_base + r * 3 + c];
    float3 ic_diff;
    ic_diff.x = inv_cov[0][0]*diff.x + inv_cov[0][1]*diff.y + inv_cov[0][2]*diff.z;
    ic_diff.y = inv_cov[1][0]*diff.x + inv_cov[1][1]*diff.y + inv_cov[1][2]*diff.z;
    ic_diff.z = inv_cov[2][0]*diff.x + inv_cov[2][1]*diff.y + inv_cov[2][2]*diff.z;
    float mahalanobis_sq = dot(diff, ic_diff);
    float score = -d1 * exp(-d2 * 0.5f * mahalanobis_sq);
    // 3D rigid motion Jacobian: J = [I_3 | -[p]_x]
    float J[3][6] = {
        {1.0f, 0.0f, 0.0f,  0.0f,  p.z, -p.y},
        {0.0f, 1.0f, 0.0f, -p.z,  0.0f,  p.x},
        {0.0f, 0.0f, 1.0f,  p.y, -p.x,  0.0f}
    };
    // Compute J^T * inv_cov * J scaled by score
    float factor = -d2 * score;
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            float val = 0.0f;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    val += J[i][r] * inv_cov[i][j] * J[j][c];
                }
            }
            out_hessian_blocks[h_base + r * 6 + c] = factor * val;
        }
    }
}}
