#include "common/math.clh"
#include "common/memory.clh"

#define RANSAC_WG_SIZE 256

// Solves 4-point planar projective homography H (3x3) using Direct Linear Transform (DLT)
inline bool solve_homography_4point(
    const float2 src[4],
    const float2 dst[4],
    float H[9]) {
    
    // Gauss-Jordan elimination on 8x8 linear system for H (with H[8] = 1.0)
    float A[8][9];

    for (int i = 0; i < 4; ++i) {
        float sx = src[i].x;
        float sy = src[i].y;
        float dx = dst[i].x;
        float dy = dst[i].y;

        int r0 = 2 * i;
        int r1 = 2 * i + 1;

        A[r0][0] = -sx;    A[r0][1] = -sy;    A[r0][2] = -1.0f;
        A[r0][3] = 0.0f;   A[r0][4] = 0.0f;   A[r0][5] = 0.0f;
        A[r0][6] = dx*sx;  A[r0][7] = dx*sy;  A[r0][8] = dx;

        A[r1][0] = 0.0f;   A[r1][1] = 0.0f;   A[r1][2] = 0.0f;
        A[r1][3] = -sx;    A[r1][4] = -sy;    A[r1][5] = -1.0f;
        A[r1][6] = dy*sx;  A[r1][7] = dy*sy;  A[r1][8] = dy;
    }

    // Solve 8x8 system
    for (int i = 0; i < 8; ++i) {
        // Pivot
        int max_r = i;
        float max_v = fabs(A[i][i]);
        for (int r = i + 1; r < 8; ++r) {
            if (fabs(A[r][i]) > max_v) {
                max_v = fabs(A[r][i]);
                max_r = r;
            }
        }
        if (max_v < 1.0e-7f) return false;

        // Swap rows
        if (max_r != i) {
            for (int c = i; c <= 8; ++c) {
                float tmp = A[i][c];
                A[i][c] = A[max_r][c];
                A[max_r][c] = tmp;
            }
        }

        // Eliminate
        float diag = A[i][i];
        for (int c = i; c <= 8; ++c) {
            A[i][c] /= diag;
        }

        for (int r = 0; r < 8; ++r) {
            if (r != i) {
                float factor = A[r][i];
                for (int c = i; c <= 8; ++c) {
                    A[r][c] -= factor * A[i][c];
                }
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        H[i] = -A[i][8];
    }
    H[8] = 1.0f;
    return true;
}

// Parallel RANSAC Homography Evaluation Kernel
// Each work-group evaluates one randomized 4-point hypothesis against all matches.
__kernel void kernel_ransac_homography_hypotheses(
    __global const float2* src_pts,
    __global const float2* dst_pts,
    __global const uint4* sample_indices, // 4 point indices per hypothesis
    __global float* out_hypotheses_H,     // 9 floats per hypothesis
    __global uint* out_inlier_counts,     // 1 uint per hypothesis
    const uint total_matches,
    const float inlier_threshold_sq) {
    
    __local float l_H[9];
    __local uint l_inliers[RANSAC_WG_SIZE];
    __local bool l_valid;

    uint hyp_id = get_group_id(0);
    uint tid = get_local_id(0);

    if (tid == 0) {
        uint4 idx = sample_indices[hyp_id];
        float2 s[4] = {src_pts[idx.x], src_pts[idx.y], src_pts[idx.z], src_pts[idx.w]};
        float2 d[4] = {dst_pts[idx.x], dst_pts[idx.y], dst_pts[idx.z], dst_pts[idx.w]};

        float H[9];
        l_valid = solve_homography_4point(s, d, H);
        if (l_valid) {
            for (int i = 0; i < 9; ++i) {
                l_H[i] = H[i];
                out_hypotheses_H[hyp_id * 9 + i] = H[i];
            }
        }
    }
    sync_local();

    if (!l_valid) {
        if (tid == 0) out_inlier_counts[hyp_id] = 0;
        return;
    }

    // Count inliers across all matched pairs
    uint local_count = 0;
    for (uint i = tid; i < total_matches; i += RANSAC_WG_SIZE) {
        float2 sp = src_pts[i];
        float2 dp = dst_pts[i];

        // Forward map: H * sp
        float hx = l_H[0] * sp.x + l_H[1] * sp.y + l_H[2];
        float hy = l_H[3] * sp.x + l_H[4] * sp.y + l_H[5];
        float hw = l_H[6] * sp.x + l_H[7] * sp.y + l_H[8];

        if (fabs(hw) > 1.0e-7f) {
            float inv_w = 1.0f / hw;
            float pred_x = hx * inv_w;
            float pred_y = hy * inv_w;

            float err_sq = (pred_x - dp.x) * (pred_x - dp.x) + (pred_y - dp.y) * (pred_y - dp.y);
            if (err_sq < inlier_threshold_sq) {
                local_count++;
            }
        }
    }

    l_inliers[tid] = local_count;
    sync_local();

    // Reduce local inlier count
    for (uint s = RANSAC_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            l_inliers[tid] += l_inliers[tid + s];
        }
        sync_local();
    }

    if (tid == 0) {
        out_inlier_counts[hyp_id] = l_inliers[0];
    }
}
