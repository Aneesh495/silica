#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

#define ICP_WG_SIZE 256

// Point-to-Plane ICP: Accumulates 6x6 Matrix (J^T * J) and 6x1 Vector (J^T * r)
// Parameter vector x = [alpha, beta, gamma, tx, ty, tz]^T (rotation angles and translations)
__kernel void kernel_icp_point_to_plane_accumulate(
    __global const float4* src_points,
    __global const float4* dst_points,
    __global const float4* dst_normals,
    __global float* out_block_JTJ, // [num_groups x 36]
    __global float* out_block_JTr, // [num_groups x 6]
    const uint num_correspondences,
    const float max_corr_dist_sq) {
    
    // Thread-local accumulation of upper-triangular JTJ (21 elements) and JTr (6 elements)
    float l_JTJ[21];
    float l_JTr[6];

    #pragma unroll
    for (int i = 0; i < 21; ++i) l_JTJ[i] = 0.0f;
    #pragma unroll
    for (int i = 0; i < 6; ++i)  l_JTr[i] = 0.0f;

    uint tid = get_local_id(0);
    uint gid = get_global_id(0);
    uint stride = get_global_size(0);

    for (uint i = gid; i < num_correspondences; i += stride) {
        float4 s4 = src_points[i];
        float4 d4 = dst_points[i];
        float4 n4 = dst_normals[i];

        float3 s = (float3)(s4.x, s4.y, s4.z);
        float3 d = (float3)(d4.x, d4.y, d4.z);
        float3 n = (float3)(n4.x, n4.y, n4.z);

        float dist_sq = dot(s - d, s - d);
        if (dist_sq > max_corr_dist_sq) continue;

        // Jacobian row: J = [cross(s, n), n] (6 elements)
        float3 s_cross_n = cross(s, n);
        float J[6] = {
            s_cross_n.x, s_cross_n.y, s_cross_n.z,
            n.x, n.y, n.z
        };

        // Residual r = -dot(s - d, n)
        float residual = -dot(s - d, n);

        // Huber robust weight
        float k_huber = 0.05f;
        float abs_r = fabs(residual);
        float weight = (abs_r <= k_huber) ? 1.0f : (k_huber / abs_r);

        // Accumulate upper-triangular JTJ: J_j * J_k
        int idx = 0;
        for (int j = 0; j < 6; ++j) {
            for (int k = j; k < 6; ++k) {
                l_JTJ[idx++] += weight * J[j] * J[k];
            }
            l_JTr[j] += weight * J[j] * residual;
        }
    }

    // Work-group local reduction across threads
    __local float s_JTJ[21][ICP_WG_SIZE];
    __local float s_JTr[6][ICP_WG_SIZE];

    for (int i = 0; i < 21; ++i) s_JTJ[i][tid] = l_JTJ[i];
    for (int i = 0; i < 6; ++i)  s_JTr[i][tid] = l_JTr[i];
    sync_local();

    for (uint s = ICP_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int i = 0; i < 21; ++i) s_JTJ[i][tid] += s_JTJ[i][tid + s];
            for (int i = 0; i < 6; ++i)  s_JTr[i][tid] += s_JTr[i][tid + s];
        }
        sync_local();
    }

    if (tid == 0) {
        uint group_idx = get_group_id(0);

        // Unpack upper-triangular to full 6x6
        int p = 0;
        for (int j = 0; j < 6; ++j) {
            for (int k = j; k < 6; ++k) {
                float val = s_JTJ[p++][0];
                out_block_JTJ[group_idx * 36 + j * 6 + k] = val;
                out_block_JTJ[group_idx * 36 + k * 6 + j] = val;
            }
            out_block_JTr[group_idx * 6 + j] = s_JTr[j][0];
        }
    }
}
