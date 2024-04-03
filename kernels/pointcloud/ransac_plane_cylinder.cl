#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

#define RANSAC_PRIM_WG_SIZE 256

// Parallel RANSAC 3D Plane Detection: ax + by + cz + d = 0
// Each work-group evaluates 1 candidate plane defined by 3 sampled points.
__kernel void kernel_ransac_fit_plane_hypotheses(
    __global const float4* points,
    __global const uint4* sample_triplets, // (p0, p1, p2, _)
    __global float4* out_plane_equations,  // (a, b, c, d) with normal (a,b,c) unit length
    __global uint* out_inlier_counts,
    const uint total_points,
    const float distance_threshold) {
    
    __local float4 l_plane;
    __local uint l_inliers[RANSAC_PRIM_WG_SIZE];
    __local bool l_valid;

    uint hyp_id = get_group_id(0);
    uint tid = get_local_id(0);

    if (tid == 0) {
        uint4 trip = sample_triplets[hyp_id];
        float4 p0 = points[trip.x];
        float4 p1 = points[trip.y];
        float4 p2 = points[trip.z];

        float3 v0 = (float3)(p0.x, p0.y, p0.z);
        float3 v1 = (float3)(p1.x, p1.y, p1.z);
        float3 v2 = (float3)(p2.x, p2.y, p2.z);

        float3 normal = cross(v1 - v0, v2 - v0);
        float len = length(normal);

        if (len > 1.0e-5f) {
            normal /= len;
            float d = -dot(normal, v0);
            l_plane = (float4)(normal.x, normal.y, normal.z, d);
            out_plane_equations[hyp_id] = l_plane;
            l_valid = true;
        } else {
            l_valid = false;
        }
    }
    sync_local();

    if (!l_valid) {
        if (tid == 0) out_inlier_counts[hyp_id] = 0;
        return;
    }

    float4 plane = l_plane;
    float3 n = (float3)(plane.x, plane.y, plane.z);
    float d = plane.w;

    uint local_count = 0;
    for (uint i = tid; i < total_points; i += RANSAC_PRIM_WG_SIZE) {
        float4 pt4 = points[i];
        float3 p = (float3)(pt4.x, pt4.y, pt4.z);

        float dist = fabs(dot(n, p) + d);
        if (dist <= distance_threshold) {
            local_count++;
        }
    }

    l_inliers[tid] = local_count;
    sync_local();

    for (uint s = RANSAC_PRIM_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            l_inliers[tid] += l_inliers[tid + s];
        }
        sync_local();
    }

    if (tid == 0) {
        out_inlier_counts[hyp_id] = l_inliers[0];
    }
}
