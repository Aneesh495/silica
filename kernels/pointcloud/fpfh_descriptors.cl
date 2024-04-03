#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

#define FPFH_BINS 11
#define FPFH_TOTAL_BINS (3 * FPFH_BINS) // 33 dimensions

// Computes 3 Darboux Frame Angular Features between Source Point (ps, ns) and Target Point (pt, nt)
inline float3 compute_darboux_features(float3 ps, float3 ns, float3 pt, float3 nt) {
    float3 delta = pt - ps;
    float dist = length(delta);
    if (dist < 1.0e-5f) return (float3)(0.0f, 0.0f, 0.0f);

    float3 d_norm = delta / dist;

    // Darboux frame: u = ns, v = cross(d_norm, u), w = cross(u, v)
    float3 u = ns;
    float3 v = cross(d_norm, u);
    float len_v = length(v);
    if (len_v > 1.0e-5f) {
        v /= len_v;
    } else {
        v = (float3)(0.0f, 1.0f, 0.0f);
    }
    float3 w = cross(u, v);

    // Alpha = dot(v, nt) in [-1, 1]
    float alpha = dot(v, nt);

    // Phi = dot(u, d_norm) in [-1, 1]
    float phi = dot(u, d_norm);

    // Theta = atan2(dot(w, nt), dot(u, nt)) in [-pi, pi]
    float theta = atan2(dot(w, nt), dot(u, nt));

    return (float3)(alpha, phi, theta);
}

// Stage 1: Simplified Point Feature Histogram (SPFH 33D) per point
__kernel void kernel_compute_spfh_descriptors(
    __global const float4* points,
    __global const float4* normals,
    __global float* out_spfh, // [num_points x 33 floats]
    const uint num_points,
    const int search_radius_points) {
    
    uint gid = get_global_id(0);
    if (gid >= num_points) return;

    float4 ps4 = points[gid];
    float4 ns4 = normals[gid];
    float3 ps = (float3)(ps4.x, ps4.y, ps4.z);
    float3 ns = (float3)(ns4.x, ns4.y, ns4.z);

    float hist[FPFH_TOTAL_BINS];
    #pragma unroll
    for (int i = 0; i < FPFH_TOTAL_BINS; ++i) hist[i] = 0.0f;

    int start_idx = max(0, (int)gid - search_radius_points / 2);
    int end_idx   = min((int)num_points - 1, (int)gid + search_radius_points / 2);
    int pair_count = 0;

    for (int j = start_idx; j <= end_idx; ++j) {
        if ((uint)j == gid) continue;

        float4 pt4 = points[j];
        float4 nt4 = normals[j];
        float3 pt = (float3)(pt4.x, pt4.y, pt4.z);
        float3 nt = (float3)(nt4.x, nt4.y, nt4.z);

        float3 feats = compute_darboux_features(ps, ns, pt, nt);

        // Bin alpha: [-1, 1] -> [0, 10]
        int b_alpha = clamp_i((int)floor((feats.x + 1.0f) * 0.5f * (float)FPFH_BINS), 0, FPFH_BINS - 1);
        // Bin phi: [-1, 1] -> [0, 10]
        int b_phi   = clamp_i((int)floor((feats.y + 1.0f) * 0.5f * (float)FPFH_BINS), 0, FPFH_BINS - 1);
        // Bin theta: [-pi, pi] -> [0, 10]
        int b_theta = clamp_i((int)floor((feats.z + PI_F) * (0.5f * INV_PI_F) * (float)FPFH_BINS), 0, FPFH_BINS - 1);

        hist[b_alpha]++;
        hist[FPFH_BINS + b_phi]++;
        hist[2 * FPFH_BINS + b_theta]++;
        pair_count++;
    }

    float inv_pairs = (pair_count > 0) ? (100.0f / (float)pair_count) : 0.0f;
    size_t out_base = (size_t)gid * FPFH_TOTAL_BINS;

    for (int i = 0; i < FPFH_TOTAL_BINS; ++i) {
        out_spfh[out_base + i] = hist[i] * inv_pairs;
    }
}
