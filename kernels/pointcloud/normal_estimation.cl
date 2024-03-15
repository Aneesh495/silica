#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

#define NORMAL_KNN 16

// Surface Normal and Local Curvature Estimation via Covariance Matrix PCA
__kernel void kernel_estimate_normals_pca(
    __global const float4* in_points,
    __global float4* out_normals, // (nx, ny, nz, curvature)
    const float3 viewpoint,
    const uint num_points,
    const int search_window) {
    
    uint gid = get_global_id(0);
    if (gid >= num_points) return;

    float4 p4 = in_points[gid];
    float3 p = (float3)(p4.x, p4.y, p4.z);

    // Compute centroid of k-nearest neighbors
    float3 centroid = (float3)(0.0f, 0.0f, 0.0f);
    int start_idx = max(0, (int)gid - search_window / 2);
    int end_idx   = min((int)num_points - 1, (int)gid + search_window / 2);
    int count = 0;

    for (int j = start_idx; j <= end_idx; ++j) {
        float4 q4 = in_points[j];
        centroid += (float3)(q4.x, q4.y, q4.z);
        count++;
    }

    if (count < 3) {
        out_normals[gid] = (float4)(0.0f, 0.0f, 1.0f, 0.0f);
        return;
    }

    centroid /= (float)count;

    // Build 3x3 covariance matrix: C = sum((q - c) * (q - c)^T)
    float cxx = 0.0f, cyy = 0.0f, czz = 0.0f;
    float cxy = 0.0f, cxz = 0.0f, cyz = 0.0f;

    for (int j = start_idx; j <= end_idx; ++j) {
        float4 q4 = in_points[j];
        float3 d = (float3)(q4.x, q4.y, q4.z) - centroid;

        cxx += d.x * d.x;
        cyy += d.y * d.y;
        czz += d.z * d.z;
        cxy += d.x * d.y;
        cxz += d.x * d.z;
        cyz += d.y * d.z;
    }

    float inv_n = 1.0f / (float)count;
    cxx *= inv_n; cyy *= inv_n; czz *= inv_n;
    cxy *= inv_n; cxz *= inv_n; cyz *= inv_n;

    // Power iteration / cross-product approximation of smallest eigenvector
    float3 v1 = (float3)(cxx, cxy, cxz);
    float3 v2 = (float3)(cxy, cyy, cyz);
    float3 normal = cross(v1, v2);

    float len = length(normal);
    if (len > 1.0e-5f) {
        normal /= len;
    } else {
        normal = (float3)(0.0f, 0.0f, 1.0f);
    }

    // Orient normal consistently towards viewpoint: dot(n, viewpoint - p) > 0
    float3 view_dir = viewpoint - p;
    if (dot(normal, view_dir) < 0.0f) {
        normal = -normal;
    }

    float tr = cxx + cyy + czz;
    float curvature = (tr > EPSILON_F) ? (len / tr) : 0.0f;

    out_normals[gid] = (float4)(normal.x, normal.y, normal.z, curvature);
}
