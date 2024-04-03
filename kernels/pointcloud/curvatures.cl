#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

// Estimates Principal Curvatures (k1, k2), Mean Curvature, and Gaussian Curvature
__kernel void kernel_compute_pointcloud_curvatures(
    __global const float4* points,
    __global const float4* normals,
    __global float4* out_curvatures, // (k1, k2, gaussian_K, mean_H)
    const uint num_points,
    const int search_window) {
    
    uint gid = get_global_id(0);
    if (gid >= num_points) return;

    float4 p4 = points[gid];
    float4 n4 = normals[gid];
    float3 p0 = (float3)(p4.x, p4.y, p4.z);
    float3 n0 = normalize((float3)(n4.x, n4.y, n4.z));

    // Construct local tangent coordinate frame (u, v, n0)
    float3 up = (fabs(n0.z) < 0.99f) ? (float3)(0.0f, 0.0f, 1.0f) : (float3)(1.0f, 0.0f, 0.0f);
    float3 u = normalize(cross(up, n0));
    float3 v = cross(n0, u);

    // Fit quadratic Monge patch: z = 0.5 * (e * x^2 + 2 * f * x * y + g * y^2)
    // Build 3x3 normal equations for parameters (e, f, g)
    float A[3][3] = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
    float b[3] = {0.0f, 0.0f, 0.0f};

    int start_idx = max(0, (int)gid - search_window / 2);
    int end_idx   = min((int)num_points - 1, (int)gid + search_window / 2);
    int count = 0;

    for (int j = start_idx; j <= end_idx; ++j) {
        if ((uint)j == gid) continue;

        float4 q4 = points[j];
        float3 diff = (float3)(q4.x, q4.y, q4.z) - p0;

        float x = dot(diff, u);
        float y = dot(diff, v);
        float z = dot(diff, n0);

        float basis[3] = {0.5f * x * x, x * y, 0.5f * y * y};

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                A[r][c] += basis[r] * basis[c];
            }
            b[r] += basis[r] * z;
        }
        count++;
    }

    if (count < 6) {
        out_curvatures[gid] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // Solve 3x3 system via Cramer's rule
    A[0][0] += 0.001f; A[1][1] += 0.001f; A[2][2] += 0.001f; // Tikhonov regularization
    float det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    if (fabs(det) < 1.0e-7f) {
        out_curvatures[gid] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float inv_det = 1.0f / det;
    float e = (b[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
               A[0][1] * (b[1] * A[2][2] - A[1][2] * b[2]) +
               A[0][2] * (b[1] * A[2][1] - A[1][1] * b[2])) * inv_det;
    float f = (A[0][0] * (b[1] * A[2][2] - A[1][2] * b[2]) -
               b[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
               A[0][2] * (A[1][0] * b[2] - b[1] * A[2][0])) * inv_det;
    float g = (A[0][0] * (A[1][1] * b[2] - b[1] * A[2][1]) -
               A[0][1] * (A[1][0] * b[2] - b[1] * A[2][0]) +
               b[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])) * inv_det;

    // Weingarten shape matrix S = [e, f; f, g]
    // Gaussian curvature K = det(S) = e*g - f^2
    // Mean curvature H = 0.5 * trace(S) = 0.5 * (e + g)
    float K = e * g - f * f;
    float H = 0.5f * (e + g);

    float disc = sqrt(fmax(0.0f, H * H - K));
    float k1 = H + disc;
    float k2 = H - disc;

    out_curvatures[gid] = (float4)(k1, k2, K, H);
}
