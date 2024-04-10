#include "common/math.clh"
#include "common/memory.clh"

// Diffeomorphic Demons Non-Rigid Image Registration

// Diffeomorphic Demons Velocity Field Computation (Thirion / Vercauteren formulation)
__kernel void kernel_demons_velocity_field_3d(
    __global const float* moving_image,
    __global const float* static_image,
    __global const float4* static_gradients,
    __global float4* out_velocity_update,
    const float alpha_sigma,
    const uint total_voxels) {
    uint gid = get_global_id(0);
    if (gid >= total_voxels) return;
    float m = moving_image[gid];
    float s = static_image[gid];
    float diff = m - s;
    float4 grad = static_gradients[gid];
    float grad_sq = grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
    float denom = grad_sq + alpha_sigma * diff * diff;
    if (denom > 1.0e-7f) {
        float factor = diff / denom;
        out_velocity_update[gid] = (float4)(-grad.x * factor, -grad.y * factor, -grad.z * factor, 0.0f);
    } else {
        out_velocity_update[gid] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

// Lie Algebra Vector Field Composition: exp(v) o exp(u) via Scaling and Squaring
__kernel void kernel_vector_field_compose_3d(
    __global const float4* field_a,
    __global const float4* field_b,
    __global float4* out_composed,
    const int nx, const int ny, const int nz) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    if (x >= nx || y >= ny || z >= nz) return;
    size_t slice = (size_t)nx * (size_t)ny;
    size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;
    float4 va = field_a[idx];
    // Sample field_b at position (x + va.x, y + va.y, z + va.z)
    float px = (float)x + va.x;
    float py = (float)y + va.y;
    float pz = (float)z + va.z;
    int ix = clamp_i((int)floor(px), 0, nx - 2);
    int iy = clamp_i((int)floor(py), 0, ny - 2);
    int iz = clamp_i((int)floor(pz), 0, nz - 2);
    float fx = px - (float)ix;
    float fy = py - (float)iy;
    float fz = pz - (float)iz;
    float4 c000 = field_b[(size_t)iz * slice + (size_t)iy * (size_t)nx + (size_t)ix];
    float4 c100 = field_b[(size_t)iz * slice + (size_t)iy * (size_t)nx + (size_t)(ix + 1)];
    float4 vb = mix(c000, c100, fx);
    out_composed[idx] = va + vb;
}
