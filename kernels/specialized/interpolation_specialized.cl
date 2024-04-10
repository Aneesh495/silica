#include "common/math.clh"
#include "common/memory.clh"

// ============================================================================
// Multidimensional Continuous Interpolators, Splines, and Deformation Fields
// ============================================================================

inline float4 bspline_weights(float t) {
    float it = 1.0f - t;
    float w0 = it * it * it * (1.0f / 6.0f);
    float w1 = (3.0f * t * t * t - 6.0f * t * t + 4.0f) * (1.0f / 6.0f);
    float w2 = (-3.0f * t * t * t + 3.0f * t * t + 3.0f * t + 1.0f) * (1.0f / 6.0f);
    float w3 = t * t * t * (1.0f / 6.0f);
    return (float4)(w0, w1, w2, w3);
}

inline float4 catmull_rom_weights(float t) {
    float t2 = t * t; float t3 = t2 * t;
    float w0 = -0.5f * t3 + t2 - 0.5f * t;
    float w1 =  1.5f * t3 - 2.5f * t2 + 1.0f;
    float w2 = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
    float w3 =  0.5f * t3 - 0.5f * t2;
    return (float4)(w0, w1, w2, w3);
}

inline float lanczos3_weight(float x) {
    if (fabs(x) < 1.0e-6f) return 1.0f;
    if (fabs(x) >= 3.0f) return 0.0f;
    float px = PI_F * x;
    return (sin(px) / px) * (sin(px / 3.0f) / (px / 3.0f));
}

// 2D Lanczos-3 High-Fidelity Resampler
__kernel void kernel_interp2d_lanczos3(
    __global const float* in_img, __global float* out_img,
    const int in_w, const int in_h, const int out_w, const int out_h,
    const float scale_x, const float scale_y) {
    int x = get_global_id(0); int y = get_global_id(1);
    if (x >= out_w || y >= out_h) return;
    float fx = (float)x * scale_x; float fy = (float)y * scale_y;
    int ix = (int)floor(fx); int iy = (int)floor(fy);
    float sum = 0.0f; float total_weight = 0.0f;
    for (int dy = -2; dy <= 3; ++dy) {
        int cy = clamp_i(iy + dy, 0, in_h - 1);
        float wy = lanczos3_weight(fy - (float)(iy + dy));
        for (int dx = -2; dx <= 3; ++dx) {
            int cx = clamp_i(ix + dx, 0, in_w - 1);
            float wx = lanczos3_weight(fx - (float)(ix + dx));
            float w = wx * wy;
            sum += in_img[(size_t)cy * (size_t)in_w + (size_t)cx] * w;
            total_weight += w;
        }
    }
    out_img[(size_t)y * (size_t)out_w + (size_t)x] = (total_weight > 1.0e-6f) ? (sum / total_weight) : 0.0f;
}}


// 3D Tricubic B-Spline Continuous Volume Resampling
__kernel void kernel_interp3d_cubic_bspline(
    __global const float* in_volume, __global float* out_volume,
    const int in_nx, const int in_ny, const int in_nz,
    const int out_nx, const int out_ny, const int out_nz,
    const float scale_x, const float scale_y, const float scale_z) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x >= out_nx || y >= out_ny || z >= out_nz) return;
    float fx = (float)x * scale_x; float fy = (float)y * scale_y; float fz = (float)z * scale_z;
    int ix = (int)floor(fx); int iy = (int)floor(fy); int iz = (int)floor(fz);
    float4 wx = bspline_weights(fx - (float)ix);
    float4 wy = bspline_weights(fy - (float)iy);
    float4 wz = bspline_weights(fz - (float)iz);
    float weights_x[4] = {wx.x, wx.y, wx.z, wx.w};
    float weights_y[4] = {wy.x, wy.y, wy.z, wy.w};
    float weights_z[4] = {wz.x, wz.y, wz.z, wz.w};
    float sum = 0.0f; size_t in_slice = (size_t)in_nx * (size_t)in_ny;
    for (int dz = -1; dz <= 2; ++dz) {
        int cz = clamp_i(iz + dz, 0, in_nz - 1); float wz_val = weights_z[dz + 1];
        for (int dy = -1; dy <= 2; ++dy) {
            int cy = clamp_i(iy + dy, 0, in_ny - 1); float wy_val = weights_y[dy + 1];
            for (int dx = -1; dx <= 2; ++dx) {
                int cx = clamp_i(ix + dx, 0, in_nx - 1); float wx_val = weights_x[dx + 1];
                sum += in_volume[(size_t)cz * in_slice + (size_t)cy * (size_t)in_nx + (size_t)cx] * (wx_val * wy_val * wz_val);
            }
        }
    }
    out_volume[(size_t)z * ((size_t)out_nx * (size_t)out_ny) + (size_t)y * (size_t)out_nx + (size_t)x] = sum;
}}


// Thin-Plate Spline (TPS) Radial Basis Warp Grid Evaluator: U(r) = r^2 * ln(r)
__kernel void kernel_tps_warp_2d(
    __global const float2* control_points, // [num_ctrl]
    __global const float2* tps_weights,    // [num_ctrl]
    __global const float* affine_coeffs,   // [2 x 3] (a0, ax, ay)
    __global float2* out_displacement_field, // [h x w]
    const int num_ctrl, const int width, const int height) {
    int x = get_global_id(0); int y = get_global_id(1);
    if (x >= width || y >= height) return;
    float px = (float)x; float py = (float)y;
    // Affine component
    float dx = affine_coeffs[0] + affine_coeffs[1] * px + affine_coeffs[2] * py;
    float dy = affine_coeffs[3] + affine_coeffs[4] * px + affine_coeffs[5] * py;
    // Non-linear TPS RBF kernel summation
    for (int k = 0; k < num_ctrl; ++k) {
        float2 cp = control_points[k];
        float rx = px - cp.x; float ry = py - cp.y;
        float r_sq = rx * rx + ry * ry;
        if (r_sq > 1.0e-8f) {
            float u = 0.5f * r_sq * log(r_sq);
            float2 w = tps_weights[k];
            dx += w.x * u; dy += w.y * u;
        }
    }
    out_displacement_field[(size_t)y * (size_t)width + (size_t)x] = (float2)(dx, dy);
}}


// ============================================================================
// Radial Basis Function (RBF) Kernel Interpolators: Wendland and Multiquadric
// ============================================================================

// 3D RBF Evaluation: gaussian
__kernel void kernel_rbf_interp3d_gaussian(
    __global const float4* center_nodes, // [num_centers] (x, y, z, 0)
    __global const float* weights,        // [num_centers]
    __global float* out_volume,
    const float epsilon_sq, const float support_radius,
    const int nx, const int ny, const int nz, const float inv_dx, const int num_centers) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x >= nx || y >= ny || z >= nz) return;
    float px = (float)x * inv_dx; float py = (float)y * inv_dx; float pz = (float)z * inv_dx;
    float sum = 0.0f;
    for (int k = 0; k < num_centers; ++k) {
        float4 c = center_nodes[k];
        float rx = px - c.x; float ry = py - c.y; float rz = pz - c.z;
        float r_sq = rx * rx + ry * ry + rz * rz;
        float r_norm = sqrt(r_sq) / support_radius;
        float phi = exp(-epsilon_sq * r_sq);
        sum += weights[k] * phi;
    }
    out_volume[(size_t)z * ((size_t)nx * (size_t)ny) + (size_t)y * (size_t)nx + (size_t)x] = sum;
}

// 3D RBF Evaluation: multiquadric
__kernel void kernel_rbf_interp3d_multiquadric(
    __global const float4* center_nodes, // [num_centers] (x, y, z, 0)
    __global const float* weights,        // [num_centers]
    __global float* out_volume,
    const float epsilon_sq, const float support_radius,
    const int nx, const int ny, const int nz, const float inv_dx, const int num_centers) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x >= nx || y >= ny || z >= nz) return;
    float px = (float)x * inv_dx; float py = (float)y * inv_dx; float pz = (float)z * inv_dx;
    float sum = 0.0f;
    for (int k = 0; k < num_centers; ++k) {
        float4 c = center_nodes[k];
        float rx = px - c.x; float ry = py - c.y; float rz = pz - c.z;
        float r_sq = rx * rx + ry * ry + rz * rz;
        float r_norm = sqrt(r_sq) / support_radius;
        float phi = sqrt(1.0f + epsilon_sq * r_sq);
        sum += weights[k] * phi;
    }
    out_volume[(size_t)z * ((size_t)nx * (size_t)ny) + (size_t)y * (size_t)nx + (size_t)x] = sum;
}

// 3D RBF Evaluation: inverse_multiquadric
__kernel void kernel_rbf_interp3d_inverse_multiquadric(
    __global const float4* center_nodes, // [num_centers] (x, y, z, 0)
    __global const float* weights,        // [num_centers]
    __global float* out_volume,
    const float epsilon_sq, const float support_radius,
    const int nx, const int ny, const int nz, const float inv_dx, const int num_centers) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x >= nx || y >= ny || z >= nz) return;
    float px = (float)x * inv_dx; float py = (float)y * inv_dx; float pz = (float)z * inv_dx;
    float sum = 0.0f;
    for (int k = 0; k < num_centers; ++k) {
        float4 c = center_nodes[k];
        float rx = px - c.x; float ry = py - c.y; float rz = pz - c.z;
        float r_sq = rx * rx + ry * ry + rz * rz;
        float r_norm = sqrt(r_sq) / support_radius;
        float phi = 1.0f / sqrt(1.0f + epsilon_sq * r_sq);
        sum += weights[k] * phi;
    }
    out_volume[(size_t)z * ((size_t)nx * (size_t)ny) + (size_t)y * (size_t)nx + (size_t)x] = sum;
}

// 3D RBF Evaluation: inverse_quadratic
__kernel void kernel_rbf_interp3d_inverse_quadratic(
    __global const float4* center_nodes, // [num_centers] (x, y, z, 0)
    __global const float* weights,        // [num_centers]
    __global float* out_volume,
    const float epsilon_sq, const float support_radius,
    const int nx, const int ny, const int nz, const float inv_dx, const int num_centers) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x >= nx || y >= ny || z >= nz) return;
    float px = (float)x * inv_dx; float py = (float)y * inv_dx; float pz = (float)z * inv_dx;
    float sum = 0.0f;
    for (int k = 0; k < num_centers; ++k) {
        float4 c = center_nodes[k];
        float rx = px - c.x; float ry = py - c.y; float rz = pz - c.z;
        float r_sq = rx * rx + ry * ry + rz * rz;
        float r_norm = sqrt(r_sq) / support_radius;
        float phi = 1.0f / (1.0f + epsilon_sq * r_sq);
        sum += weights[k] * phi;
    }
    out_volume[(size_t)z * ((size_t)nx * (size_t)ny) + (size_t)y * (size_t)nx + (size_t)x] = sum;
}

// 3D RBF Evaluation: wendland_c2
__kernel void kernel_rbf_interp3d_wendland_c2(
    __global const float4* center_nodes, // [num_centers] (x, y, z, 0)
    __global const float* weights,        // [num_centers]
    __global float* out_volume,
    const float epsilon_sq, const float support_radius,
    const int nx, const int ny, const int nz, const float inv_dx, const int num_centers) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x >= nx || y >= ny || z >= nz) return;
    float px = (float)x * inv_dx; float py = (float)y * inv_dx; float pz = (float)z * inv_dx;
    float sum = 0.0f;
    for (int k = 0; k < num_centers; ++k) {
        float4 c = center_nodes[k];
        float rx = px - c.x; float ry = py - c.y; float rz = pz - c.z;
        float r_sq = rx * rx + ry * ry + rz * rz;
        float r_norm = sqrt(r_sq) / support_radius;
        float phi = select(0.0f, (1.0f - r_norm) * (1.0f - r_norm) * (1.0f - r_norm) * (1.0f - r_norm) * (4.0f * r_norm + 1.0f), r_norm < 1.0f);
        sum += weights[k] * phi;
    }
    out_volume[(size_t)z * ((size_t)nx * (size_t)ny) + (size_t)y * (size_t)nx + (size_t)x] = sum;
}

// 3D RBF Evaluation: wendland_c4
__kernel void kernel_rbf_interp3d_wendland_c4(
    __global const float4* center_nodes, // [num_centers] (x, y, z, 0)
    __global const float* weights,        // [num_centers]
    __global float* out_volume,
    const float epsilon_sq, const float support_radius,
    const int nx, const int ny, const int nz, const float inv_dx, const int num_centers) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x >= nx || y >= ny || z >= nz) return;
    float px = (float)x * inv_dx; float py = (float)y * inv_dx; float pz = (float)z * inv_dx;
    float sum = 0.0f;
    for (int k = 0; k < num_centers; ++k) {
        float4 c = center_nodes[k];
        float rx = px - c.x; float ry = py - c.y; float rz = pz - c.z;
        float r_sq = rx * rx + ry * ry + rz * rz;
        float r_norm = sqrt(r_sq) / support_radius;
        float phi = select(0.0f, (1.0f - r_norm) * (1.0f - r_norm) * (1.0f - r_norm) * (1.0f - r_norm) * (1.0f - r_norm) * (1.0f - r_norm) * (35.0f * r_norm * r_norm + 18.0f * r_norm + 3.0f), r_norm < 1.0f);
        sum += weights[k] * phi;
    }
    out_volume[(size_t)z * ((size_t)nx * (size_t)ny) + (size_t)y * (size_t)nx + (size_t)x] = sum;
}
