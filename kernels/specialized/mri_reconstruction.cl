#include "common/math.clh"
#include "common/memory.clh"

// ============================================================================
// Advanced Multi-Channel Non-Cartesian and Compressed Sensing MRI Reconstruction
// ============================================================================

// 1. Kaiser-Bessel Kernel Convolution Function
inline float evaluate_kaiser_bessel(float r, float beta, float width) {
    if (r > width * 0.5f) return 0.0f;
    float norm_r = (2.0f * r) / width;
    float arg = sqrt(max(0.0f, 1.0f - norm_r * norm_r));
    return exp(beta * (arg - 1.0f));
}

// 2. 3D Non-Cartesian Stack-of-Stars Gridding
__kernel void kernel_mri_gridding_3d_stack_of_stars(
    __global const float4* trajectory_samples, // (kx, ky, kz, weight)
    __global const float2* kspace_data,
    __global float2* out_cartesian_volume,
    const int nx, const int ny, const int nz,
    const float kaiser_beta, const float kernel_width, const uint num_samples) {
    uint sid = get_global_id(0); if (sid >= num_samples) return;
    float4 traj = trajectory_samples[sid]; float2 val = kspace_data[sid] * traj.w;
    float gx = (traj.x + 0.5f) * (float)nx; float gy = (traj.y + 0.5f) * (float)ny; float gz = (traj.z + 0.5f) * (float)nz;
    int x0 = clamp_i((int)ceil(gx - kernel_width * 0.5f), 0, nx - 1);
    int x1 = clamp_i((int)floor(gx + kernel_width * 0.5f), 0, nx - 1);
    int y0 = clamp_i((int)ceil(gy - kernel_width * 0.5f), 0, ny - 1);
    int y1 = clamp_i((int)floor(gy + kernel_width * 0.5f), 0, ny - 1);
    int iz = clamp_i((int)round(gz), 0, nz - 1);
    size_t slice = (size_t)nx * (size_t)ny;
    for (int y = y0; y <= y1; ++y) {
        float wy = evaluate_kaiser_bessel(fabs((float)y - gy), kaiser_beta, kernel_width);
        for (int x = x0; x <= x1; ++x) {
            float wx = evaluate_kaiser_bessel(fabs((float)x - gx), kaiser_beta, kernel_width);
            float w = wx * wy;
            size_t idx = (size_t)iz * slice + (size_t)y * (size_t)nx + (size_t)x;
            atomic_add_float_global((volatile __global float*)&out_cartesian_volume[idx].x, val.x * w);
            atomic_add_float_global((volatile __global float*)&out_cartesian_volume[idx].y, val.y * w);
        }
    }
}}


// 3. Fast Iterative Soft-Thresholding (FISTA) Momentum Acceleration Step
__kernel void kernel_mri_fista_momentum_update(
    __global float2* x_curr, __global const float2* x_prev,
    __global float2* y_momentum, const float t_curr, const float t_next,
    const uint total_elements) {
    uint gid = get_global_id(0); if (gid >= total_elements) return;
    float momentum_factor = (t_curr - 1.0f) / t_next;
    float2 xc = x_curr[gid]; float2 xp = x_prev[gid];
    y_momentum[gid] = xc + (xc - xp) * momentum_factor;
}}


// ============================================================================
// 2D/3D GRAPPA Kernel Convolution and ESPIRiT Sensitivity Calibration
// ============================================================================

// 2D GRAPPA Synthesis: Acceleration R=2, Source Kernel 3x3
__kernel void kernel_mri_grappa_synthesize_r2_k3(
    __global const float2* acquired_kspace, // [num_coils x ky x kx]
    __global const float2* grappa_weights,
    __global float2* out_full_kspace,
    const int num_coils, const int nx, const int ny) {
    int kx = get_global_id(0); int ky = get_global_id(1); int target_coil = get_global_id(2);
    if (kx >= nx || ky >= ny || target_coil >= num_coils) return;
    int r_offset = ky % 2;
    size_t coil_slice = (size_t)ny * (size_t)nx;
    if (r_offset == 0) {
        // Acquired line: copy directly
        size_t idx = (size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx;
        out_full_kspace[idx] = acquired_kspace[idx];
        return;
    }
    // Synthesize unacquired line from neighboring acquired lines
    float2 synthesized = (float2)(0.0f, 0.0f);
    int base_ky = ky - r_offset;
    for (int sc = 0; sc < num_coils; ++sc) {
        for (int dy = -(1); dy <= (1); ++dy) {
            int src_y = clamp_i(base_ky + dy * 2, 0, ny - 1);
            for (int dx = -1; dx <= 1; ++dx) {
                int src_x = clamp_i(kx + dx, 0, nx - 1);
                size_t src_idx = (size_t)sc * coil_slice + (size_t)src_y * (size_t)nx + (size_t)src_x;
                float2 s_val = acquired_kspace[src_idx];
                // Weight index look-up
                size_t w_idx = (((size_t)target_coil * (2 - 1) + (r_offset - 1)) * num_coils + sc) * (3 * 3) + (dy + 1) * 3 + (dx + 1);
                float2 w_val = grappa_weights[w_idx];
                synthesized.x += s_val.x * w_val.x - s_val.y * w_val.y;
                synthesized.y += s_val.x * w_val.y + s_val.y * w_val.x;
            }
        }
    }
    out_full_kspace[(size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx] = synthesized;
}

// 2D GRAPPA Synthesis: Acceleration R=2, Source Kernel 4x3
__kernel void kernel_mri_grappa_synthesize_r2_k4(
    __global const float2* acquired_kspace, // [num_coils x ky x kx]
    __global const float2* grappa_weights,
    __global float2* out_full_kspace,
    const int num_coils, const int nx, const int ny) {
    int kx = get_global_id(0); int ky = get_global_id(1); int target_coil = get_global_id(2);
    if (kx >= nx || ky >= ny || target_coil >= num_coils) return;
    int r_offset = ky % 2;
    size_t coil_slice = (size_t)ny * (size_t)nx;
    if (r_offset == 0) {
        // Acquired line: copy directly
        size_t idx = (size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx;
        out_full_kspace[idx] = acquired_kspace[idx];
        return;
    }
    // Synthesize unacquired line from neighboring acquired lines
    float2 synthesized = (float2)(0.0f, 0.0f);
    int base_ky = ky - r_offset;
    for (int sc = 0; sc < num_coils; ++sc) {
        for (int dy = -(2); dy <= (2); ++dy) {
            int src_y = clamp_i(base_ky + dy * 2, 0, ny - 1);
            for (int dx = -1; dx <= 1; ++dx) {
                int src_x = clamp_i(kx + dx, 0, nx - 1);
                size_t src_idx = (size_t)sc * coil_slice + (size_t)src_y * (size_t)nx + (size_t)src_x;
                float2 s_val = acquired_kspace[src_idx];
                // Weight index look-up
                size_t w_idx = (((size_t)target_coil * (2 - 1) + (r_offset - 1)) * num_coils + sc) * (4 * 3) + (dy + 2) * 3 + (dx + 1);
                float2 w_val = grappa_weights[w_idx];
                synthesized.x += s_val.x * w_val.x - s_val.y * w_val.y;
                synthesized.y += s_val.x * w_val.y + s_val.y * w_val.x;
            }
        }
    }
    out_full_kspace[(size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx] = synthesized;
}

// 2D GRAPPA Synthesis: Acceleration R=2, Source Kernel 5x3
__kernel void kernel_mri_grappa_synthesize_r2_k5(
    __global const float2* acquired_kspace, // [num_coils x ky x kx]
    __global const float2* grappa_weights,
    __global float2* out_full_kspace,
    const int num_coils, const int nx, const int ny) {
    int kx = get_global_id(0); int ky = get_global_id(1); int target_coil = get_global_id(2);
    if (kx >= nx || ky >= ny || target_coil >= num_coils) return;
    int r_offset = ky % 2;
    size_t coil_slice = (size_t)ny * (size_t)nx;
    if (r_offset == 0) {
        // Acquired line: copy directly
        size_t idx = (size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx;
        out_full_kspace[idx] = acquired_kspace[idx];
        return;
    }
    // Synthesize unacquired line from neighboring acquired lines
    float2 synthesized = (float2)(0.0f, 0.0f);
    int base_ky = ky - r_offset;
    for (int sc = 0; sc < num_coils; ++sc) {
        for (int dy = -(2); dy <= (2); ++dy) {
            int src_y = clamp_i(base_ky + dy * 2, 0, ny - 1);
            for (int dx = -1; dx <= 1; ++dx) {
                int src_x = clamp_i(kx + dx, 0, nx - 1);
                size_t src_idx = (size_t)sc * coil_slice + (size_t)src_y * (size_t)nx + (size_t)src_x;
                float2 s_val = acquired_kspace[src_idx];
                // Weight index look-up
                size_t w_idx = (((size_t)target_coil * (2 - 1) + (r_offset - 1)) * num_coils + sc) * (5 * 3) + (dy + 2) * 3 + (dx + 1);
                float2 w_val = grappa_weights[w_idx];
                synthesized.x += s_val.x * w_val.x - s_val.y * w_val.y;
                synthesized.y += s_val.x * w_val.y + s_val.y * w_val.x;
            }
        }
    }
    out_full_kspace[(size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx] = synthesized;
}

// 2D GRAPPA Synthesis: Acceleration R=3, Source Kernel 3x3
__kernel void kernel_mri_grappa_synthesize_r3_k3(
    __global const float2* acquired_kspace, // [num_coils x ky x kx]
    __global const float2* grappa_weights,
    __global float2* out_full_kspace,
    const int num_coils, const int nx, const int ny) {
    int kx = get_global_id(0); int ky = get_global_id(1); int target_coil = get_global_id(2);
    if (kx >= nx || ky >= ny || target_coil >= num_coils) return;
    int r_offset = ky % 3;
    size_t coil_slice = (size_t)ny * (size_t)nx;
    if (r_offset == 0) {
        // Acquired line: copy directly
        size_t idx = (size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx;
        out_full_kspace[idx] = acquired_kspace[idx];
        return;
    }
    // Synthesize unacquired line from neighboring acquired lines
    float2 synthesized = (float2)(0.0f, 0.0f);
    int base_ky = ky - r_offset;
    for (int sc = 0; sc < num_coils; ++sc) {
        for (int dy = -(1); dy <= (1); ++dy) {
            int src_y = clamp_i(base_ky + dy * 3, 0, ny - 1);
            for (int dx = -1; dx <= 1; ++dx) {
                int src_x = clamp_i(kx + dx, 0, nx - 1);
                size_t src_idx = (size_t)sc * coil_slice + (size_t)src_y * (size_t)nx + (size_t)src_x;
                float2 s_val = acquired_kspace[src_idx];
                // Weight index look-up
                size_t w_idx = (((size_t)target_coil * (3 - 1) + (r_offset - 1)) * num_coils + sc) * (3 * 3) + (dy + 1) * 3 + (dx + 1);
                float2 w_val = grappa_weights[w_idx];
                synthesized.x += s_val.x * w_val.x - s_val.y * w_val.y;
                synthesized.y += s_val.x * w_val.y + s_val.y * w_val.x;
            }
        }
    }
    out_full_kspace[(size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx] = synthesized;
}

// 2D GRAPPA Synthesis: Acceleration R=3, Source Kernel 4x3
__kernel void kernel_mri_grappa_synthesize_r3_k4(
    __global const float2* acquired_kspace, // [num_coils x ky x kx]
    __global const float2* grappa_weights,
    __global float2* out_full_kspace,
    const int num_coils, const int nx, const int ny) {
    int kx = get_global_id(0); int ky = get_global_id(1); int target_coil = get_global_id(2);
    if (kx >= nx || ky >= ny || target_coil >= num_coils) return;
    int r_offset = ky % 3;
    size_t coil_slice = (size_t)ny * (size_t)nx;
    if (r_offset == 0) {
        // Acquired line: copy directly
        size_t idx = (size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx;
        out_full_kspace[idx] = acquired_kspace[idx];
        return;
    }
    // Synthesize unacquired line from neighboring acquired lines
    float2 synthesized = (float2)(0.0f, 0.0f);
    int base_ky = ky - r_offset;
    for (int sc = 0; sc < num_coils; ++sc) {
        for (int dy = -(2); dy <= (2); ++dy) {
            int src_y = clamp_i(base_ky + dy * 3, 0, ny - 1);
            for (int dx = -1; dx <= 1; ++dx) {
                int src_x = clamp_i(kx + dx, 0, nx - 1);
                size_t src_idx = (size_t)sc * coil_slice + (size_t)src_y * (size_t)nx + (size_t)src_x;
                float2 s_val = acquired_kspace[src_idx];
                // Weight index look-up
                size_t w_idx = (((size_t)target_coil * (3 - 1) + (r_offset - 1)) * num_coils + sc) * (4 * 3) + (dy + 2) * 3 + (dx + 1);
                float2 w_val = grappa_weights[w_idx];
                synthesized.x += s_val.x * w_val.x - s_val.y * w_val.y;
                synthesized.y += s_val.x * w_val.y + s_val.y * w_val.x;
            }
        }
    }
    out_full_kspace[(size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx] = synthesized;
}

// 2D GRAPPA Synthesis: Acceleration R=3, Source Kernel 5x3
__kernel void kernel_mri_grappa_synthesize_r3_k5(
    __global const float2* acquired_kspace, // [num_coils x ky x kx]
    __global const float2* grappa_weights,
    __global float2* out_full_kspace,
    const int num_coils, const int nx, const int ny) {
    int kx = get_global_id(0); int ky = get_global_id(1); int target_coil = get_global_id(2);
    if (kx >= nx || ky >= ny || target_coil >= num_coils) return;
    int r_offset = ky % 3;
    size_t coil_slice = (size_t)ny * (size_t)nx;
    if (r_offset == 0) {
        // Acquired line: copy directly
        size_t idx = (size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx;
        out_full_kspace[idx] = acquired_kspace[idx];
        return;
    }
    // Synthesize unacquired line from neighboring acquired lines
    float2 synthesized = (float2)(0.0f, 0.0f);
    int base_ky = ky - r_offset;
    for (int sc = 0; sc < num_coils; ++sc) {
        for (int dy = -(2); dy <= (2); ++dy) {
            int src_y = clamp_i(base_ky + dy * 3, 0, ny - 1);
            for (int dx = -1; dx <= 1; ++dx) {
                int src_x = clamp_i(kx + dx, 0, nx - 1);
                size_t src_idx = (size_t)sc * coil_slice + (size_t)src_y * (size_t)nx + (size_t)src_x;
                float2 s_val = acquired_kspace[src_idx];
                // Weight index look-up
                size_t w_idx = (((size_t)target_coil * (3 - 1) + (r_offset - 1)) * num_coils + sc) * (5 * 3) + (dy + 2) * 3 + (dx + 1);
                float2 w_val = grappa_weights[w_idx];
                synthesized.x += s_val.x * w_val.x - s_val.y * w_val.y;
                synthesized.y += s_val.x * w_val.y + s_val.y * w_val.x;
            }
        }
    }
    out_full_kspace[(size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx] = synthesized;
}

// 2D GRAPPA Synthesis: Acceleration R=4, Source Kernel 3x3
__kernel void kernel_mri_grappa_synthesize_r4_k3(
    __global const float2* acquired_kspace, // [num_coils x ky x kx]
    __global const float2* grappa_weights,
    __global float2* out_full_kspace,
    const int num_coils, const int nx, const int ny) {
    int kx = get_global_id(0); int ky = get_global_id(1); int target_coil = get_global_id(2);
    if (kx >= nx || ky >= ny || target_coil >= num_coils) return;
    int r_offset = ky % 4;
    size_t coil_slice = (size_t)ny * (size_t)nx;
    if (r_offset == 0) {
        // Acquired line: copy directly
        size_t idx = (size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx;
        out_full_kspace[idx] = acquired_kspace[idx];
        return;
    }
    // Synthesize unacquired line from neighboring acquired lines
    float2 synthesized = (float2)(0.0f, 0.0f);
    int base_ky = ky - r_offset;
    for (int sc = 0; sc < num_coils; ++sc) {
        for (int dy = -(1); dy <= (1); ++dy) {
            int src_y = clamp_i(base_ky + dy * 4, 0, ny - 1);
            for (int dx = -1; dx <= 1; ++dx) {
                int src_x = clamp_i(kx + dx, 0, nx - 1);
                size_t src_idx = (size_t)sc * coil_slice + (size_t)src_y * (size_t)nx + (size_t)src_x;
                float2 s_val = acquired_kspace[src_idx];
                // Weight index look-up
                size_t w_idx = (((size_t)target_coil * (4 - 1) + (r_offset - 1)) * num_coils + sc) * (3 * 3) + (dy + 1) * 3 + (dx + 1);
                float2 w_val = grappa_weights[w_idx];
                synthesized.x += s_val.x * w_val.x - s_val.y * w_val.y;
                synthesized.y += s_val.x * w_val.y + s_val.y * w_val.x;
            }
        }
    }
    out_full_kspace[(size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx] = synthesized;
}

// 2D GRAPPA Synthesis: Acceleration R=4, Source Kernel 4x3
__kernel void kernel_mri_grappa_synthesize_r4_k4(
    __global const float2* acquired_kspace, // [num_coils x ky x kx]
    __global const float2* grappa_weights,
    __global float2* out_full_kspace,
    const int num_coils, const int nx, const int ny) {
    int kx = get_global_id(0); int ky = get_global_id(1); int target_coil = get_global_id(2);
    if (kx >= nx || ky >= ny || target_coil >= num_coils) return;
    int r_offset = ky % 4;
    size_t coil_slice = (size_t)ny * (size_t)nx;
    if (r_offset == 0) {
        // Acquired line: copy directly
        size_t idx = (size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx;
        out_full_kspace[idx] = acquired_kspace[idx];
        return;
    }
    // Synthesize unacquired line from neighboring acquired lines
    float2 synthesized = (float2)(0.0f, 0.0f);
    int base_ky = ky - r_offset;
    for (int sc = 0; sc < num_coils; ++sc) {
        for (int dy = -(2); dy <= (2); ++dy) {
            int src_y = clamp_i(base_ky + dy * 4, 0, ny - 1);
            for (int dx = -1; dx <= 1; ++dx) {
                int src_x = clamp_i(kx + dx, 0, nx - 1);
                size_t src_idx = (size_t)sc * coil_slice + (size_t)src_y * (size_t)nx + (size_t)src_x;
                float2 s_val = acquired_kspace[src_idx];
                // Weight index look-up
                size_t w_idx = (((size_t)target_coil * (4 - 1) + (r_offset - 1)) * num_coils + sc) * (4 * 3) + (dy + 2) * 3 + (dx + 1);
                float2 w_val = grappa_weights[w_idx];
                synthesized.x += s_val.x * w_val.x - s_val.y * w_val.y;
                synthesized.y += s_val.x * w_val.y + s_val.y * w_val.x;
            }
        }
    }
    out_full_kspace[(size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx] = synthesized;
}

// 2D GRAPPA Synthesis: Acceleration R=4, Source Kernel 5x3
__kernel void kernel_mri_grappa_synthesize_r4_k5(
    __global const float2* acquired_kspace, // [num_coils x ky x kx]
    __global const float2* grappa_weights,
    __global float2* out_full_kspace,
    const int num_coils, const int nx, const int ny) {
    int kx = get_global_id(0); int ky = get_global_id(1); int target_coil = get_global_id(2);
    if (kx >= nx || ky >= ny || target_coil >= num_coils) return;
    int r_offset = ky % 4;
    size_t coil_slice = (size_t)ny * (size_t)nx;
    if (r_offset == 0) {
        // Acquired line: copy directly
        size_t idx = (size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx;
        out_full_kspace[idx] = acquired_kspace[idx];
        return;
    }
    // Synthesize unacquired line from neighboring acquired lines
    float2 synthesized = (float2)(0.0f, 0.0f);
    int base_ky = ky - r_offset;
    for (int sc = 0; sc < num_coils; ++sc) {
        for (int dy = -(2); dy <= (2); ++dy) {
            int src_y = clamp_i(base_ky + dy * 4, 0, ny - 1);
            for (int dx = -1; dx <= 1; ++dx) {
                int src_x = clamp_i(kx + dx, 0, nx - 1);
                size_t src_idx = (size_t)sc * coil_slice + (size_t)src_y * (size_t)nx + (size_t)src_x;
                float2 s_val = acquired_kspace[src_idx];
                // Weight index look-up
                size_t w_idx = (((size_t)target_coil * (4 - 1) + (r_offset - 1)) * num_coils + sc) * (5 * 3) + (dy + 2) * 3 + (dx + 1);
                float2 w_val = grappa_weights[w_idx];
                synthesized.x += s_val.x * w_val.x - s_val.y * w_val.y;
                synthesized.y += s_val.x * w_val.y + s_val.y * w_val.x;
            }
        }
    }
    out_full_kspace[(size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx] = synthesized;
}

// ESPIRiT Calibration Matrix Eigenvector Projection
__kernel void kernel_mri_espirit_project_eigenvector(
    __global const float2* calibration_eigenvectors, // [num_coils x ny x nx]
    __global const float* singular_values,            // [ny x nx]
    __global float2* out_sensitivity_maps,
    const float eigenvalue_threshold, const int num_coils, const int width, const int height) {
    int x = get_global_id(0); int y = get_global_id(1); int c = get_global_id(2);
    if (x >= width || y >= height || c >= num_coils) return;
    size_t pixel_idx = (size_t)y * (size_t)width + (size_t)x;
    float sigma = singular_values[pixel_idx];
    size_t coil_idx = (size_t)c * ((size_t)width * (size_t)height) + pixel_idx;
    if (sigma >= eigenvalue_threshold) {
        out_sensitivity_maps[coil_idx] = calibration_eigenvectors[coil_idx];
    } else {
        out_sensitivity_maps[coil_idx] = (float2)(0.0f, 0.0f);
    }
}
