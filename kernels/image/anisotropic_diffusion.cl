#include "common/math.clh"
#include "common/memory.clh"

// Catté-Lions Regularized Anisotropic Diffusion
// Preserves sharp organ and bone boundaries while eliminating intra-tissue CT noise.
__kernel void kernel_anisotropic_diffusion_iteration(
    __global const float* in_image,
    __global float* out_image,
    const int width,
    const int height,
    const float kappa,
    const float lambda_dt) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t idx = (size_t)y * (size_t)width + (size_t)x;
    float center = in_image[idx];

    // 4 cardinal differences
    float north = (y > 0) ? in_image[idx - width] : center;
    float south = (y + 1 < height) ? in_image[idx + width] : center;
    float east  = (x + 1 < width) ? in_image[idx + 1] : center;
    float west  = (x > 0) ? in_image[idx - 1] : center;

    float delta_n = north - center;
    float delta_s = south - center;
    float delta_e = east  - center;
    float delta_w = west  - center;

    // Perona-Malik g2 conduction function: g(grad) = 1 / (1 + (grad / kappa)^2)
    float inv_k2 = 1.0f / (kappa * kappa + EPSILON_F);

    float c_n = 1.0f / (1.0f + (delta_n * delta_n) * inv_k2);
    float c_s = 1.0f / (1.0f + (delta_s * delta_s) * inv_k2);
    float c_e = 1.0f / (1.0f + (delta_e * delta_e) * inv_k2);
    float c_w = 1.0f / (1.0f + (delta_w * delta_w) * inv_k2);

    float updated = center + lambda_dt * (c_n * delta_n + c_s * delta_s + c_e * delta_e + c_w * delta_w);
    out_image[idx] = updated;
}
