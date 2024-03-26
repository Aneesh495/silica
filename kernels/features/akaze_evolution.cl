#include "common/math.clh"
#include "common/memory.clh"

#define DIFFUSION_PERONA_MALIK_1 0
#define DIFFUSION_PERONA_MALIK_2 1
#define DIFFUSION_WEICKERT       2

// Computes Nonlinear Conductivity Coefficient c(x, y) from Gaussian Smoothed Gradients
__kernel void kernel_compute_diffusivity(
    __global const float* smoothed_img,
    __global float* out_diffusivity,
    const int width,
    const int height,
    const float k_contrast,
    const int conductivity_type) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    float p10 = read_image_2d_safe(smoothed_img, x + 1, y, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float pm10 = read_image_2d_safe(smoothed_img, x - 1, y, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float p01 = read_image_2d_safe(smoothed_img, x, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float p0m1 = read_image_2d_safe(smoothed_img, x, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f);

    float gx = 0.5f * (p10 - pm10);
    float gy = 0.5f * (p01 - p0m1);
    float grad_sq = gx * gx + gy * gy;

    float inv_k2 = 1.0f / (k_contrast * k_contrast + EPSILON_F);
    float c = 1.0f;

    if (conductivity_type == DIFFUSION_PERONA_MALIK_1) {
        // g(s) = exp(-s^2 / k^2)
        c = exp(-grad_sq * inv_k2);
    } else if (conductivity_type == DIFFUSION_PERONA_MALIK_2) {
        // g(s) = 1 / (1 + s^2 / k^2)
        c = 1.0f / (1.0f + grad_sq * inv_k2);
    } else {
        // Weickert conductivity
        if (grad_sq <= 0.0f) {
            c = 1.0f;
        } else {
            float r = grad_sq * inv_k2;
            c = 1.0f - exp(-3.31488f / (r * r * r * r + EPSILON_F));
        }
    }

    out_diffusivity[(size_t)y * (size_t)width + (size_t)x] = c;
}

// Fast Explicit Diffusion (FED) Step: L^{j+1} = L^j + tau * div(c * grad(L))
__kernel void kernel_fed_diffusion_step(
    __global const float* in_L,
    __global const float* in_c,
    __global float* out_L,
    const int width,
    const int height,
    const float tau_step) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t idx = (size_t)y * (size_t)width + (size_t)x;

    float L_c = in_L[idx];
    float c_c = in_c[idx];

    // Neighbor values
    float L_r = (x + 1 < width) ? in_L[idx + 1] : L_c;
    float L_l = (x > 0) ? in_L[idx - 1] : L_c;
    float L_d = (y + 1 < height) ? in_L[idx + width] : L_c;
    float L_u = (y > 0) ? in_L[idx - width] : L_c;

    float c_r = (x + 1 < width) ? in_c[idx + 1] : c_c;
    float c_l = (x > 0) ? in_c[idx - 1] : c_c;
    float c_d = (y + 1 < height) ? in_c[idx + width] : c_c;
    float c_u = (y > 0) ? in_c[idx - width] : c_c;

    // Harmonic / arithmetic mean conductivities on grid cell interfaces
    float c_east  = 0.5f * (c_c + c_r);
    float c_west  = 0.5f * (c_c + c_l);
    float c_south = 0.5f * (c_c + c_d);
    float c_north = 0.5f * (c_c + c_u);

    // Divergence: d/dx(c * dL/dx) + d/dy(c * dL/dy)
    float flux_x = c_east * (L_r - L_c) - c_west * (L_c - L_l);
    float flux_y = c_south * (L_d - L_c) - c_north * (L_c - L_u);

    out_L[idx] = L_c + tau_step * (flux_x + flux_y);
}
