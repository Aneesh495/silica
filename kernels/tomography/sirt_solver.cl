#include "common/math.clh"
#include "common/memory.clh"

// SIRT Step 1: Computes Projection Residual: r = measured_sinogram - forward_project(current_x)
__kernel void kernel_sirt_compute_residual(
    __global const float* measured_sinogram,
    __global const float* current_simulated_sinogram,
    __global float* out_residual,
    const uint total_sinogram_bins) {
    
    uint gid = get_global_id(0);
    if (gid >= total_sinogram_bins) return;

    out_residual[gid] = measured_sinogram[gid] - current_simulated_sinogram[gid];
}

// SIRT Step 2: Back-projects Normalized Residual and Updates Image Grid
// x^{k+1} = x^k + alpha * C * A^T * R * (b - A*x^k)
__kernel void kernel_sirt_update_image_grid(
    __global float* image_grid,
    __global const float* residual,
    const int img_size,
    const int num_angles,
    const int num_detectors,
    const float pixel_size,
    const float detector_spacing,
    const float relaxation_param) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= img_size || y >= img_size) return;

    float half_grid = 0.5f * (float)(img_size - 1);
    float px = ((float)x - half_grid) * pixel_size;
    float py = ((float)y - half_grid) * pixel_size;

    float d_theta = PI_F / (float)num_angles;
    float half_detectors = 0.5f * (float)(num_detectors - 1);
    float inv_det_spacing = 1.0f / detector_spacing;

    float backprojected_error = 0.0f;

    for (int a = 0; a < num_angles; ++a) {
        float theta = (float)a * d_theta;
        float cos_t = cos(theta);
        float sin_t = sin(theta);

        float s = px * cos_t + py * sin_t;
        float det_pos = s * inv_det_spacing + half_detectors;

        int det_0 = (int)floor(det_pos);
        int det_1 = det_0 + 1;
        float frac = det_pos - (float)det_0;

        if (det_0 >= 0 && det_1 < num_detectors) {
            size_t row_base = (size_t)a * (size_t)num_detectors;
            float r0 = residual[row_base + det_0];
            float r1 = residual[row_base + det_1];
            backprojected_error += lerp_f(r0, r1, frac);
        }
    }

    size_t idx = (size_t)y * (size_t)img_size + (size_t)x;
    float current_val = image_grid[idx];
    float updated_val = current_val + relaxation_param * (backprojected_error / (float)num_angles);

    // Non-negativity physical constraint for X-ray attenuation coefficients
    image_grid[idx] = fmax(0.0f, updated_val);
}

// SIRT Step 3: Total Variation (TV) Regularization Pass on Image Grid
// Minimizes gradient norm |grad(x)| to suppress noise while preserving sharp boundaries
__kernel void kernel_sirt_tv_regularization(
    __global float* image_grid,
    const int img_size,
    const float tv_weight) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < 1 || x >= img_size - 1 || y < 1 || y >= img_size - 1) return;

    size_t idx = (size_t)y * (size_t)img_size + (size_t)x;
    float v00 = image_grid[idx];

    float dx_plus  = image_grid[idx + 1] - v00;
    float dx_minus = v00 - image_grid[idx - 1];
    float dy_plus  = image_grid[idx + img_size] - v00;
    float dy_minus = v00 - image_grid[idx - img_size];

    // Divergence of normalized gradient
    float norm_plus = sqrt(dx_plus * dx_plus + dy_plus * dy_plus + 1.0e-6f);
    float norm_minus = sqrt(dx_minus * dx_minus + dy_minus * dy_minus + 1.0e-6f);

    float div_tv = (dx_plus / norm_plus - dx_minus / norm_minus) +
                   (dy_plus / norm_plus - dy_minus / norm_minus);

    float regularized = v00 + tv_weight * div_tv;
    image_grid[idx] = fmax(0.0f, regularized);
}
