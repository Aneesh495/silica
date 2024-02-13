#include "common/math.clh"
#include "common/memory.clh"

// 2D Filtered Back-Projection (FBP) Reconstruction
// Reconstructs pixel grid f(x, y) from pre-filtered sinogram projections.
__kernel void kernel_filtered_backprojection_2d(
    __global const float* filtered_sinogram, // [num_angles x num_detectors]
    __global float* out_reconstruction,      // [img_size x img_size]
    const int img_size,
    const int num_angles,
    const int num_detectors,
    const float pixel_size,
    const float detector_spacing) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= img_size || y >= img_size) return;

    // Physical coordinates centered at (0, 0)
    float half_grid = 0.5f * (float)(img_size - 1);
    float px = ((float)x - half_grid) * pixel_size;
    float py = ((float)y - half_grid) * pixel_size;

    float d_theta = PI_F / (float)num_angles;
    float half_detectors = 0.5f * (float)(num_detectors - 1);
    float inv_det_spacing = 1.0f / detector_spacing;

    float sum = 0.0f;

    for (int a = 0; a < num_angles; ++a) {
        float theta = (float)a * d_theta;
        float cos_t = cos(theta);
        float sin_t = sin(theta);

        // Ray distance s from center of detector
        float s = px * cos_t + py * sin_t;

        // Detector continuous index
        float det_pos = s * inv_det_spacing + half_detectors;

        int det_0 = (int)floor(det_pos);
        int det_1 = det_0 + 1;
        float frac = det_pos - (float)det_0;

        if (det_0 >= 0 && det_1 < num_detectors) {
            size_t row_base = (size_t)a * (size_t)num_detectors;
            float v0 = filtered_sinogram[row_base + det_0];
            float v1 = filtered_sinogram[row_base + det_1];
            sum += lerp_f(v0, v1, frac);
        } else if (det_0 >= 0 && det_0 < num_detectors) {
            size_t row_base = (size_t)a * (size_t)num_detectors;
            sum += filtered_sinogram[row_base + det_0];
        }
    }

    size_t out_idx = (size_t)y * (size_t)img_size + (size_t)x;
    out_reconstruction[out_idx] = sum * d_theta;
}

// Circular Reconstruction Mask: clears out-of-field-of-view artifact corners
__kernel void kernel_apply_circular_mask(
    __global float* grid,
    const int img_size,
    const float radius) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= img_size || y >= img_size) return;

    float half_grid = 0.5f * (float)(img_size - 1);
    float dx = (float)x - half_grid;
    float dy = (float)y - half_grid;

    if (dx * dx + dy * dy > radius * radius) {
        grid[(size_t)y * (size_t)img_size + (size_t)x] = 0.0f;
    }
}
