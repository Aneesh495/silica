#include "common/math.clh"
#include "common/memory.clh"

// Forward Radon Transform: generates 2D sinogram from phantom density grid
// Uses ray-marching line integration with bilinear texture/grid sampling.
__kernel void kernel_forward_radon_parallel(
    __global const float* image_grid,
    __global float* out_sinogram, // [num_angles x num_detectors]
    const int img_size,
    const int num_angles,
    const int num_detectors,
    const float pixel_size,
    const float detector_spacing) {
    
    int det_idx = get_global_id(0);
    int angle_idx = get_global_id(1);

    if (det_idx >= num_detectors || angle_idx >= num_angles) return;

    // Angle theta in radians
    float theta = (float)angle_idx * PI_F / (float)num_angles;
    float cos_t = cos(theta);
    float sin_t = sin(theta);

    // Distance s from origin to ray
    float s = ((float)det_idx - 0.5f * (float)(num_detectors - 1)) * detector_spacing;

    // Ray origin on the projection normal line
    float ox = s * cos_t;
    float oy = s * sin_t;

    // Ray direction perpendicular to normal line
    float dx = -sin_t;
    float dy =  cos_t;

    float half_grid = 0.5f * (float)img_size * pixel_size;
    float step = pixel_size * 0.5f; // Step size for trapezoidal integration
    float ray_length = sqrt(2.0f) * (float)img_size * pixel_size;

    float sum = 0.0f;
    int num_steps = (int)(ray_length / step);

    for (int i = -num_steps / 2; i <= num_steps / 2; ++i) {
        float t = (float)i * step;
        float rx = ox + t * dx;
        float ry = oy + t * dy;

        // Map world coordinate (rx, ry) to image pixel grid [0, img_size-1]
        float px = (rx + half_grid) / pixel_size;
        float py = (ry + half_grid) / pixel_size;

        if (px >= 0.0f && px < (float)(img_size - 1) &&
            py >= 0.0f && py < (float)(img_size - 1)) {
            
            int x0 = (int)floor(px);
            int y0 = (int)floor(py);
            float fx = px - (float)x0;
            float fy = py - (float)y0;

            float v00 = image_grid[(size_t)y0 * (size_t)img_size + (size_t)x0];
            float v10 = image_grid[(size_t)y0 * (size_t)img_size + (size_t)(x0 + 1)];
            float v01 = image_grid[(size_t)(y0 + 1) * (size_t)img_size + (size_t)x0];
            float v11 = image_grid[(size_t)(y0 + 1) * (size_t)img_size + (size_t)(x0 + 1)];

            sum += bilinear_interp_unit(v00, v10, v01, v11, fx, fy);
        }
    }

    size_t out_idx = (size_t)angle_idx * (size_t)num_detectors + (size_t)det_idx;
    out_sinogram[out_idx] = sum * step;
}
