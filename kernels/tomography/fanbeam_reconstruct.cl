#include "common/math.clh"
#include "common/memory.clh"

// Parker Short-Scan Redundancy Weighting for Fan-Beam CT
// Smoothly scales redundant ray samples for scan angles between 0 and pi + 2*gamma_max.
inline float compute_parker_weight(float beta, float gamma, float total_scan_angle, float fan_angle) {
    float gamma_m = 0.5f * fan_angle;

    if (beta < 2.0f * (gamma_m - gamma)) {
        float arg = (0.25f * PI_F * beta) / (gamma_m - gamma + EPSILON_F);
        return sqr_f(sin(arg));
    } else if (beta <= total_scan_angle - 2.0f * (gamma_m + gamma)) {
        return 1.0f;
    } else if (beta <= total_scan_angle) {
        float arg = (0.25f * PI_F * (total_scan_angle - beta)) / (gamma_m + gamma + EPSILON_F);
        return sqr_f(sin(arg));
    }
    return 0.0f;
}

// Direct Equiangular Curved-Detector Fan-Beam Back-Projection
__kernel void kernel_fanbeam_equiangular_backproject(
    __global const float* filtered_sinogram, // [num_views x num_channels]
    __global float* out_reconstruction,
    const int img_size,
    const int num_views,
    const int num_channels,
    const float source_to_isocenter_dist, // D
    const float pixel_size,
    const float angular_channel_spacing,  // d_gamma
    const float total_scan_angle) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= img_size || y >= img_size) return;

    float half_grid = 0.5f * (float)(img_size - 1);
    float px = ((float)x - half_grid) * pixel_size;
    float py = ((float)y - half_grid) * pixel_size;

    float D = source_to_isocenter_dist;
    float d_beta = total_scan_angle / (float)num_views;
    float fan_angle = (float)num_channels * angular_channel_spacing;
    float half_channels = 0.5f * (float)(num_channels - 1);

    float sum = 0.0f;

    for (int v = 0; v < num_views; ++v) {
        float beta = (float)v * d_beta;
        float cos_b = cos(beta);
        float sin_b = sin(beta);

        // Distance from X-ray source to voxel
        float source_x = -D * sin_b;
        float source_y =  D * cos_b;

        float rx = px - source_x;
        float ry = py - source_y;

        float dist_sq = rx * rx + ry * ry;
        if (dist_sq < EPSILON_F) continue;

        // Angle gamma of ray connecting source to voxel
        // In local source coordinate frame
        float local_x =  rx * cos_b + ry * sin_b;
        float local_y = -rx * sin_b + ry * cos_b;

        float gamma = atan2(local_x, local_y);

        // Continuous detector channel index
        float chan_pos = gamma / angular_channel_spacing + half_channels;

        int ch0 = (int)floor(chan_pos);
        int ch1 = ch0 + 1;
        float fch = chan_pos - (float)ch0;

        if (ch0 >= 0 && ch1 < num_channels) {
            size_t row_base = (size_t)v * (size_t)num_channels;
            float val0 = filtered_sinogram[row_base + ch0];
            float val1 = filtered_sinogram[row_base + ch1];
            float val = lerp_f(val0, val1, fch);

            float parker_w = compute_parker_weight(beta, gamma, total_scan_angle, fan_angle);

            // 1/L^2 geometric weight
            float L2 = dist_sq;
            sum += (val * parker_w) / (L2 + EPSILON_F);
        }
    }

    size_t out_idx = (size_t)y * (size_t)img_size + (size_t)x;
    out_reconstruction[out_idx] = sum * (D * d_beta);
}
