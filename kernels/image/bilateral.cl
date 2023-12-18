#include "common/math.clh"
#include "common/memory.clh"

#define BILATERAL_WG_X 16
#define BILATERAL_WG_Y 16
#define MAX_BILATERAL_RADIUS 8

// Edge-Preserving Bilateral Filter for Grayscale Images
// Caches local tile in LDS to minimize global bandwidth.
__kernel void kernel_bilateral_filter_gray(
    __global const float* input,
    __global float* output,
    const int width,
    const int height,
    const int radius,
    const float sigma_spatial,
    const float sigma_range) {
    
    __local float l_tile[BILATERAL_WG_Y + 2 * MAX_BILATERAL_RADIUS][BILATERAL_WG_X + 2 * MAX_BILATERAL_RADIUS + 1];

    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    int r = min(radius, MAX_BILATERAL_RADIUS);
    int tile_w = BILATERAL_WG_X + 2 * r;
    int tile_h = BILATERAL_WG_Y + 2 * r;
    int total_tile_pixels = tile_w * tile_h;

    int tid = ty * BILATERAL_WG_X + tx;
    int num_threads = BILATERAL_WG_X * BILATERAL_WG_Y;

    // Cooperative load of tile into local memory
    int base_x = get_group_id(0) * BILATERAL_WG_X - r;
    int base_y = get_group_id(1) * BILATERAL_WG_Y - r;

    for (int i = tid; i < total_tile_pixels; i += num_threads) {
        int lx = i % tile_w;
        int ly = i / tile_w;
        int sx = base_x + lx;
        int sy = base_y + ly;
        l_tile[ly][lx] = read_image_2d_safe(input, sx, sy, width, height, BORDER_CLAMP_EDGE, 0.0f);
    }

    sync_local();

    if (gx >= width || gy >= height) return;

    float center_val = l_tile[r + ty][r + tx];
    float spatial_coeff = -0.5f / (sigma_spatial * sigma_spatial);
    float range_coeff   = -0.5f / (sigma_range * sigma_range);

    float sum_val = 0.0f;
    float sum_weight = 0.0f;

    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            float neighbor_val = l_tile[r + ty + dy][r + tx + dx];
            float spatial_dist_sq = (float)(dx * dx + dy * dy);
            float range_diff = neighbor_val - center_val;
            float range_dist_sq = range_diff * range_diff;

            float w = exp(spatial_dist_sq * spatial_coeff + range_dist_sq * range_coeff);
            sum_val += neighbor_val * w;
            sum_weight += w;
        }
    }

    output[gy * width + gx] = sum_val / (sum_weight + EPSILON_F);
}

// Joint / Cross Bilateral Filter: Filters target depth/intensity using RGB guidance image
__kernel void kernel_joint_bilateral_filter(
    __global const float* target,
    __global const float* guide_rgb,
    __global float* output,
    const int width,
    const int height,
    const int radius,
    const float sigma_spatial,
    const float sigma_range) {
    
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    if (gx >= width || gy >= height) return;

    size_t guide_center_idx = ((size_t)gy * (size_t)width + (size_t)gx) * 3;
    float center_r = guide_rgb[guide_center_idx + 0];
    float center_g = guide_rgb[guide_center_idx + 1];
    float center_b = guide_rgb[guide_center_idx + 2];

    float spatial_coeff = -0.5f / (sigma_spatial * sigma_spatial);
    float range_coeff   = -0.5f / (sigma_range * sigma_range);

    float sum_val = 0.0f;
    float sum_weight = 0.0f;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int sx = handle_border_1d(gx + dx, width, BORDER_CLAMP_EDGE);
            int sy = handle_border_1d(gy + dy, height, BORDER_CLAMP_EDGE);

            size_t g_idx = ((size_t)sy * (size_t)width + (size_t)sx) * 3;
            float nr = guide_rgb[g_idx + 0];
            float ng = guide_rgb[g_idx + 1];
            float nb = guide_rgb[g_idx + 2];

            float dr = nr - center_r;
            float dg = ng - center_g;
            float db = nb - center_b;
            float range_dist_sq = dr * dr + dg * dg + db * db;

            float spatial_dist_sq = (float)(dx * dx + dy * dy);
            float w = exp(spatial_dist_sq * spatial_coeff + range_dist_sq * range_coeff);

            float target_val = target[(size_t)sy * (size_t)width + (size_t)sx];
            sum_val += target_val * w;
            sum_weight += w;
        }
    }

    output[(size_t)gy * (size_t)width + (size_t)gx] = sum_val / (sum_weight + EPSILON_F);
}
