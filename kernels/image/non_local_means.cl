#include "common/math.clh"
#include "common/memory.clh"

#define NLM_PATCH_RADIUS 3     // 7x7 patch
#define NLM_SEARCH_RADIUS 6    // 13x13 search window

// Non-Local Means (NLM) Patch-Based Denoising
// Compares similarity of pixel neighborhood patches rather than isolated pixel values.
__kernel void kernel_non_local_means_gray(
    __global const float* input,
    __global float* output,
    const int width,
    const int height,
    const float filter_h) {
    
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    if (gx >= width || gy >= height) return;

    float inv_h2 = 1.0f / (filter_h * filter_h + EPSILON_F);
    float sum_val = 0.0f;
    float sum_weight = 0.0f;
    float max_weight = 0.0f;

    // Iterate over search window
    for (int sy = -NLM_SEARCH_RADIUS; sy <= NLM_SEARCH_RADIUS; ++sy) {
        for (int sx = -NLM_SEARCH_RADIUS; sx <= NLM_SEARCH_RADIUS; ++sx) {
            if (sx == 0 && sy == 0) continue; // Handle self-patch separately

            int qx = handle_border_1d(gx + sx, width, BORDER_CLAMP_EDGE);
            int qy = handle_border_1d(gy + sy, height, BORDER_CLAMP_EDGE);

            // Compute L2 Euclidean patch distance between patch(p) and patch(q)
            float patch_dist_sq = 0.0f;
            float patch_weight_norm = 0.0f;

            for (int py = -NLM_PATCH_RADIUS; py <= NLM_PATCH_RADIUS; ++py) {
                for (int px = -NLM_PATCH_RADIUS; px <= NLM_PATCH_RADIUS; ++px) {
                    float val_p = read_image_2d_safe(input, gx + px, gy + py, width, height, BORDER_CLAMP_EDGE, 0.0f);
                    float val_q = read_image_2d_safe(input, qx + px, qy + py, width, height, BORDER_CLAMP_EDGE, 0.0f);
                    
                    float diff = val_p - val_q;
                    // Spatial decay inside patch
                    float spatial_w = 1.0f / (1.0f + (float)(px * px + py * py));
                    patch_dist_sq += diff * diff * spatial_w;
                    patch_weight_norm += spatial_w;
                }
            }

            patch_dist_sq /= (patch_weight_norm + EPSILON_F);

            float weight = exp(-patch_dist_sq * inv_h2);
            float q_val = read_image_2d_safe(input, qx, qy, width, height, BORDER_CLAMP_EDGE, 0.0f);

            sum_val += q_val * weight;
            sum_weight += weight;
            max_weight = fmax(max_weight, weight);
        }
    }

    // Add center pixel with maximum neighbor weight to avoid over-smoothing
    float center_val = read_image_2d_safe(input, gx, gy, width, height, BORDER_CLAMP_EDGE, 0.0f);
    sum_val += center_val * max_weight;
    sum_weight += max_weight;

    output[(size_t)gy * (size_t)width + (size_t)gx] = sum_val / (sum_weight + EPSILON_F);
}
