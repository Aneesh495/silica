#include "common/math.clh"
#include "common/memory.clh"

// 3D Scale-Space Extrema Detection in Difference of Gaussians (DoG)
// Compares center pixel against 26 space-scale neighbors.
__kernel void kernel_dog_extrema_detect(
    __global const float* dog_prev,
    __global const float* dog_curr,
    __global const float* dog_next,
    __global float4* out_keypoints, // (x, y, scale, response)
    __global volatile uint* keypoint_counter,
    const int width,
    const int height,
    const float contrast_threshold,
    const float edge_ratio_threshold,
    const uint max_keypoints) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    size_t center_idx = (size_t)y * (size_t)width + (size_t)x;
    float val = dog_curr[center_idx];

    if (fabs(val) < contrast_threshold) return;

    bool is_max = val > 0.0f;
    bool is_min = val < 0.0f;

    // Check 26 neighbors: 8 in curr, 9 in prev, 9 in next
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            float v = dog_curr[(size_t)(y + dy) * (size_t)width + (size_t)(x + dx)];
            if (is_max && v >= val) is_max = false;
            if (is_min && v <= val) is_min = false;
        }
    }

    if (!is_max && !is_min) return;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            float vp = dog_prev[(size_t)(y + dy) * (size_t)width + (size_t)(x + dx)];
            float vn = dog_next[(size_t)(y + dy) * (size_t)width + (size_t)(x + dx)];
            if (is_max && (vp >= val || vn >= val)) is_max = false;
            if (is_min && (vp <= val || vn <= val)) is_min = false;
        }
    }

    if (!is_max && !is_min) return;

    // Reject strong edge responses using 2D spatial Hessian
    float dxx = dog_curr[center_idx + 1] + dog_curr[center_idx - 1] - 2.0f * val;
    float dyy = dog_curr[center_idx + width] + dog_curr[center_idx - width] - 2.0f * val;
    float dxy = 0.25f * (dog_curr[center_idx + width + 1] - dog_curr[center_idx + width - 1] -
                         dog_curr[center_idx - width + 1] + dog_curr[center_idx - width - 1]);

    float tr = dxx + dyy;
    float det = dxx * dyy - dxy * dxy;

    if (det <= 0.0f) return;

    float edge_score = (tr * tr) / det;
    float edge_limit = (edge_ratio_threshold + 1.0f) * (edge_ratio_threshold + 1.0f) / edge_ratio_threshold;
    if (edge_score > edge_limit) return;

    // Subpixel refinement via 1st order Taylor step
    float dx = 0.5f * (dog_curr[center_idx + 1] - dog_curr[center_idx - 1]);
    float dy = 0.5f * (dog_curr[center_idx + width] - dog_curr[center_idx - width]);

    float inv_det = 1.0f / det;
    float sub_x = -inv_det * (dyy * dx - dxy * dy);
    float sub_y = -inv_det * (-dxy * dx + dxx * dy);

    if (fabs(sub_x) > 1.5f || fabs(sub_y) > 1.5f) return;

    uint slot = atomic_inc(keypoint_counter);
    if (slot < max_keypoints) {
        out_keypoints[slot] = (float4)((float)x + sub_x, (float)y + sub_y, 1.0f, val);
    }
}
