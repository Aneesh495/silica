#include "common/math.clh"
#include "common/memory.clh"

// 16 pixels on Bresenham circle of radius 3 centered at (x, y)
// Offsets: (dx, dy)
__constant int2 FAST_CIRCLE_OFFSETS[16] = {
    (int2)( 0, -3), (int2)( 1, -3), (int2)( 2, -2), (int2)( 3, -1),
    (int2)( 3,  0), (int2)( 3,  1), (int2)( 2,  2), (int2)( 1,  3),
    (int2)( 0,  3), (int2)(-1,  3), (int2)(-2,  2), (int2)(-3,  1),
    (int2)(-3,  0), (int2)(-3, -1), (int2)(-2, -2), (int2)(-1, -3)
};

// FAST-9 Corner Detection with Early Rejection
__kernel void kernel_fast9_detect(
    __global const float* in_gray,
    __global float* out_corner_score,
    const int width,
    const int height,
    const float threshold) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < 3 || x >= width - 3 || y < 3 || y >= height - 3) {
        if (x < width && y < height) {
            out_corner_score[(size_t)y * (size_t)width + (size_t)x] = 0.0f;
        }
        return;
    }

    size_t center_idx = (size_t)y * (size_t)width + (size_t)x;
    float p0 = in_gray[center_idx];

    // Early rejection test: check pixels 0, 4, 8, 12 (North, East, South, West)
    float p_north = in_gray[(size_t)(y - 3) * (size_t)width + (size_t)x];
    float p_east  = in_gray[(size_t)y * (size_t)width + (size_t)(x + 3)];
    float p_south = in_gray[(size_t)(y + 3) * (size_t)width + (size_t)x];
    float p_west  = in_gray[(size_t)y * (size_t)width + (size_t)(x - 3)];

    int bright_count = (p_north > p0 + threshold) +
                       (p_east  > p0 + threshold) +
                       (p_south > p0 + threshold) +
                       (p_west  > p0 + threshold);

    int dark_count = (p_north < p0 - threshold) +
                     (p_east  < p0 - threshold) +
                     (p_south < p0 - threshold) +
                     (p_west  < p0 - threshold);

    if (bright_count < 2 && dark_count < 2) {
        out_corner_score[center_idx] = 0.0f;
        return;
    }

    // Read full 16 circle pixels
    float circle[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int cx = x + FAST_CIRCLE_OFFSETS[i].x;
        int cy = y + FAST_CIRCLE_OFFSETS[i].y;
        circle[i] = in_gray[(size_t)cy * (size_t)width + (size_t)cx];
    }

    // Check for 9 contiguous arc pixels
    bool is_corner = false;
    float score = 0.0f;

    // Test brighter arc
    for (int start = 0; start < 16; ++start) {
        bool all_bright = true;
        float current_score = 0.0f;
        for (int k = 0; k < 9; ++k) {
            int idx = (start + k) & 15;
            float diff = circle[idx] - p0;
            if (diff <= threshold) {
                all_bright = false;
                break;
            }
            current_score += diff;
        }
        if (all_bright) {
            is_corner = true;
            score = fmax(score, current_score);
        }
    }

    // Test darker arc
    if (!is_corner) {
        for (int start = 0; start < 16; ++start) {
            bool all_dark = true;
            float current_score = 0.0f;
            for (int k = 0; k < 9; ++k) {
                int idx = (start + k) & 15;
                float diff = p0 - circle[idx];
                if (diff <= threshold) {
                    all_dark = false;
                    break;
                }
                current_score += diff;
            }
            if (all_dark) {
                is_corner = true;
                score = fmax(score, current_score);
            }
        }
    }

    out_corner_score[center_idx] = is_corner ? score : 0.0f;
}

// 3x3 Non-Maximum Suppression (NMS) for corner scores
__kernel void kernel_corner_non_max_suppression(
    __global const float* in_scores,
    __global float* out_corners,
    const int width,
    const int height) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        if (x < width && y < height) {
            out_corners[(size_t)y * (size_t)width + (size_t)x] = 0.0f;
        }
        return;
    }

    size_t center_idx = (size_t)y * (size_t)width + (size_t)x;
    float center_score = in_scores[center_idx];

    if (center_score <= 0.0f) {
        out_corners[center_idx] = 0.0f;
        return;
    }

    // Compare with 8 neighbors
    bool is_max = true;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            float neighbor_score = in_scores[(size_t)(y + dy) * (size_t)width + (size_t)(x + dx)];
            if (neighbor_score >= center_score) {
                is_max = false;
                break;
            }
        }
        if (!is_max) break;
    }

    out_corners[center_idx] = is_max ? center_score : 0.0f;
}
