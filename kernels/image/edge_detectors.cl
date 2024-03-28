#include "common/math.clh"
#include "common/memory.clh"

#define EDGE_SECTOR_0   0 // Horizontal
#define EDGE_SECTOR_45  1 // Diagonal positive
#define EDGE_SECTOR_90  2 // Vertical
#define EDGE_SECTOR_135 3 // Diagonal negative

// Canny Stage 1: Non-Maximum Suppression along Gradient Direction
__kernel void kernel_canny_non_max_suppression(
    __global const float* grad_mag,
    __global const float* grad_angle, // In radians
    __global float* out_nms,
    const int width,
    const int height) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        if (x < width && y < height) out_nms[(size_t)y * (size_t)width + (size_t)x] = 0.0f;
        return;
    }

    size_t idx = (size_t)y * (size_t)width + (size_t)x;
    float mag = grad_mag[idx];
    float angle = grad_angle[idx] * RAD_TO_DEG_F;
    if (angle < 0.0f) angle += 180.0f;

    float neighbor_a = 0.0f;
    float neighbor_b = 0.0f;

    // Sector classification (4 direction sectors: 0, 45, 90, 135 deg)
    if ((angle >= 0.0f && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f)) {
        // Sector 0: East - West
        neighbor_a = grad_mag[idx + 1];
        neighbor_b = grad_mag[idx - 1];
    } else if (angle >= 22.5f && angle < 67.5f) {
        // Sector 45: North-East - South-West
        neighbor_a = grad_mag[(size_t)(y - 1) * (size_t)width + (size_t)(x + 1)];
        neighbor_b = grad_mag[(size_t)(y + 1) * (size_t)width + (size_t)(x - 1)];
    } else if (angle >= 67.5f && angle < 112.5f) {
        // Sector 90: North - South
        neighbor_a = grad_mag[(size_t)(y - 1) * (size_t)width + (size_t)x];
        neighbor_b = grad_mag[(size_t)(y + 1) * (size_t)width + (size_t)x];
    } else {
        // Sector 135: North-West - South-East
        neighbor_a = grad_mag[(size_t)(y - 1) * (size_t)width + (size_t)(x - 1)];
        neighbor_b = grad_mag[(size_t)(y + 1) * (size_t)width + (size_t)(x + 1)];
    }

    if (mag >= neighbor_a && mag >= neighbor_b) {
        out_nms[idx] = mag;
    } else {
        out_nms[idx] = 0.0f;
    }
}

// Canny Stage 2: Dual Thresholding and Hysteresis Linkage
__kernel void kernel_canny_hysteresis(
    __global const float* in_nms,
    __global uchar* out_edges,
    const int width,
    const int height,
    const float low_threshold,
    const float high_threshold) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        if (x < width && y < height) out_edges[(size_t)y * (size_t)width + (size_t)x] = 0;
        return;
    }

    size_t idx = (size_t)y * (size_t)width + (size_t)x;
    float val = in_nms[idx];

    if (val >= high_threshold) {
        out_edges[idx] = 255; // Strong edge
    } else if (val >= low_threshold) {
        // Weak edge: check 8-connected neighbors for strong edge
        bool has_strong_neighbor = false;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                if (in_nms[(size_t)(y + dy) * (size_t)width + (size_t)(x + dx)] >= high_threshold) {
                    has_strong_neighbor = true;
                    break;
                }
            }
            if (has_strong_neighbor) break;
        }
        out_edges[idx] = has_strong_neighbor ? 255 : 0;
    } else {
        out_edges[idx] = 0;
    }
}
