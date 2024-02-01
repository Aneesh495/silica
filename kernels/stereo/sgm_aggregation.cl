#include "common/math.clh"
#include "common/memory.clh"

#define SGM_MAX_DISPARITY 64

// Horizontal Left-to-Right SGM Path Cost Aggregation
// Each thread processes one row across all disparities.
__kernel void kernel_sgm_aggregate_left_to_right(
    __global const uchar* cost_volume,
    __global ushort* out_path_cost,
    const int width,
    const int height,
    const int max_disparity,
    const ushort P1,
    const ushort P2) {
    
    int y = get_global_id(0);
    if (y >= height) return;

    ushort prev_L[SGM_MAX_DISPARITY];
    #pragma unroll
    for (int d = 0; d < SGM_MAX_DISPARITY; ++d) {
        prev_L[d] = 0;
    }

    size_t row_base = (size_t)y * (size_t)width * (size_t)max_disparity;

    for (int x = 0; x < width; ++x) {
        size_t pixel_base = row_base + (size_t)x * (size_t)max_disparity;

        ushort min_prev = 0xFFFF;
        for (int d = 0; d < max_disparity; ++d) {
            min_prev = min(min_prev, prev_L[d]);
        }

        ushort curr_L[SGM_MAX_DISPARITY];

        for (int d = 0; d < max_disparity; ++d) {
            uchar c = cost_volume[pixel_base + d];

            if (x == 0) {
                curr_L[d] = (ushort)c;
            } else {
                ushort l_same = prev_L[d];
                ushort l_prev = (d > 0) ? (prev_L[d - 1] + P1) : 0xFFFF;
                ushort l_next = (d + 1 < max_disparity) ? (prev_L[d + 1] + P1) : 0xFFFF;
                ushort l_other = min_prev + P2;

                ushort min_term = min(min(l_same, l_prev), min(l_next, l_other));
                curr_L[d] = (ushort)c + min_term - min_prev;
            }

            out_path_cost[pixel_base + d] = curr_L[d];
        }

        for (int d = 0; d < max_disparity; ++d) {
            prev_L[d] = curr_L[d];
        }
    }
}

// Horizontal Right-to-Left SGM Path Cost Aggregation
__kernel void kernel_sgm_aggregate_right_to_left(
    __global const uchar* cost_volume,
    __global ushort* in_out_total_cost,
    const int width,
    const int height,
    const int max_disparity,
    const ushort P1,
    const ushort P2) {
    
    int y = get_global_id(0);
    if (y >= height) return;

    ushort prev_L[SGM_MAX_DISPARITY];
    #pragma unroll
    for (int d = 0; d < SGM_MAX_DISPARITY; ++d) {
        prev_L[d] = 0;
    }

    size_t row_base = (size_t)y * (size_t)width * (size_t)max_disparity;

    for (int x = width - 1; x >= 0; --x) {
        size_t pixel_base = row_base + (size_t)x * (size_t)max_disparity;

        ushort min_prev = 0xFFFF;
        for (int d = 0; d < max_disparity; ++d) {
            min_prev = min(min_prev, prev_L[d]);
        }

        ushort curr_L[SGM_MAX_DISPARITY];

        for (int d = 0; d < max_disparity; ++d) {
            uchar c = cost_volume[pixel_base + d];

            if (x == width - 1) {
                curr_L[d] = (ushort)c;
            } else {
                ushort l_same = prev_L[d];
                ushort l_prev = (d > 0) ? (prev_L[d - 1] + P1) : 0xFFFF;
                ushort l_next = (d + 1 < max_disparity) ? (prev_L[d + 1] + P1) : 0xFFFF;
                ushort l_other = min_prev + P2;

                ushort min_term = min(min(l_same, l_prev), min(l_next, l_other));
                curr_L[d] = (ushort)c + min_term - min_prev;
            }

            in_out_total_cost[pixel_base + d] += curr_L[d];
        }

        for (int d = 0; d < max_disparity; ++d) {
            prev_L[d] = curr_L[d];
        }
    }
}
