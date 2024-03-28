#include "common/math.clh"
#include "common/memory.clh"

#define SGM_MAX_DISPARITY 64

// Diagonal Path 1: Top-Left to Bottom-Right (+1, +1) Cost Aggregation
// Thread id corresponds to diagonal ray index: d_id = x - y + height - 1
__kernel void kernel_sgm_aggregate_diagonal_tl_br(
    __global const uchar* cost_volume,
    __global ushort* in_out_total_cost,
    const int width,
    const int height,
    const int max_disparity,
    const ushort P1,
    const ushort P2) {
    
    int diag_id = get_global_id(0);
    int total_diags = width + height - 1;
    if (diag_id >= total_diags) return;

    // Start coordinates for this diagonal line
    int start_x = (diag_id < height) ? 0 : (diag_id - height + 1);
    int start_y = (diag_id < height) ? (height - 1 - diag_id) : 0;

    ushort prev_L[SGM_MAX_DISPARITY];
    #pragma unroll
    for (int d = 0; d < SGM_MAX_DISPARITY; ++d) {
        prev_L[d] = 0;
    }

    int cx = start_x;
    int cy = start_y;
    bool first_step = true;

    while (cx < width && cy < height) {
        size_t pixel_base = (((size_t)cy * (size_t)width) + (size_t)cx) * (size_t)max_disparity;

        ushort min_prev = 0xFFFF;
        for (int d = 0; d < max_disparity; ++d) {
            min_prev = min(min_prev, prev_L[d]);
        }

        ushort curr_L[SGM_MAX_DISPARITY];

        for (int d = 0; d < max_disparity; ++d) {
            uchar c = cost_volume[pixel_base + d];

            if (first_step) {
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

        first_step = false;
        cx++;
        cy++;
    }
}

// Diagonal Path 2: Top-Right to Bottom-Left (-1, +1) Cost Aggregation
__kernel void kernel_sgm_aggregate_diagonal_tr_bl(
    __global const uchar* cost_volume,
    __global ushort* in_out_total_cost,
    const int width,
    const int height,
    const int max_disparity,
    const ushort P1,
    const ushort P2) {
    
    int diag_id = get_global_id(0);
    int total_diags = width + height - 1;
    if (diag_id >= total_diags) return;

    int start_x = (diag_id < width) ? diag_id : (width - 1);
    int start_y = (diag_id < width) ? 0 : (diag_id - width + 1);

    ushort prev_L[SGM_MAX_DISPARITY];
    #pragma unroll
    for (int d = 0; d < SGM_MAX_DISPARITY; ++d) {
        prev_L[d] = 0;
    }

    int cx = start_x;
    int cy = start_y;
    bool first_step = true;

    while (cx >= 0 && cy < height) {
        size_t pixel_base = (((size_t)cy * (size_t)width) + (size_t)cx) * (size_t)max_disparity;

        ushort min_prev = 0xFFFF;
        for (int d = 0; d < max_disparity; ++d) {
            min_prev = min(min_prev, prev_L[d]);
        }

        ushort curr_L[SGM_MAX_DISPARITY];

        for (int d = 0; d < max_disparity; ++d) {
            uchar c = cost_volume[pixel_base + d];

            if (first_step) {
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

        first_step = false;
        cx--;
        cy++;
    }
}
