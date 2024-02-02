#include "common/math.clh"
#include "common/memory.clh"

// Winner-Take-All (WTA) Disparity Selection with Quadratic Parabola Subpixel Refinement
__kernel void kernel_disparity_wta_subpixel(
    __global const ushort* total_cost_volume,
    __global float* out_disparity,
    const int width,
    const int height,
    const int max_disparity,
    const float uniqueness_ratio) { // e.g. 0.95f
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t pixel_base = ((size_t)y * (size_t)width + (size_t)x) * (size_t)max_disparity;

    ushort best_cost = 0xFFFF;
    ushort second_cost = 0xFFFF;
    int best_d = 0;

    for (int d = 0; d < max_disparity; ++d) {
        ushort c = total_cost_volume[pixel_base + d];
        if (c < best_cost) {
            second_cost = best_cost;
            best_cost = c;
            best_d = d;
        } else if (c < second_cost) {
            second_cost = c;
        }
    }

    // Uniqueness check: reject ambiguous matches
    if ((float)best_cost > uniqueness_ratio * (float)second_cost) {
        out_disparity[(size_t)y * (size_t)width + (size_t)x] = -1.0f; // Invalid
        return;
    }

    float sub_d = (float)best_d;

    // Subpixel parabola fit if inside interior disparities
    if (best_d > 0 && best_d < max_disparity - 1) {
        float c_prev = (float)total_cost_volume[pixel_base + best_d - 1];
        float c_curr = (float)best_cost;
        float c_next = (float)total_cost_volume[pixel_base + best_d + 1];

        float denom = 2.0f * (c_prev + c_next - 2.0f * c_curr);
        if (fabs(denom) > 1.0e-5f) {
            float delta = (c_prev - c_next) / denom;
            if (fabs(delta) <= 1.0f) {
                sub_d += delta;
            }
        }
    }

    out_disparity[(size_t)y * (size_t)width + (size_t)x] = sub_d;
}

// Left-Right Consistency Check for Occlusion Detection
__kernel void kernel_disparity_left_right_check(
    __global const float* disp_left,
    __global const float* disp_right,
    __global float* out_filtered_disp,
    const int width,
    const int height,
    const float lrc_threshold) { // typically 1.0f px
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t idx = (size_t)y * (size_t)width + (size_t)x;
    float dl = disp_left[idx];

    if (dl < 0.0f) {
        out_filtered_disp[idx] = -1.0f;
        return;
    }

    // Matching pixel in right view
    int match_xr = (int)round((float)x - dl);

    if (match_xr < 0 || match_xr >= width) {
        out_filtered_disp[idx] = -1.0f; // Out of bounds
        return;
    }

    float dr = disp_right[(size_t)y * (size_t)width + (size_t)match_xr];

    if (fabs(dl - dr) > lrc_threshold) {
        out_filtered_disp[idx] = -1.0f; // Occlusion / mismatch
    } else {
        out_filtered_disp[idx] = dl;
    }
}

// Horizontal Left-to-Right Hole Filling for Occluded Regions
__kernel void kernel_disparity_fill_holes(
    __global const float* in_disp,
    __global float* out_disp,
    const int width,
    const int height) {
    
    int y = get_global_id(0);
    if (y >= height) return;

    size_t row_base = (size_t)y * (size_t)width;

    float last_valid = 0.0f;

    for (int x = 0; x < width; ++x) {
        float val = in_disp[row_base + x];
        if (val >= 0.0f) {
            last_valid = val;
            out_disp[row_base + x] = val;
        } else {
            out_disp[row_base + x] = last_valid; // Extrapolate from background
        }
    }
}
