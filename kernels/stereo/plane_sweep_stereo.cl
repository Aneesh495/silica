#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

// Multi-View Plane-Sweep Stereo: Warps target camera view onto depth plane d
// Evaluates photo-consistency against reference view
__kernel void kernel_plane_sweep_sample_view(
    __global const float* in_target_rgb,
    __global const float* in_ref_rgb,
    __global float* out_plane_cost,
    __constant const float* homography_matrix, // 3x3 homography H_i(d)
    const int width,
    const int height,
    const int plane_idx,
    const int total_planes) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t ref_base = ((size_t)y * (size_t)width + (size_t)x) * 3;
    float ref_r = in_ref_rgb[ref_base + 0];
    float ref_g = in_ref_rgb[ref_base + 1];
    float ref_b = in_ref_rgb[ref_base + 2];

    // Project reference (x, y, 1) through homography H into target view
    float u_target = homography_matrix[0] * (float)x + homography_matrix[1] * (float)y + homography_matrix[2];
    float v_target = homography_matrix[3] * (float)x + homography_matrix[4] * (float)y + homography_matrix[5];
    float w_target = homography_matrix[6] * (float)x + homography_matrix[7] * (float)y + homography_matrix[8];

    float cost = 1.0f; // Default high penalty if out of bounds or behind camera

    if (w_target > 1.0e-5f) {
        float inv_w = 1.0f / w_target;
        float tx = u_target * inv_w;
        float ty = v_target * inv_w;

        int x0 = (int)floor(tx);
        int y0 = (int)floor(ty);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        if (x0 >= 0 && x1 < width && y0 >= 0 && y1 < height) {
            float fx = tx - (float)x0;
            float fy = ty - (float)y0;

            // Bilinear interpolate R, G, B in target image
            float tgt_r = bilinear_interp_unit(
                in_target_rgb[((size_t)y0 * width + x0) * 3 + 0],
                in_target_rgb[((size_t)y0 * width + x1) * 3 + 0],
                in_target_rgb[((size_t)y1 * width + x0) * 3 + 0],
                in_target_rgb[((size_t)y1 * width + x1) * 3 + 0],
                fx, fy);

            float tgt_g = bilinear_interp_unit(
                in_target_rgb[((size_t)y0 * width + x0) * 3 + 1],
                in_target_rgb[((size_t)y0 * width + x1) * 3 + 1],
                in_target_rgb[((size_t)y1 * width + x0) * 3 + 1],
                in_target_rgb[((size_t)y1 * width + x1) * 3 + 1],
                fx, fy);

            float tgt_b = bilinear_interp_unit(
                in_target_rgb[((size_t)y0 * width + x0) * 3 + 2],
                in_target_rgb[((size_t)y0 * width + x1) * 3 + 2],
                in_target_rgb[((size_t)y1 * width + x0) * 3 + 2],
                in_target_rgb[((size_t)y1 * width + x1) * 3 + 2],
                fx, fy);

            // Photometric color difference (L1 norm)
            cost = 0.333333f * (fabs(ref_r - tgt_r) + fabs(ref_g - tgt_g) + fabs(ref_b - tgt_b));
        }
    }

    size_t cost_idx = (((size_t)plane_idx * (size_t)height + (size_t)y) * (size_t)width) + (size_t)x;
    out_plane_cost[cost_idx] = cost;
}
