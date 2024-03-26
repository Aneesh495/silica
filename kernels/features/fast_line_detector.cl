#include "common/math.clh"
#include "common/memory.clh"

// Hough Transform Accumulator for 2D Straight Line Detection: rho = x*cos(theta) + y*sin(theta)
__kernel void kernel_hough_line_transform(
    __global const uchar* edge_image,
    __global uint* out_hough_accumulator, // [num_thetas x num_rhos]
    const int width,
    const int height,
    const int num_thetas,
    const int num_rhos,
    const float max_rho) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    if (edge_image[(size_t)y * (size_t)width + (size_t)x] == 0) return; // Not an edge

    float half_rhos = 0.5f * (float)(num_rhos - 1);
    float d_theta = PI_F / (float)num_thetas;

    for (int t = 0; t < num_thetas; ++t) {
        float theta = (float)t * d_theta;
        float rho = (float)x * cos(theta) + (float)y * sin(theta);

        // Map rho in [-max_rho, max_rho] to [0, num_rhos - 1]
        int r_idx = (int)round((rho / max_rho) * half_rhos + half_rhos);

        if (r_idx >= 0 && r_idx < num_rhos) {
            size_t accum_idx = (size_t)t * (size_t)num_rhos + (size_t)r_idx;
            atomic_inc(&out_hough_accumulator[accum_idx]);
        }
    }
}
