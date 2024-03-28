#include "common/math.clh"
#include "common/memory.clh"

// Lens Undistortion Mapping: Generates Rectified Image from Distorted Sensor Data
// Brown-Conrady 5-parameter model (k1, k2, k3 radial; p1, p2 tangential)
__kernel void kernel_undistort_image(
    __global const float* distorted_image,
    __global float* out_rectified_image,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float k1,
    const float k2,
    const float p1,
    const float p2,
    const float k3) {
    
    int u = get_global_id(0);
    int v = get_global_id(1);

    if (u >= width || v >= height) return;

    size_t out_idx = (size_t)v * (size_t)width + (size_t)u;

    // Normalized rectified camera coordinates
    float x = ((float)u - cx) / fx;
    float y = ((float)v - cy) / fy;

    float r2 = x * x + y * y;
    float r4 = r2 * r2;
    float r6 = r4 * r2;

    // Radial distortion factor
    float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;

    // Tangential distortion
    float x_dist = x * radial + (2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x));
    float y_dist = y * radial + (p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y);

    // Reproject to distorted pixel coordinates
    float u_src = fx * x_dist + cx;
    float v_src = fy * y_dist + cy;

    int u0 = (int)floor(u_src);
    int v0 = (int)floor(v_src);
    int u1 = u0 + 1;
    int v1 = v0 + 1;

    if (u0 >= 0 && u1 < width && v0 >= 0 && v1 < height) {
        float fu = u_src - (float)u0;
        float fv = v_src - (float)v0;

        float p00 = distorted_image[(size_t)v0 * (size_t)width + (size_t)u0];
        float p10 = distorted_image[(size_t)v0 * (size_t)width + (size_t)u1];
        float p01 = distorted_image[(size_t)v1 * (size_t)width + (size_t)u0];
        float p11 = distorted_image[(size_t)v1 * (size_t)width + (size_t)u1];

        out_rectified_image[out_idx] = bilinear_interp_unit(p00, p10, p01, p11, fu, fv);
    } else {
        out_rectified_image[out_idx] = 0.0f; // Outside sensor
    }
}
