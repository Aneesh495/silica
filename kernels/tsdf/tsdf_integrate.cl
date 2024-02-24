#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

// Truncated Signed Distance Field (TSDF) Volumetric Integration
// Updates distance and weight grids with new depth frame and camera pose.
__kernel void kernel_tsdf_integrate_frame(
    __global float* tsdf_grid,          // [depth x height x width] in [-1.0, 1.0]
    __global float* weight_grid,        // [depth x height x width]
    __global const float* depth_image,  // [img_h x img_w] in meters
    __global const float* rgb_image,    // [img_h x img_w x 3] in [0, 1]
    __global uint* color_grid,          // [depth x height x width] packed RGB
    __constant const float* c2w_matrix, // 4x4 camera-to-world matrix
    __constant const float* w2c_matrix, // 4x4 world-to-camera matrix
    const int vx,
    const int vy,
    const int vz,
    const float voxel_size,
    const int img_w,
    const int img_h,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float truncation_mu,
    const float max_weight) {
    
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if (ix >= vx || iy >= vy || iz >= vz) return;

    // Voxel world coordinates
    float3 p_world = (float3)(
        ((float)ix - 0.5f * (float)(vx - 1)) * voxel_size,
        ((float)iy - 0.5f * (float)(vy - 1)) * voxel_size,
        ((float)iz - 0.5f * (float)(vz - 1)) * voxel_size
    );

    // Transform world point into camera coordinate frame: p_cam = w2c * p_world
    float3 p_cam = mat4_transform_point(w2c_matrix, p_world);

    if (p_cam.z <= 0.1f) return; // Behind camera plane

    // Project to 2D image coordinates
    float2 uv = camera_project(p_cam, fx, fy, cx, cy);

    int u = (int)round(uv.x);
    int v = (int)round(uv.y);

    if (u < 0 || u >= img_w || v < 0 || v >= img_h) return;

    float raw_depth = depth_image[(size_t)v * (size_t)img_w + (size_t)u];
    if (raw_depth <= 0.05f || isnan(raw_depth)) return;

    // Signed distance eta: positive in front of surface, negative behind
    float eta = raw_depth - p_cam.z;

    // Truncation boundary condition: ignore points further behind surface than -mu
    if (eta <= -truncation_mu) return;

    float sdf = clamp_f(eta / truncation_mu, -1.0f, 1.0f);

    size_t vol_idx = (((size_t)iz * (size_t)vy + (size_t)iy) * (size_t)vx) + (size_t)ix;

    float current_tsdf = tsdf_grid[vol_idx];
    float current_w = weight_grid[vol_idx];

    // Weight inversely proportional to squared depth for sensor noise model
    float add_w = 1.0f;
    float next_w = fmin(max_weight, current_w + add_w);
    float next_tsdf = (current_tsdf * current_w + sdf * add_w) / next_w;

    tsdf_grid[vol_idx] = next_tsdf;
    weight_grid[vol_idx] = next_w;

    // Update color if RGB provided
    if (rgb_image != 0 && color_grid != 0 && fabs(sdf) < 0.2f) {
        size_t c_idx = ((size_t)v * (size_t)img_w + (size_t)u) * 3;
        uchar r = (uchar)clamp_f(rgb_image[c_idx + 0] * 255.0f, 0.0f, 255.0f);
        uchar g = (uchar)clamp_f(rgb_image[c_idx + 1] * 255.0f, 0.0f, 255.0f);
        uchar b = (uchar)clamp_f(rgb_image[c_idx + 2] * 255.0f, 0.0f, 255.0f);
        color_grid[vol_idx] = ((uint)r << 16) | ((uint)g << 8) | (uint)b;
    }
}
