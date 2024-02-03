#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

// Backprojects Disparity Map to Metric 3D Depth Map (Z in meters)
__kernel void kernel_disparity_to_depth(
    __global const float* disparity,
    __global float* out_depth,
    const int width,
    const int height,
    const float focal_length,
    const float baseline,
    const float min_depth,
    const float max_depth) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t idx = (size_t)y * (size_t)width + (size_t)x;
    float d = disparity[idx];

    if (d <= 0.0f) {
        out_depth[idx] = 0.0f; // Invalid
        return;
    }

    float z = (focal_length * baseline) / d;
    if (z < min_depth || z > max_depth || isnan(z) || isinf(z)) {
        out_depth[idx] = 0.0f;
    } else {
        out_depth[idx] = z;
    }
}

// Re-projects Depth Map and RGB to 3D Point Cloud: (X, Y, Z, RGB)
__kernel void kernel_reproject_to_pointcloud(
    __global const float* depth,
    __global const float* rgb,
    __global float4* out_points, // (x, y, z, packed_color)
    __global volatile uint* valid_point_counter,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint max_points) {
    
    int u = get_global_id(0);
    int v = get_global_id(1);

    if (u >= width || v >= height) return;

    size_t idx = (size_t)v * (size_t)width + (size_t)u;
    float z = depth[idx];

    if (z <= 0.0f) return;

    float3 pt = camera_backproject((float)u, (float)v, z, fx, fy, cx, cy);

    // Pack RGB color if provided
    float packed_rgb = 0.0f;
    if (rgb != 0) {
        size_t c_idx = idx * 3;
        uchar r = (uchar)clamp_f(rgb[c_idx + 0] * 255.0f, 0.0f, 255.0f);
        uchar g = (uchar)clamp_f(rgb[c_idx + 1] * 255.0f, 0.0f, 255.0f);
        uchar b = (uchar)clamp_f(rgb[c_idx + 2] * 255.0f, 0.0f, 255.0f);
        uint color_u32 = ((uint)r << 16) | ((uint)g << 8) | (uint)b;
        packed_rgb = as_float(color_u32);
    }

    uint slot = atomic_inc(valid_point_counter);
    if (slot < max_points) {
        out_points[slot] = (float4)(pt.x, pt.y, pt.z, packed_rgb);
    }
}
