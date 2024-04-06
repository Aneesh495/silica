#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

#define MPR_AXIAL    0
#define MPR_CORONAL  1
#define MPR_SAGITTAL 2
#define MPR_OBLIQUE  3

// Multi-Planar Reconstruction (MPR) Slicing Kernel with Window-Level Contrast
__kernel void kernel_mpr_slice_view(
    __global const float* volume_grid,
    __global float* out_slice_image,
    const int slice_w,
    const int slice_h,
    const int3 volume_dims,
    const float3 voxel_size,
    const int mpr_mode,
    const float slice_pos,
    const float window_width,
    const float window_level) {
    
    int u = get_global_id(0);
    int v = get_global_id(1);

    if (u >= slice_w || v >= slice_h) return;

    size_t out_idx = (size_t)v * (size_t)slice_w + (size_t)u;

    int3 dims = volume_dims;
    float3 p_voxel = (float3)(0.0f, 0.0f, 0.0f);

    if (mpr_mode == MPR_AXIAL) {
        // X = u, Y = v, Z = slice_pos
        p_voxel.x = (float)u * (float)dims.x / (float)slice_w;
        p_voxel.y = (float)v * (float)dims.y / (float)slice_h;
        p_voxel.z = clamp_f(slice_pos, 0.0f, (float)(dims.z - 1));
    } else if (mpr_mode == MPR_CORONAL) {
        // X = u, Z = v, Y = slice_pos
        p_voxel.x = (float)u * (float)dims.x / (float)slice_w;
        p_voxel.z = (float)v * (float)dims.z / (float)slice_h;
        p_voxel.y = clamp_f(slice_pos, 0.0f, (float)(dims.y - 1));
    } else { // MPR_SAGITTAL
        // Y = u, Z = v, X = slice_pos
        p_voxel.y = (float)u * (float)dims.y / (float)slice_w;
        p_voxel.z = (float)v * (float)dims.z / (float)slice_h;
        p_voxel.x = clamp_f(slice_pos, 0.0f, (float)(dims.x - 1));
    }

    // Sample volume density
    float density = read_volume_3d_safe(
        volume_grid,
        (int)p_voxel.x, (int)p_voxel.y, (int)p_voxel.z,
        dims.x, dims.y, dims.z,
        BORDER_CLAMP_EDGE, 0.0f);

    // Apply Window-Level linear mapping: [Level - Width/2, Level + Width/2] -> [0.0, 1.0]
    float low = window_level - 0.5f * window_width;
    float high = window_level + 0.5f * window_width;

    float normalized = clamp_f((density - low) / (high - low + EPSILON_F), 0.0f, 1.0f);

    out_slice_image[out_idx] = normalized;
}
