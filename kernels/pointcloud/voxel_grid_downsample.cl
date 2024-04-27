#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"
#include "common/atomic.clh"

// Stage 1: Assigns 3D Point to Voxel Grid Cell and Accumulates Centroid
__kernel void kernel_voxel_grid_accumulate(
    __global const float4* in_points,
    __global float* out_voxel_centroids, // [total_voxels * 4]
    __global uint* out_voxel_counts,
    const float3 origin,
    const float voxel_size,
    const int3 grid_dims,
    const uint num_points) {
    
    uint gid = get_global_id(0);
    if (gid >= num_points) return;

    float4 pt = in_points[gid];
    float3 p = (float3)(pt.x, pt.y, pt.z);

    int vx = (int)floor((p.x - origin.x) / voxel_size);
    int vy = (int)floor((p.y - origin.y) / voxel_size);
    int vz = (int)floor((p.z - origin.z) / voxel_size);

    if (vx < 0 || vx >= grid_dims.x ||
        vy < 0 || vy >= grid_dims.y ||
        vz < 0 || vz >= grid_dims.z) {
        return;
    }

    size_t voxel_idx = (((size_t)vz * (size_t)grid_dims.y + (size_t)vy) * (size_t)grid_dims.x) + (size_t)vx;
    size_t base = voxel_idx * 4;

    // Atomically accumulate coordinates and count
    atomic_add_float_global(&out_voxel_centroids[base + 0], p.x);
    atomic_add_float_global(&out_voxel_centroids[base + 1], p.y);
    atomic_add_float_global(&out_voxel_centroids[base + 2], p.z);
    atomic_inc(&out_voxel_counts[voxel_idx]);
}

// Stage 2: Finalizes Centroids and Compacts Occupied Voxels
__kernel void kernel_voxel_grid_finalize(
    __global const float* voxel_centroids, // [total_voxels * 4]
    __global const uint* voxel_counts,
    __global float4* out_downsampled_points,
    __global volatile uint* out_point_counter,
    const uint total_voxels,
    const uint max_output_points) {
    
    uint gid = get_global_id(0);
    if (gid >= total_voxels) return;

    uint count = voxel_counts[gid];
    if (count == 0) return;

    size_t base = (size_t)gid * 4;
    float inv_count = 1.0f / (float)count;

    float3 centroid = (float3)(voxel_centroids[base + 0] * inv_count,
                               voxel_centroids[base + 1] * inv_count,
                               voxel_centroids[base + 2] * inv_count);

    uint slot = atomic_inc(out_point_counter);
    if (slot < max_output_points) {
        out_downsampled_points[slot] = (float4)(centroid.x, centroid.y, centroid.z, 1.0f);
    }
}
