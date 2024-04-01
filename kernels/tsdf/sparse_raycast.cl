#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"
#include "voxel_hash_table.cl"

// Directly Raycasts Sparse Hash Table of Voxel Blocks
__kernel void kernel_sparse_hash_raycast(
    __global const HashEntry* hash_table,
    __global const float* block_voxel_tsdf, // [max_blocks x 512 voxels]
    __global float* out_depth_image,
    __constant const float* c2w_matrix,
    const int img_w,
    const int img_h,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float voxel_size,
    const float block_world_size,
    const uint num_buckets,
    const float max_ray_dist) {
    
    int u = get_global_id(0);
    int v = get_global_id(1);

    if (u >= img_w || v >= img_h) return;

    size_t out_pixel = (size_t)v * (size_t)img_w + (size_t)u;

    float3 ray_dir_cam = normalize((float3)(((float)u - cx) / fx, ((float)v - cy) / fy, 1.0f));
    float3 ray_orig_world = mat4_transform_point(c2w_matrix, (float3)(0.0f, 0.0f, 0.0f));
    float3 ray_dir_world  = mat4_transform_direction(c2w_matrix, ray_dir_cam);

    float step = voxel_size * 0.8f;
    float prev_sdf = 1.0f;
    float prev_t = 0.2f;
    bool hit = false;
    float hit_t = 0.0f;

    for (float t = 0.2f; t <= max_ray_dist; t += step) {
        float3 p_world = ray_orig_world + t * ray_dir_world;

        int3 block_coord = (int3)(
            (int)floor(p_world.x / block_world_size),
            (int)floor(p_world.y / block_world_size),
            (int)floor(p_world.z / block_world_size)
        );

        uint bucket = compute_spatial_hash(block_coord, num_buckets);
        size_t bucket_base = (size_t)bucket * BUCKET_ENTRY_COUNT;

        int found_block_idx = -1;
        for (int s = 0; s < BUCKET_ENTRY_COUNT; ++s) {
            __global const HashEntry* entry = hash_table + bucket_base + s;
            if (entry->block_index >= 0 &&
                entry->block_coord.x == block_coord.x &&
                entry->block_coord.y == block_coord.y &&
                entry->block_coord.z == block_coord.z) {
                found_block_idx = entry->block_index;
                break;
            }
        }

        if (found_block_idx >= 0) {
            // Find local voxel index within 8x8x8 block
            float3 local_p = (p_world - (float3)(
                (float)block_coord.x * block_world_size,
                (float)block_coord.y * block_world_size,
                (float)block_coord.z * block_world_size
            )) / voxel_size;

            int vx = clamp_i((int)floor(local_p.x), 0, BLOCK_SIZE_VOXELS - 1);
            int vy = clamp_i((int)floor(local_p.y), 0, BLOCK_SIZE_VOXELS - 1);
            int vz = clamp_i((int)floor(local_p.z), 0, BLOCK_SIZE_VOXELS - 1);

            size_t voxel_offset = ((size_t)found_block_idx * BLOCK_TOTAL_VOXELS) +
                                  (((size_t)vz * BLOCK_SIZE_VOXELS + vy) * BLOCK_SIZE_VOXELS + vx);

            float sdf = block_voxel_tsdf[voxel_offset];

            if (prev_sdf > 0.0f && sdf < 0.0f) {
                hit_t = prev_t + (prev_sdf / (prev_sdf - sdf + EPSILON_F)) * (t - prev_t);
                hit = true;
                break;
            }

            prev_sdf = sdf;
            prev_t = t;
        } else {
            prev_sdf = 1.0f;
            prev_t = t;
        }
    }

    out_depth_image[out_pixel] = hit ? hit_t : 0.0f;
}
