#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

#define HASH_PRIME_X 73856093
#define HASH_PRIME_Y 19349663
#define HASH_PRIME_Z 83492791

#define BUCKET_ENTRY_COUNT 8
#define BLOCK_SIZE_VOXELS 8 // 8x8x8 = 512 voxels per block
#define BLOCK_TOTAL_VOXELS (BLOCK_SIZE_VOXELS * BLOCK_SIZE_VOXELS * BLOCK_SIZE_VOXELS)

typedef struct {
    int3 block_coord; // Spatial coordinate (bx, by, bz) in units of blocks
    int block_index;  // Index into block voxel pool (-1 if empty)
} HashEntry;

// Computes 3D spatial hash index
inline uint compute_spatial_hash(int3 coord, uint num_buckets) {
    uint h = (uint)((coord.x * HASH_PRIME_X) ^ (coord.y * HASH_PRIME_Y) ^ (coord.z * HASH_PRIME_Z));
    return h % num_buckets;
}

// Allocates Voxel Blocks in Sparse Hash Table along Camera Frustum
__kernel void kernel_voxel_hash_frustum_alloc(
    __global HashEntry* hash_table,             // [num_buckets x BUCKET_ENTRY_COUNT]
    __global volatile int* free_block_counter,  // Points to next free block in pool
    __global const float* depth_image,
    __constant const float* c2w_matrix,
    const int img_w,
    const int img_h,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float block_world_size,
    const float truncation_mu,
    const uint num_buckets,
    const int max_blocks) {
    
    int u = get_global_id(0);
    int v = get_global_id(1);

    if (u >= img_w || v >= img_h) return;

    // Subsample image by 4x for allocation efficiency
    if ((u & 3) != 0 || (v & 3) != 0) return;

    float depth = depth_image[(size_t)v * (size_t)img_w + (size_t)u];
    if (depth <= 0.1f || isnan(depth)) return;

    // Ray bounds along depth: [depth - mu, depth + mu]
    float d_start = fmax(0.1f, depth - truncation_mu);
    float d_end   = depth + truncation_mu;

    for (float d = d_start; d <= d_end; d += block_world_size * 0.5f) {
        float3 p_cam = camera_backproject((float)u, (float)v, d, fx, fy, cx, cy);
        float3 p_world = mat4_transform_point(c2w_matrix, p_cam);

        // Compute 3D block coordinate
        int3 block_coord = (int3)(
            (int)floor(p_world.x / block_world_size),
            (int)floor(p_world.y / block_world_size),
            (int)floor(p_world.z / block_world_size)
        );

        uint bucket = compute_spatial_hash(block_coord, num_buckets);
        size_t bucket_base = (size_t)bucket * BUCKET_ENTRY_COUNT;

        // Linear probing within bucket slots
        bool exists = false;
        int empty_slot = -1;

        for (int s = 0; s < BUCKET_ENTRY_COUNT; ++s) {
            __global HashEntry* entry = hash_table + bucket_base + s;
            if (entry->block_index >= 0) {
                if (entry->block_coord.x == block_coord.x &&
                    entry->block_coord.y == block_coord.y &&
                    entry->block_coord.z == block_coord.z) {
                    exists = true;
                    break;
                }
            } else if (empty_slot < 0) {
                empty_slot = s;
            }
        }

        if (!exists && empty_slot >= 0) {
            __global HashEntry* entry = hash_table + bucket_base + empty_slot;
            int old_idx = atomic_cmpxchg(&entry->block_index, -1, -2); // Claim lock
            if (old_idx == -1) {
                int new_block = atomic_inc(free_block_counter);
                if (new_block < max_blocks) {
                    entry->block_coord = block_coord;
                    entry->block_index = new_block;
                } else {
                    entry->block_index = -1; // Out of pool memory
                }
            }
        }
    }
}
