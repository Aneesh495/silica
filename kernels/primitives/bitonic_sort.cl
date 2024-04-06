#include "common/math.clh"
#include "common/memory.clh"

#define BITONIC_WG_SIZE 256
#define BITONIC_ITEMS_PER_WG (BITONIC_WG_SIZE * 2) // 512 items

// Local Work-Group Bitonic Sort for Blocks of 512 elements
__kernel void kernel_bitonic_sort_local(
    __global uint* keys,
    __global uint* values,
    const int dir) { // 1 = ascending, 0 = descending
    
    __local uint l_keys[BITONIC_ITEMS_PER_WG];
    __local uint l_vals[BITONIC_ITEMS_PER_WG];

    uint tid = get_local_id(0);
    uint gid = get_group_id(0);
    uint base_idx = gid * BITONIC_ITEMS_PER_WG;

    l_keys[2 * tid]     = keys[base_idx + 2 * tid];
    l_keys[2 * tid + 1] = keys[base_idx + 2 * tid + 1];

    if (values != 0) {
        l_vals[2 * tid]     = values[base_idx + 2 * tid];
        l_vals[2 * tid + 1] = values[base_idx + 2 * tid + 1];
    }
    sync_local();

    // Bitonic merge network inside local memory
    for (uint size = 2; size <= BITONIC_ITEMS_PER_WG; size <<= 1) {
        for (uint stride = size / 2; stride > 0; stride >>= 1) {
            uint pos = 2 * tid - (tid & (stride - 1));
            uint idx1 = pos;
            uint idx2 = pos + stride;

            bool is_ascending = ((pos & size) == 0) == (dir != 0);

            if (idx2 < BITONIC_ITEMS_PER_WG) {
                uint k1 = l_keys[idx1];
                uint k2 = l_keys[idx2];

                if ((k1 > k2) == is_ascending) {
                    l_keys[idx1] = k2;
                    l_keys[idx2] = k1;

                    if (values != 0) {
                        uint v1 = l_vals[idx1];
                        uint v2 = l_vals[idx2];
                        l_vals[idx1] = v2;
                        l_vals[idx2] = v1;
                    }
                }
            }
            sync_local();
        }
    }

    keys[base_idx + 2 * tid]     = l_keys[2 * tid];
    keys[base_idx + 2 * tid + 1] = l_keys[2 * tid + 1];

    if (values != 0) {
        values[base_idx + 2 * tid]     = l_vals[2 * tid];
        values[base_idx + 2 * tid + 1] = l_vals[2 * tid + 1];
    }
}

// Global Bitonic Merge Step for Large Arrays
__kernel void kernel_bitonic_merge_global(
    __global uint* keys,
    __global uint* values,
    const uint stage_size,
    const uint stride,
    const int dir) {
    
    uint tid = get_global_id(0);

    uint pos = 2 * tid - (tid & (stride - 1));
    uint idx1 = pos;
    uint idx2 = pos + stride;

    bool is_ascending = ((pos & stage_size) == 0) == (dir != 0);

    uint k1 = keys[idx1];
    uint k2 = keys[idx2];

    if ((k1 > k2) == is_ascending) {
        keys[idx1] = k2;
        keys[idx2] = k1;

        if (values != 0) {
            uint v1 = values[idx1];
            uint v2 = values[idx2];
            values[idx1] = v2;
            values[idx2] = v1;
        }
    }
}
