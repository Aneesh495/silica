#include "common/math.clh"
#include "common/memory.clh"

#define RADIX_BITS 4
#define RADIX_BUCKETS (1 << RADIX_BITS) // 16 buckets
#define RADIX_WG_SIZE 256
#define RADIX_ITEMS_PER_THREAD 4
#define RADIX_ITEMS_PER_WG (RADIX_WG_SIZE * RADIX_ITEMS_PER_THREAD) // 1024 items

// Phase 1: Count local histogram for each 4-bit digit pass
__kernel void kernel_radix_sort_histogram(
    __global const uint* in_keys,
    __global uint* out_histograms,
    const uint total_elements,
    const uint bit_shift) {
    
    __local uint l_hist[RADIX_BUCKETS][RADIX_WG_SIZE];

    uint tid = get_local_id(0);
    uint gid = get_group_id(0);
    uint num_groups = get_num_groups(0);

    // Initialize local histograms to zero
    for (uint b = 0; b < RADIX_BUCKETS; ++b) {
        l_hist[b][tid] = 0;
    }
    sync_local();

    // Accumulate digit occurrences for this workgroup
    uint base_idx = gid * RADIX_ITEMS_PER_WG + tid;
    for (uint i = 0; i < RADIX_ITEMS_PER_THREAD; ++i) {
        uint idx = base_idx + i * RADIX_WG_SIZE;
        if (idx < total_elements) {
            uint key = in_keys[idx];
            uint bucket = (key >> bit_shift) & (RADIX_BUCKETS - 1);
            l_hist[bucket][tid]++;
        }
    }
    sync_local();

    // Reduce histograms across threads for each bucket
    for (uint b = 0; b < RADIX_BUCKETS; ++b) {
        // Tree reduction within workgroup
        for (uint s = RADIX_WG_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) {
                l_hist[b][tid] += l_hist[b][tid + s];
            }
            sync_local();
        }
    }

    // Write group histogram to global memory in bucket-major order
    // out_histograms layout: [bucket * num_groups + gid]
    if (tid < RADIX_BUCKETS) {
        out_histograms[tid * num_groups + gid] = l_hist[tid][0];
    }
}

// Phase 3: Scatter keys and values based on local rank and global bucket offset
__kernel void kernel_radix_sort_scatter(
    __global const uint* in_keys,
    __global const uint* in_values,
    __global uint* out_keys,
    __global uint* out_values,
    __global const uint* global_bucket_offsets,
    const uint total_elements,
    const uint bit_shift) {
    
    __local uint l_keys[RADIX_ITEMS_PER_WG];
    __local uint l_values[RADIX_ITEMS_PER_WG];
    __local uint l_digit_offsets[RADIX_BUCKETS];

    uint tid = get_local_id(0);
    uint gid = get_group_id(0);
    uint num_groups = get_num_groups(0);
    uint base_idx = gid * RADIX_ITEMS_PER_WG;

    // Load elements into local memory
    for (uint i = 0; i < RADIX_ITEMS_PER_THREAD; ++i) {
        uint local_idx = tid + i * RADIX_WG_SIZE;
        uint global_idx = base_idx + local_idx;
        if (global_idx < total_elements) {
            l_keys[local_idx]   = in_keys[global_idx];
            l_values[local_idx] = in_values ? in_values[global_idx] : global_idx;
        } else {
            l_keys[local_idx]   = 0xFFFFFFFF; // Max key sentinel
            l_values[local_idx] = 0;
        }
    }
    sync_local();

    // Read global bucket offset for this workgroup
    if (tid < RADIX_BUCKETS) {
        l_digit_offsets[tid] = global_bucket_offsets[tid * num_groups + gid];
    }
    sync_local();

    // Reorder locally using 16-bucket split and scatter
    for (uint i = 0; i < RADIX_ITEMS_PER_THREAD; ++i) {
        uint local_idx = tid * RADIX_ITEMS_PER_THREAD + i;
        uint global_in = base_idx + local_idx;

        if (global_in < total_elements) {
            uint key = in_keys[global_in];
            uint val = in_values ? in_values[global_in] : global_in;
            uint bucket = (key >> bit_shift) & (RADIX_BUCKETS - 1);

            // Compute local rank: count elements before this with same bucket
            uint local_rank = 0;
            for (uint j = 0; j < local_idx; ++j) {
                uint other_bucket = (l_keys[j] >> bit_shift) & (RADIX_BUCKETS - 1);
                if (other_bucket == bucket) {
                    local_rank++;
                }
            }

            uint out_pos = l_digit_offsets[bucket] + local_rank;
            if (out_pos < total_elements) {
                out_keys[out_pos] = key;
                if (out_values != 0) {
                    out_values[out_pos] = val;
                }
            }
        }
    }
}
