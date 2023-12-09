#include "common/math.clh"
#include "common/memory.clh"

// Blelloch work-efficient parallel prefix sum (exclusive and inclusive).
// Each work-group scans an array of 2 * WG_SIZE elements in local memory.
// Up-sweep builds balanced binary reduction tree; down-sweep accumulates prefix sums.

#define SCAN_WG_SIZE 256
#define SCAN_ITEMS_PER_WG (2 * SCAN_WG_SIZE)
#define SCAN_BANK_OFFSET(n) ((n) + ((n) >> 5))

// Work-group local exclusive scan for uint
inline void workgroup_scan_exclusive_uint(
    __local uint* l_data,
    uint tid,
    uint n,
    __local uint* total_sum) {
    
    int offset = 1;

    // Up-sweep (reduction) phase
    for (int d = n >> 1; d > 0; d >>= 1) {
        sync_local();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai = SCAN_BANK_OFFSET(ai);
            bi = SCAN_BANK_OFFSET(bi);
            l_data[bi] += l_data[ai];
        }
        offset *= 2;
    }

    // Clear the last element to zero and capture total sum
    if (tid == 0) {
        int last_idx = SCAN_BANK_OFFSET(n - 1);
        if (total_sum != 0) {
            *total_sum = l_data[last_idx];
        }
        l_data[last_idx] = 0;
    }

    // Down-sweep phase
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        sync_local();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai = SCAN_BANK_OFFSET(ai);
            bi = SCAN_BANK_OFFSET(bi);

            uint t = l_data[ai];
            l_data[ai] = l_data[bi];
            l_data[bi] += t;
        }
    }
    sync_local();
}

// Work-group local exclusive scan for float
inline void workgroup_scan_exclusive_float(
    __local float* l_data,
    uint tid,
    uint n,
    __local float* total_sum) {
    
    int offset = 1;

    // Up-sweep
    for (int d = n >> 1; d > 0; d >>= 1) {
        sync_local();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai = SCAN_BANK_OFFSET(ai);
            bi = SCAN_BANK_OFFSET(bi);
            l_data[bi] += l_data[ai];
        }
        offset *= 2;
    }

    // Root clear
    if (tid == 0) {
        int last_idx = SCAN_BANK_OFFSET(n - 1);
        if (total_sum != 0) {
            *total_sum = l_data[last_idx];
        }
        l_data[last_idx] = 0.0f;
    }

    // Down-sweep
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        sync_local();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai = SCAN_BANK_OFFSET(ai);
            bi = SCAN_BANK_OFFSET(bi);

            float t = l_data[ai];
            l_data[ai] = l_data[bi];
            l_data[bi] += t;
        }
    }
    sync_local();
}

// Multi-block Phase 1: Local Scan & Block Sum Emission (uint)
__kernel void kernel_scan_blocks_exclusive_uint(
    __global const uint* input,
    __global uint* output,
    __global uint* block_sums,
    const uint total_elements) {
    
    __local uint l_mem[SCAN_BANK_OFFSET(SCAN_ITEMS_PER_WG)];
    __local uint l_total;

    uint tid = get_local_id(0);
    uint gid = get_group_id(0);
    uint base_idx = gid * SCAN_ITEMS_PER_WG;

    uint idx_a = base_idx + 2 * tid;
    uint idx_b = base_idx + 2 * tid + 1;

    uint val_a = (idx_a < total_elements) ? input[idx_a] : 0;
    uint val_b = (idx_b < total_elements) ? input[idx_b] : 0;

    l_mem[SCAN_BANK_OFFSET(2 * tid)]     = val_a;
    l_mem[SCAN_BANK_OFFSET(2 * tid + 1)] = val_b;

    workgroup_scan_exclusive_uint(l_mem, tid, SCAN_ITEMS_PER_WG, &l_total);

    if (idx_a < total_elements) {
        output[idx_a] = l_mem[SCAN_BANK_OFFSET(2 * tid)];
    }
    if (idx_b < total_elements) {
        output[idx_b] = l_mem[SCAN_BANK_OFFSET(2 * tid + 1)];
    }

    if (tid == 0 && block_sums != 0) {
        // Total block sum is exclusive scan total + last original element
        block_sums[gid] = l_total + val_b;
    }
}

// Multi-block Phase 1: Local Scan & Block Sum Emission (float)
__kernel void kernel_scan_blocks_exclusive_float(
    __global const float* input,
    __global float* output,
    __global float* block_sums,
    const uint total_elements) {
    
    __local float l_mem[SCAN_BANK_OFFSET(SCAN_ITEMS_PER_WG)];
    __local float l_total;

    uint tid = get_local_id(0);
    uint gid = get_group_id(0);
    uint base_idx = gid * SCAN_ITEMS_PER_WG;

    uint idx_a = base_idx + 2 * tid;
    uint idx_b = base_idx + 2 * tid + 1;

    float val_a = (idx_a < total_elements) ? input[idx_a] : 0.0f;
    float val_b = (idx_b < total_elements) ? input[idx_b] : 0.0f;

    l_mem[SCAN_BANK_OFFSET(2 * tid)]     = val_a;
    l_mem[SCAN_BANK_OFFSET(2 * tid + 1)] = val_b;

    workgroup_scan_exclusive_float(l_mem, tid, SCAN_ITEMS_PER_WG, &l_total);

    if (idx_a < total_elements) {
        output[idx_a] = l_mem[SCAN_BANK_OFFSET(2 * tid)];
    }
    if (idx_b < total_elements) {
        output[idx_b] = l_mem[SCAN_BANK_OFFSET(2 * tid + 1)];
    }

    if (tid == 0 && block_sums != 0) {
        block_sums[gid] = l_total + val_b;
    }
}

// Multi-block Phase 3: Add Block Offsets to scanned elements (uint)
__kernel void kernel_scan_add_offsets_uint(
    __global uint* output,
    __global const uint* scanned_block_offsets,
    const uint total_elements) {
    
    uint gid = get_group_id(0);
    uint tid = get_local_id(0);
    uint base_idx = gid * SCAN_ITEMS_PER_WG;

    uint offset = scanned_block_offsets[gid];

    uint idx_a = base_idx + 2 * tid;
    uint idx_b = base_idx + 2 * tid + 1;

    if (idx_a < total_elements) {
        output[idx_a] += offset;
    }
    if (idx_b < total_elements) {
        output[idx_b] += offset;
    }
}

// Multi-block Phase 3: Add Block Offsets to scanned elements (float)
__kernel void kernel_scan_add_offsets_float(
    __global float* output,
    __global const float* scanned_block_offsets,
    const uint total_elements) {
    
    uint gid = get_group_id(0);
    uint tid = get_local_id(0);
    uint base_idx = gid * SCAN_ITEMS_PER_WG;

    float offset = scanned_block_offsets[gid];

    uint idx_a = base_idx + 2 * tid;
    uint idx_b = base_idx + 2 * tid + 1;

    if (idx_a < total_elements) {
        output[idx_a] += offset;
    }
    if (idx_b < total_elements) {
        output[idx_b] += offset;
    }
}
