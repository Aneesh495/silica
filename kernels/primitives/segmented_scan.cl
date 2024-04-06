#include "common/math.clh"
#include "common/memory.clh"

#define SEG_SCAN_WG_SIZE 256

// Segmented Inclusive Prefix Sum (Float)
// Scans each contiguous segment delineated by flag[i] == 1
__kernel void kernel_segmented_scan_float(
    __global const float* in_values,
    __global const uint* in_flags, // 1 at segment head, 0 elsewhere
    __global float* out_values,
    const uint total_elements) {
    
    __local float l_val[SEG_SCAN_WG_SIZE];
    __local uint  l_flag[SEG_SCAN_WG_SIZE];

    uint tid = get_local_id(0);
    uint gid = get_global_id(0);

    float val = (gid < total_elements) ? in_values[gid] : 0.0f;
    uint flag = (gid < total_elements) ? in_flags[gid] : 0;

    l_val[tid]  = val;
    l_flag[tid] = flag;
    sync_local();

    // Hillis-Steele step-efficient segmented scan
    for (uint offset = 1; offset < SEG_SCAN_WG_SIZE; offset <<= 1) {
        float temp_val = 0.0f;
        uint temp_flag = 0;

        if (tid >= offset) {
            if (l_flag[tid] == 0) {
                temp_val = l_val[tid - offset];
            }
            temp_flag = l_flag[tid - offset];
        }
        sync_local();

        if (tid >= offset) {
            l_val[tid] += temp_val;
            l_flag[tid] |= temp_flag;
        }
        sync_local();
    }

    if (gid < total_elements) {
        out_values[gid] = l_val[tid];
    }
}
