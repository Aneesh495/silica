#include "common/math.clh"
#include "common/memory.clh"

// Stream Compaction Scatter Phase (Float4)
// Scatters elements where predicate flag is set using scanned prefix offsets
__kernel void kernel_stream_compact_scatter_float4(
    __global const float4* in_data,
    __global const uint* in_flags,
    __global const uint* in_scanned_offsets,
    __global float4* out_compacted_data,
    const uint total_elements) {
    
    uint gid = get_global_id(0);
    if (gid >= total_elements) return;

    if (in_flags[gid] != 0) {
        uint target_idx = in_scanned_offsets[gid];
        out_compacted_data[target_idx] = in_data[gid];
    }
}

// Stream Compaction Scatter Phase (Float)
__kernel void kernel_stream_compact_scatter_float(
    __global const float* in_data,
    __global const uint* in_flags,
    __global const uint* in_scanned_offsets,
    __global float* out_compacted_data,
    const uint total_elements) {
    
    uint gid = get_global_id(0);
    if (gid >= total_elements) return;

    if (in_flags[gid] != 0) {
        uint target_idx = in_scanned_offsets[gid];
        out_compacted_data[target_idx] = in_data[gid];
    }
}
