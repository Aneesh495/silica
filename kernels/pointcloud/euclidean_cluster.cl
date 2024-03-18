#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

// Stage 1: Initialize Each Point Label to its Global Index
__kernel void kernel_cluster_init_labels(
    __global uint* out_labels,
    const uint num_points) {
    
    uint gid = get_global_id(0);
    if (gid >= num_points) return;

    out_labels[gid] = gid;
}

// Stage 2: Parallel Label Propagation: propagates minimum label among Euclidean neighbors
__kernel void kernel_cluster_propagate_labels(
    __global const float4* in_points,
    __global const uint* in_labels,
    __global uint* out_labels,
    __global volatile uint* out_change_flag,
    const uint num_points,
    const float cluster_tolerance,
    const int search_window) {
    
    uint gid = get_global_id(0);
    if (gid >= num_points) return;

    float4 p4 = in_points[gid];
    float3 p = (float3)(p4.x, p4.y, p4.z);
    uint current_label = in_labels[gid];
    uint min_label = current_label;

    int start_idx = max(0, (int)gid - search_window / 2);
    int end_idx   = min((int)num_points - 1, (int)gid + search_window / 2);

    for (int j = start_idx; j <= end_idx; ++j) {
        if ((uint)j == gid) continue;

        float4 q4 = in_points[j];
        float3 q = (float3)(q4.x, q4.y, q4.z);

        if (distance(p, q) <= cluster_tolerance) {
            uint neighbor_label = in_labels[j];
            if (neighbor_label < min_label) {
                min_label = neighbor_label;
            }
        }
    }

    out_labels[gid] = min_label;

    if (min_label < current_label) {
        *out_change_flag = 1;
    }
}
