#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

#define SOR_K_NEIGHBORS 16

// Stage 1: Computes Mean Distance to k Nearest Neighbors within Local Search Window
__kernel void kernel_sor_compute_knn_mean_distances(
    __global const float4* in_points,
    __global float* out_mean_distances,
    const uint num_points,
    const int search_window_size) {
    
    uint gid = get_global_id(0);
    if (gid >= num_points) return;

    float4 p4 = in_points[gid];
    float3 p = (float3)(p4.x, p4.y, p4.z);

    // Track k smallest distances using insertion sort in private registers
    float smallest_dists[SOR_K_NEIGHBORS];
    #pragma unroll
    for (int i = 0; i < SOR_K_NEIGHBORS; ++i) {
        smallest_dists[i] = LARGE_FLOAT_F;
    }

    int start_idx = max(0, (int)gid - search_window_size / 2);
    int end_idx   = min((int)num_points - 1, (int)gid + search_window_size / 2);

    for (int j = start_idx; j <= end_idx; ++j) {
        if ((uint)j == gid) continue;

        float4 q4 = in_points[j];
        float3 q = (float3)(q4.x, q4.y, q4.z);

        float dist = distance(p, q);

        if (dist < smallest_dists[SOR_K_NEIGHBORS - 1]) {
            smallest_dists[SOR_K_NEIGHBORS - 1] = dist;
            for (int k = SOR_K_NEIGHBORS - 1; k > 0; --k) {
                if (smallest_dists[k] < smallest_dists[k - 1]) {
                    float tmp = smallest_dists[k];
                    smallest_dists[k] = smallest_dists[k - 1];
                    smallest_dists[k - 1] = tmp;
                } else {
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    int valid_k = 0;
    for (int i = 0; i < SOR_K_NEIGHBORS; ++i) {
        if (smallest_dists[i] < LARGE_FLOAT_F) {
            sum += smallest_dists[i];
            valid_k++;
        }
    }

    out_mean_distances[gid] = (valid_k > 0) ? (sum / (float)valid_k) : 0.0f;
}

// Stage 2: Filter Points exceeding (mean + std_mul * std_dev)
__kernel void kernel_sor_filter_outliers(
    __global const float4* in_points,
    __global const float* mean_distances,
    __global float4* out_inlier_points,
    __global volatile uint* inlier_counter,
    const uint num_points,
    const float distance_threshold,
    const uint max_inliers) {
    
    uint gid = get_global_id(0);
    if (gid >= num_points) return;

    float mean_dist = mean_distances[gid];

    if (mean_dist > 0.0f && mean_dist <= distance_threshold) {
        uint slot = atomic_inc(inlier_counter);
        if (slot < max_inliers) {
            out_inlier_points[slot] = in_points[gid];
        }
    }
}
