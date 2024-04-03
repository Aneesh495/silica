#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

#define MAX_STACK_DEPTH 32

typedef struct {
    float3 bbox_min;
    float3 bbox_max;
    int left_child;
    int right_child;
    int point_index; // -1 if internal node
} OctreeNode;

// Computes Minimum Euclidean Distance Squared between Query Point and Node AABB
inline float distance_point_to_aabb_sq(float3 p, float3 bmin, float3 bmax) {
    float dist_sq = 0.0f;
    for (int i = 0; i < 3; ++i) {
        float v = (i == 0) ? p.x : ((i == 1) ? p.y : p.z);
        float min_v = (i == 0) ? bmin.x : ((i == 1) ? bmin.y : bmin.z);
        float max_v = (i == 0) ? bmax.x : ((i == 1) ? bmax.y : bmax.z);

        if (v < min_v) dist_sq += (min_v - v) * (min_v - v);
        else if (v > max_v) dist_sq += (v - max_v) * (v - max_v);
    }
    return dist_sq;
}

// Parallel Nearest-Neighbor Query against Octree BVH
__kernel void kernel_octree_nearest_neighbor(
    __global const OctreeNode* octree_nodes,
    __global const float4* point_cloud,
    __global const float4* query_points,
    __global int* out_nearest_point_idx,
    __global float* out_nearest_distance_sq,
    const uint num_queries) {
    
    uint gid = get_global_id(0);
    if (gid >= num_queries) return;

    float4 q4 = query_points[gid];
    float3 query = (float3)(q4.x, q4.y, q4.z);

    // Private traversal stack
    int stack[MAX_STACK_DEPTH];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0; // Push root node

    float best_dist_sq = LARGE_FLOAT_F;
    int best_idx = -1;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        OctreeNode node = octree_nodes[node_idx];

        float d_box = distance_point_to_aabb_sq(query, node.bbox_min, node.bbox_max);
        if (d_box >= best_dist_sq) continue;

        if (node.point_index >= 0) {
            // Leaf node: test actual point
            float4 pt4 = point_cloud[node.point_index];
            float3 pt = (float3)(pt4.x, pt4.y, pt4.z);
            float d_pt = distance(query, pt);
            float d_pt_sq = d_pt * d_pt;

            if (d_pt_sq < best_dist_sq) {
                best_dist_sq = d_pt_sq;
                best_idx = node.point_index;
            }
        } else {
            // Internal node: push children
            if (stack_ptr < MAX_STACK_DEPTH - 2) {
                if (node.left_child >= 0)  stack[stack_ptr++] = node.left_child;
                if (node.right_child >= 0) stack[stack_ptr++] = node.right_child;
            }
        }
    }

    out_nearest_point_idx[gid] = best_idx;
    out_nearest_distance_sq[gid] = best_dist_sq;
}
