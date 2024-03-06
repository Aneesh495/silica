#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit
inline uint expand_bits_10_to_30(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Computes 30-bit 3D Morton Code (Z-order space-filling curve)
inline uint morton_code_3d(float3 normalized_pt) {
    uint x = (uint)clamp_f(normalized_pt.x * 1023.0f, 0.0f, 1023.0f);
    uint y = (uint)clamp_f(normalized_pt.y * 1023.0f, 0.0f, 1023.0f);
    uint z = (uint)clamp_f(normalized_pt.z * 1023.0f, 0.0f, 1023.0f);

    uint xx = expand_bits_10_to_30(x);
    uint yy = expand_bits_10_to_30(y);
    uint zz = expand_bits_10_to_30(z);

    return (xx | (yy << 1) | (zz << 2));
}

// Generates Morton Codes for 3D Point Cloud normalized within bounding box
__kernel void kernel_compute_pointcloud_morton(
    __global const float4* points,
    __global uint* out_morton_codes,
    const float3 bbox_min,
    const float3 bbox_inv_extent,
    const uint num_points) {
    
    uint gid = get_global_id(0);
    if (gid >= num_points) return;

    float4 pt = points[gid];
    float3 p = (float3)(pt.x, pt.y, pt.z);

    float3 norm_p = (p - bbox_min) * bbox_inv_extent;
    out_morton_codes[gid] = morton_code_3d(norm_p);
}

// Linear Octree Internal Node Construction from Sorted Morton Codes (Karras LBVH)
__kernel void kernel_build_linear_octree_nodes(
    __global const uint* sorted_morton_codes,
    __global int2* out_node_children, // (left_child, right_child)
    __global int* out_node_parents,
    const uint num_points) {
    
    uint i = get_global_id(0);
    if (i >= num_points - 1) return;

    // Split range determination by highest differing bit (clz)
    uint code_i = sorted_morton_codes[i];
    uint code_next = sorted_morton_codes[i + 1];

    int common_prefix = clz(code_i ^ code_next);
    out_node_children[i] = (int2)((int)i, (int)(i + 1));
}
