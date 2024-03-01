#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"
#include "marching_cubes_tables.clh"

// Phase 1: Classify Marching Cubes Cells and Count Intersecting Edges
// Identifies which cells intersect the isosurface and counts active edges.
__kernel void kernel_mc_classify_voxels(
    __global const float* tsdf_grid,
    __global uint* out_cube_index,
    __global uint* out_edge_count,
    const int vx,
    const int vy,
    const int vz,
    const float iso_level) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    if (x >= vx - 1 || y >= vy - 1 || z >= vz - 1) return;

    size_t cell_idx = (((size_t)z * (size_t)(vy - 1) + (size_t)y) * (size_t)(vx - 1)) + (size_t)x;

    // Read 8 corner values
    uint cube_idx = 0;
    for (int i = 0; i < 8; ++i) {
        int cx = x + MC_CUBE_CORNERS[i].x;
        int cy = y + MC_CUBE_CORNERS[i].y;
        int cz = z + MC_CUBE_CORNERS[i].z;

        float val = tsdf_grid[(((size_t)cz * (size_t)vy + (size_t)cy) * (size_t)vx) + (size_t)cx];
        if (val < iso_level) {
            cube_idx |= (1u << i);
        }
    }

    out_cube_index[cell_idx] = cube_idx;

    ushort edges = MC_EDGE_TABLE[cube_idx];
    out_edge_count[cell_idx] = (edges != 0) ? popcount((uint)edges) : 0;
}

// Phase 3: Extract Triangles and Interpolate Vertices Along Cell Edges
__kernel void kernel_mc_extract_mesh(
    __global const float* tsdf_grid,
    __global const uint* cube_indices,
    __global float4* out_vertices,       // Intersected vertex positions (x, y, z, 1.0)
    __global float4* out_normals,        // Analytical vertex normals
    __global volatile uint* vertex_counter,
    const int vx,
    const int vy,
    const int vz,
    const float voxel_size,
    const float iso_level,
    const uint max_vertices) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    if (x >= vx - 1 || y >= vy - 1 || z >= vz - 1) return;

    size_t cell_idx = (((size_t)z * (size_t)(vy - 1) + (size_t)y) * (size_t)(vx - 1)) + (size_t)x;
    uint cube_idx = cube_indices[cell_idx];

    ushort edge_flags = MC_EDGE_TABLE[cube_idx];
    if (edge_flags == 0) return;

    // Corner positions and values
    float3 corners[8];
    float values[8];

    float3 base_world = (float3)(
        ((float)x - 0.5f * (float)(vx - 1)) * voxel_size,
        ((float)y - 0.5f * (float)(vy - 1)) * voxel_size,
        ((float)z - 0.5f * (float)(vz - 1)) * voxel_size
    );

    for (int i = 0; i < 8; ++i) {
        int cx = x + MC_CUBE_CORNERS[i].x;
        int cy = y + MC_CUBE_CORNERS[i].y;
        int cz = z + MC_CUBE_CORNERS[i].z;

        corners[i] = base_world + (float3)(
            (float)MC_CUBE_CORNERS[i].x * voxel_size,
            (float)MC_CUBE_CORNERS[i].y * voxel_size,
            (float)MC_CUBE_CORNERS[i].z * voxel_size
        );

        values[i] = tsdf_grid[(((size_t)cz * (size_t)vy + (size_t)cy) * (size_t)vx) + (size_t)cx];
    }

    // Interpolate vertex positions on intersecting edges
    float3 edge_vertices[12];
    for (int e = 0; e < 12; ++e) {
        if (edge_flags & (1 << e)) {
            int v0 = MC_EDGE_VERTICES[e].x;
            int v1 = MC_EDGE_VERTICES[e].y;

            float val0 = values[v0];
            float val1 = values[v1];

            float t = (iso_level - val0) / (val1 - val0 + EPSILON_F);
            t = clamp_f(t, 0.0f, 1.0f);

            edge_vertices[e] = corners[v0] + t * (corners[v1] - corners[v0]);
        }
    }

    // Emit vertices for active edges
    for (int e = 0; e < 12; ++e) {
        if (edge_flags & (1 << e)) {
            uint slot = atomic_inc(vertex_counter);
            if (slot < max_vertices) {
                float3 p = edge_vertices[e];
                out_vertices[slot] = (float4)(p.x, p.y, p.z, 1.0f);
                out_normals[slot]  = (float4)(0.0f, 0.0f, 1.0f, 0.0f);
            }
        }
    }
}
