#include "common/math.clh"
#include "common/memory.clh"

// Loop and Catmull-Clark Parallel Subdivision Surface Geometry Masks

// Loop Subdivision Edge Vertex Interpolation: v_new = (3/8)*(v0 + v1) + (1/8)*(v2 + v3)
__kernel void kernel_subdiv_loop_edge_vertex(
    __global const float4* vertex_positions,
    __global const int4* edge_topo, // (v0, v1, opp_v2, opp_v3)
    __global float4* out_edge_vertices,
    const uint num_edges) {
    uint eid = get_global_id(0);
    if (eid >= num_edges) return;
    int4 topo = edge_topo[eid];
    float4 v0 = vertex_positions[topo.x];
    float4 v1 = vertex_positions[topo.y];
    float4 v2 = vertex_positions[topo.z];
    float4 v3 = vertex_positions[topo.w];
    // Interior edge: 3/8 * (v0 + v1) + 1/8 * (v2 + v3)
    if (topo.w >= 0) {
        out_edge_vertices[eid] = 0.375f * (v0 + v1) + 0.125f * (v2 + v3);
    } else {
        // Boundary edge: 1/2 * (v0 + v1)
        out_edge_vertices[eid] = 0.5f * (v0 + v1);
    }
}
