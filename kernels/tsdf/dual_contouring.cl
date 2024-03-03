#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

// Quadric Error Metric (QEM) Dual Vertex Solver
// Solves optimal vertex location x within cell minimizing distance to tangent planes:
// E(x) = sum (n_i * (x - p_i))^2  =>  A * x = b
__kernel void kernel_dual_contouring_qem_vertex(
    __global const float* tsdf_grid,
    __global float4* out_dual_vertices,
    __global uchar* out_cell_has_vertex,
    const int vx,
    const int vy,
    const int vz,
    const float voxel_size,
    const float iso_level) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    if (x >= vx - 1 || y >= vy - 1 || z >= vz - 1) return;

    size_t cell_idx = (((size_t)z * (size_t)(vy - 1) + (size_t)y) * (size_t)(vx - 1)) + (size_t)x;

    // Center of voxel cell
    float3 cell_center = (float3)(
        ((float)x - 0.5f * (float)(vx - 1) + 0.5f) * voxel_size,
        ((float)y - 0.5f * (float)(vy - 1) + 0.5f) * voxel_size,
        ((float)z - 0.5f * (float)(vz - 1) + 0.5f) * voxel_size
    );

    // Accumulate A = sum(n * n^T) and b = sum(n * (n . p))
    float A[3][3] = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
    float b[3] = {0.0f, 0.0f, 0.0f};
    int edge_crossings = 0;

    // Check 3 primary edges sharing corner (x, y, z): X-axis, Y-axis, Z-axis
    float v000 = tsdf_grid[(((size_t)z * (size_t)vy + (size_t)y) * (size_t)vx) + (size_t)x];
    bool sign0 = (v000 < iso_level);

    // Edge along X
    float v100 = tsdf_grid[(((size_t)z * (size_t)vy + (size_t)y) * (size_t)vx) + (size_t)(x + 1)];
    if ((v100 < iso_level) != sign0) {
        float t = (iso_level - v000) / (v100 - v000 + EPSILON_F);
        float3 p = cell_center + (float3)((t - 0.5f) * voxel_size, -0.5f * voxel_size, -0.5f * voxel_size);
        float3 n = (float3)(1.0f, 0.0f, 0.0f); // Approximate normal

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                float nr = (r == 0) ? n.x : ((r == 1) ? n.y : n.z);
                float nc = (c == 0) ? n.x : ((c == 1) ? n.y : n.z);
                A[r][c] += nr * nc;
            }
            float nr = (r == 0) ? n.x : ((r == 1) ? n.y : n.z);
            b[r] += nr * dot(n, p);
        }
        edge_crossings++;
    }

    // Edge along Y
    float v010 = tsdf_grid[(((size_t)z * (size_t)vy + (size_t)(y + 1)) * (size_t)vx) + (size_t)x];
    if ((v010 < iso_level) != sign0) {
        float t = (iso_level - v000) / (v010 - v000 + EPSILON_F);
        float3 p = cell_center + (float3)(-0.5f * voxel_size, (t - 0.5f) * voxel_size, -0.5f * voxel_size);
        float3 n = (float3)(0.0f, 1.0f, 0.0f);

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                float nr = (r == 0) ? n.x : ((r == 1) ? n.y : n.z);
                float nc = (c == 0) ? n.x : ((c == 1) ? n.y : n.z);
                A[r][c] += nr * nc;
            }
            float nr = (r == 0) ? n.x : ((r == 1) ? n.y : n.z);
            b[r] += nr * dot(n, p);
        }
        edge_crossings++;
    }

    // Edge along Z
    float v001 = tsdf_grid[(((size_t)(z + 1) * (size_t)vy + (size_t)y) * (size_t)vx) + (size_t)x];
    if ((v001 < iso_level) != sign0) {
        float t = (iso_level - v000) / (v001 - v000 + EPSILON_F);
        float3 p = cell_center + (float3)(-0.5f * voxel_size, -0.5f * voxel_size, (t - 0.5f) * voxel_size);
        float3 n = (float3)(0.0f, 0.0f, 1.0f);

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                float nr = (r == 0) ? n.x : ((r == 1) ? n.y : n.z);
                float nc = (c == 0) ? n.x : ((c == 1) ? n.y : n.z);
                A[r][c] += nr * nc;
            }
            float nr = (r == 0) ? n.x : ((r == 1) ? n.y : n.z);
            b[r] += nr * dot(n, p);
        }
        edge_crossings++;
    }

    if (edge_crossings == 0) {
        out_cell_has_vertex[cell_idx] = 0;
        return;
    }

    // Regularize A and solve A * x = b, fallback to cell center
    A[0][0] += 0.05f;
    A[1][1] += 0.05f;
    A[2][2] += 0.05f;

    // Simple Cramer's rule for 3x3
    float det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    float3 qem_pos = cell_center;
    if (fabs(det) > 1.0e-5f) {
        float inv_det = 1.0f / det;
        float sol_x = (b[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                       A[0][1] * (b[1] * A[2][2] - A[1][2] * b[2]) +
                       A[0][2] * (b[1] * A[2][1] - A[1][1] * b[2])) * inv_det;
        float sol_y = (A[0][0] * (b[1] * A[2][2] - A[1][2] * b[2]) -
                       b[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                       A[0][2] * (A[1][0] * b[2] - b[1] * A[2][0])) * inv_det;
        float sol_z = (A[0][0] * (A[1][1] * b[2] - b[1] * A[2][1]) -
                       A[0][1] * (A[1][0] * b[2] - b[1] * A[2][0]) +
                       b[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])) * inv_det;

        float3 solved = (float3)(sol_x, sol_y, sol_z);
        // Clamp vertex within cell bounds
        float half_v = 0.5f * voxel_size;
        qem_pos.x = clamp_f(solved.x, cell_center.x - half_v, cell_center.x + half_v);
        qem_pos.y = clamp_f(solved.y, cell_center.y - half_v, cell_center.y + half_v);
        qem_pos.z = clamp_f(solved.z, cell_center.z - half_v, cell_center.z + half_v);
    }

    out_dual_vertices[cell_idx] = (float4)(qem_pos.x, qem_pos.y, qem_pos.z, 1.0f);
    out_cell_has_vertex[cell_idx] = 1;
}
