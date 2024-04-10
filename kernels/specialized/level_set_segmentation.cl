#include "common/math.clh"
#include "common/memory.clh"

// Geodesic Active Contours and Chan-Vese 3D Level Set Evolution

// 3D Level Set Curvature Motion and Edge Advection Force
__kernel void kernel_level_set_evolution_3d(
    __global const float* phi_curr,
    __global const float* edge_indicator_g,
    __global const float4* edge_gradient,
    __global float* phi_next,
    const float balloon_weight,
    const float dt,
    const int nx, const int ny, const int nz) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;
    size_t slice = (size_t)nx * (size_t)ny;
    size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;
    float p = phi_curr[idx];
    float px_plus = phi_curr[idx + 1] - p;
    float px_minus = p - phi_curr[idx - 1];
    float py_plus = phi_curr[idx + nx] - p;
    float py_minus = p - phi_curr[idx - nx];
    float pz_plus = phi_curr[idx + slice] - p;
    float pz_minus = p - phi_curr[idx - slice];
    float g = edge_indicator_g[idx];
    float4 grad_g = edge_gradient[idx];
    // Central differences for curvature
    float dx = (phi_curr[idx + 1] - phi_curr[idx - 1]) * 0.5f;
    float dy = (phi_curr[idx + nx] - phi_curr[idx - nx]) * 0.5f;
    float dz = (phi_curr[idx + slice] - phi_curr[idx - slice]) * 0.5f;
    float grad_norm = sqrt(dx*dx + dy*dy + dz*dz + 1.0e-6f);
    // Upwind scheme for advection and balloon force
    float adv = grad_g.x * dx + grad_g.y * dy + grad_g.z * dz;
    float balloon = balloon_weight * g * grad_norm;
    phi_next[idx] = p + dt * (g * grad_norm + adv + balloon);
}
