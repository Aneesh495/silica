#include "common/math.clh"
#include "common/memory.clh"

// ============================================================================
// High-Order 2D and 3D Finite-Difference Stencils for Wavefields and Transport
// ============================================================================

// 2D Finite-Difference Laplacian: 2nd_order (Radius 1)
__kernel void kernel_stencil2d_2nd_order(
    __global const float* input, __global float* output,
    const int nx, const int ny, const float inv_dx2) {
    int x = get_global_id(0); int y = get_global_id(1);
    if (x < 1 || x >= nx - 1 || y < 1 || y >= ny - 1) return;
    size_t idx = (size_t)y * (size_t)nx + (size_t)x;
    float sum = -4.0f * input[idx];
    sum += (1.0f) * (input[idx - 1] + input[idx + 1] + input[idx - 1 * nx] + input[idx + 1 * nx]);
    output[idx] = sum * inv_dx2;
}

// 2D Finite-Difference Laplacian: 4th_order (Radius 2)
__kernel void kernel_stencil2d_4th_order(
    __global const float* input, __global float* output,
    const int nx, const int ny, const float inv_dx2) {
    int x = get_global_id(0); int y = get_global_id(1);
    if (x < 2 || x >= nx - 2 || y < 2 || y >= ny - 2) return;
    size_t idx = (size_t)y * (size_t)nx + (size_t)x;
    float sum = -5.0f * input[idx];
    sum += (1.3333333333333333f) * (input[idx - 1] + input[idx + 1] + input[idx - 1 * nx] + input[idx + 1 * nx]);
    sum += (-0.08333333333333333f) * (input[idx - 2] + input[idx + 2] + input[idx - 2 * nx] + input[idx + 2 * nx]);
    output[idx] = sum * inv_dx2;
}

// 2D Finite-Difference Laplacian: 6th_order (Radius 3)
__kernel void kernel_stencil2d_6th_order(
    __global const float* input, __global float* output,
    const int nx, const int ny, const float inv_dx2) {
    int x = get_global_id(0); int y = get_global_id(1);
    if (x < 3 || x >= nx - 3 || y < 3 || y >= ny - 3) return;
    size_t idx = (size_t)y * (size_t)nx + (size_t)x;
    float sum = -2.7222222222222223f * input[idx];
    sum += (1.5f) * (input[idx - 1] + input[idx + 1] + input[idx - 1 * nx] + input[idx + 1 * nx]);
    sum += (-0.15f) * (input[idx - 2] + input[idx + 2] + input[idx - 2 * nx] + input[idx + 2 * nx]);
    sum += (0.011111111111111112f) * (input[idx - 3] + input[idx + 3] + input[idx - 3 * nx] + input[idx + 3 * nx]);
    output[idx] = sum * inv_dx2;
}

// 2D Finite-Difference Laplacian: 8th_order (Radius 4)
__kernel void kernel_stencil2d_8th_order(
    __global const float* input, __global float* output,
    const int nx, const int ny, const float inv_dx2) {
    int x = get_global_id(0); int y = get_global_id(1);
    if (x < 4 || x >= nx - 4 || y < 4 || y >= ny - 4) return;
    size_t idx = (size_t)y * (size_t)nx + (size_t)x;
    float sum = -2.8472222222222223f * input[idx];
    sum += (1.6f) * (input[idx - 1] + input[idx + 1] + input[idx - 1 * nx] + input[idx + 1 * nx]);
    sum += (-0.2f) * (input[idx - 2] + input[idx + 2] + input[idx - 2 * nx] + input[idx + 2 * nx]);
    sum += (0.025396825396825397f) * (input[idx - 3] + input[idx + 3] + input[idx - 3 * nx] + input[idx + 3 * nx]);
    sum += (-0.0017857142857142857f) * (input[idx - 4] + input[idx + 4] + input[idx - 4 * nx] + input[idx + 4 * nx]);
    output[idx] = sum * inv_dx2;
}

// 3D Acoustic Wavefield Propagation: 7pt_2nd_order (Radius 1)
__kernel void kernel_stencil3d_7pt_2nd_order(
    __global const float* u_curr, __global const float* u_prev,
    __global const float* velocity_sq, __global float* u_next,
    const float dt_sq_dx_sq, const float damping, const int nx, const int ny, const int nz) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x < 1 || x >= nx - 1 || y < 1 || y >= ny - 1 || z < 1 || z >= nz - 1) return;
    size_t slice = (size_t)nx * (size_t)ny;
    size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;
    float curr = u_curr[idx]; float sum = -6.0f * curr;
    sum += (1.0f) * (u_curr[idx - 1] + u_curr[idx + 1] + u_curr[idx - 1 * nx] + u_curr[idx + 1 * nx] + u_curr[idx - 1 * slice] + u_curr[idx + 1 * slice]);
    float c2 = velocity_sq ? velocity_sq[idx] : 1.0f;
    float prev = u_prev ? u_prev[idx] : curr;
    u_next[idx] = (2.0f * curr - prev + c2 * dt_sq_dx_sq * sum) / (1.0f + damping);
}

// 3D Acoustic Wavefield Propagation: 13pt_4th_order (Radius 2)
__kernel void kernel_stencil3d_13pt_4th_order(
    __global const float* u_curr, __global const float* u_prev,
    __global const float* velocity_sq, __global float* u_next,
    const float dt_sq_dx_sq, const float damping, const int nx, const int ny, const int nz) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x < 2 || x >= nx - 2 || y < 2 || y >= ny - 2 || z < 2 || z >= nz - 2) return;
    size_t slice = (size_t)nx * (size_t)ny;
    size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;
    float curr = u_curr[idx]; float sum = -7.5f * curr;
    sum += (1.3333333333333333f) * (u_curr[idx - 1] + u_curr[idx + 1] + u_curr[idx - 1 * nx] + u_curr[idx + 1 * nx] + u_curr[idx - 1 * slice] + u_curr[idx + 1 * slice]);
    sum += (-0.08333333333333333f) * (u_curr[idx - 2] + u_curr[idx + 2] + u_curr[idx - 2 * nx] + u_curr[idx + 2 * nx] + u_curr[idx - 2 * slice] + u_curr[idx + 2 * slice]);
    float c2 = velocity_sq ? velocity_sq[idx] : 1.0f;
    float prev = u_prev ? u_prev[idx] : curr;
    u_next[idx] = (2.0f * curr - prev + c2 * dt_sq_dx_sq * sum) / (1.0f + damping);
}

// 3D Acoustic Wavefield Propagation: 19pt_6th_order (Radius 3)
__kernel void kernel_stencil3d_19pt_6th_order(
    __global const float* u_curr, __global const float* u_prev,
    __global const float* velocity_sq, __global float* u_next,
    const float dt_sq_dx_sq, const float damping, const int nx, const int ny, const int nz) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x < 3 || x >= nx - 3 || y < 3 || y >= ny - 3 || z < 3 || z >= nz - 3) return;
    size_t slice = (size_t)nx * (size_t)ny;
    size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;
    float curr = u_curr[idx]; float sum = -4.083333333333333f * curr;
    sum += (1.5f) * (u_curr[idx - 1] + u_curr[idx + 1] + u_curr[idx - 1 * nx] + u_curr[idx + 1 * nx] + u_curr[idx - 1 * slice] + u_curr[idx + 1 * slice]);
    sum += (-0.15f) * (u_curr[idx - 2] + u_curr[idx + 2] + u_curr[idx - 2 * nx] + u_curr[idx + 2 * nx] + u_curr[idx - 2 * slice] + u_curr[idx + 2 * slice]);
    sum += (0.011111111111111112f) * (u_curr[idx - 3] + u_curr[idx + 3] + u_curr[idx - 3 * nx] + u_curr[idx + 3 * nx] + u_curr[idx - 3 * slice] + u_curr[idx + 3 * slice]);
    float c2 = velocity_sq ? velocity_sq[idx] : 1.0f;
    float prev = u_prev ? u_prev[idx] : curr;
    u_next[idx] = (2.0f * curr - prev + c2 * dt_sq_dx_sq * sum) / (1.0f + damping);
}

// 3D Acoustic Wavefield Propagation: 25pt_8th_order (Radius 4)
__kernel void kernel_stencil3d_25pt_8th_order(
    __global const float* u_curr, __global const float* u_prev,
    __global const float* velocity_sq, __global float* u_next,
    const float dt_sq_dx_sq, const float damping, const int nx, const int ny, const int nz) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x < 4 || x >= nx - 4 || y < 4 || y >= ny - 4 || z < 4 || z >= nz - 4) return;
    size_t slice = (size_t)nx * (size_t)ny;
    size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;
    float curr = u_curr[idx]; float sum = -4.270833333333333f * curr;
    sum += (1.6f) * (u_curr[idx - 1] + u_curr[idx + 1] + u_curr[idx - 1 * nx] + u_curr[idx + 1 * nx] + u_curr[idx - 1 * slice] + u_curr[idx + 1 * slice]);
    sum += (-0.2f) * (u_curr[idx - 2] + u_curr[idx + 2] + u_curr[idx - 2 * nx] + u_curr[idx + 2 * nx] + u_curr[idx - 2 * slice] + u_curr[idx + 2 * slice]);
    sum += (0.025396825396825397f) * (u_curr[idx - 3] + u_curr[idx + 3] + u_curr[idx - 3 * nx] + u_curr[idx + 3 * nx] + u_curr[idx - 3 * slice] + u_curr[idx + 3 * slice]);
    sum += (-0.0017857142857142857f) * (u_curr[idx - 4] + u_curr[idx + 4] + u_curr[idx - 4 * nx] + u_curr[idx + 4 * nx] + u_curr[idx - 4 * slice] + u_curr[idx + 4 * slice]);
    float c2 = velocity_sq ? velocity_sq[idx] : 1.0f;
    float prev = u_prev ? u_prev[idx] : curr;
    u_next[idx] = (2.0f * curr - prev + c2 * dt_sq_dx_sq * sum) / (1.0f + damping);
}

// 3D Split-Field PML Absorbing Boundary Condition for Acoustic Wavefields
__kernel void kernel_pml_acoustic_boundary_3d(
    __global float* px, __global float* py, __global float* pz,
    __global const float* vx, __global const float* vy, __global const float* vz,
    __global const float* sigma_x, __global const float* sigma_y, __global const float* sigma_z,
    const float bulk_modulus, const float dt, const float inv_dx,
    const int nx, const int ny, const int nz) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;
    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;
    float sx = sigma_x[x]; float sy = sigma_y[y]; float sz = sigma_z[z];
    float dvx_dx = (vx[idx] - vx[idx - 1]) * inv_dx;
    float dvy_dy = (vy[idx] - vy[idx - nx]) * inv_dx;
    float dvz_dz = (vz[idx] - vz[idx - slice]) * inv_dx;
    px[idx] = (px[idx] * (1.0f - 0.5f * dt * sx) - dt * bulk_modulus * dvx_dx) / (1.0f + 0.5f * dt * sx);
    py[idx] = (py[idx] * (1.0f - 0.5f * dt * sy) - dt * bulk_modulus * dvy_dy) / (1.0f + 0.5f * dt * sy);
    pz[idx] = (pz[idx] * (1.0f - 0.5f * dt * sz) - dt * bulk_modulus * dvz_dz) / (1.0f + 0.5f * dt * sz);
}


// ============================================================================
// Advanced 3D Scientific Physics Stencils: FDTD, Navier-Stokes, Cahn-Hilliard
// ============================================================================

// 1. 3D FDTD Maxwell Yee Cell: Electric Field Update (Ex, Ey, Ez)
__kernel void kernel_fdtd_yee_e_field_update(
    __global float* Ex, __global float* Ey, __global float* Ez,
    __global const float* Hx, __global const float* Hy, __global const float* Hz,
    __global const float* eps_inv, const float dt_dx,
    const int nx, const int ny, const int nz) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;
    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;
    float c = dt_dx * eps_inv[idx];
    // Ex update: dHz/dy - dHy/dz
    float dHz_dy = Hz[idx] - Hz[idx - nx];
    float dHy_dz = Hy[idx] - Hy[idx - slice];
    Ex[idx] += c * (dHz_dy - dHy_dz);
    // Ey update: dHx/dz - dHz/dx
    float dHx_dz = Hx[idx] - Hx[idx - slice];
    float dHz_dx = Hz[idx] - Hz[idx - 1];
    Ey[idx] += c * (dHx_dz - dHz_dx);
    // Ez update: dHy/dx - dHx/dy
    float dHy_dx = Hy[idx] - Hy[idx - 1];
    float dHx_dy = Hx[idx] - Hx[idx - nx];
    Ez[idx] += c * (dHy_dx - dHx_dy);
}}


// 2. 3D FDTD Maxwell Yee Cell: Magnetic Field Update (Hx, Hy, Hz)
__kernel void kernel_fdtd_yee_h_field_update(
    __global float* Hx, __global float* Hy, __global float* Hz,
    __global const float* Ex, __global const float* Ey, __global const float* Ez,
    __global const float* mu_inv, const float dt_dx,
    const int nx, const int ny, const int nz) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x < 0 || x >= nx - 1 || y < 0 || y >= ny - 1 || z < 0 || z >= nz - 1) return;
    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;
    float c = dt_dx * mu_inv[idx];
    // Hx update: -(dEz/dy - dEy/dz)
    float dEz_dy = Ez[idx + nx] - Ez[idx];
    float dEy_dz = Ey[idx + slice] - Ey[idx];
    Hx[idx] -= c * (dEz_dy - dEy_dz);
    // Hy update: -(dEx/dz - dEz/dx)
    float dEx_dz = Ex[idx + slice] - Ex[idx];
    float dEz_dx = Ez[idx + 1] - Ez[idx];
    Hy[idx] -= c * (dEx_dz - dEz_dx);
    // Hz update: -(dEy/dx - dEx/dy)
    float dEy_dx = Ey[idx + 1] - Ey[idx];
    float dEx_dy = Ex[idx + nx] - Ex[idx];
    Hz[idx] -= c * (dEy_dx - dEx_dy);
}}


// 3. 3D Incompressible Navier-Stokes MAC Staggered Grid Advection
__kernel void kernel_navier_stokes_advection_3d(
    __global const float* u, __global const float* v, __global const float* w,
    __global float* u_out, __global float* v_out, __global float* w_out,
    const float dt, const float inv_dx, const int nx, const int ny, const int nz) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;
    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;
    // Trace back particle position via RK2 midpoint
    float pos_x = (float)x - 0.5f * dt * inv_dx * u[idx];
    float pos_y = (float)y - 0.5f * dt * inv_dx * v[idx];
    float pos_z = (float)z - 0.5f * dt * inv_dx * w[idx];
    int ix = clamp_i((int)floor(pos_x), 0, nx - 2);
    int iy = clamp_i((int)floor(pos_y), 0, ny - 2);
    int iz = clamp_i((int)floor(pos_z), 0, nz - 2);
    float fx = pos_x - (float)ix; float fy = pos_y - (float)iy; float fz = pos_z - (float)iz;
    size_t i000 = (size_t)iz * slice + (size_t)iy * (size_t)nx + (size_t)ix;
    u_out[idx] = mix(mix(mix(u[i000], u[i000 + 1], fx), mix(u[i000 + nx], u[i000 + nx + 1], fx), fy),
                     mix(mix(u[i000 + slice], u[i000 + slice + 1], fx), mix(u[i000 + slice + nx], u[i000 + slice + nx + 1], fx), fy), fz);
    v_out[idx] = mix(mix(mix(v[i000], v[i000 + 1], fx), mix(v[i000 + nx], v[i000 + nx + 1], fx), fy),
                     mix(mix(v[i000 + slice], v[i000 + slice + 1], fx), mix(v[i000 + slice + nx], v[i000 + slice + nx + 1], fx), fy), fz);
    w_out[idx] = mix(mix(mix(w[i000], w[i000 + 1], fx), mix(w[i000 + nx], w[i000 + nx + 1], fx), fy),
                     mix(mix(w[i000 + slice], w[i000 + slice + 1], fx), mix(w[i000 + slice + nx], w[i000 + slice + nx + 1], fx), fy), fz);
}}


// 4. 3D Incompressible Navier-Stokes Pressure Poisson Divergence
__kernel void kernel_navier_stokes_divergence_3d(
    __global const float* u, __global const float* v, __global const float* w,
    __global float* div_out, const float inv_dx,
    const int nx, const int ny, const int nz) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;
    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;
    float du_dx = (u[idx + 1] - u[idx]) * inv_dx;
    float dv_dy = (v[idx + nx] - v[idx]) * inv_dx;
    float dw_dz = (w[idx + slice] - w[idx]) * inv_dx;
    div_out[idx] = du_dx + dv_dy + dw_dz;
}}


// 5. 3D Incompressible Navier-Stokes Pressure Gradient Projection Step
__kernel void kernel_navier_stokes_project_velocity_3d(
    __global float* u, __global float* v, __global float* w,
    __global const float* pressure, const float dt_rho_inv_dx,
    const int nx, const int ny, const int nz) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;
    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;
    u[idx] -= dt_rho_inv_dx * (pressure[idx] - pressure[idx - 1]);
    v[idx] -= dt_rho_inv_dx * (pressure[idx] - pressure[idx - nx]);
    w[idx] -= dt_rho_inv_dx * (pressure[idx] - pressure[idx - slice]);
}}


// 6. 3D Cahn-Hilliard Phase Field 4th-Order Biharmonic Evolution
__kernel void kernel_cahn_hilliard_chemical_potential_3d(
    __global const float* c, __global float* mu,
    const float gamma, const float inv_dx2,
    const int nx, const int ny, const int nz) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;
    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;
    float c_val = c[idx];
    float lap_c = (c[idx - 1] + c[idx + 1] + c[idx - nx] + c[idx + nx] + c[idx - slice] + c[idx + slice] - 6.0f * c_val) * inv_dx2;
    // Chemical potential: mu = c^3 - c - gamma * laplacian(c)
    mu[idx] = c_val * c_val * c_val - c_val - gamma * lap_c;
}}


__kernel void kernel_cahn_hilliard_phase_update_3d(
    __global float* c, __global const float* mu,
    const float mobility_dt, const float inv_dx2,
    const int nx, const int ny, const int nz) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;
    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;
    float mu_val = mu[idx];
    float lap_mu = (mu[idx - 1] + mu[idx + 1] + mu[idx - nx] + mu[idx + nx] + mu[idx - slice] + mu[idx + slice] - 6.0f * mu_val) * inv_dx2;
    c[idx] += mobility_dt * lap_mu;
}}
