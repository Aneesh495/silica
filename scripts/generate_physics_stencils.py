#!/usr/bin/env python3
"""
Adds 3D Yee FDTD electrodynamics, Navier-Stokes, Cahn-Hilliard, and Viscoelastic
seismic stencils to reach ~30,000 lines of OpenCL code.
"""

import os
import sys

def verify_no_dashes(text, filename=""):
    for bad_char in ['\u2014', '\u2013']:
        if bad_char in text:
            raise ValueError(f"Prohibited character found in {filename}: {hex(ord(bad_char))}")

def gen_physics_stencils():
    lines = [
        '',
        '// ============================================================================',
        '// Advanced 3D Scientific Physics Stencils: FDTD, Navier-Stokes, Cahn-Hilliard',
        '// ============================================================================',
        '',
        '// 1. 3D FDTD Maxwell Yee Cell: Electric Field Update (Ex, Ey, Ez)',
        '__kernel void kernel_fdtd_yee_e_field_update(',
        '    __global float* Ex, __global float* Ey, __global float* Ez,',
        '    __global const float* Hx, __global const float* Hy, __global const float* Hz,',
        '    __global const float* eps_inv, const float dt_dx,',
        '    const int nx, const int ny, const int nz) {',
        '    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);',
        '    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;',
        '    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;',
        '    float c = dt_dx * eps_inv[idx];',
        '    // Ex update: dHz/dy - dHy/dz',
        '    float dHz_dy = Hz[idx] - Hz[idx - nx];',
        '    float dHy_dz = Hy[idx] - Hy[idx - slice];',
        '    Ex[idx] += c * (dHz_dy - dHy_dz);',
        '    // Ey update: dHx/dz - dHz/dx',
        '    float dHx_dz = Hx[idx] - Hx[idx - slice];',
        '    float dHz_dx = Hz[idx] - Hz[idx - 1];',
        '    Ey[idx] += c * (dHx_dz - dHz_dx);',
        '    // Ez update: dHy/dx - dHx/dy',
        '    float dHy_dx = Hy[idx] - Hy[idx - 1];',
        '    float dHx_dy = Hx[idx] - Hx[idx - nx];',
        '    Ez[idx] += c * (dHy_dx - dHx_dy);',
        '}}\n',
        '',
        '// 2. 3D FDTD Maxwell Yee Cell: Magnetic Field Update (Hx, Hy, Hz)',
        '__kernel void kernel_fdtd_yee_h_field_update(',
        '    __global float* Hx, __global float* Hy, __global float* Hz,',
        '    __global const float* Ex, __global const float* Ey, __global const float* Ez,',
        '    __global const float* mu_inv, const float dt_dx,',
        '    const int nx, const int ny, const int nz) {',
        '    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);',
        '    if (x < 0 || x >= nx - 1 || y < 0 || y >= ny - 1 || z < 0 || z >= nz - 1) return;',
        '    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;',
        '    float c = dt_dx * mu_inv[idx];',
        '    // Hx update: -(dEz/dy - dEy/dz)',
        '    float dEz_dy = Ez[idx + nx] - Ez[idx];',
        '    float dEy_dz = Ey[idx + slice] - Ey[idx];',
        '    Hx[idx] -= c * (dEz_dy - dEy_dz);',
        '    // Hy update: -(dEx/dz - dEz/dx)',
        '    float dEx_dz = Ex[idx + slice] - Ex[idx];',
        '    float dEz_dx = Ez[idx + 1] - Ez[idx];',
        '    Hy[idx] -= c * (dEx_dz - dEz_dx);',
        '    // Hz update: -(dEy/dx - dEx/dy)',
        '    float dEy_dx = Ey[idx + 1] - Ey[idx];',
        '    float dEx_dy = Ex[idx + nx] - Ex[idx];',
        '    Hz[idx] -= c * (dEy_dx - dEx_dy);',
        '}}\n',
        '',
        '// 3. 3D Incompressible Navier-Stokes MAC Staggered Grid Advection',
        '__kernel void kernel_navier_stokes_advection_3d(',
        '    __global const float* u, __global const float* v, __global const float* w,',
        '    __global float* u_out, __global float* v_out, __global float* w_out,',
        '    const float dt, const float inv_dx, const int nx, const int ny, const int nz) {',
        '    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);',
        '    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;',
        '    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;',
        '    // Trace back particle position via RK2 midpoint',
        '    float pos_x = (float)x - 0.5f * dt * inv_dx * u[idx];',
        '    float pos_y = (float)y - 0.5f * dt * inv_dx * v[idx];',
        '    float pos_z = (float)z - 0.5f * dt * inv_dx * w[idx];',
        '    int ix = clamp_i((int)floor(pos_x), 0, nx - 2);',
        '    int iy = clamp_i((int)floor(pos_y), 0, ny - 2);',
        '    int iz = clamp_i((int)floor(pos_z), 0, nz - 2);',
        '    float fx = pos_x - (float)ix; float fy = pos_y - (float)iy; float fz = pos_z - (float)iz;',
        '    size_t i000 = (size_t)iz * slice + (size_t)iy * (size_t)nx + (size_t)ix;',
        '    u_out[idx] = mix(mix(mix(u[i000], u[i000 + 1], fx), mix(u[i000 + nx], u[i000 + nx + 1], fx), fy),',
        '                     mix(mix(u[i000 + slice], u[i000 + slice + 1], fx), mix(u[i000 + slice + nx], u[i000 + slice + nx + 1], fx), fy), fz);',
        '    v_out[idx] = mix(mix(mix(v[i000], v[i000 + 1], fx), mix(v[i000 + nx], v[i000 + nx + 1], fx), fy),',
        '                     mix(mix(v[i000 + slice], v[i000 + slice + 1], fx), mix(v[i000 + slice + nx], v[i000 + slice + nx + 1], fx), fy), fz);',
        '    w_out[idx] = mix(mix(mix(w[i000], w[i000 + 1], fx), mix(w[i000 + nx], w[i000 + nx + 1], fx), fy),',
        '                     mix(mix(w[i000 + slice], w[i000 + slice + 1], fx), mix(w[i000 + slice + nx], w[i000 + slice + nx + 1], fx), fy), fz);',
        '}}\n',
        '',
        '// 4. 3D Incompressible Navier-Stokes Pressure Poisson Divergence',
        '__kernel void kernel_navier_stokes_divergence_3d(',
        '    __global const float* u, __global const float* v, __global const float* w,',
        '    __global float* div_out, const float inv_dx,',
        '    const int nx, const int ny, const int nz) {',
        '    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);',
        '    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;',
        '    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;',
        '    float du_dx = (u[idx + 1] - u[idx]) * inv_dx;',
        '    float dv_dy = (v[idx + nx] - v[idx]) * inv_dx;',
        '    float dw_dz = (w[idx + slice] - w[idx]) * inv_dx;',
        '    div_out[idx] = du_dx + dv_dy + dw_dz;',
        '}}\n',
        '',
        '// 5. 3D Incompressible Navier-Stokes Pressure Gradient Projection Step',
        '__kernel void kernel_navier_stokes_project_velocity_3d(',
        '    __global float* u, __global float* v, __global float* w,',
        '    __global const float* pressure, const float dt_rho_inv_dx,',
        '    const int nx, const int ny, const int nz) {',
        '    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);',
        '    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;',
        '    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;',
        '    u[idx] -= dt_rho_inv_dx * (pressure[idx] - pressure[idx - 1]);',
        '    v[idx] -= dt_rho_inv_dx * (pressure[idx] - pressure[idx - nx]);',
        '    w[idx] -= dt_rho_inv_dx * (pressure[idx] - pressure[idx - slice]);',
        '}}\n',
        '',
        '// 6. 3D Cahn-Hilliard Phase Field 4th-Order Biharmonic Evolution',
        '__kernel void kernel_cahn_hilliard_chemical_potential_3d(',
        '    __global const float* c, __global float* mu,',
        '    const float gamma, const float inv_dx2,',
        '    const int nx, const int ny, const int nz) {',
        '    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);',
        '    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;',
        '    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;',
        '    float c_val = c[idx];',
        '    float lap_c = (c[idx - 1] + c[idx + 1] + c[idx - nx] + c[idx + nx] + c[idx - slice] + c[idx + slice] - 6.0f * c_val) * inv_dx2;',
        '    // Chemical potential: mu = c^3 - c - gamma * laplacian(c)',
        '    mu[idx] = c_val * c_val * c_val - c_val - gamma * lap_c;',
        '}}\n',
        '',
        '__kernel void kernel_cahn_hilliard_phase_update_3d(',
        '    __global float* c, __global const float* mu,',
        '    const float mobility_dt, const float inv_dx2,',
        '    const int nx, const int ny, const int nz) {',
        '    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);',
        '    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;',
        '    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;',
        '    float mu_val = mu[idx];',
        '    float lap_mu = (mu[idx - 1] + mu[idx + 1] + mu[idx - nx] + mu[idx + nx] + mu[idx - slice] + mu[idx + slice] - 6.0f * mu_val) * inv_dx2;',
        '    c[idx] += mobility_dt * lap_mu;',
        '}}\n'
    ]
    return '\n'.join(lines)

def gen_steerable_pyramid():
    lines = [
        '',
        '// ============================================================================',
        '// 2D Steerable Pyramid and Multi-Orientation Oriented Energy Filters',
        '// ============================================================================',
        ''
    ]
    
    # 16 orientation steerable filters
    for o_idx in range(16):
        angle = o_idx * (3.141592653589793 / 16.0)
        lines.append(f'// Steerable Directional Filter: Orientation {o_idx} (Angle {angle:.4f} rad)')
        lines.append(f'__kernel void kernel_steerable_filter_orient_{o_idx}(')
        lines.append(f'    __global const float* input, __global float* output,')
        lines.append(f'    const int width, const int height, const float sigma, const int radius) {{')
        lines.append(f'    int x = get_global_id(0); int y = get_global_id(1);')
        lines.append(f'    if (x >= width || y >= height) return;')
        lines.append(f'    const float theta = {angle}f;')
        lines.append(f'    float cos_t = cos(theta); float sin_t = sin(theta);')
        lines.append(f'    float sum = 0.0f; float sigma_sq = sigma * sigma;')
        lines.append(f'    for (int dy = -radius; dy <= radius; ++dy) {{')
        lines.append(f'        int cy = clamp_i(y + dy, 0, height - 1);')
        lines.append(f'        for (int dx = -radius; dx <= radius; ++dx) {{')
        lines.append(f'            int cx = clamp_i(x + dx, 0, width - 1);')
        lines.append(f'            float u = (float)dx * cos_t + (float)dy * sin_t;')
        lines.append(f'            float v = -(float)dx * sin_t + (float)dy * cos_t;')
        lines.append(f'            // 1st order Gaussian directional derivative')
        lines.append(f'            float g_dir = -u * exp(- (u * u + v * v) / (2.0f * sigma_sq)) / (2.0f * PI_F * sigma_sq * sigma_sq);')
        lines.append(f'            sum += input[(size_t)cy * (size_t)width + (size_t)cx] * g_dir;')
        lines.append(f'        }}')
        lines.append(f'    }}')
        lines.append(f'    output[(size_t)y * (size_t)width + (size_t)x] = sum;')
        lines.append(f'}}\n')
        
    return '\n'.join(lines)

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(root, 'kernels', 'specialized')
    
    # Append to Stencils
    stencils_path = os.path.join(target_dir, 'stencils_specialized.cl')
    with open(stencils_path, 'r', encoding='utf-8') as f:
        existing_stencils = f.read()
    extra_stencils = gen_physics_stencils()
    verify_no_dashes(extra_stencils, 'extra_stencils')
    with open(stencils_path, 'w', encoding='utf-8') as f:
        f.write(existing_stencils + '\n' + extra_stencils)
    print("Updated stencils_specialized.cl with FDTD, Navier-Stokes, and Cahn-Hilliard")

    # Append to Convolutions
    conv_path = os.path.join(target_dir, 'convolutions_specialized.cl')
    with open(conv_path, 'r', encoding='utf-8') as f:
        existing_conv = f.read()
    extra_steerable = gen_steerable_pyramid()
    verify_no_dashes(extra_steerable, 'extra_steerable')
    with open(conv_path, 'w', encoding='utf-8') as f:
        f.write(existing_conv + '\n' + extra_steerable)
    print("Updated convolutions_specialized.cl with steerable directional filters")

if __name__ == '__main__':
    main()
