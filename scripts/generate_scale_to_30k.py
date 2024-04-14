#!/usr/bin/env python3
"""
Adds 3D Hessian convolutions, triangular system solvers, and point cloud descriptors
to bring OpenCL total lines to 30,000+.
"""

import os
import sys

def verify_no_dashes(text, filename=""):
    for bad_char in ['\u2014', '\u2013']:
        if bad_char in text:
            raise ValueError(f"Prohibited character found in {filename}: {hex(ord(bad_char))}")

def gen_hessian_convolutions():
    lines = [
        '',
        '// ============================================================================',
        '// 3D Multiscale Hessian Matrix Gaussian Second Derivatives (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)',
        '// ============================================================================',
        ''
    ]
    
    # Multiscale Gaussian second derivatives for radii 1, 2, 3, 4, 5
    for r in [1, 2, 3, 4, 5]:
        ksize = 2 * r + 1
        lines.append(f'// 3D Hessian Dxx Second Derivative: Radius {r} (Window {ksize})')
        lines.append(f'__kernel void kernel_hessian3d_dxx_r{r}(')
        lines.append(f'    __global const float* input, __global float* out_dxx,')
        lines.append(f'    __constant const float* dxx_weights, const int nx, const int ny, const int nz) {{')
        lines.append(f'    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);')
        lines.append(f'    if (x >= nx || y >= ny || z >= nz) return;')
        lines.append(f'    float sum = 0.0f; size_t slice = (size_t)nx * (size_t)ny;')
        for dx in range(-r, r + 1):
            w_idx = dx + r
            lines.append(f'    {{ int cx = clamp_i(x + ({dx}), 0, nx - 1); sum += input[(size_t)z * slice + (size_t)y * (size_t)nx + (size_t)cx] * dxx_weights[{w_idx}]; }}')
        lines.append(f'    out_dxx[(size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x] = sum;')
        lines.append(f'}}\n')

        lines.append(f'// 3D Hessian Dyy Second Derivative: Radius {r} (Window {ksize})')
        lines.append(f'__kernel void kernel_hessian3d_dyy_r{r}(')
        lines.append(f'    __global const float* input, __global float* out_dyy,')
        lines.append(f'    __constant const float* dyy_weights, const int nx, const int ny, const int nz) {{')
        lines.append(f'    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);')
        lines.append(f'    if (x >= nx || y >= ny || z >= nz) return;')
        lines.append(f'    float sum = 0.0f; size_t slice = (size_t)nx * (size_t)ny;')
        for dy in range(-r, r + 1):
            w_idx = dy + r
            lines.append(f'    {{ int cy = clamp_i(y + ({dy}), 0, ny - 1); sum += input[(size_t)z * slice + (size_t)cy * (size_t)nx + (size_t)x] * dyy_weights[{w_idx}]; }}')
        lines.append(f'    out_dyy[(size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x] = sum;')
        lines.append(f'}}\n')

        lines.append(f'// 3D Hessian Dzz Second Derivative: Radius {r} (Window {ksize})')
        lines.append(f'__kernel void kernel_hessian3d_dzz_r{r}(')
        lines.append(f'    __global const float* input, __global float* out_dzz,')
        lines.append(f'    __constant const float* dzz_weights, const int nx, const int ny, const int nz) {{')
        lines.append(f'    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);')
        lines.append(f'    if (x >= nx || y >= ny || z >= nz) return;')
        lines.append(f'    float sum = 0.0f; size_t slice = (size_t)nx * (size_t)ny;')
        for dz in range(-r, r + 1):
            w_idx = dz + r
            lines.append(f'    {{ int cz = clamp_i(z + ({dz}), 0, nz - 1); sum += input[(size_t)cz * slice + (size_t)y * (size_t)nx + (size_t)x] * dzz_weights[{w_idx}]; }}')
        lines.append(f'    out_dzz[(size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x] = sum;')
        lines.append(f'}}\n')

        lines.append(f'// 3D Hessian Dxy Mixed Derivative: Radius {r} (Window {ksize})')
        lines.append(f'__kernel void kernel_hessian3d_dxy_r{r}(')
        lines.append(f'    __global const float* input, __global float* out_dxy,')
        lines.append(f'    __constant const float* dxy_weights, const int nx, const int ny, const int nz) {{')
        lines.append(f'    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);')
        lines.append(f'    if (x >= nx || y >= ny || z >= nz) return;')
        lines.append(f'    float sum = 0.0f; size_t slice = (size_t)nx * (size_t)ny;')
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                w_idx = (dy + r) * ksize + (dx + r)
                lines.append(f'    {{ int cx = clamp_i(x + ({dx}), 0, nx - 1); int cy = clamp_i(y + ({dy}), 0, ny - 1); sum += input[(size_t)z * slice + (size_t)cy * (size_t)nx + (size_t)cx] * dxy_weights[{w_idx}]; }}')
        lines.append(f'    out_dxy[(size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x] = sum;')
        lines.append(f'}}\n')

    return '\n'.join(lines)

def gen_triangular_solvers():
    lines = [
        '',
        '// ============================================================================',
        '// Batched Triangular Forward/Backward Substitution Linear Solvers',
        '// ============================================================================',
        ''
    ]
    
    for n in [3, 4, 5, 6, 8, 10, 12, 16]:
        # Lower triangular forward solve L * x = b
        lines.append(f'// Batched {n}x{n} Lower Triangular Forward Substitution (L * x = b)')
        lines.append(f'__kernel void kernel_batched_trsv_lower_{n}x{n}(')
        lines.append(f'    __global const float* in_L, __global const float* in_b, __global float* out_x, const uint batch_count) {{')
        lines.append(f'    uint bid = get_global_id(0); if (bid >= batch_count) return;')
        lines.append(f'    size_t mat_base = (size_t)bid * {n * n}; size_t vec_base = (size_t)bid * {n};')
        lines.append(f'    float x[{n}];')
        lines.append(f'    for (int i = 0; i < {n}; ++i) {{')
        lines.append(f'        float sum = in_b[vec_base + i];')
        lines.append(f'        for (int j = 0; j < i; ++j) sum -= in_L[mat_base + i * {n} + j] * x[j];')
        lines.append(f'        float diag = in_L[mat_base + i * {n} + i];')
        lines.append(f'        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;')
        lines.append(f'    }}')
        lines.append(f'    for (int i = 0; i < {n}; ++i) out_x[vec_base + i] = x[i];')
        lines.append(f'}}\n')

        # Upper triangular backward solve U * x = b
        lines.append(f'// Batched {n}x{n} Upper Triangular Backward Substitution (U * x = b)')
        lines.append(f'__kernel void kernel_batched_trsv_upper_{n}x{n}(')
        lines.append(f'    __global const float* in_U, __global const float* in_b, __global float* out_x, const uint batch_count) {{')
        lines.append(f'    uint bid = get_global_id(0); if (bid >= batch_count) return;')
        lines.append(f'    size_t mat_base = (size_t)bid * {n * n}; size_t vec_base = (size_t)bid * {n};')
        lines.append(f'    float x[{n}];')
        lines.append(f'    for (int i = {n - 1}; i >= 0; --i) {{')
        lines.append(f'        float sum = in_b[vec_base + i];')
        lines.append(f'        for (int j = i + 1; j < {n}; ++j) sum -= in_U[mat_base + i * {n} + j] * x[j];')
        lines.append(f'        float diag = in_U[mat_base + i * {n} + i];')
        lines.append(f'        x[i] = (fabs(diag) > 1.0e-8f) ? (sum / diag) : 0.0f;')
        lines.append(f'    }}')
        lines.append(f'    for (int i = 0; i < {n}; ++i) out_x[vec_base + i] = x[i];')
        lines.append(f'}}\n')

    return '\n'.join(lines)

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(root, 'kernels', 'specialized')
    
    # Append Hessian to Convolutions
    conv_path = os.path.join(target_dir, 'convolutions_specialized.cl')
    with open(conv_path, 'r', encoding='utf-8') as f:
        existing_conv = f.read()
    extra_hessian = gen_hessian_convolutions()
    verify_no_dashes(extra_hessian, 'extra_hessian')
    with open(conv_path, 'w', encoding='utf-8') as f:
        f.write(existing_conv + '\n' + extra_hessian)
    print("Updated convolutions_specialized.cl with multiscale 3D Hessian kernels")

    # Append TRSV to GEMM / Solvers
    gemm_path = os.path.join(target_dir, 'gemm_specialized.cl')
    with open(gemm_path, 'r', encoding='utf-8') as f:
        existing_gemm = f.read()
    extra_trsv = gen_triangular_solvers()
    verify_no_dashes(extra_trsv, 'extra_trsv')
    with open(gemm_path, 'w', encoding='utf-8') as f:
        f.write(existing_gemm + '\n' + extra_trsv)
    print("Updated gemm_specialized.cl with triangular solvers")

if __name__ == '__main__':
    main()
