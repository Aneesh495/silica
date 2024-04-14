#!/usr/bin/env python3
"""
Master generator for the comprehensive Silica OpenCL specialization kernel suites.
Strictly verified to contain zero em-dashes and zero en-dashes.
Conforms strictly to OpenCL 1.2 standards with high-performance memory tiling and vectorization.
"""

import os
import sys

def verify_no_dashes(text, filename=""):
    for bad_char in ['\u2014', '\u2013']:
        if bad_char in text:
            raise ValueError(f"Prohibited character found in {filename}: {hex(ord(bad_char))}")

def gen_convolutions():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// Complete Suite of Separable, Dilated, Strided, and Direct N-D Convolutions',
        ''
    ]
    
    radii = [1, 2, 3, 4, 5, 7, 9, 11, 15]
    types = [
        ('float', 'float', '1.0f'),
        ('float2', 'float2', '(float2)(1.0f, 1.0f)'),
        ('float4', 'float4', '(float4)(1.0f, 1.0f, 1.0f, 1.0f)')
    ]
    
    # 1. Separable Convolutions
    for tname, scalar_t, def_val in types:
        for r in radii:
            ksize = 2 * r + 1
            # Horizontal
            lines.append(f'// Unrolled 2D Horizontal Convolution: Radius {r} (Window {ksize}), Type {tname}')
            lines.append(f'__kernel void kernel_conv2d_h_r{r}_{tname}(')
            lines.append(f'    __global const {tname}* input,')
            lines.append(f'    __global {tname}* output,')
            lines.append(f'    __constant const float* weights,')
            lines.append(f'    const int width,')
            lines.append(f'    const int height) {{')
            lines.append(f'    int x = get_global_id(0);')
            lines.append(f'    int y = get_global_id(1);')
            lines.append(f'    if (x >= width || y >= height) return;')
            lines.append(f'    {tname} sum = 0.0f;')
            lines.append(f'    size_t row_base = (size_t)y * (size_t)width;')
            for k in range(-r, r + 1):
                w_idx = k + r
                lines.append(f'    {{ int cx = clamp_i(x + ({k}), 0, width - 1); sum += input[row_base + cx] * weights[{w_idx}]; }}')
            lines.append(f'    output[row_base + x] = sum;')
            lines.append(f'}}\n')

            # Vertical
            lines.append(f'// Unrolled 2D Vertical Convolution: Radius {r} (Window {ksize}), Type {tname}')
            lines.append(f'__kernel void kernel_conv2d_v_r{r}_{tname}(')
            lines.append(f'    __global const {tname}* input,')
            lines.append(f'    __global {tname}* output,')
            lines.append(f'    __constant const float* weights,')
            lines.append(f'    const int width,')
            lines.append(f'    const int height) {{')
            lines.append(f'    int x = get_global_id(0);')
            lines.append(f'    int y = get_global_id(1);')
            lines.append(f'    if (x >= width || y >= height) return;')
            lines.append(f'    {tname} sum = 0.0f;')
            for k in range(-r, r + 1):
                w_idx = k + r
                lines.append(f'    {{ int cy = clamp_i(y + ({k}), 0, height - 1); sum += input[(size_t)cy * (size_t)width + (size_t)x] * weights[{w_idx}]; }}')
            lines.append(f'    output[(size_t)y * (size_t)width + (size_t)x] = sum;')
            lines.append(f'}}\n')

            # 3D Depth Slice
            lines.append(f'// Unrolled 3D Depth Z Convolution: Radius {r} (Window {ksize}), Type {tname}')
            lines.append(f'__kernel void kernel_conv3d_z_r{r}_{tname}(')
            lines.append(f'    __global const {tname}* input,')
            lines.append(f'    __global {tname}* output,')
            lines.append(f'    __constant const float* weights,')
            lines.append(f'    const int width, const int height, const int depth) {{')
            lines.append(f'    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);')
            lines.append(f'    if (x >= width || y >= height || z >= depth) return;')
            lines.append(f'    {tname} sum = 0.0f;')
            lines.append(f'    size_t slice = (size_t)width * (size_t)height;')
            lines.append(f'    size_t xy = (size_t)y * (size_t)width + (size_t)x;')
            for k in range(-r, r + 1):
                w_idx = k + r
                lines.append(f'    {{ int cz = clamp_i(z + ({k}), 0, depth - 1); sum += input[(size_t)cz * slice + xy] * weights[{w_idx}]; }}')
            lines.append(f'    output[(size_t)z * slice + xy] = sum;')
            lines.append(f'}}\n')

    # 2. Dilated / Atrous Convolutions
    dilations = [2, 3, 4]
    for d in dilations:
        for r in [1, 2, 3, 5]:
            lines.append(f'// Dilated 2D Convolution: Radius {r}, Dilation Rate {d}')
            lines.append(f'__kernel void kernel_conv2d_dilated_r{r}_d{d}(')
            lines.append(f'    __global const float* input, __global float* output,')
            lines.append(f'    __constant const float* weights, const int width, const int height) {{')
            lines.append(f'    int x = get_global_id(0); int y = get_global_id(1);')
            lines.append(f'    if (x >= width || y >= height) return;')
            lines.append(f'    float sum = 0.0f;')
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    w_idx = (dy + r) * (2 * r + 1) + (dx + r)
                    lines.append(f'    {{ int cx = clamp_i(x + ({dx * d}), 0, width - 1); int cy = clamp_i(y + ({dy * d}), 0, height - 1); sum += input[(size_t)cy * (size_t)width + (size_t)cx] * weights[{w_idx}]; }}')
            lines.append(f'    output[(size_t)y * (size_t)width + (size_t)x] = sum;')
            lines.append(f'}}\n')

    # 3. Direct 2D Convolutions
    for r in [1, 2, 3, 4, 5]:
        ksize = 2 * r + 1
        lines.append(f'// Direct 2D Non-Separable Convolution: Kernel {ksize}x{ksize}')
        lines.append(f'__kernel void kernel_conv2d_direct_{ksize}x{ksize}(')
        lines.append(f'    __global const float* input, __global float* output,')
        lines.append(f'    __constant const float* mask, const int width, const int height) {{')
        lines.append(f'    int x = get_global_id(0); int y = get_global_id(1);')
        lines.append(f'    if (x >= width || y >= height) return;')
        lines.append(f'    float sum = 0.0f;')
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                m_idx = (dy + r) * ksize + (dx + r)
                lines.append(f'    {{ int cx = clamp_i(x + ({dx}), 0, width - 1); int cy = clamp_i(y + ({dy}), 0, height - 1); sum += input[(size_t)cy * (size_t)width + (size_t)cx] * mask[{m_idx}]; }}')
        lines.append(f'    output[(size_t)y * (size_t)width + (size_t)x] = sum;')
        lines.append(f'}}\n')

    # 4. Direct 3D Convolutions (3x3x3, 5x5x5)
    for r in [1, 2]:
        ksize = 2 * r + 1
        lines.append(f'// Direct 3D Non-Separable Convolution: Kernel {ksize}x{ksize}x{ksize}')
        lines.append(f'__kernel void kernel_conv3d_direct_{ksize}x{ksize}x{ksize}(')
        lines.append(f'    __global const float* input, __global float* output,')
        lines.append(f'    __constant const float* mask, const int nx, const int ny, const int nz) {{')
        lines.append(f'    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);')
        lines.append(f'    if (x >= nx || y >= ny || z >= nz) return;')
        lines.append(f'    float sum = 0.0f;')
        lines.append(f'    size_t slice = (size_t)nx * (size_t)ny;')
        for dz in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    m_idx = (dz + r) * ksize * ksize + (dy + r) * ksize + (dx + r)
                    lines.append(f'    {{ int cx = clamp_i(x + ({dx}), 0, nx - 1); int cy = clamp_i(y + ({dy}), 0, ny - 1); int cz = clamp_i(z + ({dz}), 0, nz - 1); sum += input[(size_t)cz * slice + (size_t)cy * (size_t)nx + (size_t)cx] * mask[{m_idx}]; }}')
        lines.append(f'    output[(size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x] = sum;')
        lines.append(f'}}\n')

    return '\n'.join(lines)

def gen_gemm():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// Register-Tiled Matrix Multiplications (GEMM), Complex CGEMM, and Batched Solvers',
        ''
    ]
    
    # 1. Real GEMM variants (NN, NT, TN, TT)
    modes = ['NN', 'NT', 'TN', 'TT']
    for mode in modes:
        lines.append(f'// GEMM Mode {mode}: Tiled Matrix Multiply with Local Memory')
        lines.append(f'__kernel void kernel_sgemm_{mode.lower()}(')
        lines.append(f'    __global const float* A, __global const float* B, __global float* C,')
        lines.append(f'    const uint M, const uint N, const uint K, const float alpha, const float beta) {{')
        lines.append(f'    __local float l_A[16][16];')
        lines.append(f'    __local float l_B[16][16];')
        lines.append(f'    uint tx = get_local_id(0); uint ty = get_local_id(1);')
        lines.append(f'    uint row = get_group_id(1) * 16 + ty;')
        lines.append(f'    uint col = get_group_id(0) * 16 + tx;')
        lines.append(f'    float acc = 0.0f;')
        lines.append(f'    uint num_tiles = (K + 15) / 16;')
        lines.append(f'    for (uint t = 0; t < num_tiles; ++t) {{')
        if mode == 'NN':
            lines.append(f'        l_A[ty][tx] = (row < M && (t * 16 + tx) < K) ? A[row * K + t * 16 + tx] : 0.0f;')
            lines.append(f'        l_B[ty][tx] = ((t * 16 + ty) < K && col < N) ? B[(t * 16 + ty) * N + col] : 0.0f;')
        elif mode == 'NT':
            lines.append(f'        l_A[ty][tx] = (row < M && (t * 16 + tx) < K) ? A[row * K + t * 16 + tx] : 0.0f;')
            lines.append(f'        l_B[ty][tx] = ((t * 16 + ty) < K && col < N) ? B[col * K + t * 16 + ty] : 0.0f;')
        elif mode == 'TN':
            lines.append(f'        l_A[ty][tx] = (row < M && (t * 16 + tx) < K) ? A[(t * 16 + tx) * M + row] : 0.0f;')
            lines.append(f'        l_B[ty][tx] = ((t * 16 + ty) < K && col < N) ? B[(t * 16 + ty) * N + col] : 0.0f;')
        else: # TT
            lines.append(f'        l_A[ty][tx] = (row < M && (t * 16 + tx) < K) ? A[(t * 16 + tx) * M + row] : 0.0f;')
            lines.append(f'        l_B[ty][tx] = ((t * 16 + ty) < K && col < N) ? B[col * K + t * 16 + ty] : 0.0f;')
        lines.append(f'        sync_local();')
        lines.append(f'        #pragma unroll')
        lines.append(f'        for (int k = 0; k < 16; ++k) acc = fma(l_A[ty][k], l_B[k][tx], acc);')
        lines.append(f'        sync_local();')
        lines.append(f'    }}')
        lines.append(f'    if (row < M && col < N) {{')
        lines.append(f'        uint idx = row * N + col;')
        lines.append(f'        float orig = (beta != 0.0f) ? C[idx] : 0.0f;')
        lines.append(f'        C[idx] = alpha * acc + beta * orig;')
        lines.append(f'    }}')
        lines.append(f'}}\n')

    # 2. Complex Matrix Multiply (CGEMM)
    lines.append(f'// Complex Single-Precision Matrix Multiply (CGEMM)')
    lines.append(f'__kernel void kernel_cgemm_nn(')
    lines.append(f'    __global const float2* A, __global const float2* B, __global float2* C,')
    lines.append(f'    const uint M, const uint N, const uint K, const float2 alpha, const float2 beta) {{')
    lines.append(f'    uint row = get_global_id(1); uint col = get_global_id(0);')
    lines.append(f'    if (row >= M || col >= N) return;')
    lines.append(f'    float2 acc = (float2)(0.0f, 0.0f);')
    lines.append(f'    for (uint k = 0; k < K; ++k) {{')
    lines.append(f'        float2 a_val = A[row * K + k];')
    lines.append(f'        float2 b_val = B[k * N + col];')
    lines.append(f'        acc.x += a_val.x * b_val.x - a_val.y * b_val.y;')
    lines.append(f'        acc.y += a_val.x * b_val.y + a_val.y * b_val.x;')
    lines.append(f'    }}')
    lines.append(f'    uint idx = row * N + col;')
    lines.append(f'    float2 orig = (beta.x != 0.0f || beta.y != 0.0f) ? C[idx] : (float2)(0.0f, 0.0f);')
    lines.append(f'    float2 res;')
    lines.append(f'    res.x = (alpha.x * acc.x - alpha.y * acc.y) + (beta.x * orig.x - beta.y * orig.y);')
    lines.append(f'    res.y = (alpha.x * acc.y + alpha.y * acc.x) + (beta.x * orig.y + beta.y * orig.x);')
    lines.append(f'    C[idx] = res;')
    lines.append(f'}}\n')

    # 3. Batched small matrix inversion (2x2 up to 8x8)
    for n in [2, 3, 4, 5, 6, 8]:
        lines.append(f'// Batched {n}x{n} Matrix Inversion with Partial Pivoting')
        lines.append(f'__kernel void kernel_batched_inv_{n}x{n}(')
        lines.append(f'    __global const float* in_mats, __global float* out_invs, const uint batch_count) {{')
        lines.append(f'    uint bid = get_global_id(0); if (bid >= batch_count) return;')
        lines.append(f'    size_t base = (size_t)bid * {n * n};')
        lines.append(f'    float A[{n}][{n}]; float Inv[{n}][{n}];')
        lines.append(f'    for (int i = 0; i < {n}; ++i) for (int j = 0; j < {n}; ++j) {{ A[i][j] = in_mats[base + i * {n} + j]; Inv[i][j] = (i == j) ? 1.0f : 0.0f; }}')
        lines.append(f'    for (int i = 0; i < {n}; ++i) {{')
        lines.append(f'        int pivot = i; float max_v = fabs(A[i][i]);')
        lines.append(f'        for (int k = i + 1; k < {n}; ++k) {{ float v = fabs(A[k][i]); if (v > max_v) {{ max_v = v; pivot = k; }} }}')
        lines.append(f'        if (pivot != i) {{')
        lines.append(f'            for (int j = 0; j < {n}; ++j) {{')
        lines.append(f'                float ta = A[i][j]; A[i][j] = A[pivot][j]; A[pivot][j] = ta;')
        lines.append(f'                float ti = Inv[i][j]; Inv[i][j] = Inv[pivot][j]; Inv[pivot][j] = ti;')
        lines.append(f'            }}')
        lines.append(f'        }}')
        lines.append(f'        float diag = A[i][i]; float inv_d = (fabs(diag) > 1.0e-9f) ? (1.0f / diag) : 0.0f;')
        lines.append(f'        for (int j = 0; j < {n}; ++j) {{ A[i][j] *= inv_d; Inv[i][j] *= inv_d; }}')
        lines.append(f'        for (int k = 0; k < {n}; ++k) {{')
        lines.append(f'            if (k != i) {{')
        lines.append(f'                float f = A[k][i];')
        lines.append(f'                for (int j = 0; j < {n}; ++j) {{ A[k][j] -= f * A[i][j]; Inv[k][j] -= f * Inv[i][j]; }}')
        lines.append(f'            }}')
        lines.append(f'        }}')
        lines.append(f'    }}')
        lines.append(f'    for (int i = 0; i < {n}; ++i) for (int j = 0; j < {n}; ++j) out_invs[base + i * {n} + j] = Inv[i][j];')
        lines.append(f'}}\n')

    # 4. Batched Cholesky Decomposition (3x3 up to 8x8)
    for n in [3, 4, 6]:
        lines.append(f'// Batched {n}x{n} Cholesky Decomposition (A = L * L^T)')
        lines.append(f'__kernel void kernel_batched_cholesky_{n}x{n}(')
        lines.append(f'    __global const float* in_mats, __global float* out_L, const uint batch_count) {{')
        lines.append(f'    uint bid = get_global_id(0); if (bid >= batch_count) return;')
        lines.append(f'    size_t base = (size_t)bid * {n * n};')
        lines.append(f'    float L[{n}][{n}];')
        lines.append(f'    for (int i = 0; i < {n}; ++i) for (int j = 0; j < {n}; ++j) L[i][j] = 0.0f;')
        lines.append(f'    for (int i = 0; i < {n}; ++i) {{')
        lines.append(f'        for (int j = 0; j <= i; ++j) {{')
        lines.append(f'            float sum = 0.0f;')
        lines.append(f'            for (int k = 0; k < j; ++k) sum += L[i][k] * L[j][k];')
        lines.append(f'            if (i == j) {{')
        lines.append(f'                float val = in_mats[base + i * {n} + i] - sum;')
        lines.append(f'                L[i][j] = (val > 1.0e-9f) ? sqrt(val) : 0.0f;')
        lines.append(f'            }} else {{')
        lines.append(f'                float diag = L[j][j];')
        lines.append(f'                L[i][j] = (diag > 1.0e-9f) ? ((in_mats[base + i * {n} + j] - sum) / diag) : 0.0f;')
        lines.append(f'            }}')
        lines.append(f'        }}')
        lines.append(f'    }}')
        lines.append(f'    for (int i = 0; i < {n}; ++i) for (int j = 0; j < {n}; ++j) out_L[base + i * {n} + j] = L[i][j];')
        lines.append(f'}}\n')

    return '\n'.join(lines)

def gen_stencils():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// High-Order 2D and 3D Finite-Difference PDE Stencils for Wavefields and Transport',
        ''
    ]
    
    # 2D Stencils (5-pt, 9-pt, 13-pt)
    stencils_2d = [
        ('5pt_2nd_order', 1),
        ('9pt_4th_order', 2),
        ('13pt_6th_order', 3),
    ]
    for sname, r in stencils_2d:
        lines.append(f'// 2D Finite Difference Laplacian: {sname} (Radius {r})')
        lines.append(f'__kernel void kernel_stencil2d_{sname}(')
        lines.append(f'    __global const float* input, __global float* output,')
        lines.append(f'    const int nx, const int ny, const float inv_dx2) {{')
        lines.append(f'    int x = get_global_id(0); int y = get_global_id(1);')
        lines.append(f'    if (x < {r} || x >= nx - {r} || y < {r} || y >= ny - {r}) return;')
        lines.append(f'    size_t idx = (size_t)y * (size_t)nx + (size_t)x;')
        lines.append(f'    float center = input[idx];')
        if r == 1:
            lines.append(f'    float lap = (input[idx - 1] + input[idx + 1] + input[idx - nx] + input[idx + nx] - 4.0f * center);')
        elif r == 2:
            lines.append(f'    float lap = (- (input[idx - 2] + input[idx + 2] + input[idx - 2*nx] + input[idx + 2*nx])')
            lines.append(f'                 + 16.0f * (input[idx - 1] + input[idx + 1] + input[idx - nx] + input[idx + nx])')
            lines.append(f'                 - 60.0f * center) * (1.0f / 12.0f);')
        else:
            lines.append(f'    float lap = (2.0f * (input[idx - 3] + input[idx + 3] + input[idx - 3*nx] + input[idx + 3*nx])')
            lines.append(f'                 - 27.0f * (input[idx - 2] + input[idx + 2] + input[idx - 2*nx] + input[idx + 2*nx])')
            lines.append(f'                 + 270.0f * (input[idx - 1] + input[idx + 1] + input[idx - nx] + input[idx + nx])')
            lines.append(f'                 - 490.0f * center) * (1.0f / 180.0f);')
        lines.append(f'    output[idx] = lap * inv_dx2;')
        lines.append(f'}}\n')

    # 3D Stencils (7-pt, 13-pt, 19-pt, 25-pt)
    stencils_3d = [
        ('7pt_2nd_order', 1),
        ('13pt_4th_order', 2),
        ('19pt_isotropic', 1),
        ('25pt_6th_order', 3),
    ]
    for sname, r in stencils_3d:
        lines.append(f'// 3D Acoustic Wavefield Stencil: {sname} (Radius {r})')
        lines.append(f'__kernel void kernel_stencil3d_{sname}(')
        lines.append(f'    __global const float* u_curr, __global const float* u_prev,')
        lines.append(f'    __global const float* velocity_sq, __global float* u_next,')
        lines.append(f'    const float dt_sq_dx_sq, const float damping, const int nx, const int ny, const int nz) {{')
        lines.append(f'    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);')
        lines.append(f'    if (x < {r} || x >= nx - {r} || y < {r} || y >= ny - {r} || z < {r} || z >= nz - {r}) return;')
        lines.append(f'    size_t slice = (size_t)nx * (size_t)ny;')
        lines.append(f'    size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;')
        lines.append(f'    float curr = u_curr[idx];')
        if r == 1 and '7pt' in sname:
            lines.append(f'    float lap = (u_curr[idx - 1] + u_curr[idx + 1] + u_curr[idx - nx] + u_curr[idx + nx] + u_curr[idx - slice] + u_curr[idx + slice] - 6.0f * curr);')
        elif r == 2:
            lines.append(f'    float lap = (- (u_curr[idx - 2] + u_curr[idx + 2] + u_curr[idx - 2*nx] + u_curr[idx + 2*nx] + u_curr[idx - 2*slice] + u_curr[idx + 2*slice])')
            lines.append(f'                 + 16.0f * (u_curr[idx - 1] + u_curr[idx + 1] + u_curr[idx - nx] + u_curr[idx + nx] + u_curr[idx - slice] + u_curr[idx + slice])')
            lines.append(f'                 - 90.0f * curr) * (1.0f / 12.0f);')
        elif 'isotropic' in sname:
            lines.append(f'    float lap = (u_curr[idx - 1] + u_curr[idx + 1] + u_curr[idx - nx] + u_curr[idx + nx] + u_curr[idx - slice] + u_curr[idx + slice] - 6.0f * curr);')
        else: # 6th order
            lines.append(f'    float lap = (2.0f * (u_curr[idx - 3] + u_curr[idx + 3] + u_curr[idx - 3*nx] + u_curr[idx + 3*nx] + u_curr[idx - 3*slice] + u_curr[idx + 3*slice])')
            lines.append(f'                 - 27.0f * (u_curr[idx - 2] + u_curr[idx + 2] + u_curr[idx - 2*nx] + u_curr[idx + 2*nx] + u_curr[idx - 2*slice] + u_curr[idx + 2*slice])')
            lines.append(f'                 + 270.0f * (u_curr[idx - 1] + u_curr[idx + 1] + u_curr[idx - nx] + u_curr[idx + nx] + u_curr[idx - slice] + u_curr[idx + slice])')
            lines.append(f'                 - 735.0f * curr) * (1.0f / 180.0f);')
        lines.append(f'    float c2 = velocity_sq ? velocity_sq[idx] : 1.0f;')
        lines.append(f'    float prev = u_prev ? u_prev[idx] : curr;')
        lines.append(f'    u_next[idx] = (2.0f * curr - prev + c2 * dt_sq_dx_sq * lap) / (1.0f + damping);')
        lines.append(f'}}\n')

    # 3D Staggered Grid Elastodynamics (Stress-Velocity)
    lines.append(f'// 3D Staggered-Grid Elastodynamic Velocity Update')
    lines.append(f'__kernel void kernel_elastodynamics_velocity_update(')
    lines.append(f'    __global float* vx, __global float* vy, __global float* vz,')
    lines.append(f'    __global const float* sxx, __global const float* syy, __global const float* szz,')
    lines.append(f'    __global const float* sxy, __global const float* sxz, __global const float* syz,')
    lines.append(f'    __global const float* buoyancy, const float dt_dx, const int nx, const int ny, const int nz) {{')
    lines.append(f'    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);')
    lines.append(f'    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;')
    lines.append(f'    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;')
    lines.append(f'    float b = buoyancy[idx];')
    lines.append(f'    float dsxx_dx = sxx[idx] - sxx[idx - 1];')
    lines.append(f'    float dsxy_dy = sxy[idx] - sxy[idx - nx];')
    lines.append(f'    float dsxz_dz = sxz[idx] - sxz[idx - slice];')
    lines.append(f'    vx[idx] += dt_dx * b * (dsxx_dx + dsxy_dy + dsxz_dz);')
    lines.append(f'    float dsxy_dx = sxy[idx + 1] - sxy[idx];')
    lines.append(f'    float dsyy_dy = syy[idx] - syy[idx - nx];')
    lines.append(f'    float dsyz_dz = syz[idx] - syz[idx - slice];')
    lines.append(f'    vy[idx] += dt_dx * b * (dsxy_dx + dsyy_dy + dsyz_dz);')
    lines.append(f'    float dsxz_dx = sxz[idx + 1] - sxz[idx];')
    lines.append(f'    float dsyz_dy = syz[idx + nx] - syz[idx];')
    lines.append(f'    float dszz_dz = szz[idx] - szz[idx - slice];')
    lines.append(f'    vz[idx] += dt_dx * b * (dsxz_dx + dsyz_dy + dszz_dz);')
    lines.append(f'}}\n')

    return '\n'.join(lines)

def gen_vectorized_math():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// Vectorized Transcendental, Activation, and Arithmetic Primitives',
        ''
    ]
    
    vec_types = ['float', 'float2', 'float4', 'float8', 'float16']
    
    for vtype in vec_types:
        lines.append(f'// ========================================================')
        lines.append(f'// Type: {vtype}')
        lines.append(f'// ========================================================')
        
        ops = [
            ('add', 'a[gid] + b[gid]', 'binary'),
            ('sub', 'a[gid] - b[gid]', 'binary'),
            ('mul', 'a[gid] * b[gid]', 'binary'),
            ('div', 'a[gid] / b[gid]', 'binary'),
            ('exp', 'exp(a[gid])', 'unary'),
            ('log', 'log(a[gid])', 'unary'),
            ('sqrt', 'sqrt(a[gid])', 'unary'),
            ('rsqrt', 'rsqrt(a[gid])', 'unary'),
            ('sin', 'sin(a[gid])', 'unary'),
            ('cos', 'cos(a[gid])', 'unary'),
            ('tan', 'tan(a[gid])', 'unary'),
            ('sinh', 'sinh(a[gid])', 'unary'),
            ('cosh', 'cosh(a[gid])', 'unary'),
            ('tanh', 'tanh(a[gid])', 'unary'),
            ('abs', 'fabs(a[gid])', 'unary'),
            ('sigmoid', f'({vtype})(1.0f) / (({vtype})(1.0f) + exp(-a[gid]))', 'unary'),
            ('relu', f'max(a[gid], ({vtype})(0.0f))', 'unary'),
            ('leaky_relu', f'select(a[gid] * ({vtype})(0.01f), a[gid], a[gid] > ({vtype})(0.0f))', 'unary'),
            ('gelu', f'({vtype})(0.5f) * a[gid] * (({vtype})(1.0f) + tanh(({vtype})(0.79788456f) * (a[gid] + ({vtype})(0.044715f) * a[gid] * a[gid] * a[gid])))', 'unary'),
            ('elu', f'select(({vtype})(1.0f) * (exp(a[gid]) - ({vtype})(1.0f)), a[gid], a[gid] > ({vtype})(0.0f))', 'unary'),
            ('softplus', f'log(({vtype})(1.0f) + exp(a[gid]))', 'unary'),
            ('swish', f'a[gid] / (({vtype})(1.0f) + exp(-a[gid]))', 'unary'),
        ]
        
        for name, expr, arity in ops:
            if arity == 'binary':
                lines.append(f'__kernel void kernel_vec_{name}_{vtype}(__global const {vtype}* a, __global const {vtype}* b, __global {vtype}* c, const uint n) {{')
                lines.append(f'    uint gid = get_global_id(0); if (gid < n) c[gid] = {expr};')
                lines.append(f'}}\n')
            else:
                lines.append(f'__kernel void kernel_vec_{name}_{vtype}(__global const {vtype}* a, __global {vtype}* b, const uint n) {{')
                lines.append(f'    uint gid = get_global_id(0); if (gid < n) b[gid] = {expr};')
                lines.append(f'}}\n')

        lines.append(f'__kernel void kernel_vec_fma_{vtype}(__global const {vtype}* a, __global const {vtype}* b, __global const {vtype}* c, __global {vtype}* d, const uint n) {{')
        lines.append(f'    uint gid = get_global_id(0); if (gid < n) d[gid] = fma(a[gid], b[gid], c[gid]);')
        lines.append(f'}}\n')
        lines.append(f'__kernel void kernel_vec_clamp_{vtype}(__global const {vtype}* a, __global {vtype}* b, const float min_val, const float max_val, const uint n) {{')
        lines.append(f'    uint gid = get_global_id(0); if (gid < n) b[gid] = clamp(a[gid], ({vtype})(min_val), ({vtype})(max_val));')
        lines.append(f'}}\n')

    return '\n'.join(lines)

def gen_interpolation():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// High-Order 1D, 2D, and 3D Continuous Resampling and Free-Form Deformation Solvers',
        '',
        'inline float4 bspline_weights(float t) {',
        '    float it = 1.0f - t;',
        '    float w0 = it * it * it * (1.0f / 6.0f);',
        '    float w1 = (3.0f * t * t * t - 6.0f * t * t + 4.0f) * (1.0f / 6.0f);',
        '    float w2 = (-3.0f * t * t * t + 3.0f * t * t + 3.0f * t + 1.0f) * (1.0f / 6.0f);',
        '    float w3 = t * t * t * (1.0f / 6.0f);',
        '    return (float4)(w0, w1, w2, w3);',
        '}',
        '',
        'inline float4 catmull_rom_weights(float t) {',
        '    float t2 = t * t; float t3 = t2 * t;',
        '    float w0 = -0.5f * t3 + t2 - 0.5f * t;',
        '    float w1 =  1.5f * t3 - 2.5f * t2 + 1.0f;',
        '    float w2 = -1.5f * t3 + 2.0f * t2 + 0.5f * t;',
        '    float w3 =  0.5f * t3 - 0.5f * t2;',
        '    return (float4)(w0, w1, w2, w3);',
        '}',
        '',
        'inline float lanczos2_weight(float x) {',
        '    if (fabs(x) < 1.0e-6f) return 1.0f;',
        '    if (fabs(x) >= 2.0f) return 0.0f;',
        '    float px = PI_F * x;',
        '    return (sin(px) / px) * (sin(0.5f * px) / (0.5f * px));',
        '}',
        '',
        '// 2D Bicubic Catmull-Rom Image Resampler',
        '__kernel void kernel_interp2d_catmull_rom(',
        '    __global const float* in_img, __global float* out_img,',
        '    const int in_w, const int in_h, const int out_w, const int out_h,',
        '    const float scale_x, const float scale_y) {',
        '    int x = get_global_id(0); int y = get_global_id(1);',
        '    if (x >= out_w || y >= out_h) return;',
        '    float fx = (float)x * scale_x; float fy = (float)y * scale_y;',
        '    int ix = (int)floor(fx); int iy = (int)floor(fy);',
        '    float4 wx = catmull_rom_weights(fx - (float)ix);',
        '    float4 wy = catmull_rom_weights(fy - (float)iy);',
        '    float weights_x[4] = {wx.x, wx.y, wx.z, wx.w};',
        '    float weights_y[4] = {wy.x, wy.y, wy.z, wy.w};',
        '    float sum = 0.0f;',
        '    for (int dy = -1; dy <= 2; ++dy) {',
        '        int cy = clamp_i(iy + dy, 0, in_h - 1);',
        '        float wy_val = weights_y[dy + 1];',
        '        for (int dx = -1; dx <= 2; ++dx) {',
        '            int cx = clamp_i(ix + dx, 0, in_w - 1);',
        '            float wx_val = weights_x[dx + 1];',
        '            sum += in_img[(size_t)cy * (size_t)in_w + (size_t)cx] * (wx_val * wy_val);',
        '        }',
        '    }',
        '    out_img[(size_t)y * (size_t)out_w + (size_t)x] = sum;',
        '}}\n',
        '',
        '// 3D Tricubic B-Spline Continuous Volume Resampling',
        '__kernel void kernel_interp3d_cubic_bspline(',
        '    __global const float* in_volume, __global float* out_volume,',
        '    const int in_nx, const int in_ny, const int in_nz,',
        '    const int out_nx, const int out_ny, const int out_nz,',
        '    const float scale_x, const float scale_y, const float scale_z) {',
        '    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);',
        '    if (x >= out_nx || y >= out_ny || z >= out_nz) return;',
        '    float fx = (float)x * scale_x; float fy = (float)y * scale_y; float fz = (float)z * scale_z;',
        '    int ix = (int)floor(fx); int iy = (int)floor(fy); int iz = (int)floor(fz);',
        '    float4 wx = bspline_weights(fx - (float)ix);',
        '    float4 wy = bspline_weights(fy - (float)iy);',
        '    float4 wz = bspline_weights(fz - (float)iz);',
        '    float weights_x[4] = {wx.x, wx.y, wx.z, wx.w};',
        '    float weights_y[4] = {wy.x, wy.y, wy.z, wy.w};',
        '    float weights_z[4] = {wz.x, wz.y, wz.z, wz.w};',
        '    float sum = 0.0f; size_t in_slice = (size_t)in_nx * (size_t)in_ny;',
        '    for (int dz = -1; dz <= 2; ++dz) {',
        '        int cz = clamp_i(iz + dz, 0, in_nz - 1); float wz_val = weights_z[dz + 1];',
        '        for (int dy = -1; dy <= 2; ++dy) {',
        '            int cy = clamp_i(iy + dy, 0, in_ny - 1); float wy_val = weights_y[dy + 1];',
        '            for (int dx = -1; dx <= 2; ++dx) {',
        '                int cx = clamp_i(ix + dx, 0, in_nx - 1); float wx_val = weights_x[dx + 1];',
        '                sum += in_volume[(size_t)cz * in_slice + (size_t)cy * (size_t)in_nx + (size_t)cx] * (wx_val * wy_val * wz_val);',
        '            }',
        '        }',
        '    }',
        '    out_volume[(size_t)z * ((size_t)out_nx * (size_t)out_ny) + (size_t)y * (size_t)out_nx + (size_t)x] = sum;',
        '}}\n',
        '',
        '// 3D Free-Form Deformation (FFD) Displacement Grid Evaluator',
        '__kernel void kernel_ffd_transform_grid(',
        '    __global const float4* control_points, __global float4* out_deformation_field,',
        '    const int grid_nx, const int grid_ny, const int grid_nz,',
        '    const int nx, const int ny, const int nz,',
        '    const float spacing_x, const float spacing_y, const float spacing_z) {',
        '    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);',
        '    if (x >= nx || y >= ny || z >= nz) return;',
        '    float gx = (float)x / spacing_x; float gy = (float)y / spacing_y; float gz = (float)z / spacing_z;',
        '    int cx = (int)floor(gx); int cy = (int)floor(gy); int cz = (int)floor(gz);',
        '    float4 wx = bspline_weights(gx - (float)cx);',
        '    float4 wy = bspline_weights(gy - (float)cy);',
        '    float4 wz = bspline_weights(gz - (float)cz);',
        '    float weights_x[4] = {wx.x, wx.y, wx.z, wx.w};',
        '    float weights_y[4] = {wy.x, wy.y, wy.z, wy.w};',
        '    float weights_z[4] = {wz.x, wz.y, wz.z, wz.w};',
        '    float4 disp = (float4)(0.0f, 0.0f, 0.0f, 0.0f); size_t grid_slice = (size_t)grid_nx * (size_t)grid_ny;',
        '    for (int l = 0; l < 4; ++l) {',
        '        int pz = clamp_i(cz - 1 + l, 0, grid_nz - 1); float wz_val = weights_z[l];',
        '        for (int m = 0; m < 4; ++m) {',
        '            int py = clamp_i(cy - 1 + m, 0, grid_ny - 1); float wy_val = weights_y[m];',
        '            for (int n = 0; n < 4; ++n) {',
        '                int px = clamp_i(cx - 1 + n, 0, grid_nx - 1); float wx_val = weights_x[n];',
        '                disp += control_points[(size_t)pz * grid_slice + (size_t)py * (size_t)grid_nx + (size_t)px] * (wx_val * wy_val * wz_val);',
        '            }',
        '        }',
        '    }',
        '    out_deformation_field[(size_t)z * ((size_t)nx * (size_t)ny) + (size_t)y * (size_t)nx + (size_t)x] = disp;',
        '}}\n'
    ]
    return '\n'.join(lines)

def gen_wavelets():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// Discrete Wavelet Transform (DWT) and Gabor Multiscale Decomposition Filter Banks',
        '',
        '// 1D Haar Lifting Forward Transform Row Pass',
        '__kernel void kernel_haar_forward_1d_rows(',
        '    __global const float* input, __global float* output, const int width, const int height) {',
        '    int x = get_global_id(0); int y = get_global_id(1); int half_w = width / 2;',
        '    if (x >= half_w || y >= height) return;',
        '    size_t base = (size_t)y * (size_t)width;',
        '    float even_val = input[base + 2 * x]; float odd_val  = input[base + 2 * x + 1];',
        '    float d = (odd_val - even_val) * 0.70710678f;',
        '    float s = (even_val + odd_val) * 0.70710678f;',
        '    output[base + x] = s; output[base + half_w + x] = d;',
        '}',
        '',
        '// 1D Haar Inverse Transform Row Pass',
        '__kernel void kernel_haar_inverse_1d_rows(',
        '    __global const float* input, __global float* output, const int width, const int height) {',
        '    int x = get_global_id(0); int y = get_global_id(1); int half_w = width / 2;',
        '    if (x >= half_w || y >= height) return;',
        '    size_t base = (size_t)y * (size_t)width;',
        '    float s = input[base + x]; float d = input[base + half_w + x];',
        '    float inv_sqrt2 = 0.70710678f;',
        '    output[base + 2 * x]     = (s - d) * inv_sqrt2;',
        '    output[base + 2 * x + 1] = (s + d) * inv_sqrt2;',
        '}',
        '',
        '// CDF 9/7 Biorthogonal Wavelet Forward Lifting Steps',
        '__kernel void kernel_cdf97_predict_step(__global float* d, __global const float* s, const float alpha, const int length) {',
        '    int i = get_global_id(0); if (i >= length) return;',
        '    int prev = clamp_i(i, 0, length - 1); int next = clamp_i(i + 1, 0, length - 1);',
        '    d[i] += alpha * (s[prev] + s[next]);',
        '}',
        '',
        '__kernel void kernel_cdf97_update_step(__global float* s, __global const float* d, const float beta, const int length) {',
        '    int i = get_global_id(0); if (i >= length) return;',
        '    int prev = clamp_i(i - 1, 0, length - 1); int curr = clamp_i(i, 0, length - 1);',
        '    s[i] += beta * (d[prev] + d[curr]);',
        '}',
        '',
        '// Multiscale 2D Gabor Filter Bank (Texture & Edge Frequency Analysis)',
        '__kernel void kernel_gabor_filter_2d(',
        '    __global const float* input, __global float2* out_gabor_complex,',
        '    const int width, const int height, const float lambda, const float theta,',
        '    const float psi, const float sigma, const float gamma, const int radius) {',
        '    int x = get_global_id(0); int y = get_global_id(1);',
        '    if (x >= width || y >= height) return;',
        '    float cos_t = cos(theta); float sin_t = sin(theta);',
        '    float2 acc = (float2)(0.0f, 0.0f);',
        '    float sigma_sq = sigma * sigma;',
        '    for (int dy = -radius; dy <= radius; ++dy) {',
        '        int cy = clamp_i(y + dy, 0, height - 1);',
        '        for (int dx = -radius; dx <= radius; ++dx) {',
        '            int cx = clamp_i(x + dx, 0, width - 1);',
        '            float x_p =  (float)dx * cos_t + (float)dy * sin_t;',
        '            float y_p = -(float)dx * sin_t + (float)dy * cos_t;',
        '            float envelope = exp(- (x_p * x_p + gamma * gamma * y_p * y_p) / (2.0f * sigma_sq));',
        '            float carrier_phase = (2.0f * PI_F * x_p / lambda) + psi;',
        '            float g_real = envelope * cos(carrier_phase);',
        '            float g_imag = envelope * sin(carrier_phase);',
        '            float val = input[(size_t)cy * (size_t)width + (size_t)cx];',
        '            acc.x += val * g_real; acc.y += val * g_imag;',
        '        }',
        '    }',
        '    out_gabor_complex[(size_t)y * (size_t)width + (size_t)x] = acc;',
        '}}\n'
    ]
    return '\n'.join(lines)

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(root, 'kernels', 'specialized')
    os.makedirs(target_dir, exist_ok=True)
    
    # Generate large modules
    files = {
        'convolutions_specialized.cl': gen_convolutions(),
        'gemm_specialized.cl': gen_gemm(),
        'stencils_specialized.cl': gen_stencils(),
        'interpolation_specialized.cl': gen_interpolation(),
        'wavelets_specialized.cl': gen_wavelets(),
        'vectorized_math_ops.cl': gen_vectorized_math(),
    }
    
    for filename, content in files.items():
        verify_no_dashes(content, filename)
        path = os.path.join(target_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filename}: {len(content.splitlines())} lines")

if __name__ == '__main__':
    main()
