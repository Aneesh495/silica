#!/usr/bin/env python3
"""
Full specialization kernel suite generator for Silica.
Generates comprehensive, mathematically rigorous OpenCL 1.2 kernel libraries
spanning separable/direct convolutions, tiled SGEMM/CGEMM, batched solvers,
high-order PDE stencils, continuous spline/RBF resamplers, wavelets, and vector intrinsics.
Verified to contain zero em dashes and zero en dashes.
"""

import os
import sys

def verify_no_dashes(text, filename=""):
    for bad_char in ['\u2014', '\u2013']:
        if bad_char in text:
            raise ValueError(f"Prohibited character found in {filename}: {hex(ord(bad_char))}")

def gen_convolutions_specialized():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// ============================================================================',
        '// Specialized N-D Convolutions: Separable, Direct, Dilated, and Multi-Boundary',
        '// ============================================================================',
        ''
    ]
    
    radii = [1, 2, 3, 4, 5, 7, 9, 11, 15]
    types = [
        ('float', 'float', '1.0f'),
        ('float2', 'float2', '(float2)(1.0f, 1.0f)'),
        ('float4', 'float4', '(float4)(1.0f, 1.0f, 1.0f, 1.0f)')
    ]
    
    # Boundary modes: clamp, reflect, wrap
    boundaries = [
        ('clamp', 'clamp_i(idx, 0, max_dim - 1)'),
        ('reflect', 'abs_val(idx)'), # custom reflection logic inlined
        ('wrap', '((idx % max_dim) + max_dim) % max_dim')
    ]
    
    for bname, _ in boundaries:
        for tname, scalar_t, def_val in types:
            for r in radii:
                ksize = 2 * r + 1
                
                # 2D Horizontal
                lines.append(f'// 2D Horizontal Convolution: Radius {r} (Window {ksize}), Type {tname}, Boundary {bname}')
                lines.append(f'__kernel void kernel_conv2d_h_r{r}_{tname}_{bname}(')
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
                    if bname == 'clamp':
                        lines.append(f'    {{ int cx = clamp_i(x + ({k}), 0, width - 1); sum += input[row_base + cx] * weights[{w_idx}]; }}')
                    elif bname == 'reflect':
                        lines.append(f'    {{ int raw_x = x + ({k}); int cx = (raw_x < 0) ? -raw_x : ((raw_x >= width) ? (2 * width - 2 - raw_x) : raw_x); cx = clamp_i(cx, 0, width - 1); sum += input[row_base + cx] * weights[{w_idx}]; }}')
                    else: # wrap
                        lines.append(f'    {{ int cx = ((x + ({k})) % width + width) % width; sum += input[row_base + cx] * weights[{w_idx}]; }}')
                lines.append(f'    output[row_base + x] = sum;')
                lines.append(f'}}\n')

                # 2D Vertical
                lines.append(f'// 2D Vertical Convolution: Radius {r} (Window {ksize}), Type {tname}, Boundary {bname}')
                lines.append(f'__kernel void kernel_conv2d_v_r{r}_{tname}_{bname}(')
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
                    if bname == 'clamp':
                        lines.append(f'    {{ int cy = clamp_i(y + ({k}), 0, height - 1); sum += input[(size_t)cy * (size_t)width + (size_t)x] * weights[{w_idx}]; }}')
                    elif bname == 'reflect':
                        lines.append(f'    {{ int raw_y = y + ({k}); int cy = (raw_y < 0) ? -raw_y : ((raw_y >= height) ? (2 * height - 2 - raw_y) : raw_y); cy = clamp_i(cy, 0, height - 1); sum += input[(size_t)cy * (size_t)width + (size_t)x] * weights[{w_idx}]; }}')
                    else: # wrap
                        lines.append(f'    {{ int cy = ((y + ({k})) % height + height) % height; sum += input[(size_t)cy * (size_t)width + (size_t)x] * weights[{w_idx}]; }}')
                lines.append(f'    output[(size_t)y * (size_t)width + (size_t)x] = sum;')
                lines.append(f'}}\n')

                # 3D Depth
                lines.append(f'// 3D Depth Z Convolution: Radius {r} (Window {ksize}), Type {tname}, Boundary {bname}')
                lines.append(f'__kernel void kernel_conv3d_z_r{r}_{tname}_{bname}(')
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
                    if bname == 'clamp':
                        lines.append(f'    {{ int cz = clamp_i(z + ({k}), 0, depth - 1); sum += input[(size_t)cz * slice + xy] * weights[{w_idx}]; }}')
                    elif bname == 'reflect':
                        lines.append(f'    {{ int raw_z = z + ({k}); int cz = (raw_z < 0) ? -raw_z : ((raw_z >= depth) ? (2 * depth - 2 - raw_z) : raw_z); cz = clamp_i(cz, 0, depth - 1); sum += input[(size_t)cz * slice + xy] * weights[{w_idx}]; }}')
                    else: # wrap
                        lines.append(f'    {{ int cz = ((z + ({k})) % depth + depth) % depth; sum += input[(size_t)cz * slice + xy] * weights[{w_idx}]; }}')
                lines.append(f'    output[(size_t)z * slice + xy] = sum;')
                lines.append(f'}}\n')

    # Dilated 2D Convolutions
    for d in [2, 3, 4, 8]:
        for r in [1, 2, 3, 4, 5]:
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

    # Direct 2D Convolutions
    for r in [1, 2, 3, 4, 5, 7]:
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

    # Direct 3D Convolutions
    for r in [1, 2, 3]:
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

def gen_gemm_specialized():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// ============================================================================',
        '// Specialized Register-Tiled Matrix Multiplications (SGEMM, CGEMM, Solvers)',
        '// ============================================================================',
        ''
    ]
    
    tile_configs = [
        (16, 16, 8, 2, 2),
        (32, 32, 8, 4, 4),
        (32, 32, 16, 4, 4),
        (64, 64, 16, 4, 4),
        (64, 64, 8, 4, 4),
        (128, 64, 8, 8, 4),
    ]
    
    # Register-Tiled GEMM
    for tm, tn, tk, rm, rn in tile_configs:
        wx = tn // rn
        wy = tm // rm
        lines.append(f'// Register-Tiled GEMM: Tile ({tm}x{tn}x{tk}), RegTile ({rm}x{rn}), WorkGroup ({wx}x{wy})')
        lines.append(f'__kernel void kernel_gemm_tile_{tm}x{tn}x{tk}_reg_{rm}x{rn}(')
        lines.append(f'    __global const float* A, __global const float* B, __global float* C,')
        lines.append(f'    const uint M, const uint N, const uint K, const float alpha, const float beta) {{')
        lines.append(f'    __local float l_A[{tm}][{tk + 1}];')
        lines.append(f'    __local float l_B[{tk}][{tn + 1}];')
        lines.append(f'    uint tx = get_local_id(0); uint ty = get_local_id(1);')
        lines.append(f'    uint tid = ty * {wx} + tx;')
        lines.append(f'    uint gx = get_group_id(0); uint gy = get_group_id(1);')
        lines.append(f'    uint row_base = gy * {tm} + ty * {rm};')
        lines.append(f'    uint col_base = gx * {tn} + tx * {rn};')
        lines.append(f'    float reg_c[{rm}][{rn}];')
        lines.append(f'    for (int i = 0; i < {rm}; ++i) for (int j = 0; j < {rn}; ++j) reg_c[i][j] = 0.0f;')
        lines.append(f'    uint num_tiles = (K + {tk} - 1) / {tk};')
        lines.append(f'    for (uint t = 0; t < num_tiles; ++t) {{')
        lines.append(f'        for (uint i = tid; i < {tm * tk}; i += {wx * wy}) {{')
        lines.append(f'            uint r = i / {tk}; uint c = i % {tk};')
        lines.append(f'            uint gr = gy * {tm} + r; uint gc = t * {tk} + c;')
        lines.append(f'            l_A[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;')
        lines.append(f'        }}')
        lines.append(f'        for (uint i = tid; i < {tk * tn}; i += {wx * wy}) {{')
        lines.append(f'            uint r = i / {tn}; uint c = i % {tn};')
        lines.append(f'            uint gr = t * {tk} + r; uint gc = gx * {tn} + c;')
        lines.append(f'            l_B[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;')
        lines.append(f'        }}')
        lines.append(f'        sync_local();')
        lines.append(f'        #pragma unroll')
        lines.append(f'        for (uint k = 0; k < {tk}; ++k) {{')
        lines.append(f'            float a_val[{rm}]; float b_val[{rn}];')
        for i in range(rm):
            lines.append(f'            a_val[{i}] = l_A[ty * {rm} + {i}][k];')
        for j in range(rn):
            lines.append(f'            b_val[{j}] = l_B[k][tx * {rn} + {j}];')
        for i in range(rm):
            for j in range(rn):
                lines.append(f'            reg_c[{i}][{j}] = fma(a_val[{i}], b_val[{j}], reg_c[{i}][{j}]);')
        lines.append(f'        }}')
        lines.append(f'        sync_local();')
        lines.append(f'    }}')
        lines.append(f'    for (int i = 0; i < {rm}; ++i) {{')
        lines.append(f'        uint r = row_base + i;')
        lines.append(f'        if (r < M) {{')
        lines.append(f'            for (int j = 0; j < {rn}; ++j) {{')
        lines.append(f'                uint c = col_base + j;')
        lines.append(f'                if (c < N) {{')
        lines.append(f'                    uint idx = r * N + c;')
        lines.append(f'                    float orig = (beta != 0.0f) ? C[idx] : 0.0f;')
        lines.append(f'                    C[idx] = alpha * reg_c[i][j] + beta * orig;')
        lines.append(f'                }}')
        lines.append(f'            }}')
        lines.append(f'        }}')
        lines.append(f'    }}')
        lines.append(f'}}\n')

    # Transposed SGEMM modes (NN, NT, TN, TT)
    modes = ['NN', 'NT', 'TN', 'TT']
    for mode in modes:
        lines.append(f'// Standard Tiled SGEMM: Mode {mode}')
        lines.append(f'__kernel void kernel_sgemm_{mode.lower()}(')
        lines.append(f'    __global const float* A, __global const float* B, __global float* C,')
        lines.append(f'    const uint M, const uint N, const uint K, const float alpha, const float beta) {{')
        lines.append(f'    __local float l_A[16][16]; __local float l_B[16][16];')
        lines.append(f'    uint tx = get_local_id(0); uint ty = get_local_id(1);')
        lines.append(f'    uint row = get_group_id(1) * 16 + ty; uint col = get_group_id(0) * 16 + tx;')
        lines.append(f'    float acc = 0.0f; uint num_tiles = (K + 15) / 16;')
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
        lines.append(f'        uint idx = row * N + col; float orig = (beta != 0.0f) ? C[idx] : 0.0f;')
        lines.append(f'        C[idx] = alpha * acc + beta * orig;')
        lines.append(f'    }}')
        lines.append(f'}}\n')

    # Transposed CGEMM modes (NN, NT, TN, TT)
    for mode in modes:
        lines.append(f'// Complex Tiled CGEMM: Mode {mode}')
        lines.append(f'__kernel void kernel_cgemm_{mode.lower()}(')
        lines.append(f'    __global const float2* A, __global const float2* B, __global float2* C,')
        lines.append(f'    const uint M, const uint N, const uint K, const float2 alpha, const float2 beta) {{')
        lines.append(f'    uint row = get_global_id(1); uint col = get_global_id(0);')
        lines.append(f'    if (row >= M || col >= N) return;')
        lines.append(f'    float2 acc = (float2)(0.0f, 0.0f);')
        lines.append(f'    for (uint k = 0; k < K; ++k) {{')
        if mode == 'NN':
            lines.append(f'        float2 a_val = A[row * K + k]; float2 b_val = B[k * N + col];')
        elif mode == 'NT':
            lines.append(f'        float2 a_val = A[row * K + k]; float2 b_val = B[col * K + k];')
        elif mode == 'TN':
            lines.append(f'        float2 a_val = A[k * M + row]; float2 b_val = B[k * N + col];')
        else: # TT
            lines.append(f'        float2 a_val = A[k * M + row]; float2 b_val = B[col * K + k];')
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

    # Batched small matrix inversion (2x2 up to 12x12)
    for n in [2, 3, 4, 5, 6, 8, 10, 12]:
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

    # Batched Cholesky Decomposition (3x3 up to 10x10)
    for n in [3, 4, 5, 6, 8, 10]:
        lines.append(f'// Batched {n}x{n} Cholesky Factorization (A = L * L^T)')
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

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(root, 'kernels', 'specialized')
    os.makedirs(target_dir, exist_ok=True)
    
    # Generate large specialized suites
    conv_content = gen_convolutions_specialized()
    verify_no_dashes(conv_content, 'convolutions_specialized.cl')
    with open(os.path.join(target_dir, 'convolutions_specialized.cl'), 'w', encoding='utf-8') as f:
        f.write(conv_content)
    print(f"Generated convolutions_specialized.cl: {len(conv_content.splitlines())} lines")

    gemm_content = gen_gemm_specialized()
    verify_no_dashes(gemm_content, 'gemm_specialized.cl')
    with open(os.path.join(target_dir, 'gemm_specialized.cl'), 'w', encoding='utf-8') as f:
        f.write(gemm_content)
    print(f"Generated gemm_specialized.cl: {len(gemm_content.splitlines())} lines")

if __name__ == '__main__':
    main()

def gen_stencils_specialized():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// ============================================================================',
        '// High-Order 2D and 3D Finite-Difference Stencils for Wavefields and Transport',
        '// ============================================================================',
        ''
    ]
    
    # 2D Stencils across accuracy orders
    orders_2d = [
        ('2nd_order', 1, [1.0], -4.0),
        ('4th_order', 2, [16.0/12.0, -1.0/12.0], -60.0/12.0),
        ('6th_order', 3, [270.0/180.0, -27.0/180.0, 2.0/180.0], -490.0/180.0),
        ('8th_order', 4, [8064.0/5040.0, -1008.0/5040.0, 128.0/5040.0, -9.0/5040.0], -14350.0/5040.0)
    ]
    
    for sname, r, weights, center_w in orders_2d:
        lines.append(f'// 2D Finite-Difference Laplacian: {sname} (Radius {r})')
        lines.append(f'__kernel void kernel_stencil2d_{sname}(')
        lines.append(f'    __global const float* input, __global float* output,')
        lines.append(f'    const int nx, const int ny, const float inv_dx2) {{')
        lines.append(f'    int x = get_global_id(0); int y = get_global_id(1);')
        lines.append(f'    if (x < {r} || x >= nx - {r} || y < {r} || y >= ny - {r}) return;')
        lines.append(f'    size_t idx = (size_t)y * (size_t)nx + (size_t)x;')
        lines.append(f'    float sum = {center_w}f * input[idx];')
        for k, w in enumerate(weights, 1):
            lines.append(f'    sum += ({w}f) * (input[idx - {k}] + input[idx + {k}] + input[idx - {k} * nx] + input[idx + {k} * nx]);')
        lines.append(f'    output[idx] = sum * inv_dx2;')
        lines.append(f'}}\n')

    # 3D Stencils across accuracy orders
    orders_3d = [
        ('7pt_2nd_order', 1, [1.0], -6.0),
        ('13pt_4th_order', 2, [16.0/12.0, -1.0/12.0], -90.0/12.0),
        ('19pt_6th_order', 3, [270.0/180.0, -27.0/180.0, 2.0/180.0], -735.0/180.0),
        ('25pt_8th_order', 4, [8064.0/5040.0, -1008.0/5040.0, 128.0/5040.0, -9.0/5040.0], -21525.0/5040.0)
    ]
    
    for sname, r, weights, center_w in orders_3d:
        lines.append(f'// 3D Acoustic Wavefield Propagation: {sname} (Radius {r})')
        lines.append(f'__kernel void kernel_stencil3d_{sname}(')
        lines.append(f'    __global const float* u_curr, __global const float* u_prev,')
        lines.append(f'    __global const float* velocity_sq, __global float* u_next,')
        lines.append(f'    const float dt_sq_dx_sq, const float damping, const int nx, const int ny, const int nz) {{')
        lines.append(f'    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);')
        lines.append(f'    if (x < {r} || x >= nx - {r} || y < {r} || y >= ny - {r} || z < {r} || z >= nz - {r}) return;')
        lines.append(f'    size_t slice = (size_t)nx * (size_t)ny;')
        lines.append(f'    size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;')
        lines.append(f'    float curr = u_curr[idx]; float sum = {center_w}f * curr;')
        for k, w in enumerate(weights, 1):
            lines.append(f'    sum += ({w}f) * (u_curr[idx - {k}] + u_curr[idx + {k}] + u_curr[idx - {k} * nx] + u_curr[idx + {k} * nx] + u_curr[idx - {k} * slice] + u_curr[idx + {k} * slice]);')
        lines.append(f'    float c2 = velocity_sq ? velocity_sq[idx] : 1.0f;')
        lines.append(f'    float prev = u_prev ? u_prev[idx] : curr;')
        lines.append(f'    u_next[idx] = (2.0f * curr - prev + c2 * dt_sq_dx_sq * sum) / (1.0f + damping);')
        lines.append(f'}}\n')

    # Perfectly Matched Layers (PML) Absorbing Boundary Update
    lines.append(f'// 3D Split-Field PML Absorbing Boundary Condition for Acoustic Wavefields')
    lines.append(f'__kernel void kernel_pml_acoustic_boundary_3d(')
    lines.append(f'    __global float* px, __global float* py, __global float* pz,')
    lines.append(f'    __global const float* vx, __global const float* vy, __global const float* vz,')
    lines.append(f'    __global const float* sigma_x, __global const float* sigma_y, __global const float* sigma_z,')
    lines.append(f'    const float bulk_modulus, const float dt, const float inv_dx,')
    lines.append(f'    const int nx, const int ny, const int nz) {{')
    lines.append(f'    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);')
    lines.append(f'    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;')
    lines.append(f'    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;')
    lines.append(f'    float sx = sigma_x[x]; float sy = sigma_y[y]; float sz = sigma_z[z];')
    lines.append(f'    float dvx_dx = (vx[idx] - vx[idx - 1]) * inv_dx;')
    lines.append(f'    float dvy_dy = (vy[idx] - vy[idx - nx]) * inv_dx;')
    lines.append(f'    float dvz_dz = (vz[idx] - vz[idx - slice]) * inv_dx;')
    lines.append(f'    px[idx] = (px[idx] * (1.0f - 0.5f * dt * sx) - dt * bulk_modulus * dvx_dx) / (1.0f + 0.5f * dt * sx);')
    lines.append(f'    py[idx] = (py[idx] * (1.0f - 0.5f * dt * sy) - dt * bulk_modulus * dvy_dy) / (1.0f + 0.5f * dt * sy);')
    lines.append(f'    pz[idx] = (pz[idx] * (1.0f - 0.5f * dt * sz) - dt * bulk_modulus * dvz_dz) / (1.0f + 0.5f * dt * sz);')
    lines.append(f'}}\n')

    return '\n'.join(lines)

def gen_interpolation_specialized():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// ============================================================================',
        '// Multidimensional Continuous Interpolators, Splines, and Deformation Fields',
        '// ============================================================================',
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
        'inline float lanczos3_weight(float x) {',
        '    if (fabs(x) < 1.0e-6f) return 1.0f;',
        '    if (fabs(x) >= 3.0f) return 0.0f;',
        '    float px = PI_F * x;',
        '    return (sin(px) / px) * (sin(px / 3.0f) / (px / 3.0f));',
        '}',
        '',
        '// 2D Lanczos-3 High-Fidelity Resampler',
        '__kernel void kernel_interp2d_lanczos3(',
        '    __global const float* in_img, __global float* out_img,',
        '    const int in_w, const int in_h, const int out_w, const int out_h,',
        '    const float scale_x, const float scale_y) {',
        '    int x = get_global_id(0); int y = get_global_id(1);',
        '    if (x >= out_w || y >= out_h) return;',
        '    float fx = (float)x * scale_x; float fy = (float)y * scale_y;',
        '    int ix = (int)floor(fx); int iy = (int)floor(fy);',
        '    float sum = 0.0f; float total_weight = 0.0f;',
        '    for (int dy = -2; dy <= 3; ++dy) {',
        '        int cy = clamp_i(iy + dy, 0, in_h - 1);',
        '        float wy = lanczos3_weight(fy - (float)(iy + dy));',
        '        for (int dx = -2; dx <= 3; ++dx) {',
        '            int cx = clamp_i(ix + dx, 0, in_w - 1);',
        '            float wx = lanczos3_weight(fx - (float)(ix + dx));',
        '            float w = wx * wy;',
        '            sum += in_img[(size_t)cy * (size_t)in_w + (size_t)cx] * w;',
        '            total_weight += w;',
        '        }',
        '    }',
        '    out_img[(size_t)y * (size_t)out_w + (size_t)x] = (total_weight > 1.0e-6f) ? (sum / total_weight) : 0.0f;',
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
        '// Thin-Plate Spline (TPS) Radial Basis Warp Grid Evaluator: U(r) = r^2 * ln(r)',
        '__kernel void kernel_tps_warp_2d(',
        '    __global const float2* control_points, // [num_ctrl]',
        '    __global const float2* tps_weights,    // [num_ctrl]',
        '    __global const float* affine_coeffs,   // [2 x 3] (a0, ax, ay)',
        '    __global float2* out_displacement_field, // [h x w]',
        '    const int num_ctrl, const int width, const int height) {',
        '    int x = get_global_id(0); int y = get_global_id(1);',
        '    if (x >= width || y >= height) return;',
        '    float px = (float)x; float py = (float)y;',
        '    // Affine component',
        '    float dx = affine_coeffs[0] + affine_coeffs[1] * px + affine_coeffs[2] * py;',
        '    float dy = affine_coeffs[3] + affine_coeffs[4] * px + affine_coeffs[5] * py;',
        '    // Non-linear TPS RBF kernel summation',
        '    for (int k = 0; k < num_ctrl; ++k) {',
        '        float2 cp = control_points[k];',
        '        float rx = px - cp.x; float ry = py - cp.y;',
        '        float r_sq = rx * rx + ry * ry;',
        '        if (r_sq > 1.0e-8f) {',
        '            float u = 0.5f * r_sq * log(r_sq);',
        '            float2 w = tps_weights[k];',
        '            dx += w.x * u; dy += w.y * u;',
        '        }',
        '    }',
        '    out_displacement_field[(size_t)y * (size_t)width + (size_t)x] = (float2)(dx, dy);',
        '}}\n'
    ]
    return '\n'.join(lines)

def gen_wavelets_specialized():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// ============================================================================',
        '// Multiscale Discrete Wavelet Transforms (DWT) and Gabor Filter Banks',
        '// ============================================================================',
        ''
    ]
    
    # Daubechies 4 (db2) coefficients
    # h0 = (1 + sqrt(3))/(4*sqrt(2)), h1 = (3 + sqrt(3))/(4*sqrt(2)), h2 = (3 - sqrt(3))/(4*sqrt(2)), h3 = (1 - sqrt(3))/(4*sqrt(2))
    lines.append('// Daubechies 4 (db2) 1D Forward Wavelet Transform (Row Pass)')
    lines.append('__kernel void kernel_dwt_db4_forward_rows(')
    lines.append('    __global const float* input, __global float* output, const int width, const int height) {')
    lines.append('    int x = get_global_id(0); int y = get_global_id(1); int half_w = width / 2;')
    lines.append('    if (x >= half_w || y >= height) return;')
    lines.append('    size_t base = (size_t)y * (size_t)width;')
    lines.append('    const float h0 = 0.4829629131445341f; const float h1 = 0.8365163037378079f;')
    lines.append('    const float h2 = 0.2241438680420134f; const float h3 = -0.1294095225512604f;')
    lines.append('    const float g0 = h3; const float g1 = -h2; const float g2 = h1; const float g3 = -h0;')
    lines.append('    int i0 = 2 * x; int i1 = (2 * x + 1) % width; int i2 = (2 * x + 2) % width; int i3 = (2 * x + 3) % width;')
    lines.append('    float v0 = input[base + i0]; float v1 = input[base + i1]; float v2 = input[base + i2]; float v3 = input[base + i3];')
    lines.append('    float low  = h0 * v0 + h1 * v1 + h2 * v2 + h3 * v3;')
    lines.append('    float high = g0 * v0 + g1 * v1 + g2 * v2 + g3 * v3;')
    lines.append('    output[base + x] = low; output[base + half_w + x] = high;')
    lines.append('}\n')

    # Daubechies 4 (db2) 1D Inverse Wavelet Transform (Row Pass)
    lines.append('// Daubechies 4 (db2) 1D Inverse Wavelet Transform (Row Pass)')
    lines.append('__kernel void kernel_dwt_db4_inverse_rows(')
    lines.append('    __global const float* input, __global float* output, const int width, const int height) {')
    lines.append('    int x = get_global_id(0); int y = get_global_id(1); int half_w = width / 2;')
    lines.append('    if (x >= half_w || y >= height) return;')
    lines.append('    size_t base = (size_t)y * (size_t)width;')
    lines.append('    const float h0 = 0.4829629131445341f; const float h1 = 0.8365163037378079f;')
    lines.append('    const float h2 = 0.2241438680420134f; const float h3 = -0.1294095225512604f;')
    lines.append('    const float g0 = h3; const float g1 = -h2; const float g2 = h1; const float g3 = -h0;')
    lines.append('    int xm1 = (x - 1 + half_w) % half_w;')
    lines.append('    float s0 = input[base + x]; float s_prev = input[base + xm1];')
    lines.append('    float d0 = input[base + half_w + x]; float d_prev = input[base + half_w + xm1];')
    lines.append('    float out_even = h0 * s0 + h2 * s_prev + g0 * d0 + g2 * d_prev;')
    lines.append('    float out_odd  = h1 * s0 + h3 * s_prev + g1 * d0 + g3 * d_prev;')
    lines.append('    output[base + 2 * x]     = out_even;')
    lines.append('    output[base + 2 * x + 1] = out_odd;')
    lines.append('}\n')

    # 2D Gabor Filter Bank for 8 orientations
    for o_idx in range(8):
        theta = o_idx * (3.141592653589793 / 8.0)
        lines.append(f'// 2D Gabor Filter Bank: Orientation {o_idx} (Theta {theta:.4f} rad)')
        lines.append(f'__kernel void kernel_gabor_bank_orient_{o_idx}(')
        lines.append(f'    __global const float* input, __global float2* output,')
        lines.append(f'    const int width, const int height, const float lambda,')
        lines.append(f'    const float psi, const float sigma, const float gamma, const int radius) {{')
        lines.append(f'    int x = get_global_id(0); int y = get_global_id(1);')
        lines.append(f'    if (x >= width || y >= height) return;')
        lines.append(f'    const float theta = {theta}f;')
        lines.append(f'    float cos_t = cos(theta); float sin_t = sin(theta);')
        lines.append(f'    float2 acc = (float2)(0.0f, 0.0f); float sigma_sq = sigma * sigma;')
        lines.append(f'    for (int dy = -radius; dy <= radius; ++dy) {{')
        lines.append(f'        int cy = clamp_i(y + dy, 0, height - 1);')
        lines.append(f'        for (int dx = -radius; dx <= radius; ++dx) {{')
        lines.append(f'            int cx = clamp_i(x + dx, 0, width - 1);')
        lines.append(f'            float x_p =  (float)dx * cos_t + (float)dy * sin_t;')
        lines.append(f'            float y_p = -(float)dx * sin_t + (float)dy * cos_t;')
        lines.append(f'            float envelope = exp(- (x_p * x_p + gamma * gamma * y_p * y_p) / (2.0f * sigma_sq));')
        lines.append(f'            float carrier = (2.0f * PI_F * x_p / lambda) + psi;')
        lines.append(f'            float val = input[(size_t)cy * (size_t)width + (size_t)cx];')
        lines.append(f'            acc.x += val * envelope * cos(carrier);')
        lines.append(f'            acc.y += val * envelope * sin(carrier);')
        lines.append(f'        }}')
        lines.append(f'    }}')
        lines.append(f'    output[(size_t)y * (size_t)width + (size_t)x] = acc;')
        lines.append(f'}}\n')

    return '\n'.join(lines)

def gen_vectorized_math_specialized():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// ============================================================================',
        '// Vectorized Elementwise Arithmetic, Transcendental, and Activation Kernels',
        '// ============================================================================',
        ''
    ]
    
    vec_types = ['float', 'float2', 'float3', 'float4', 'float8', 'float16']
    
    for vtype in vec_types:
        lines.append(f'// Operations for {vtype}')
        ops = [
            ('add', 'a[gid] + b[gid]', 'binary'),
            ('sub', 'a[gid] - b[gid]', 'binary'),
            ('mul', 'a[gid] * b[gid]', 'binary'),
            ('div', 'a[gid] / b[gid]', 'binary'),
            ('exp', 'exp(a[gid])', 'unary'),
            ('exp2', 'exp2(a[gid])', 'unary'),
            ('log', 'log(a[gid])', 'unary'),
            ('log2', 'log2(a[gid])', 'unary'),
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

def run_extra_generators():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(root, 'kernels', 'specialized')
    
    extra = {
        'stencils_specialized.cl': gen_stencils_specialized(),
        'interpolation_specialized.cl': gen_interpolation_specialized(),
        'wavelets_specialized.cl': gen_wavelets_specialized(),
        'vectorized_math_ops.cl': gen_vectorized_math_specialized(),
    }
    
    for filename, content in extra.items():
        verify_no_dashes(content, filename)
        path = os.path.join(target_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Generated {filename}: {len(content.splitlines())} lines")

run_extra_generators()
