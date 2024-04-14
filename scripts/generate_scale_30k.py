#!/usr/bin/env python3
"""
Scales the Silica OpenCL kernel suite to ~30,000 lines of mathematically rich,
unrolled, and vectorized code. Strictly conforms to OpenCL 1.2 with zero em-dashes and zero en-dashes.
"""

import os
import sys

def verify_no_dashes(text, filename=""):
    for bad_char in ['\u2014', '\u2013']:
        if bad_char in text:
            raise ValueError(f"Prohibited character found in {filename}: {hex(ord(bad_char))}")

def gen_convolutions_scale():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// ============================================================================',
        '// Comprehensive Suite of Separable, Strided, Dilated, and Steerable Convolutions',
        '// ============================================================================',
        ''
    ]
    
    radii = [1, 2, 3, 4, 5, 7, 9, 11, 15]
    types = [
        ('float', 'float', '1.0f'),
        ('float2', 'float2', '(float2)(1.0f, 1.0f)'),
        ('float4', 'float4', '(float4)(1.0f, 1.0f, 1.0f, 1.0f)'),
        ('float8', 'float8', '(float8)(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f)')
    ]
    
    boundaries = [
        ('clamp', 'clamp_i(idx, 0, max_dim - 1)'),
        ('reflect', 'reflection'),
        ('wrap', 'wrap')
    ]
    
    # 1. Separable Convolutions (H, V, Z)
    for bname, _ in boundaries:
        for tname, scalar_t, def_val in types:
            for r in radii:
                ksize = 2 * r + 1
                # 2D Horizontal
                lines.append(f'// 2D Horizontal Convolution: Radius {r} (Window {ksize}), Type {tname}, Boundary {bname}')
                lines.append(f'__kernel void kernel_conv2d_h_r{r}_{tname}_{bname}(')
                lines.append(f'    __global const {tname}* input, __global {tname}* output,')
                lines.append(f'    __constant const float* weights, const int width, const int height) {{')
                lines.append(f'    int x = get_global_id(0); int y = get_global_id(1);')
                lines.append(f'    if (x >= width || y >= height) return;')
                lines.append(f'    {tname} sum = 0.0f; size_t row_base = (size_t)y * (size_t)width;')
                for k in range(-r, r + 1):
                    w_idx = k + r
                    if bname == 'clamp':
                        lines.append(f'    {{ int cx = clamp_i(x + ({k}), 0, width - 1); sum += input[row_base + cx] * weights[{w_idx}]; }}')
                    elif bname == 'reflect':
                        lines.append(f'    {{ int raw_x = x + ({k}); int cx = (raw_x < 0) ? -raw_x : ((raw_x >= width) ? (2 * width - 2 - raw_x) : raw_x); cx = clamp_i(cx, 0, width - 1); sum += input[row_base + cx] * weights[{w_idx}]; }}')
                    else:
                        lines.append(f'    {{ int cx = ((x + ({k})) % width + width) % width; sum += input[row_base + cx] * weights[{w_idx}]; }}')
                lines.append(f'    output[row_base + x] = sum;')
                lines.append(f'}}\n')

                # 2D Vertical
                lines.append(f'// 2D Vertical Convolution: Radius {r} (Window {ksize}), Type {tname}, Boundary {bname}')
                lines.append(f'__kernel void kernel_conv2d_v_r{r}_{tname}_{bname}(')
                lines.append(f'    __global const {tname}* input, __global {tname}* output,')
                lines.append(f'    __constant const float* weights, const int width, const int height) {{')
                lines.append(f'    int x = get_global_id(0); int y = get_global_id(1);')
                lines.append(f'    if (x >= width || y >= height) return;')
                lines.append(f'    {tname} sum = 0.0f;')
                for k in range(-r, r + 1):
                    w_idx = k + r
                    if bname == 'clamp':
                        lines.append(f'    {{ int cy = clamp_i(y + ({k}), 0, height - 1); sum += input[(size_t)cy * (size_t)width + (size_t)x] * weights[{w_idx}]; }}')
                    elif bname == 'reflect':
                        lines.append(f'    {{ int raw_y = y + ({k}); int cy = (raw_y < 0) ? -raw_y : ((raw_y >= height) ? (2 * height - 2 - raw_y) : raw_y); cy = clamp_i(cy, 0, height - 1); sum += input[(size_t)cy * (size_t)width + (size_t)x] * weights[{w_idx}]; }}')
                    else:
                        lines.append(f'    {{ int cy = ((y + ({k})) % height + height) % height; sum += input[(size_t)cy * (size_t)width + (size_t)x] * weights[{w_idx}]; }}')
                lines.append(f'    output[(size_t)y * (size_t)width + (size_t)x] = sum;')
                lines.append(f'}}\n')

                # 3D Depth
                lines.append(f'// 3D Depth Z Convolution: Radius {r} (Window {ksize}), Type {tname}, Boundary {bname}')
                lines.append(f'__kernel void kernel_conv3d_z_r{r}_{tname}_{bname}(')
                lines.append(f'    __global const {tname}* input, __global {tname}* output,')
                lines.append(f'    __constant const float* weights, const int width, const int height, const int depth) {{')
                lines.append(f'    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);')
                lines.append(f'    if (x >= width || y >= height || z >= depth) return;')
                lines.append(f'    {tname} sum = 0.0f; size_t slice = (size_t)width * (size_t)height; size_t xy = (size_t)y * (size_t)width + (size_t)x;')
                for k in range(-r, r + 1):
                    w_idx = k + r
                    if bname == 'clamp':
                        lines.append(f'    {{ int cz = clamp_i(z + ({k}), 0, depth - 1); sum += input[(size_t)cz * slice + xy] * weights[{w_idx}]; }}')
                    elif bname == 'reflect':
                        lines.append(f'    {{ int raw_z = z + ({k}); int cz = (raw_z < 0) ? -raw_z : ((raw_z >= depth) ? (2 * depth - 2 - raw_z) : raw_z); cz = clamp_i(cz, 0, depth - 1); sum += input[(size_t)cz * slice + xy] * weights[{w_idx}]; }}')
                    else:
                        lines.append(f'    {{ int cz = ((z + ({k})) % depth + depth) % depth; sum += input[(size_t)cz * slice + xy] * weights[{w_idx}]; }}')
                lines.append(f'    output[(size_t)z * slice + xy] = sum;')
                lines.append(f'}}\n')

    # 2. Strided Convolutions for Fast Downsampling (Stride 2 and Stride 4)
    for s in [2, 4]:
        for r in [1, 2, 3, 5]:
            ksize = 2 * r + 1
            lines.append(f'// Strided 2D Downsampling Convolution: Radius {r}, Stride {s}')
            lines.append(f'__kernel void kernel_conv2d_strided_r{r}_s{s}(')
            lines.append(f'    __global const float* input, __global float* output,')
            lines.append(f'    __constant const float* weights, const int in_w, const int in_h, const int out_w, const int out_h) {{')
            lines.append(f'    int ox = get_global_id(0); int oy = get_global_id(1);')
            lines.append(f'    if (ox >= out_w || oy >= out_h) return;')
            lines.append(f'    int cx_center = ox * {s}; int cy_center = oy * {s};')
            lines.append(f'    float sum = 0.0f;')
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    w_idx = (dy + r) * ksize + (dx + r)
                    lines.append(f'    {{ int ix = clamp_i(cx_center + ({dx}), 0, in_w - 1); int iy = clamp_i(cy_center + ({dy}), 0, in_h - 1); sum += input[(size_t)iy * (size_t)in_w + (size_t)ix] * weights[{w_idx}]; }}')
            lines.append(f'    output[(size_t)oy * (size_t)out_w + (size_t)ox] = sum;')
            lines.append(f'}}\n')

    # 3. Direct 2D Convolutions across kernel sizes (3x3 up to 15x15)
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

    # 4. Direct 3D Convolutions
    for r in [1, 2, 3]:
        ksize = 2 * r + 1
        lines.append(f'// Direct 3D Non-Separable Convolution: Kernel {ksize}x{ksize}x{ksize}')
        lines.append(f'__kernel void kernel_conv3d_direct_{ksize}x{ksize}x{ksize}(')
        lines.append(f'    __global const float* input, __global float* output,')
        lines.append(f'    __constant const float* mask, const int nx, const int ny, const int nz) {{')
        lines.append(f'    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);')
        lines.append(f'    if (x >= nx || y >= ny || z >= nz) return;')
        lines.append(f'    float sum = 0.0f; size_t slice = (size_t)nx * (size_t)ny;')
        for dz in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    m_idx = (dz + r) * ksize * ksize + (dy + r) * ksize + (dx + r)
                    lines.append(f'    {{ int cx = clamp_i(x + ({dx}), 0, nx - 1); int cy = clamp_i(y + ({dy}), 0, ny - 1); int cz = clamp_i(z + ({dz}), 0, nz - 1); sum += input[(size_t)cz * slice + (size_t)cy * (size_t)nx + (size_t)cx] * mask[{m_idx}]; }}')
        lines.append(f'    output[(size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x] = sum;')
        lines.append(f'}}\n')

    return '\n'.join(lines)

def gen_pcr_solvers_scale():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// ============================================================================',
        '// Extended Parallel Cyclic Reduction (PCR) Tridiagonal Solvers (16 up to 512)',
        '// ============================================================================',
        ''
    ]
    
    import math
    for p_size in [16, 32, 64, 128, 256, 512]:
        steps = int(math.log2(p_size))
        lines.append(f'// Batched Parallel Cyclic Reduction: System Size {p_size}')
        lines.append(f'__kernel void kernel_pcr_tridiagonal_sys_{p_size}(')
        lines.append(f'    __global const float* in_lower, __global const float* in_diag,')
        lines.append(f'    __global const float* in_upper, __global const float* in_rhs,')
        lines.append(f'    __global float* out_x, const uint batch_count) {{')
        lines.append(f'    uint bid = get_group_id(0); if (bid >= batch_count) return;')
        lines.append(f'    uint tid = get_local_id(0);')
        lines.append(f'    __local float a[{p_size}], b[{p_size}], c[{p_size}], d[{p_size}];')
        lines.append(f'    __local float a_new[{p_size}], b_new[{p_size}], c_new[{p_size}], d_new[{p_size}];')
        lines.append(f'    size_t base = (size_t)bid * {p_size};')
        lines.append(f'    if (tid < {p_size}) {{')
        lines.append(f'        a[tid] = in_lower[base + tid]; b[tid] = in_diag[base + tid];')
        lines.append(f'        c[tid] = in_upper[base + tid]; d[tid] = in_rhs[base + tid];')
        lines.append(f'    }}')
        lines.append(f'    sync_local();')
        
        for s in range(steps):
            stride = 1 << s
            lines.append(f'    // Step {s}: Stride {stride}')
            lines.append(f'    if (tid < {p_size}) {{')
            lines.append(f'        float i_a = a[tid]; float i_b = b[tid]; float i_c = c[tid]; float i_d = d[tid];')
            lines.append(f'        float k1 = (tid >= {stride}) ? (i_a / b[tid - {stride}]) : 0.0f;')
            lines.append(f'        float k2 = (tid + {stride} < {p_size}) ? (i_c / b[tid + {stride}]) : 0.0f;')
            lines.append(f'        b_new[tid] = i_b - k1 * ((tid >= {stride}) ? c[tid - {stride}] : 0.0f) - k2 * ((tid + {stride} < {p_size}) ? a[tid + {stride}] : 0.0f);')
            lines.append(f'        d_new[tid] = i_d - k1 * ((tid >= {stride}) ? d[tid - {stride}] : 0.0f) - k2 * ((tid + {stride} < {p_size}) ? d[tid + {stride}] : 0.0f);')
            lines.append(f'        a_new[tid] = -k1 * ((tid >= {stride}) ? a[tid - {stride}] : 0.0f);')
            lines.append(f'        c_new[tid] = -k2 * ((tid + {stride} < {p_size}) ? c[tid + {stride}] : 0.0f);')
            lines.append(f'    }}')
            lines.append(f'    sync_local();')
            lines.append(f'    if (tid < {p_size}) {{ a[tid] = a_new[tid]; b[tid] = b_new[tid]; c[tid] = c_new[tid]; d[tid] = d_new[tid]; }}')
            lines.append(f'    sync_local();')
            
        lines.append(f'    if (tid < {p_size}) {{ out_x[base + tid] = d[tid] / b[tid]; }}')
        lines.append(f'}}\n')

    return '\n'.join(lines)

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(root, 'kernels', 'specialized')
    
    # Scale convolutions
    conv_content = gen_convolutions_scale()
    verify_no_dashes(conv_content, 'convolutions_specialized.cl')
    with open(os.path.join(target_dir, 'convolutions_specialized.cl'), 'w', encoding='utf-8') as f:
        f.write(conv_content)
    print(f"Updated convolutions_specialized.cl: {len(conv_content.splitlines())} lines")

    # Update solvers
    from generate_deep_suites import gen_solvers_deep
    solvers_base = gen_solvers_deep()
    pcr_scale = gen_pcr_solvers_scale()
    full_solvers = solvers_base + '\n' + pcr_scale
    verify_no_dashes(full_solvers, 'solvers_specialized.cl')
    with open(os.path.join(target_dir, 'solvers_specialized.cl'), 'w', encoding='utf-8') as f:
        f.write(full_solvers)
    print(f"Updated solvers_specialized.cl: {len(full_solvers.splitlines())} lines")

if __name__ == '__main__':
    main()
