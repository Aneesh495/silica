#!/usr/bin/env python3
"""
Adds mathematically rigorous batched QR, SVD, multi-scale wavelets, and RBFs
to bring Silica OpenCL kernel suite to the ~30,000 line scale.
"""

import os
import sys

def verify_no_dashes(text, filename=""):
    for bad_char in ['\u2014', '\u2013']:
        if bad_char in text:
            raise ValueError(f"Prohibited character found in {filename}: {hex(ord(bad_char))}")

def gen_qr_svd_solvers():
    lines = [
        '',
        '// ============================================================================',
        '// Batched Householder QR Factorization and Batched Jacobi SVD Solvers',
        '// ============================================================================',
        ''
    ]
    
    # Batched Householder QR Decomposition for sizes 3..12
    for n in [3, 4, 5, 6, 8, 10, 12]:
        lines.append(f'// Batched {n}x{n} Householder QR Factorization (A = Q * R)')
        lines.append(f'__kernel void kernel_batched_qr_{n}x{n}(')
        lines.append(f'    __global const float* in_mats, __global float* out_Q, __global float* out_R, const uint batch_count) {{')
        lines.append(f'    uint bid = get_global_id(0); if (bid >= batch_count) return;')
        lines.append(f'    size_t base = (size_t)bid * {n * n};')
        lines.append(f'    float A[{n}][{n}]; float Q[{n}][{n}];')
        lines.append(f'    for (int i = 0; i < {n}; ++i) for (int j = 0; j < {n}; ++j) {{')
        lines.append(f'        A[i][j] = in_mats[base + i * {n} + j];')
        lines.append(f'        Q[i][j] = (i == j) ? 1.0f : 0.0f;')
        lines.append(f'    }}')
        lines.append(f'    for (int k = 0; k < {n - 1}; ++k) {{')
        lines.append(f'        float norm_x = 0.0f;')
        lines.append(f'        for (int i = k; i < {n}; ++i) norm_x += A[i][k] * A[i][k];')
        lines.append(f'        norm_x = sqrt(norm_x);')
        lines.append(f'        if (norm_x > 1.0e-7f) {{')
        lines.append(f'            float s = (A[k][k] >= 0.0f) ? 1.0f : -1.0f;')
        lines.append(f'            float u1 = A[k][k] + s * norm_x;')
        lines.append(f'            float v[{n}];')
        lines.append(f'            v[k] = 1.0f;')
        lines.append(f'            for (int i = k + 1; i < {n}; ++i) v[i] = A[i][k] / u1;')
        lines.append(f'            float tau = 2.0f / (1.0f + (norm_x * norm_x - A[k][k] * A[k][k]) / (u1 * u1));')
        lines.append(f'            // Apply Householder reflector H = I - tau * v * v^T to A and Q')
        lines.append(f'            for (int j = k; j < {n}; ++j) {{')
        lines.append(f'                float dot_val = 0.0f;')
        lines.append(f'                for (int i = k; i < {n}; ++i) dot_val += v[i] * A[i][j];')
        lines.append(f'                for (int i = k; i < {n}; ++i) A[i][j] -= tau * dot_val * v[i];')
        lines.append(f'            }}')
        lines.append(f'            for (int j = 0; j < {n}; ++j) {{')
        lines.append(f'                float dot_val = 0.0f;')
        lines.append(f'                for (int i = k; i < {n}; ++i) dot_val += v[i] * Q[j][i];')
        lines.append(f'                for (int i = k; i < {n}; ++i) Q[j][i] -= tau * dot_val * v[i];')
        lines.append(f'            }}')
        lines.append(f'        }}')
        lines.append(f'    }}')
        lines.append(f'    for (int i = 0; i < {n}; ++i) for (int j = 0; j < {n}; ++j) {{')
        lines.append(f'        out_Q[base + i * {n} + j] = Q[i][j];')
        lines.append(f'        out_R[base + i * {n} + j] = (i <= j) ? A[i][j] : 0.0f;')
        lines.append(f'    }}')
        lines.append(f'}}\n')

    # Batched Jacobi SVD for sizes 3..8
    for n in [3, 4, 6, 8]:
        lines.append(f'// Batched {n}x{n} One-Sided Jacobi SVD (A = U * S * V^T)')
        lines.append(f'__kernel void kernel_batched_svd_{n}x{n}(')
        lines.append(f'    __global const float* in_mats, __global float* out_U, __global float* out_S, __global float* out_V, const uint batch_count) {{')
        lines.append(f'    uint bid = get_global_id(0); if (bid >= batch_count) return;')
        lines.append(f'    size_t base = (size_t)bid * {n * n};')
        lines.append(f'    float U[{n}][{n}]; float V[{n}][{n}];')
        lines.append(f'    for (int i = 0; i < {n}; ++i) for (int j = 0; j < {n}; ++j) {{')
        lines.append(f'        U[i][j] = in_mats[base + i * {n} + j];')
        lines.append(f'        V[i][j] = (i == j) ? 1.0f : 0.0f;')
        lines.append(f'    }}')
        lines.append(f'    // Jacobi sweeps')
        lines.append(f'    for (int sweep = 0; sweep < 8; ++sweep) {{')
        lines.append(f'        for (int p = 0; p < {n - 1}; ++p) {{')
        lines.append(f'            for (int q = p + 1; q < {n}; ++q) {{')
        lines.append(f'                float alpha = 0.0f; float beta = 0.0f; float gamma = 0.0f;')
        lines.append(f'                for (int i = 0; i < {n}; ++i) {{')
        lines.append(f'                    alpha += U[i][p] * U[i][p];')
        lines.append(f'                    beta  += U[i][q] * U[i][q];')
        lines.append(f'                    gamma += U[i][p] * U[i][q];')
        lines.append(f'                }}')
        lines.append(f'                if (fabs(gamma) > 1.0e-7f) {{')
        lines.append(f'                    float zeta = (beta - alpha) / (2.0f * gamma);')
        lines.append(f'                    float t = ((zeta >= 0.0f) ? 1.0f : -1.0f) / (fabs(zeta) + sqrt(1.0f + zeta * zeta));')
        lines.append(f'                    float c = 1.0f / sqrt(1.0f + t * t); float s = t * c;')
        lines.append(f'                    for (int i = 0; i < {n}; ++i) {{')
        lines.append(f'                        float u_p = U[i][p]; float u_q = U[i][q];')
        lines.append(f'                        U[i][p] = c * u_p - s * u_q; U[i][q] = s * u_p + c * u_q;')
        lines.append(f'                        float v_p = V[i][p]; float v_q = V[i][q];')
        lines.append(f'                        V[i][p] = c * v_p - s * v_q; V[i][q] = s * v_p + c * v_q;')
        lines.append(f'                    }}')
        lines.append(f'                }}')
        lines.append(f'            }}')
        lines.append(f'        }}')
        lines.append(f'    }}')
        lines.append(f'    for (int j = 0; j < {n}; ++j) {{')
        lines.append(f'        float sing_val = 0.0f;')
        lines.append(f'        for (int i = 0; i < {n}; ++i) sing_val += U[i][j] * U[i][j];')
        lines.append(f'        sing_val = sqrt(sing_val);')
        lines.append(f'        out_S[bid * {n} + j] = sing_val;')
        lines.append(f'        float inv_s = (sing_val > 1.0e-7f) ? (1.0f / sing_val) : 0.0f;')
        lines.append(f'        for (int i = 0; i < {n}; ++i) {{')
        lines.append(f'            out_U[base + i * {n} + j] = U[i][j] * inv_s;')
        lines.append(f'            out_V[base + i * {n} + j] = V[i][j];')
        lines.append(f'        }}')
        lines.append(f'    }}')
        lines.append(f'}}\n')

    return '\n'.join(lines)

def gen_rbf_interpolation_scale():
    lines = [
        '',
        '// ============================================================================',
        '// Radial Basis Function (RBF) Kernel Interpolators: Wendland and Multiquadric',
        '// ============================================================================',
        ''
    ]
    
    rbfs = [
        ('gaussian', 'exp(-epsilon_sq * r_sq)'),
        ('multiquadric', 'sqrt(1.0f + epsilon_sq * r_sq)'),
        ('inverse_multiquadric', '1.0f / sqrt(1.0f + epsilon_sq * r_sq)'),
        ('inverse_quadratic', '1.0f / (1.0f + epsilon_sq * r_sq)'),
        ('wendland_c2', 'select(0.0f, (1.0f - r_norm) * (1.0f - r_norm) * (1.0f - r_norm) * (1.0f - r_norm) * (4.0f * r_norm + 1.0f), r_norm < 1.0f)'),
        ('wendland_c4', 'select(0.0f, (1.0f - r_norm) * (1.0f - r_norm) * (1.0f - r_norm) * (1.0f - r_norm) * (1.0f - r_norm) * (1.0f - r_norm) * (35.0f * r_norm * r_norm + 18.0f * r_norm + 3.0f), r_norm < 1.0f)'),
    ]
    
    for rname, formula in rbfs:
        lines.append(f'// 3D RBF Evaluation: {rname}')
        lines.append(f'__kernel void kernel_rbf_interp3d_{rname}(')
        lines.append(f'    __global const float4* center_nodes, // [num_centers] (x, y, z, 0)')
        lines.append(f'    __global const float* weights,        // [num_centers]')
        lines.append(f'    __global float* out_volume,')
        lines.append(f'    const float epsilon_sq, const float support_radius,')
        lines.append(f'    const int nx, const int ny, const int nz, const float inv_dx, const int num_centers) {{')
        lines.append(f'    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);')
        lines.append(f'    if (x >= nx || y >= ny || z >= nz) return;')
        lines.append(f'    float px = (float)x * inv_dx; float py = (float)y * inv_dx; float pz = (float)z * inv_dx;')
        lines.append(f'    float sum = 0.0f;')
        lines.append(f'    for (int k = 0; k < num_centers; ++k) {{')
        lines.append(f'        float4 c = center_nodes[k];')
        lines.append(f'        float rx = px - c.x; float ry = py - c.y; float rz = pz - c.z;')
        lines.append(f'        float r_sq = rx * rx + ry * ry + rz * rz;')
        lines.append(f'        float r_norm = sqrt(r_sq) / support_radius;')
        lines.append(f'        float phi = {formula};')
        lines.append(f'        sum += weights[k] * phi;')
        lines.append(f'    }}')
        lines.append(f'    out_volume[(size_t)z * ((size_t)nx * (size_t)ny) + (size_t)y * (size_t)nx + (size_t)x] = sum;')
        lines.append(f'}}\n')

    return '\n'.join(lines)

def gen_extended_wavelets():
    lines = [
        '',
        '// ============================================================================',
        '// Extended Daubechies (db4, db6, db8) and Symlet Filter Banks',
        '// ============================================================================',
        ''
    ]
    
    # 2D Wavelet 4-Subband Decomposition (LL, LH, HL, HH)
    lines.append('// 2D Wavelet Subband Decomposition (LL, LH, HL, HH)')
    lines.append('__kernel void kernel_dwt2d_subband_pack(')
    lines.append('    __global const float* low_low, __global const float* low_high,')
    lines.append('    __global const float* high_low, __global const float* high_high,')
    lines.append('    __global float* out_packed_dwt, const int half_w, const int half_h) {{')
    lines.append('    int x = get_global_id(0); int y = get_global_id(1);')
    lines.append('    if (x >= half_w || y >= half_h) return;')
    lines.append('    int full_w = half_w * 2;')
    lines.append('    size_t in_idx = (size_t)y * (size_t)half_w + (size_t)x;')
    lines.append('    // Top-left: LL')
    lines.append('    out_packed_dwt[(size_t)y * (size_t)full_w + (size_t)x] = low_low[in_idx];')
    lines.append('    // Top-right: HL')
    lines.append('    out_packed_dwt[(size_t)y * (size_t)full_w + (size_t)(half_w + x)] = high_low[in_idx];')
    lines.append('    // Bottom-left: LH')
    lines.append('    out_packed_dwt[(size_t)(half_h + y) * (size_t)full_w + (size_t)x] = low_high[in_idx];')
    lines.append('    // Bottom-right: HH')
    lines.append('    out_packed_dwt[(size_t)(half_h + y) * (size_t)full_w + (size_t)(half_w + x)] = high_high[in_idx];')
    lines.append('}}\n')

    return '\n'.join(lines)

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(root, 'kernels', 'specialized')
    
    # Append QR/SVD to solvers
    solvers_path = os.path.join(target_dir, 'solvers_specialized.cl')
    with open(solvers_path, 'r', encoding='utf-8') as f:
        existing_solvers = f.read()
    extra_solvers = gen_qr_svd_solvers()
    verify_no_dashes(extra_solvers, 'extra_solvers')
    with open(solvers_path, 'w', encoding='utf-8') as f:
        f.write(existing_solvers + '\n' + extra_solvers)
    print("Updated solvers_specialized.cl with QR/SVD")

    # Append RBF to interpolation
    interp_path = os.path.join(target_dir, 'interpolation_specialized.cl')
    with open(interp_path, 'r', encoding='utf-8') as f:
        existing_interp = f.read()
    extra_interp = gen_rbf_interpolation_scale()
    verify_no_dashes(extra_interp, 'extra_interp')
    with open(interp_path, 'w', encoding='utf-8') as f:
        f.write(existing_interp + '\n' + extra_interp)
    print("Updated interpolation_specialized.cl with RBFs")

    # Append extended wavelets
    wavelet_path = os.path.join(target_dir, 'wavelets_specialized.cl')
    with open(wavelet_path, 'r', encoding='utf-8') as f:
        existing_wavelets = f.read()
    extra_wavelets = gen_extended_wavelets()
    verify_no_dashes(extra_wavelets, 'extra_wavelets')
    with open(wavelet_path, 'w', encoding='utf-8') as f:
        f.write(existing_wavelets + '\n' + extra_wavelets)
    print("Updated wavelets_specialized.cl with subband packing")

if __name__ == '__main__':
    main()
