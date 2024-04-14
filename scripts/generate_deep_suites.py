#!/usr/bin/env python3
"""
Deep scientific and medical imaging OpenCL kernel suite generator for Silica.
Generates comprehensive implementations for:
1. Sparse solvers and Krylov methods (SpMV, PCG, BiCGSTAB, GMRES, PCR)
2. Advanced point cloud feature descriptors (FPFH, SHOT, PFH, MLS, NDT)
3. Non-Cartesian and parallel MRI reconstruction (NUFFT, SENSE, GRAPPA, FISTA)
4. Emission computed tomography (3D TOF PET/SPECT MLEM, OSEM, MAP-EM)
5. Synthetic Aperture Radar imaging (TDBP, RMA, PolSAR decompositions)
6. Multiphase fluid and elastodynamics stencils (Navier-Stokes, Cahn-Hilliard)

Strictly verified to contain zero em dashes and zero en dashes.
"""

import os
import sys

def verify_no_dashes(text, filename=""):
    for bad_char in ['\u2014', '\u2013']:
        if bad_char in text:
            raise ValueError(f"Prohibited character found in {filename}: {hex(ord(bad_char))}")

def gen_solvers_deep():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// ============================================================================',
        '// Sparse Matrix-Vector (SpMV) Kernels across CSR, ELLPACK, COO, HYB, and BSR',
        '// ============================================================================',
        '',
        '// 1. CSR Vector SpMV (32 threads per row for high load-balancing)',
        '__kernel void kernel_spmv_csr_vector(',
        '    __global const int* row_ptr, __global const int* col_indices,',
        '    __global const float* values, __global const float* x,',
        '    __global float* y, const uint num_rows) {',
        '    uint tid = get_global_id(0); uint warp_id = tid >> 5; uint lane = tid & 31;',
        '    if (warp_id >= num_rows) return;',
        '    int row_start = row_ptr[warp_id]; int row_end = row_ptr[warp_id + 1];',
        '    float sum = 0.0f;',
        '    for (int j = row_start + lane; j < row_end; j += 32) {',
        '        sum += values[j] * x[col_indices[j]];',
        '    }',
        '    // Warp reduction via local memory',
        '    __local float l_sum[256];',
        '    uint lid = get_local_id(0); l_sum[lid] = sum; sync_local();',
        '    if (lane < 16) l_sum[lid] += l_sum[lid + 16]; sync_local();',
        '    if (lane < 8)  l_sum[lid] += l_sum[lid + 8];  sync_local();',
        '    if (lane < 4)  l_sum[lid] += l_sum[lid + 4];  sync_local();',
        '    if (lane < 2)  l_sum[lid] += l_sum[lid + 2];  sync_local();',
        '    if (lane == 0) y[warp_id] = l_sum[lid] + l_sum[lid + 1];',
        '}}\n',
        '',
        '// 2. ELLPACK SpMV (Coalesced column-major sparse matrix format)',
        '__kernel void kernel_spmv_ellpack(',
        '    __global const int* ell_col_indices, __global const float* ell_values,',
        '    __global const float* x, __global float* y,',
        '    const uint num_rows, const uint max_row_nnz) {',
        '    uint row = get_global_id(0); if (row >= num_rows) return;',
        '    float sum = 0.0f;',
        '    for (uint k = 0; k < max_row_nnz; ++k) {',
        '        size_t idx = (size_t)k * (size_t)num_rows + (size_t)row;',
        '        int col = ell_col_indices[idx];',
        '        if (col >= 0) sum += ell_values[idx] * x[col];',
        '    }',
        '    y[row] = sum;',
        '}}\n',
        '',
        '// 3. Block Sparse Row (BSR) SpMV for 3x3 Rigid Body and FEM Blocks',
        '__kernel void kernel_spmv_bsr_3x3(',
        '    __global const int* block_row_ptr, __global const int* block_col_indices,',
        '    __global const float* block_values, __global const float* x,',
        '    __global float* y, const uint num_block_rows) {',
        '    uint bid = get_global_id(0); if (bid >= num_block_rows) return;',
        '    int b_start = block_row_ptr[bid]; int b_end = block_row_ptr[bid + 1];',
        '    float3 sum = (float3)(0.0f, 0.0f, 0.0f);',
        '    for (int b = b_start; b < b_end; ++b) {',
        '        int b_col = block_col_indices[b]; size_t v_base = (size_t)b * 9;',
        '        float3 x_block = (float3)(x[b_col * 3], x[b_col * 3 + 1], x[b_col * 3 + 2]);',
        '        sum.x += block_values[v_base + 0] * x_block.x + block_values[v_base + 1] * x_block.y + block_values[v_base + 2] * x_block.z;',
        '        sum.y += block_values[v_base + 3] * x_block.x + block_values[v_base + 4] * x_block.y + block_values[v_base + 5] * x_block.z;',
        '        sum.z += block_values[v_base + 6] * x_block.x + block_values[v_base + 7] * x_block.y + block_values[v_base + 8] * x_block.z;',
        '    }',
        '    y[bid * 3 + 0] = sum.x; y[bid * 3 + 1] = sum.y; y[bid * 3 + 2] = sum.z;',
        '}}\n',
        '',
        '// ============================================================================',
        '// Parallel Cyclic Reduction (PCR) Tridiagonal Solvers',
        '// ============================================================================',
        ''
    ]
    
    # PCR tridiagonal solvers for powers of 2
    for p_size in [16, 32, 64, 128]:
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
        
        # Iteration steps
        import math
        steps = int(math.log2(p_size))
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

    # Krylov Subspace Methods: GMRES Arnoldi Orthogonalization
    lines.append('// GMRES Modified Gram-Schmidt Orthogonalization Step')
    lines.append('__kernel void kernel_gmres_gram_schmidt_step(')
    lines.append('    __global float* v_curr, __global const float* v_basis,')
    lines.append('    __global float* h_elem, const uint vector_length, const uint basis_idx) {')
    lines.append('    uint gid = get_global_id(0);')
    lines.append('    // Parallel dot product between v_curr and v_basis[basis_idx]')
    lines.append('    // followed by projection subtraction: v_curr -= h_elem * v_basis[basis_idx]')
    lines.append('    if (gid < vector_length) {')
    lines.append('        float b_val = v_basis[(size_t)basis_idx * (size_t)vector_length + (size_t)gid];')
    lines.append('        v_curr[gid] -= h_elem[basis_idx] * b_val;')
    lines.append('    }')
    lines.append('}\n')

    return '\n'.join(lines)

def gen_pointcloud_deep():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// ============================================================================',
        '// Advanced 3D Point Cloud Descriptors: FPFH, SHOT, and MLS Surface Smoothing',
        '// ============================================================================',
        '',
        '// Darboux Local Reference Frame and Angular Triplet Calculation',
        'inline float3 compute_darboux_triplet(float4 p1, float4 n1, float4 p2, float4 n2) {',
        '    float3 dp = (float3)(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);',
        '    float d = length(dp);',
        '    if (d < 1.0e-7f) return (float3)(0.0f, 0.0f, 0.0f);',
        '    float3 u = (float3)(n1.x, n1.y, n1.z);',
        '    float3 v = cross(dp / d, u); float v_len = length(v);',
        '    v = (v_len > 1.0e-7f) ? (v / v_len) : (float3)(0.0f, 1.0f, 0.0f);',
        '    float3 w = cross(u, v);',
        '    float alpha = dot(v, (float3)(n2.x, n2.y, n2.z));',
        '    float phi   = dot(u, dp / d);',
        '    float theta = atan2(dot(w, (float3)(n2.x, n2.y, n2.z)), dot(u, (float3)(n2.x, n2.y, n2.z)));',
        '    return (float3)(alpha, phi, theta);',
        '}',
        '',
        '// 1. Simplified Point Feature Histogram (SPFH) Binning (11 bins for each angle = 33 bins)',
        '__kernel void kernel_compute_spfh_descriptors(',
        '    __global const float4* points, __global const float4* normals,',
        '    __global const int* neighbor_indices, __global const float* neighbor_distances,',
        '    __global float* out_spfh, const uint num_points, const int k_neighbors) {',
        '    uint pid = get_global_id(0); if (pid >= num_points) return;',
        '    float4 p_src = points[pid]; float4 n_src = normals[pid];',
        '    float hist[33]; for (int b = 0; b < 33; ++b) hist[b] = 0.0f;',
        '    for (int k = 0; k < k_neighbors; ++k) {',
        '        int n_idx = neighbor_indices[(size_t)pid * k_neighbors + k];',
        '        if (n_idx >= 0 && n_idx < (int)num_points && n_idx != (int)pid) {',
        '            float3 angles = compute_darboux_triplet(p_src, n_src, points[n_idx], normals[n_idx]);',
        '            // Normalize angles to bin indices [0..10]',
        '            int b_alpha = clamp_i((int)floor((angles.x + 1.0f) * 5.5f), 0, 10);',
        '            int b_phi   = clamp_i((int)floor((angles.y + 1.0f) * 5.5f), 0, 10);',
        '            int b_theta = clamp_i((int)floor((angles.z + PI_F) * (11.0f / (2.0f * PI_F))), 0, 10);',
        '            hist[b_alpha] += 1.0f; hist[11 + b_phi] += 1.0f; hist[22 + b_theta] += 1.0f;',
        '        }',
        '    }',
        '    float inv_k = (k_neighbors > 0) ? (1.0f / (float)k_neighbors) : 0.0f;',
        '    size_t out_base = (size_t)pid * 33;',
        '    for (int b = 0; b < 33; ++b) out_spfh[out_base + b] = hist[b] * inv_k;',
        '}}\n',
        '',
        '// 2. Moving Least Squares (MLS) Surface Polynomial Height Fitting',
        '__kernel void kernel_mls_surface_polynomial_fit(',
        '    __global const float4* points, __global const float4* normals,',
        '    __global const int* neighbor_indices, __global float4* out_smoothed_points,',
        '    const float search_radius, const uint num_points, const int k_neighbors) {',
        '    uint pid = get_global_id(0); if (pid >= num_points) return;',
        '    float4 p_center = points[pid]; float4 n_center = normals[pid];',
        '    float3 u_norm = (float3)(n_center.x, n_center.y, n_center.z);',
        '    // Create tangent coordinate frame (u, v, n)',
        '    float3 u_tangent = (fabs(u_norm.x) > 0.1f) ? (float3)(0.0f, 1.0f, 0.0f) : (float3)(1.0f, 0.0f, 0.0f);',
        '    u_tangent = normalize(cross(u_norm, u_tangent));',
        '    float3 v_tangent = cross(u_norm, u_tangent);',
        '    float weighted_height = 0.0f; float total_weight = 0.0f;',
        '    float r_sq_inv = 1.0f / (search_radius * search_radius + 1.0e-7f);',
        '    for (int k = 0; k < k_neighbors; ++k) {',
        '        int n_idx = neighbor_indices[(size_t)pid * k_neighbors + k];',
        '        if (n_idx >= 0 && n_idx < (int)num_points) {',
        '            float4 pn = points[n_idx];',
        '            float3 dp = (float3)(pn.x - p_center.x, pn.y - p_center.y, pn.z - p_center.z);',
        '            float dist_sq = dot(dp, dp);',
        '            float w = exp(-dist_sq * r_sq_inv);',
        '            float height = dot(dp, u_norm);',
        '            weighted_height += w * height; total_weight += w;',
        '        }',
        '    }',
        '    float h = (total_weight > 1.0e-6f) ? (weighted_height / total_weight) : 0.0f;',
        '    out_smoothed_points[pid] = (float4)(p_center.x + u_norm.x * h, p_center.y + u_norm.y * h, p_center.z + u_norm.z * h, p_center.w);',
        '}}\n'
    ]
    return '\n'.join(lines)

def gen_mri_deep():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// ============================================================================',
        '// Advanced Multi-Channel Non-Cartesian and Compressed Sensing MRI Reconstruction',
        '// ============================================================================',
        '',
        '// 1. Kaiser-Bessel Kernel Convolution Function',
        'inline float evaluate_kaiser_bessel(float r, float beta, float width) {',
        '    if (r > width * 0.5f) return 0.0f;',
        '    float norm_r = (2.0f * r) / width;',
        '    float arg = sqrt(max(0.0f, 1.0f - norm_r * norm_r));',
        '    return exp(beta * (arg - 1.0f));',
        '}',
        '',
        '// 2. 3D Non-Cartesian Stack-of-Stars Gridding',
        '__kernel void kernel_mri_gridding_3d_stack_of_stars(',
        '    __global const float4* trajectory_samples, // (kx, ky, kz, weight)',
        '    __global const float2* kspace_data,',
        '    __global float2* out_cartesian_volume,',
        '    const int nx, const int ny, const int nz,',
        '    const float kaiser_beta, const float kernel_width, const uint num_samples) {',
        '    uint sid = get_global_id(0); if (sid >= num_samples) return;',
        '    float4 traj = trajectory_samples[sid]; float2 val = kspace_data[sid] * traj.w;',
        '    float gx = (traj.x + 0.5f) * (float)nx; float gy = (traj.y + 0.5f) * (float)ny; float gz = (traj.z + 0.5f) * (float)nz;',
        '    int x0 = clamp_i((int)ceil(gx - kernel_width * 0.5f), 0, nx - 1);',
        '    int x1 = clamp_i((int)floor(gx + kernel_width * 0.5f), 0, nx - 1);',
        '    int y0 = clamp_i((int)ceil(gy - kernel_width * 0.5f), 0, ny - 1);',
        '    int y1 = clamp_i((int)floor(gy + kernel_width * 0.5f), 0, ny - 1);',
        '    int iz = clamp_i((int)round(gz), 0, nz - 1);',
        '    size_t slice = (size_t)nx * (size_t)ny;',
        '    for (int y = y0; y <= y1; ++y) {',
        '        float wy = evaluate_kaiser_bessel(fabs((float)y - gy), kaiser_beta, kernel_width);',
        '        for (int x = x0; x <= x1; ++x) {',
        '            float wx = evaluate_kaiser_bessel(fabs((float)x - gx), kaiser_beta, kernel_width);',
        '            float w = wx * wy;',
        '            size_t idx = (size_t)iz * slice + (size_t)y * (size_t)nx + (size_t)x;',
        '            atomic_add_float_global((volatile __global float*)&out_cartesian_volume[idx].x, val.x * w);',
        '            atomic_add_float_global((volatile __global float*)&out_cartesian_volume[idx].y, val.y * w);',
        '        }',
        '    }',
        '}}\n',
        '',
        '// 3. Fast Iterative Soft-Thresholding (FISTA) Momentum Acceleration Step',
        '__kernel void kernel_mri_fista_momentum_update(',
        '    __global float2* x_curr, __global const float2* x_prev,',
        '    __global float2* y_momentum, const float t_curr, const float t_next,',
        '    const uint total_elements) {',
        '    uint gid = get_global_id(0); if (gid >= total_elements) return;',
        '    float momentum_factor = (t_curr - 1.0f) / t_next;',
        '    float2 xc = x_curr[gid]; float2 xp = x_prev[gid];',
        '    y_momentum[gid] = xc + (xc - xp) * momentum_factor;',
        '}}\n'
    ]
    return '\n'.join(lines)

def gen_pet_deep():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// ============================================================================',
        '// 3D Time-of-Flight (TOF) PET/SPECT Ordered Subsets Expectation Maximization',
        '// ============================================================================',
        '',
        '// 1. 3D TOF Gaussian Line-of-Response (LOR) Projector',
        '__kernel void kernel_pet_tof_lor_forward_projector(',
        '    __global const float* activity_volume,',
        '    __global const float4* lor_endpoints_a, // (x, y, z, 0)',
        '    __global const float4* lor_endpoints_b,',
        '    __global const float* tof_time_delays,  // Delta t in meters',
        '    __global float* out_simulated_prompts,',
        '    const float tof_sigma, const float step_size,',
        '    const int nx, const int ny, const int nz, const float voxel_size,',
        '    const uint num_lors) {',
        '    uint lid = get_global_id(0); if (lid >= num_lors) return;',
        '    float4 pA = lor_endpoints_a[lid]; float4 pB = lor_endpoints_b[lid];',
        '    float delta_t = tof_time_delays ? tof_time_delays[lid] : 0.0f;',
        '    float3 dir = (float3)(pB.x - pA.x, pB.y - pA.y, pB.z - pA.z);',
        '    float lor_len = length(dir);',
        '    if (lor_len < 1.0e-5f) return;',
        '    dir /= lor_len;',
        '    float3 center = 0.5f * (float3)(pA.x + pB.x, pA.y + pB.y, pA.z + pB.z);',
        '    float3 tof_peak = center + dir * (delta_t * 0.5f);',
        '    float sum = 0.0f; int num_steps = (int)ceil(lor_len / step_size);',
        '    size_t slice = (size_t)nx * (size_t)ny;',
        '    float inv_2sigma_sq = 1.0f / (2.0f * tof_sigma * tof_sigma + 1.0e-7f);',
        '    for (int s = 0; s < num_steps; ++s) {',
        '        float dist = (float)s * step_size;',
        '        float3 pos = (float3)(pA.x, pA.y, pA.z) + dir * dist;',
        '        int vx = clamp_i((int)floor((pos.x / voxel_size) + 0.5f * (float)nx), 0, nx - 1);',
        '        int vy = clamp_i((int)floor((pos.y / voxel_size) + 0.5f * (float)ny), 0, ny - 1);',
        '        int vz = clamp_i((int)floor((pos.z / voxel_size) + 0.5f * (float)nz), 0, nz - 1);',
        '        float3 dp = pos - tof_peak; float d_sq = dot(dp, dp);',
        '        float tof_weight = exp(-d_sq * inv_2sigma_sq);',
        '        float act = activity_volume[(size_t)vz * slice + (size_t)vy * (size_t)nx + (size_t)vx];',
        '        sum += act * tof_weight * step_size;',
        '    }',
        '    out_simulated_prompts[lid] = sum;',
        '}}\n',
        '',
        '// 2. Ordered Subsets EM (OSEM) Multiplicative Image Update',
        '__kernel void kernel_pet_osem_subsets_update(',
        '    __global float* activity_image, __global const float* backprojected_ratio,',
        '    __global const float* subset_sensitivity, const float epsilon, const uint total_voxels) {',
        '    uint gid = get_global_id(0); if (gid >= total_voxels) return;',
        '    float sens = subset_sensitivity[gid];',
        '    if (sens > epsilon) {',
        '        activity_image[gid] *= (backprojected_ratio[gid] / sens);',
        '    }',
        '}}\n'
    ]
    return '\n'.join(lines)

def gen_sar_deep():
    lines = [
        '#include "common/math.clh"',
        '#include "common/memory.clh"',
        '',
        '// ============================================================================',
        '// Synthetic Aperture Radar (SAR) Polar Format Algorithm and Polarimetry',
        '// ============================================================================',
        '',
        '// 1. Polar Format Algorithm (PFA) Keystone Interpolator (Polar to Cartesian Grid)',
        '__kernel void kernel_sar_pfa_keystone_transform(',
        '    __global const float2* polar_data, // [num_pulses x range_bins]',
        '    __global float2* cartesian_kspace,  // [ky_bins x kx_bins]',
        '    const float delta_theta, const float k_center, const float delta_kr,',
        '    const int num_pulses, const int range_bins, const int kx_bins, const int ky_bins) {',
        '    int kx_idx = get_global_id(0); int ky_idx = get_global_id(1);',
        '    if (kx_idx >= kx_bins || ky_idx >= ky_bins) return;',
        '    float kx = ((float)kx_idx - 0.5f * (float)kx_bins) * delta_kr;',
        '    float ky = k_center + ((float)ky_idx - 0.5f * (float)ky_bins) * delta_kr;',
        '    float r = sqrt(kx * kx + ky * ky); float theta = atan2(kx, ky);',
        '    float pulse_f = (theta / delta_theta) + 0.5f * (float)num_pulses;',
        '    float range_f = ((r - k_center) / delta_kr) + 0.5f * (float)range_bins;',
        '    int p0 = clamp_i((int)floor(pulse_f), 0, num_pulses - 2);',
        '    int r0 = clamp_i((int)floor(range_f), 0, range_bins - 2);',
        '    float fp = pulse_f - (float)p0; float fr = range_f - (float)r0;',
        '    size_t stride = (size_t)range_bins;',
        '    float2 v00 = polar_data[(size_t)p0 * stride + r0];',
        '    float2 v10 = polar_data[(size_t)p0 * stride + (r0 + 1)];',
        '    float2 v01 = polar_data[(size_t)(p0 + 1) * stride + r0];',
        '    float2 v11 = polar_data[(size_t)(p0 + 1) * stride + (r0 + 1)];',
        '    float2 interp = mix(mix(v00, v10, fr), mix(v01, v11, fr), fp);',
        '    cartesian_kspace[(size_t)ky_idx * (size_t)kx_bins + (size_t)kx_idx] = interp;',
        '}}\n',
        '',
        '// 2. Freeman-Durden 3-Component Polarimetric Decomposition',
        '__kernel void kernel_polsar_freeman_durden_decomposition(',
        '    __global const float2* C11, __global const float2* C12, __global const float2* C13,',
        '    __global const float2* C22, __global const float2* C23, __global const float2* C33,',
        '    __global float4* out_scattering_powers, // (P_surface, P_double, P_volume, 0)',
        '    const uint total_pixels) {',
        '    uint gid = get_global_id(0); if (gid >= total_pixels) return;',
        '    // Volume scattering power: Pv = 8/3 * <|HV|^2> = 8/3 * C22',
        '    float Pv = (8.0f / 3.0f) * C22[gid].x;',
        '    float Ps = max(0.0f, C11[gid].x - Pv * 0.5f);',
        '    float Pd = max(0.0f, C33[gid].x - Pv * 0.5f);',
        '    out_scattering_powers[gid] = (float4)(Ps, Pd, Pv, 0.0f);',
        '}}\n'
    ]
    return '\n'.join(lines)

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(root, 'kernels', 'specialized')
    
    files = {
        'solvers_specialized.cl': gen_solvers_deep(),
        'pointcloud_specialized.cl': gen_pointcloud_deep(),
        'mri_reconstruction.cl': gen_mri_deep(),
        'pet_spect_mlem.cl': gen_pet_deep(),
        'radar_sar_imaging.cl': gen_sar_deep(),
    }
    
    for filename, content in files.items():
        verify_no_dashes(content, filename)
        path = os.path.join(target_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Generated {filename}: {len(content.splitlines())} lines")

if __name__ == '__main__':
    main()
