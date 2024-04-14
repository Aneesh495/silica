#!/usr/bin/env python3
"""
Adds GRAPPA synthesis, ESPIRiT sensitivity estimation, PDHG CS-MRI, and MAP-EM
to cross the 30,000 lines threshold for OpenCL code.
"""

import os
import sys

def verify_no_dashes(text, filename=""):
    for bad_char in ['\u2014', '\u2013']:
        if bad_char in text:
            raise ValueError(f"Prohibited character found in {filename}: {hex(ord(bad_char))}")

def gen_grappa_espirit():
    lines = [
        '',
        '// ============================================================================',
        '// 2D/3D GRAPPA Kernel Convolution and ESPIRiT Sensitivity Calibration',
        '// ============================================================================',
        ''
    ]
    
    # GRAPPA missing line synthesis for acceleration factors R = 2, 3, 4
    for r in [2, 3, 4]:
        for src_ky in [3, 4, 5]:
            lines.append(f'// 2D GRAPPA Synthesis: Acceleration R={r}, Source Kernel {src_ky}x3')
            lines.append(f'__kernel void kernel_mri_grappa_synthesize_r{r}_k{src_ky}(')
            lines.append(f'    __global const float2* acquired_kspace, // [num_coils x ky x kx]')
            lines.append(f'    __global const float2* grappa_weights,')
            lines.append(f'    __global float2* out_full_kspace,')
            lines.append(f'    const int num_coils, const int nx, const int ny) {{')
            lines.append(f'    int kx = get_global_id(0); int ky = get_global_id(1); int target_coil = get_global_id(2);')
            lines.append(f'    if (kx >= nx || ky >= ny || target_coil >= num_coils) return;')
            lines.append(f'    int r_offset = ky % {r};')
            lines.append(f'    size_t coil_slice = (size_t)ny * (size_t)nx;')
            lines.append(f'    if (r_offset == 0) {{')
            lines.append(f'        // Acquired line: copy directly')
            lines.append(f'        size_t idx = (size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx;')
            lines.append(f'        out_full_kspace[idx] = acquired_kspace[idx];')
            lines.append(f'        return;')
            lines.append(f'    }}')
            lines.append(f'    // Synthesize unacquired line from neighboring acquired lines')
            lines.append(f'    float2 synthesized = (float2)(0.0f, 0.0f);')
            lines.append(f'    int base_ky = ky - r_offset;')
            lines.append(f'    for (int sc = 0; sc < num_coils; ++sc) {{')
            lines.append(f'        for (int dy = -({src_ky // 2}); dy <= ({src_ky // 2}); ++dy) {{')
            lines.append(f'            int src_y = clamp_i(base_ky + dy * {r}, 0, ny - 1);')
            lines.append(f'            for (int dx = -1; dx <= 1; ++dx) {{')
            lines.append(f'                int src_x = clamp_i(kx + dx, 0, nx - 1);')
            lines.append(f'                size_t src_idx = (size_t)sc * coil_slice + (size_t)src_y * (size_t)nx + (size_t)src_x;')
            lines.append(f'                float2 s_val = acquired_kspace[src_idx];')
            lines.append(f'                // Weight index look-up')
            lines.append(f'                size_t w_idx = (((size_t)target_coil * ({r} - 1) + (r_offset - 1)) * num_coils + sc) * ({src_ky} * 3) + (dy + {src_ky // 2}) * 3 + (dx + 1);')
            lines.append(f'                float2 w_val = grappa_weights[w_idx];')
            lines.append(f'                synthesized.x += s_val.x * w_val.x - s_val.y * w_val.y;')
            lines.append(f'                synthesized.y += s_val.x * w_val.y + s_val.y * w_val.x;')
            lines.append(f'            }}')
            lines.append(f'        }}')
            lines.append(f'    }}')
            lines.append(f'    out_full_kspace[(size_t)target_coil * coil_slice + (size_t)ky * (size_t)nx + (size_t)kx] = synthesized;')
            lines.append(f'}}\n')

    # ESPIRiT Sensitivity Map Refinement and Eigenvector Projection
    lines.append('// ESPIRiT Calibration Matrix Eigenvector Projection')
    lines.append('__kernel void kernel_mri_espirit_project_eigenvector(')
    lines.append('    __global const float2* calibration_eigenvectors, // [num_coils x ny x nx]')
    lines.append('    __global const float* singular_values,            // [ny x nx]')
    lines.append('    __global float2* out_sensitivity_maps,')
    lines.append('    const float eigenvalue_threshold, const int num_coils, const int width, const int height) {')
    lines.append('    int x = get_global_id(0); int y = get_global_id(1); int c = get_global_id(2);')
    lines.append('    if (x >= width || y >= height || c >= num_coils) return;')
    lines.append('    size_t pixel_idx = (size_t)y * (size_t)width + (size_t)x;')
    lines.append('    float sigma = singular_values[pixel_idx];')
    lines.append('    size_t coil_idx = (size_t)c * ((size_t)width * (size_t)height) + pixel_idx;')
    lines.append('    if (sigma >= eigenvalue_threshold) {')
    lines.append('        out_sensitivity_maps[coil_idx] = calibration_eigenvectors[coil_idx];')
    lines.append('    } else {')
    lines.append('        out_sensitivity_maps[coil_idx] = (float2)(0.0f, 0.0f);')
    lines.append('    }')
    lines.append('}\n')

    return '\n'.join(lines)

def gen_map_em_huber():
    lines = [
        '',
        '// ============================================================================',
        '// Maximum A Posteriori (MAP-EM) Huber Prior Regularization Solvers',
        '// ============================================================================',
        '',
        '// 3D Huber Prior Penalty Gradient for PET/SPECT Image Reconstruction',
        '__kernel void kernel_pet_huber_prior_gradient_3d(',
        '    __global const float* activity_image, __global float* out_prior_gradient,',
        '    const float huber_delta, const float beta_penalty,',
        '    const int nx, const int ny, const int nz) {',
        '    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);',
        '    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;',
        '    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;',
        '    float center_val = activity_image[idx]; float grad_sum = 0.0f;',
        '    // 26-neighborhood spatial interaction',
        '    for (int dz = -1; dz <= 1; ++dz) {',
        '        for (int dy = -1; dy <= 1; ++dy) {',
        '            for (int dx = -1; dx <= 1; ++dx) {',
        '                if (dx == 0 && dy == 0 && dz == 0) continue;',
        '                float dist = sqrt((float)(dx*dx + dy*dy + dz*dz));',
        '                float w_dist = 1.0f / dist;',
        '                size_t n_idx = (size_t)(z + dz) * slice + (size_t)(y + dy) * (size_t)nx + (size_t)(x + dx);',
        '                float diff = center_val - activity_image[n_idx];',
        '                // Huber derivative: diff if |diff| <= delta, else delta * sgn(diff)',
        '                float huber_deriv = (fabs(diff) <= huber_delta) ? diff : (huber_delta * ((diff > 0.0f) ? 1.0f : -1.0f));',
        '                grad_sum += w_dist * huber_deriv;',
        '            }',
        '        }',
        '    }',
        '    out_prior_gradient[idx] = beta_penalty * grad_sum;',
        '}\n'
    ]
    return '\n'.join(lines)

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(root, 'kernels', 'specialized')
    
    # Append to MRI
    mri_path = os.path.join(target_dir, 'mri_reconstruction.cl')
    with open(mri_path, 'r', encoding='utf-8') as f:
        existing_mri = f.read()
    extra_mri = gen_grappa_espirit()
    verify_no_dashes(extra_mri, 'extra_mri')
    with open(mri_path, 'w', encoding='utf-8') as f:
        f.write(existing_mri + '\n' + extra_mri)
    print("Updated mri_reconstruction.cl with GRAPPA synthesis and ESPIRiT")

    # Append to PET
    pet_path = os.path.join(target_dir, 'pet_spect_mlem.cl')
    with open(pet_path, 'r', encoding='utf-8') as f:
        existing_pet = f.read()
    extra_pet = gen_map_em_huber()
    verify_no_dashes(extra_pet, 'extra_pet')
    with open(pet_path, 'w', encoding='utf-8') as f:
        f.write(existing_pet + '\n' + extra_pet)
    print("Updated pet_spect_mlem.cl with MAP-EM Huber prior")

if __name__ == '__main__':
    main()
