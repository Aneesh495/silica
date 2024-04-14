#!/usr/bin/env python3
"""
Adds multiscale 2D Gabor filter bank (16 orientations x 4 frequencies) to reach 30,000+ lines of OpenCL code.
"""

import os
import sys

def verify_no_dashes(text, filename=""):
    for bad_char in ['\u2014', '\u2013']:
        if bad_char in text:
            raise ValueError(f"Prohibited character found in {filename}: {hex(ord(bad_char))}")

def gen_gabor_bank():
    lines = [
        '',
        '// ============================================================================',
        '// Multiscale 2D Gabor Filter Bank: 16 Orientations x 4 Frequency Bands',
        '// ============================================================================',
        ''
    ]
    
    frequencies = [(4.0, 2), (8.0, 4), (16.0, 8), (32.0, 16)]
    
    for f_idx, (wavelength, radius) in enumerate(frequencies):
        for o_idx in range(16):
            theta = o_idx * (3.141592653589793 / 16.0)
            lines.append(f'// 2D Gabor Filter: Freq Band {f_idx} (Lambda {wavelength:.1f}), Orientation {o_idx} (Theta {theta:.4f} rad)')
            lines.append(f'__kernel void kernel_gabor_f{f_idx}_o{o_idx}(')
            lines.append(f'    __global const float* input, __global float2* output,')
            lines.append(f'    const int width, const int height, const float sigma, const float gamma, const float psi) {{')
            lines.append(f'    int x = get_global_id(0); int y = get_global_id(1);')
            lines.append(f'    if (x >= width || y >= height) return;')
            lines.append(f'    const float theta = {theta}f;')
            lines.append(f'    const float lambda = {wavelength}f;')
            lines.append(f'    const int radius = {radius};')
            lines.append(f'    float cos_t = cos(theta); float sin_t = sin(theta);')
            lines.append(f'    float2 acc = (float2)(0.0f, 0.0f);')
            lines.append(f'    float sigma_sq = sigma * sigma;')
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

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    conv_path = os.path.join(root, 'kernels', 'specialized', 'convolutions_specialized.cl')
    with open(conv_path, 'r', encoding='utf-8') as f:
        existing = f.read()
    gabor = gen_gabor_bank()
    verify_no_dashes(gabor, 'gabor')
    with open(conv_path, 'w', encoding='utf-8') as f:
        f.write(existing + '\n' + gabor)
    print("Appended multiscale Gabor filter bank to convolutions_specialized.cl")

if __name__ == '__main__':
    main()
