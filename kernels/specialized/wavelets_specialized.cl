#include "common/math.clh"
#include "common/memory.clh"

// ============================================================================
// Multiscale Discrete Wavelet Transforms (DWT) and Gabor Filter Banks
// ============================================================================

// Daubechies 4 (db2) 1D Forward Wavelet Transform (Row Pass)
__kernel void kernel_dwt_db4_forward_rows(
    __global const float* input, __global float* output, const int width, const int height) {
    int x = get_global_id(0); int y = get_global_id(1); int half_w = width / 2;
    if (x >= half_w || y >= height) return;
    size_t base = (size_t)y * (size_t)width;
    const float h0 = 0.4829629131445341f; const float h1 = 0.8365163037378079f;
    const float h2 = 0.2241438680420134f; const float h3 = -0.1294095225512604f;
    const float g0 = h3; const float g1 = -h2; const float g2 = h1; const float g3 = -h0;
    int i0 = 2 * x; int i1 = (2 * x + 1) % width; int i2 = (2 * x + 2) % width; int i3 = (2 * x + 3) % width;
    float v0 = input[base + i0]; float v1 = input[base + i1]; float v2 = input[base + i2]; float v3 = input[base + i3];
    float low  = h0 * v0 + h1 * v1 + h2 * v2 + h3 * v3;
    float high = g0 * v0 + g1 * v1 + g2 * v2 + g3 * v3;
    output[base + x] = low; output[base + half_w + x] = high;
}

// Daubechies 4 (db2) 1D Inverse Wavelet Transform (Row Pass)
__kernel void kernel_dwt_db4_inverse_rows(
    __global const float* input, __global float* output, const int width, const int height) {
    int x = get_global_id(0); int y = get_global_id(1); int half_w = width / 2;
    if (x >= half_w || y >= height) return;
    size_t base = (size_t)y * (size_t)width;
    const float h0 = 0.4829629131445341f; const float h1 = 0.8365163037378079f;
    const float h2 = 0.2241438680420134f; const float h3 = -0.1294095225512604f;
    const float g0 = h3; const float g1 = -h2; const float g2 = h1; const float g3 = -h0;
    int xm1 = (x - 1 + half_w) % half_w;
    float s0 = input[base + x]; float s_prev = input[base + xm1];
    float d0 = input[base + half_w + x]; float d_prev = input[base + half_w + xm1];
    float out_even = h0 * s0 + h2 * s_prev + g0 * d0 + g2 * d_prev;
    float out_odd  = h1 * s0 + h3 * s_prev + g1 * d0 + g3 * d_prev;
    output[base + 2 * x]     = out_even;
    output[base + 2 * x + 1] = out_odd;
}

// 2D Gabor Filter Bank: Orientation 0 (Theta 0.0000 rad)
__kernel void kernel_gabor_bank_orient_0(
    __global const float* input, __global float2* output,
    const int width, const int height, const float lambda,
    const float psi, const float sigma, const float gamma, const int radius) {
    int x = get_global_id(0); int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const float theta = 0.0f;
    float cos_t = cos(theta); float sin_t = sin(theta);
    float2 acc = (float2)(0.0f, 0.0f); float sigma_sq = sigma * sigma;
    for (int dy = -radius; dy <= radius; ++dy) {
        int cy = clamp_i(y + dy, 0, height - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int cx = clamp_i(x + dx, 0, width - 1);
            float x_p =  (float)dx * cos_t + (float)dy * sin_t;
            float y_p = -(float)dx * sin_t + (float)dy * cos_t;
            float envelope = exp(- (x_p * x_p + gamma * gamma * y_p * y_p) / (2.0f * sigma_sq));
            float carrier = (2.0f * PI_F * x_p / lambda) + psi;
            float val = input[(size_t)cy * (size_t)width + (size_t)cx];
            acc.x += val * envelope * cos(carrier);
            acc.y += val * envelope * sin(carrier);
        }
    }
    output[(size_t)y * (size_t)width + (size_t)x] = acc;
}

// 2D Gabor Filter Bank: Orientation 1 (Theta 0.3927 rad)
__kernel void kernel_gabor_bank_orient_1(
    __global const float* input, __global float2* output,
    const int width, const int height, const float lambda,
    const float psi, const float sigma, const float gamma, const int radius) {
    int x = get_global_id(0); int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const float theta = 0.39269908169872414f;
    float cos_t = cos(theta); float sin_t = sin(theta);
    float2 acc = (float2)(0.0f, 0.0f); float sigma_sq = sigma * sigma;
    for (int dy = -radius; dy <= radius; ++dy) {
        int cy = clamp_i(y + dy, 0, height - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int cx = clamp_i(x + dx, 0, width - 1);
            float x_p =  (float)dx * cos_t + (float)dy * sin_t;
            float y_p = -(float)dx * sin_t + (float)dy * cos_t;
            float envelope = exp(- (x_p * x_p + gamma * gamma * y_p * y_p) / (2.0f * sigma_sq));
            float carrier = (2.0f * PI_F * x_p / lambda) + psi;
            float val = input[(size_t)cy * (size_t)width + (size_t)cx];
            acc.x += val * envelope * cos(carrier);
            acc.y += val * envelope * sin(carrier);
        }
    }
    output[(size_t)y * (size_t)width + (size_t)x] = acc;
}

// 2D Gabor Filter Bank: Orientation 2 (Theta 0.7854 rad)
__kernel void kernel_gabor_bank_orient_2(
    __global const float* input, __global float2* output,
    const int width, const int height, const float lambda,
    const float psi, const float sigma, const float gamma, const int radius) {
    int x = get_global_id(0); int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const float theta = 0.7853981633974483f;
    float cos_t = cos(theta); float sin_t = sin(theta);
    float2 acc = (float2)(0.0f, 0.0f); float sigma_sq = sigma * sigma;
    for (int dy = -radius; dy <= radius; ++dy) {
        int cy = clamp_i(y + dy, 0, height - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int cx = clamp_i(x + dx, 0, width - 1);
            float x_p =  (float)dx * cos_t + (float)dy * sin_t;
            float y_p = -(float)dx * sin_t + (float)dy * cos_t;
            float envelope = exp(- (x_p * x_p + gamma * gamma * y_p * y_p) / (2.0f * sigma_sq));
            float carrier = (2.0f * PI_F * x_p / lambda) + psi;
            float val = input[(size_t)cy * (size_t)width + (size_t)cx];
            acc.x += val * envelope * cos(carrier);
            acc.y += val * envelope * sin(carrier);
        }
    }
    output[(size_t)y * (size_t)width + (size_t)x] = acc;
}

// 2D Gabor Filter Bank: Orientation 3 (Theta 1.1781 rad)
__kernel void kernel_gabor_bank_orient_3(
    __global const float* input, __global float2* output,
    const int width, const int height, const float lambda,
    const float psi, const float sigma, const float gamma, const int radius) {
    int x = get_global_id(0); int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const float theta = 1.1780972450961724f;
    float cos_t = cos(theta); float sin_t = sin(theta);
    float2 acc = (float2)(0.0f, 0.0f); float sigma_sq = sigma * sigma;
    for (int dy = -radius; dy <= radius; ++dy) {
        int cy = clamp_i(y + dy, 0, height - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int cx = clamp_i(x + dx, 0, width - 1);
            float x_p =  (float)dx * cos_t + (float)dy * sin_t;
            float y_p = -(float)dx * sin_t + (float)dy * cos_t;
            float envelope = exp(- (x_p * x_p + gamma * gamma * y_p * y_p) / (2.0f * sigma_sq));
            float carrier = (2.0f * PI_F * x_p / lambda) + psi;
            float val = input[(size_t)cy * (size_t)width + (size_t)cx];
            acc.x += val * envelope * cos(carrier);
            acc.y += val * envelope * sin(carrier);
        }
    }
    output[(size_t)y * (size_t)width + (size_t)x] = acc;
}

// 2D Gabor Filter Bank: Orientation 4 (Theta 1.5708 rad)
__kernel void kernel_gabor_bank_orient_4(
    __global const float* input, __global float2* output,
    const int width, const int height, const float lambda,
    const float psi, const float sigma, const float gamma, const int radius) {
    int x = get_global_id(0); int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const float theta = 1.5707963267948966f;
    float cos_t = cos(theta); float sin_t = sin(theta);
    float2 acc = (float2)(0.0f, 0.0f); float sigma_sq = sigma * sigma;
    for (int dy = -radius; dy <= radius; ++dy) {
        int cy = clamp_i(y + dy, 0, height - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int cx = clamp_i(x + dx, 0, width - 1);
            float x_p =  (float)dx * cos_t + (float)dy * sin_t;
            float y_p = -(float)dx * sin_t + (float)dy * cos_t;
            float envelope = exp(- (x_p * x_p + gamma * gamma * y_p * y_p) / (2.0f * sigma_sq));
            float carrier = (2.0f * PI_F * x_p / lambda) + psi;
            float val = input[(size_t)cy * (size_t)width + (size_t)cx];
            acc.x += val * envelope * cos(carrier);
            acc.y += val * envelope * sin(carrier);
        }
    }
    output[(size_t)y * (size_t)width + (size_t)x] = acc;
}

// 2D Gabor Filter Bank: Orientation 5 (Theta 1.9635 rad)
__kernel void kernel_gabor_bank_orient_5(
    __global const float* input, __global float2* output,
    const int width, const int height, const float lambda,
    const float psi, const float sigma, const float gamma, const int radius) {
    int x = get_global_id(0); int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const float theta = 1.9634954084936207f;
    float cos_t = cos(theta); float sin_t = sin(theta);
    float2 acc = (float2)(0.0f, 0.0f); float sigma_sq = sigma * sigma;
    for (int dy = -radius; dy <= radius; ++dy) {
        int cy = clamp_i(y + dy, 0, height - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int cx = clamp_i(x + dx, 0, width - 1);
            float x_p =  (float)dx * cos_t + (float)dy * sin_t;
            float y_p = -(float)dx * sin_t + (float)dy * cos_t;
            float envelope = exp(- (x_p * x_p + gamma * gamma * y_p * y_p) / (2.0f * sigma_sq));
            float carrier = (2.0f * PI_F * x_p / lambda) + psi;
            float val = input[(size_t)cy * (size_t)width + (size_t)cx];
            acc.x += val * envelope * cos(carrier);
            acc.y += val * envelope * sin(carrier);
        }
    }
    output[(size_t)y * (size_t)width + (size_t)x] = acc;
}

// 2D Gabor Filter Bank: Orientation 6 (Theta 2.3562 rad)
__kernel void kernel_gabor_bank_orient_6(
    __global const float* input, __global float2* output,
    const int width, const int height, const float lambda,
    const float psi, const float sigma, const float gamma, const int radius) {
    int x = get_global_id(0); int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const float theta = 2.356194490192345f;
    float cos_t = cos(theta); float sin_t = sin(theta);
    float2 acc = (float2)(0.0f, 0.0f); float sigma_sq = sigma * sigma;
    for (int dy = -radius; dy <= radius; ++dy) {
        int cy = clamp_i(y + dy, 0, height - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int cx = clamp_i(x + dx, 0, width - 1);
            float x_p =  (float)dx * cos_t + (float)dy * sin_t;
            float y_p = -(float)dx * sin_t + (float)dy * cos_t;
            float envelope = exp(- (x_p * x_p + gamma * gamma * y_p * y_p) / (2.0f * sigma_sq));
            float carrier = (2.0f * PI_F * x_p / lambda) + psi;
            float val = input[(size_t)cy * (size_t)width + (size_t)cx];
            acc.x += val * envelope * cos(carrier);
            acc.y += val * envelope * sin(carrier);
        }
    }
    output[(size_t)y * (size_t)width + (size_t)x] = acc;
}

// 2D Gabor Filter Bank: Orientation 7 (Theta 2.7489 rad)
__kernel void kernel_gabor_bank_orient_7(
    __global const float* input, __global float2* output,
    const int width, const int height, const float lambda,
    const float psi, const float sigma, const float gamma, const int radius) {
    int x = get_global_id(0); int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const float theta = 2.748893571891069f;
    float cos_t = cos(theta); float sin_t = sin(theta);
    float2 acc = (float2)(0.0f, 0.0f); float sigma_sq = sigma * sigma;
    for (int dy = -radius; dy <= radius; ++dy) {
        int cy = clamp_i(y + dy, 0, height - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int cx = clamp_i(x + dx, 0, width - 1);
            float x_p =  (float)dx * cos_t + (float)dy * sin_t;
            float y_p = -(float)dx * sin_t + (float)dy * cos_t;
            float envelope = exp(- (x_p * x_p + gamma * gamma * y_p * y_p) / (2.0f * sigma_sq));
            float carrier = (2.0f * PI_F * x_p / lambda) + psi;
            float val = input[(size_t)cy * (size_t)width + (size_t)cx];
            acc.x += val * envelope * cos(carrier);
            acc.y += val * envelope * sin(carrier);
        }
    }
    output[(size_t)y * (size_t)width + (size_t)x] = acc;
}


// ============================================================================
// Extended Daubechies (db4, db6, db8) and Symlet Filter Banks
// ============================================================================

// 2D Wavelet Subband Decomposition (LL, LH, HL, HH)
__kernel void kernel_dwt2d_subband_pack(
    __global const float* low_low, __global const float* low_high,
    __global const float* high_low, __global const float* high_high,
    __global float* out_packed_dwt, const int half_w, const int half_h) {{
    int x = get_global_id(0); int y = get_global_id(1);
    if (x >= half_w || y >= half_h) return;
    int full_w = half_w * 2;
    size_t in_idx = (size_t)y * (size_t)half_w + (size_t)x;
    // Top-left: LL
    out_packed_dwt[(size_t)y * (size_t)full_w + (size_t)x] = low_low[in_idx];
    // Top-right: HL
    out_packed_dwt[(size_t)y * (size_t)full_w + (size_t)(half_w + x)] = high_low[in_idx];
    // Bottom-left: LH
    out_packed_dwt[(size_t)(half_h + y) * (size_t)full_w + (size_t)x] = low_high[in_idx];
    // Bottom-right: HH
    out_packed_dwt[(size_t)(half_h + y) * (size_t)full_w + (size_t)(half_w + x)] = high_high[in_idx];
}}
