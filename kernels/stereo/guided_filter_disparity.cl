#include "common/math.clh"
#include "common/memory.clh"

// Guided Filter Stage 1: Computes Linear Coefficients (a, b) from Guidance I and Input p
// a_k = (cov_{Ip} + eps) / (var_I + eps),  b_k = mean_p - a_k * mean_I
__kernel void kernel_guided_filter_coeff(
    __global const float* mean_I,
    __global const float* mean_p,
    __global const float* mean_Ip,
    __global const float* var_I,
    __global float* out_a,
    __global float* out_b,
    const int width,
    const int height,
    const float eps) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t idx = (size_t)y * (size_t)width + (size_t)x;

    float mI = mean_I[idx];
    float mp = mean_p[idx];
    float mIp = mean_Ip[idx];
    float vI = var_I[idx];

    float cov_Ip = mIp - mI * mp;
    float a = cov_Ip / (vI + eps);
    float b = mp - a * mI;

    out_a[idx] = a;
    out_b[idx] = b;
}

// Guided Filter Stage 2: Evaluates Filtered Output q_i = mean_a * I_i + mean_b
__kernel void kernel_guided_filter_output(
    __global const float* guidance_I,
    __global const float* mean_a,
    __global const float* mean_b,
    __global float* out_q,
    const int width,
    const int height) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t idx = (size_t)y * (size_t)width + (size_t)x;

    float I = guidance_I[idx];
    float a = mean_a[idx];
    float b = mean_b[idx];

    out_q[idx] = a * I + b;
}
