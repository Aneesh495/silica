#include "common/math.clh"
#include "common/memory.clh"

// ============================================================================
// Vectorized Elementwise Arithmetic, Transcendental, and Activation Kernels
// ============================================================================

// Operations for float
__kernel void kernel_vec_add_float(__global const float* a, __global const float* b, __global float* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] + b[gid];
}

__kernel void kernel_vec_sub_float(__global const float* a, __global const float* b, __global float* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] - b[gid];
}

__kernel void kernel_vec_mul_float(__global const float* a, __global const float* b, __global float* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] * b[gid];
}

__kernel void kernel_vec_div_float(__global const float* a, __global const float* b, __global float* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] / b[gid];
}

__kernel void kernel_vec_exp_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = exp(a[gid]);
}

__kernel void kernel_vec_exp2_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = exp2(a[gid]);
}

__kernel void kernel_vec_log_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log(a[gid]);
}

__kernel void kernel_vec_log2_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log2(a[gid]);
}

__kernel void kernel_vec_sqrt_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sqrt(a[gid]);
}

__kernel void kernel_vec_rsqrt_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = rsqrt(a[gid]);
}

__kernel void kernel_vec_sin_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sin(a[gid]);
}

__kernel void kernel_vec_cos_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = cos(a[gid]);
}

__kernel void kernel_vec_tan_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = tan(a[gid]);
}

__kernel void kernel_vec_sinh_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sinh(a[gid]);
}

__kernel void kernel_vec_cosh_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = cosh(a[gid]);
}

__kernel void kernel_vec_tanh_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = tanh(a[gid]);
}

__kernel void kernel_vec_abs_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = fabs(a[gid]);
}

__kernel void kernel_vec_sigmoid_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = (float)(1.0f) / ((float)(1.0f) + exp(-a[gid]));
}

__kernel void kernel_vec_relu_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = max(a[gid], (float)(0.0f));
}

__kernel void kernel_vec_leaky_relu_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = select(a[gid] * (float)(0.01f), a[gid], a[gid] > (float)(0.0f));
}

__kernel void kernel_vec_gelu_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = (float)(0.5f) * a[gid] * ((float)(1.0f) + tanh((float)(0.79788456f) * (a[gid] + (float)(0.044715f) * a[gid] * a[gid] * a[gid])));
}

__kernel void kernel_vec_elu_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = select((float)(1.0f) * (exp(a[gid]) - (float)(1.0f)), a[gid], a[gid] > (float)(0.0f));
}

__kernel void kernel_vec_softplus_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log((float)(1.0f) + exp(a[gid]));
}

__kernel void kernel_vec_swish_float(__global const float* a, __global float* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = a[gid] / ((float)(1.0f) + exp(-a[gid]));
}

__kernel void kernel_vec_fma_float(__global const float* a, __global const float* b, __global const float* c, __global float* d, const uint n) {
    uint gid = get_global_id(0); if (gid < n) d[gid] = fma(a[gid], b[gid], c[gid]);
}

__kernel void kernel_vec_clamp_float(__global const float* a, __global float* b, const float min_val, const float max_val, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = clamp(a[gid], (float)(min_val), (float)(max_val));
}

// Operations for float2
__kernel void kernel_vec_add_float2(__global const float2* a, __global const float2* b, __global float2* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] + b[gid];
}

__kernel void kernel_vec_sub_float2(__global const float2* a, __global const float2* b, __global float2* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] - b[gid];
}

__kernel void kernel_vec_mul_float2(__global const float2* a, __global const float2* b, __global float2* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] * b[gid];
}

__kernel void kernel_vec_div_float2(__global const float2* a, __global const float2* b, __global float2* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] / b[gid];
}

__kernel void kernel_vec_exp_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = exp(a[gid]);
}

__kernel void kernel_vec_exp2_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = exp2(a[gid]);
}

__kernel void kernel_vec_log_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log(a[gid]);
}

__kernel void kernel_vec_log2_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log2(a[gid]);
}

__kernel void kernel_vec_sqrt_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sqrt(a[gid]);
}

__kernel void kernel_vec_rsqrt_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = rsqrt(a[gid]);
}

__kernel void kernel_vec_sin_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sin(a[gid]);
}

__kernel void kernel_vec_cos_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = cos(a[gid]);
}

__kernel void kernel_vec_tan_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = tan(a[gid]);
}

__kernel void kernel_vec_sinh_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sinh(a[gid]);
}

__kernel void kernel_vec_cosh_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = cosh(a[gid]);
}

__kernel void kernel_vec_tanh_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = tanh(a[gid]);
}

__kernel void kernel_vec_abs_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = fabs(a[gid]);
}

__kernel void kernel_vec_sigmoid_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = (float2)(1.0f) / ((float2)(1.0f) + exp(-a[gid]));
}

__kernel void kernel_vec_relu_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = max(a[gid], (float2)(0.0f));
}

__kernel void kernel_vec_leaky_relu_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = select(a[gid] * (float2)(0.01f), a[gid], a[gid] > (float2)(0.0f));
}

__kernel void kernel_vec_gelu_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = (float2)(0.5f) * a[gid] * ((float2)(1.0f) + tanh((float2)(0.79788456f) * (a[gid] + (float2)(0.044715f) * a[gid] * a[gid] * a[gid])));
}

__kernel void kernel_vec_elu_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = select((float2)(1.0f) * (exp(a[gid]) - (float2)(1.0f)), a[gid], a[gid] > (float2)(0.0f));
}

__kernel void kernel_vec_softplus_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log((float2)(1.0f) + exp(a[gid]));
}

__kernel void kernel_vec_swish_float2(__global const float2* a, __global float2* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = a[gid] / ((float2)(1.0f) + exp(-a[gid]));
}

__kernel void kernel_vec_fma_float2(__global const float2* a, __global const float2* b, __global const float2* c, __global float2* d, const uint n) {
    uint gid = get_global_id(0); if (gid < n) d[gid] = fma(a[gid], b[gid], c[gid]);
}

__kernel void kernel_vec_clamp_float2(__global const float2* a, __global float2* b, const float min_val, const float max_val, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = clamp(a[gid], (float2)(min_val), (float2)(max_val));
}

// Operations for float3
__kernel void kernel_vec_add_float3(__global const float3* a, __global const float3* b, __global float3* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] + b[gid];
}

__kernel void kernel_vec_sub_float3(__global const float3* a, __global const float3* b, __global float3* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] - b[gid];
}

__kernel void kernel_vec_mul_float3(__global const float3* a, __global const float3* b, __global float3* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] * b[gid];
}

__kernel void kernel_vec_div_float3(__global const float3* a, __global const float3* b, __global float3* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] / b[gid];
}

__kernel void kernel_vec_exp_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = exp(a[gid]);
}

__kernel void kernel_vec_exp2_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = exp2(a[gid]);
}

__kernel void kernel_vec_log_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log(a[gid]);
}

__kernel void kernel_vec_log2_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log2(a[gid]);
}

__kernel void kernel_vec_sqrt_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sqrt(a[gid]);
}

__kernel void kernel_vec_rsqrt_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = rsqrt(a[gid]);
}

__kernel void kernel_vec_sin_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sin(a[gid]);
}

__kernel void kernel_vec_cos_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = cos(a[gid]);
}

__kernel void kernel_vec_tan_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = tan(a[gid]);
}

__kernel void kernel_vec_sinh_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sinh(a[gid]);
}

__kernel void kernel_vec_cosh_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = cosh(a[gid]);
}

__kernel void kernel_vec_tanh_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = tanh(a[gid]);
}

__kernel void kernel_vec_abs_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = fabs(a[gid]);
}

__kernel void kernel_vec_sigmoid_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = (float3)(1.0f) / ((float3)(1.0f) + exp(-a[gid]));
}

__kernel void kernel_vec_relu_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = max(a[gid], (float3)(0.0f));
}

__kernel void kernel_vec_leaky_relu_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = select(a[gid] * (float3)(0.01f), a[gid], a[gid] > (float3)(0.0f));
}

__kernel void kernel_vec_gelu_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = (float3)(0.5f) * a[gid] * ((float3)(1.0f) + tanh((float3)(0.79788456f) * (a[gid] + (float3)(0.044715f) * a[gid] * a[gid] * a[gid])));
}

__kernel void kernel_vec_elu_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = select((float3)(1.0f) * (exp(a[gid]) - (float3)(1.0f)), a[gid], a[gid] > (float3)(0.0f));
}

__kernel void kernel_vec_softplus_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log((float3)(1.0f) + exp(a[gid]));
}

__kernel void kernel_vec_swish_float3(__global const float3* a, __global float3* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = a[gid] / ((float3)(1.0f) + exp(-a[gid]));
}

__kernel void kernel_vec_fma_float3(__global const float3* a, __global const float3* b, __global const float3* c, __global float3* d, const uint n) {
    uint gid = get_global_id(0); if (gid < n) d[gid] = fma(a[gid], b[gid], c[gid]);
}

__kernel void kernel_vec_clamp_float3(__global const float3* a, __global float3* b, const float min_val, const float max_val, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = clamp(a[gid], (float3)(min_val), (float3)(max_val));
}

// Operations for float4
__kernel void kernel_vec_add_float4(__global const float4* a, __global const float4* b, __global float4* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] + b[gid];
}

__kernel void kernel_vec_sub_float4(__global const float4* a, __global const float4* b, __global float4* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] - b[gid];
}

__kernel void kernel_vec_mul_float4(__global const float4* a, __global const float4* b, __global float4* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] * b[gid];
}

__kernel void kernel_vec_div_float4(__global const float4* a, __global const float4* b, __global float4* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] / b[gid];
}

__kernel void kernel_vec_exp_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = exp(a[gid]);
}

__kernel void kernel_vec_exp2_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = exp2(a[gid]);
}

__kernel void kernel_vec_log_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log(a[gid]);
}

__kernel void kernel_vec_log2_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log2(a[gid]);
}

__kernel void kernel_vec_sqrt_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sqrt(a[gid]);
}

__kernel void kernel_vec_rsqrt_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = rsqrt(a[gid]);
}

__kernel void kernel_vec_sin_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sin(a[gid]);
}

__kernel void kernel_vec_cos_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = cos(a[gid]);
}

__kernel void kernel_vec_tan_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = tan(a[gid]);
}

__kernel void kernel_vec_sinh_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sinh(a[gid]);
}

__kernel void kernel_vec_cosh_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = cosh(a[gid]);
}

__kernel void kernel_vec_tanh_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = tanh(a[gid]);
}

__kernel void kernel_vec_abs_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = fabs(a[gid]);
}

__kernel void kernel_vec_sigmoid_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = (float4)(1.0f) / ((float4)(1.0f) + exp(-a[gid]));
}

__kernel void kernel_vec_relu_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = max(a[gid], (float4)(0.0f));
}

__kernel void kernel_vec_leaky_relu_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = select(a[gid] * (float4)(0.01f), a[gid], a[gid] > (float4)(0.0f));
}

__kernel void kernel_vec_gelu_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = (float4)(0.5f) * a[gid] * ((float4)(1.0f) + tanh((float4)(0.79788456f) * (a[gid] + (float4)(0.044715f) * a[gid] * a[gid] * a[gid])));
}

__kernel void kernel_vec_elu_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = select((float4)(1.0f) * (exp(a[gid]) - (float4)(1.0f)), a[gid], a[gid] > (float4)(0.0f));
}

__kernel void kernel_vec_softplus_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log((float4)(1.0f) + exp(a[gid]));
}

__kernel void kernel_vec_swish_float4(__global const float4* a, __global float4* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = a[gid] / ((float4)(1.0f) + exp(-a[gid]));
}

__kernel void kernel_vec_fma_float4(__global const float4* a, __global const float4* b, __global const float4* c, __global float4* d, const uint n) {
    uint gid = get_global_id(0); if (gid < n) d[gid] = fma(a[gid], b[gid], c[gid]);
}

__kernel void kernel_vec_clamp_float4(__global const float4* a, __global float4* b, const float min_val, const float max_val, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = clamp(a[gid], (float4)(min_val), (float4)(max_val));
}

// Operations for float8
__kernel void kernel_vec_add_float8(__global const float8* a, __global const float8* b, __global float8* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] + b[gid];
}

__kernel void kernel_vec_sub_float8(__global const float8* a, __global const float8* b, __global float8* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] - b[gid];
}

__kernel void kernel_vec_mul_float8(__global const float8* a, __global const float8* b, __global float8* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] * b[gid];
}

__kernel void kernel_vec_div_float8(__global const float8* a, __global const float8* b, __global float8* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] / b[gid];
}

__kernel void kernel_vec_exp_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = exp(a[gid]);
}

__kernel void kernel_vec_exp2_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = exp2(a[gid]);
}

__kernel void kernel_vec_log_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log(a[gid]);
}

__kernel void kernel_vec_log2_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log2(a[gid]);
}

__kernel void kernel_vec_sqrt_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sqrt(a[gid]);
}

__kernel void kernel_vec_rsqrt_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = rsqrt(a[gid]);
}

__kernel void kernel_vec_sin_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sin(a[gid]);
}

__kernel void kernel_vec_cos_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = cos(a[gid]);
}

__kernel void kernel_vec_tan_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = tan(a[gid]);
}

__kernel void kernel_vec_sinh_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sinh(a[gid]);
}

__kernel void kernel_vec_cosh_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = cosh(a[gid]);
}

__kernel void kernel_vec_tanh_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = tanh(a[gid]);
}

__kernel void kernel_vec_abs_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = fabs(a[gid]);
}

__kernel void kernel_vec_sigmoid_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = (float8)(1.0f) / ((float8)(1.0f) + exp(-a[gid]));
}

__kernel void kernel_vec_relu_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = max(a[gid], (float8)(0.0f));
}

__kernel void kernel_vec_leaky_relu_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = select(a[gid] * (float8)(0.01f), a[gid], a[gid] > (float8)(0.0f));
}

__kernel void kernel_vec_gelu_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = (float8)(0.5f) * a[gid] * ((float8)(1.0f) + tanh((float8)(0.79788456f) * (a[gid] + (float8)(0.044715f) * a[gid] * a[gid] * a[gid])));
}

__kernel void kernel_vec_elu_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = select((float8)(1.0f) * (exp(a[gid]) - (float8)(1.0f)), a[gid], a[gid] > (float8)(0.0f));
}

__kernel void kernel_vec_softplus_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log((float8)(1.0f) + exp(a[gid]));
}

__kernel void kernel_vec_swish_float8(__global const float8* a, __global float8* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = a[gid] / ((float8)(1.0f) + exp(-a[gid]));
}

__kernel void kernel_vec_fma_float8(__global const float8* a, __global const float8* b, __global const float8* c, __global float8* d, const uint n) {
    uint gid = get_global_id(0); if (gid < n) d[gid] = fma(a[gid], b[gid], c[gid]);
}

__kernel void kernel_vec_clamp_float8(__global const float8* a, __global float8* b, const float min_val, const float max_val, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = clamp(a[gid], (float8)(min_val), (float8)(max_val));
}

// Operations for float16
__kernel void kernel_vec_add_float16(__global const float16* a, __global const float16* b, __global float16* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] + b[gid];
}

__kernel void kernel_vec_sub_float16(__global const float16* a, __global const float16* b, __global float16* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] - b[gid];
}

__kernel void kernel_vec_mul_float16(__global const float16* a, __global const float16* b, __global float16* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] * b[gid];
}

__kernel void kernel_vec_div_float16(__global const float16* a, __global const float16* b, __global float16* c, const uint n) {
    uint gid = get_global_id(0); if (gid < n) c[gid] = a[gid] / b[gid];
}

__kernel void kernel_vec_exp_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = exp(a[gid]);
}

__kernel void kernel_vec_exp2_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = exp2(a[gid]);
}

__kernel void kernel_vec_log_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log(a[gid]);
}

__kernel void kernel_vec_log2_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log2(a[gid]);
}

__kernel void kernel_vec_sqrt_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sqrt(a[gid]);
}

__kernel void kernel_vec_rsqrt_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = rsqrt(a[gid]);
}

__kernel void kernel_vec_sin_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sin(a[gid]);
}

__kernel void kernel_vec_cos_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = cos(a[gid]);
}

__kernel void kernel_vec_tan_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = tan(a[gid]);
}

__kernel void kernel_vec_sinh_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = sinh(a[gid]);
}

__kernel void kernel_vec_cosh_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = cosh(a[gid]);
}

__kernel void kernel_vec_tanh_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = tanh(a[gid]);
}

__kernel void kernel_vec_abs_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = fabs(a[gid]);
}

__kernel void kernel_vec_sigmoid_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = (float16)(1.0f) / ((float16)(1.0f) + exp(-a[gid]));
}

__kernel void kernel_vec_relu_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = max(a[gid], (float16)(0.0f));
}

__kernel void kernel_vec_leaky_relu_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = select(a[gid] * (float16)(0.01f), a[gid], a[gid] > (float16)(0.0f));
}

__kernel void kernel_vec_gelu_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = (float16)(0.5f) * a[gid] * ((float16)(1.0f) + tanh((float16)(0.79788456f) * (a[gid] + (float16)(0.044715f) * a[gid] * a[gid] * a[gid])));
}

__kernel void kernel_vec_elu_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = select((float16)(1.0f) * (exp(a[gid]) - (float16)(1.0f)), a[gid], a[gid] > (float16)(0.0f));
}

__kernel void kernel_vec_softplus_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = log((float16)(1.0f) + exp(a[gid]));
}

__kernel void kernel_vec_swish_float16(__global const float16* a, __global float16* b, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = a[gid] / ((float16)(1.0f) + exp(-a[gid]));
}

__kernel void kernel_vec_fma_float16(__global const float16* a, __global const float16* b, __global const float16* c, __global float16* d, const uint n) {
    uint gid = get_global_id(0); if (gid < n) d[gid] = fma(a[gid], b[gid], c[gid]);
}

__kernel void kernel_vec_clamp_float16(__global const float16* a, __global float16* b, const float min_val, const float max_val, const uint n) {
    uint gid = get_global_id(0); if (gid < n) b[gid] = clamp(a[gid], (float16)(min_val), (float16)(max_val));
}
