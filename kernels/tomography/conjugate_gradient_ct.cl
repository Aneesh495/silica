#include "common/math.clh"
#include "common/memory.clh"

// CGLS Step: Vector Scale and Add: y = y + alpha * x
__kernel void kernel_cgls_axpy(
    __global float* y,
    __global const float* x,
    const float alpha,
    const uint total_elements) {
    
    uint gid = get_global_id(0);
    if (gid >= total_elements) return;

    y[gid] += alpha * x[gid];
}

// CGLS Step: Direction Update: p = s + beta * p
__kernel void kernel_cgls_update_direction(
    __global float* p,
    __global const float* s,
    const float beta,
    const uint total_elements) {
    
    uint gid = get_global_id(0);
    if (gid >= total_elements) return;

    p[gid] = s[gid] + beta * p[gid];
}

// CGLS Step: Vector Inner Product Element-wise: out[i] = a[i] * b[i]
__kernel void kernel_cgls_elementwise_product(
    __global const float* a,
    __global const float* b,
    __global float* out_prod,
    const uint total_elements) {
    
    uint gid = get_global_id(0);
    if (gid >= total_elements) return;

    out_prod[gid] = a[gid] * b[gid];
}
