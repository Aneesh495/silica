#include "common/math.clh"
#include "common/memory.clh"

#define REDUCE_WG_SIZE 256

// Multi-stage Parallel Reduction for Sum (Float)
// Performs register-level grid-stride accumulation followed by workgroup local tree reduction.
__kernel void kernel_reduce_sum_float(
    __global const float* input,
    __global float* block_results,
    const uint total_elements) {
    
    __local float l_mem[REDUCE_WG_SIZE];

    uint tid = get_local_id(0);
    uint gid = get_global_id(0);
    uint grid_stride = get_global_size(0);

    // Register grid-stride accumulation
    float accum = 0.0f;
    for (uint i = gid; i < total_elements; i += grid_stride) {
        accum += input[i];
    }
    l_mem[tid] = accum;
    sync_local();

    // In-place local memory reduction tree (sequential addressing)
    for (uint s = REDUCE_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            l_mem[tid] += l_mem[tid + s];
        }
        sync_local();
    }

    if (tid == 0) {
        block_results[get_group_id(0)] = l_mem[0];
    }
}

// Multi-stage Parallel Reduction for Minimum (Float)
__kernel void kernel_reduce_min_float(
    __global const float* input,
    __global float* block_results,
    const uint total_elements) {
    
    __local float l_mem[REDUCE_WG_SIZE];

    uint tid = get_local_id(0);
    uint gid = get_global_id(0);
    uint grid_stride = get_global_size(0);

    float val_min = LARGE_FLOAT_F;
    for (uint i = gid; i < total_elements; i += grid_stride) {
        val_min = fmin(val_min, input[i]);
    }
    l_mem[tid] = val_min;
    sync_local();

    for (uint s = REDUCE_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            l_mem[tid] = fmin(l_mem[tid], l_mem[tid + s]);
        }
        sync_local();
    }

    if (tid == 0) {
        block_results[get_group_id(0)] = l_mem[0];
    }
}

// Multi-stage Parallel Reduction for Maximum (Float)
__kernel void kernel_reduce_max_float(
    __global const float* input,
    __global float* block_results,
    const uint total_elements) {
    
    __local float l_mem[REDUCE_WG_SIZE];

    uint tid = get_local_id(0);
    uint gid = get_global_id(0);
    uint grid_stride = get_global_size(0);

    float val_max = -LARGE_FLOAT_F;
    for (uint i = gid; i < total_elements; i += grid_stride) {
        val_max = fmax(val_max, input[i]);
    }
    l_mem[tid] = val_max;
    sync_local();

    for (uint s = REDUCE_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            l_mem[tid] = fmax(l_mem[tid], l_mem[tid + s]);
        }
        sync_local();
    }

    if (tid == 0) {
        block_results[get_group_id(0)] = l_mem[0];
    }
}

// Multi-stage Parallel Reduction for Simultaneous Min and Max (float2)
__kernel void kernel_reduce_minmax_float(
    __global const float* input,
    __global float2* block_results,
    const uint total_elements) {
    
    __local float2 l_mem[REDUCE_WG_SIZE];

    uint tid = get_local_id(0);
    uint gid = get_global_id(0);
    uint grid_stride = get_global_size(0);

    float2 minmax = (float2)(LARGE_FLOAT_F, -LARGE_FLOAT_F);
    for (uint i = gid; i < total_elements; i += grid_stride) {
        float val = input[i];
        minmax.x = fmin(minmax.x, val);
        minmax.y = fmax(minmax.y, val);
    }
    l_mem[tid] = minmax;
    sync_local();

    for (uint s = REDUCE_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float2 other = l_mem[tid + s];
            l_mem[tid].x = fmin(l_mem[tid].x, other.x);
            l_mem[tid].y = fmax(l_mem[tid].y, other.y);
        }
        sync_local();
    }

    if (tid == 0) {
        block_results[get_group_id(0)] = l_mem[0];
    }
}

// Multi-stage Parallel Reduction for ArgMin: returns (min_value, index)
__kernel void kernel_reduce_argmin_float(
    __global const float* input,
    __global float* out_val,
    __global uint* out_idx,
    const uint total_elements) {
    
    __local float l_val[REDUCE_WG_SIZE];
    __local uint  l_idx[REDUCE_WG_SIZE];

    uint tid = get_local_id(0);
    uint gid = get_global_id(0);
    uint grid_stride = get_global_size(0);

    float min_v = LARGE_FLOAT_F;
    uint  min_i = 0;

    for (uint i = gid; i < total_elements; i += grid_stride) {
        float val = input[i];
        if (val < min_v) {
            min_v = val;
            min_i = i;
        }
    }
    l_val[tid] = min_v;
    l_idx[tid] = min_i;
    sync_local();

    for (uint s = REDUCE_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (l_val[tid + s] < l_val[tid]) {
                l_val[tid] = l_val[tid + s];
                l_idx[tid] = l_idx[tid + s];
            }
        }
        sync_local();
    }

    if (tid == 0) {
        uint gid_out = get_group_id(0);
        out_val[gid_out] = l_val[0];
        out_idx[gid_out] = l_idx[0];
    }
}
