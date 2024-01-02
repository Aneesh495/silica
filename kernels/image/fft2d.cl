#include "common/math.clh"
#include "common/memory.clh"

// Complex multiplication: (a.x + i*a.y) * (b.x + i*b.y)
inline float2 complex_mul(float2 a, float2 b) {
    return (float2)(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

// Bit-reversal permutation index for N power of 2
inline uint bit_reverse(uint val, uint num_bits) {
    uint reversed = 0;
    for (uint i = 0; i < num_bits; ++i) {
        reversed = (reversed << 1) | (val & 1);
        val >>= 1;
    }
    return reversed;
}

// 1D Row-wise FFT Pass: Transforms each row of width N
__kernel void kernel_fft_rows(
    __global const float2* input,
    __global float2* output,
    const uint N,
    const uint log2_N,
    const int direction) { // -1 for forward FFT, +1 for inverse FFT
    
    uint row = get_global_id(1);
    uint tid = get_global_id(0);

    if (tid >= N) return;

    size_t row_base = (size_t)row * (size_t)N;

    // Bit-reversal permutation load
    uint rev_idx = bit_reverse(tid, log2_N);
    output[row_base + tid] = input[row_base + rev_idx];
}

// Butterfly stages for Row-wise FFT
__kernel void kernel_fft_rows_stage(
    __global float2* data,
    const uint N,
    const uint stage, // stage = 1 .. log2_N
    const int direction) {
    
    uint row = get_global_id(1);
    uint tid = get_global_id(0); // 0 .. N/2 - 1

    if (tid >= N / 2) return;

    size_t row_base = (size_t)row * (size_t)N;

    uint len = 1 << stage;
    uint half_len = len >> 1;

    uint block_id = tid / half_len;
    uint butterfly_idx = tid % half_len;

    uint idx0 = block_id * len + butterfly_idx;
    uint idx1 = idx0 + half_len;

    float angle = (float)direction * TWO_PI_F * (float)butterfly_idx / (float)len;
    float2 twiddle = (float2)(cos(angle), sin(angle));

    float2 u = data[row_base + idx0];
    float2 v = complex_mul(data[row_base + idx1], twiddle);

    data[row_base + idx0] = u + v;
    data[row_base + idx1] = u - v;
}

// 1D Column-wise FFT Pass: Transforms each column of height M
__kernel void kernel_fft_cols(
    __global const float2* input,
    __global float2* output,
    const uint width,
    const uint height,
    const uint log2_H,
    const int direction) {
    
    uint col = get_global_id(0);
    uint tid = get_global_id(1);

    if (col >= width || tid >= height) return;

    uint rev_idx = bit_reverse(tid, log2_H);
    output[(size_t)tid * (size_t)width + (size_t)col] = input[(size_t)rev_idx * (size_t)width + (size_t)col];
}

// Butterfly stages for Column-wise FFT
__kernel void kernel_fft_cols_stage(
    __global float2* data,
    const uint width,
    const uint height,
    const uint stage,
    const int direction) {
    
    uint col = get_global_id(0);
    uint tid = get_global_id(1); // 0 .. height/2 - 1

    if (col >= width || tid >= height / 2) return;

    uint len = 1 << stage;
    uint half_len = len >> 1;

    uint block_id = tid / half_len;
    uint butterfly_idx = tid % half_len;

    uint row0 = block_id * len + butterfly_idx;
    uint row1 = row0 + half_len;

    float angle = (float)direction * TWO_PI_F * (float)butterfly_idx / (float)len;
    float2 twiddle = (float2)(cos(angle), sin(angle));

    size_t idx0 = (size_t)row0 * (size_t)width + (size_t)col;
    size_t idx1 = (size_t)row1 * (size_t)width + (size_t)col;

    float2 u = data[idx0];
    float2 v = complex_mul(data[idx1], twiddle);

    data[idx0] = u + v;
    data[idx1] = u - v;
}

// Frequency spectrum shift: swaps quadrants 1-3 and 2-4
__kernel void kernel_fftshift_2d(
    __global const float2* input,
    __global float2* output,
    const int width,
    const int height) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int half_w = width / 2;
    int half_h = height / 2;

    int shifted_x = (x + half_w) % width;
    int shifted_y = (y + half_h) % height;

    size_t in_idx = (size_t)y * (size_t)width + (size_t)x;
    size_t out_idx = (size_t)shifted_y * (size_t)width + (size_t)shifted_x;

    output[out_idx] = input[in_idx];
}
