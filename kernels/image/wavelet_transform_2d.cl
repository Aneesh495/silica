#include "common/math.clh"
#include "common/memory.clh"

// Daubechies D4 Wavelet Filter Coefficients
#define DAUB4_H0  0.4829629131445341f
#define DAUB4_H1  0.8365163037378079f
#define DAUB4_H2  0.2241438680420134f
#define DAUB4_H3 -0.1294095225512604f

#define DAUB4_G0 -0.1294095225512604f
#define DAUB4_G1 -0.2241438680420134f
#define DAUB4_G2  0.8365163037378079f
#define DAUB4_G3 -0.4829629131445341f

// 1D Row-wise Forward Daubechies D4 Wavelet Decomposition
__kernel void kernel_dwt_daub4_rows(
    __global const float* input,
    __global float* output,
    const int width,
    const int height) {
    
    int row = get_global_id(1);
    int col = get_global_id(0); // 0 .. width/2 - 1

    if (row >= height || col >= width / 2) return;

    size_t row_base = (size_t)row * (size_t)width;
    int half_w = width / 2;

    int i0 = (2 * col - 1 + width) % width;
    int i1 = (2 * col) % width;
    int i2 = (2 * col + 1) % width;
    int i3 = (2 * col + 2) % width;

    float x0 = input[row_base + i0];
    float x1 = input[row_base + i1];
    float x2 = input[row_base + i2];
    float x3 = input[row_base + i3];

    // Low-pass approximation (LL)
    float low = DAUB4_H0 * x1 + DAUB4_H1 * x2 + DAUB4_H2 * x3 + DAUB4_H3 * x0;
    // High-pass detail (LH)
    float high = DAUB4_G0 * x1 + DAUB4_G1 * x2 + DAUB4_G2 * x3 + DAUB4_G3 * x0;

    output[row_base + col] = low;
    output[row_base + half_w + col] = high;
}

// 1D Column-wise Forward Daubechies D4 Wavelet Decomposition
__kernel void kernel_dwt_daub4_cols(
    __global const float* input,
    __global float* output,
    const int width,
    const int height) {
    
    int col = get_global_id(0);
    int row = get_global_id(1); // 0 .. height/2 - 1

    if (col >= width || row >= height / 2) return;

    int half_h = height / 2;

    int i0 = (2 * row - 1 + height) % height;
    int i1 = (2 * row) % height;
    int i2 = (2 * row + 1) % height;
    int i3 = (2 * row + 2) % height;

    float x0 = input[(size_t)i0 * (size_t)width + (size_t)col];
    float x1 = input[(size_t)i1 * (size_t)width + (size_t)col];
    float x2 = input[(size_t)i2 * (size_t)width + (size_t)col];
    float x3 = input[(size_t)i3 * (size_t)width + (size_t)col];

    float low = DAUB4_H0 * x1 + DAUB4_H1 * x2 + DAUB4_H2 * x3 + DAUB4_H3 * x0;
    float high = DAUB4_G0 * x1 + DAUB4_G1 * x2 + DAUB4_G2 * x3 + DAUB4_G3 * x0;

    output[(size_t)row * (size_t)width + (size_t)col] = low;
    output[(size_t)(half_h + row) * (size_t)width + (size_t)col] = high;
}
