#include "common/math.clh"
#include "common/memory.clh"

// Pass 1: Horizontal row prefix sum for integral image
// Each thread processes one row sequentially or in chunks
__kernel void kernel_integral_rows(
    __global const float* input,
    __global float* row_integrals,
    const int width,
    const int height) {
    
    int row = get_global_id(0);
    if (row >= height) return;

    size_t row_offset = (size_t)row * (size_t)width;
    float sum = 0.0f;

    for (int col = 0; col < width; ++col) {
        sum += input[row_offset + col];
        row_integrals[row_offset + col] = sum;
    }
}

// Pass 2: Vertical column prefix sum from row integrals
// Each thread processes one column sequentially
__kernel void kernel_integral_columns(
    __global const float* row_integrals,
    __global float* output_sat,
    const int width,
    const int height) {
    
    int col = get_global_id(0);
    if (col >= width) return;

    float sum = 0.0f;

    for (int row = 0; row < height; ++row) {
        size_t idx = (size_t)row * (size_t)width + (size_t)col;
        sum += row_integrals[idx];
        output_sat[idx] = sum;
    }
}

// O(1) Constant-Time Box Filter via Summed-Area Table Query
__kernel void kernel_box_filter_sat(
    __global const float* sat,
    __global float* output,
    const int width,
    const int height,
    const int radius) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int x1 = max(0, x - radius);
    int y1 = max(0, y - radius);
    int x2 = min(width - 1, x + radius);
    int y2 = min(height - 1, y + radius);

    float d = sat[(size_t)y2 * (size_t)width + (size_t)x2];
    float b = (y1 > 0) ? sat[(size_t)(y1 - 1) * (size_t)width + (size_t)x2] : 0.0f;
    float c = (x1 > 0) ? sat[(size_t)y2 * (size_t)width + (size_t)(x1 - 1)] : 0.0f;
    float a = (x1 > 0 && y1 > 0) ? sat[(size_t)(y1 - 1) * (size_t)width + (size_t)(x1 - 1)] : 0.0f;

    float sum = d - b - c + a;
    float area = (float)((x2 - x1 + 1) * (y2 - y1 + 1));

    output[(size_t)y * (size_t)width + (size_t)x] = sum / area;
}
