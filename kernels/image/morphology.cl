#include "common/math.clh"
#include "common/memory.clh"

#define MORPH_RADIUS 2
#define MORPH_WG_X 16
#define MORPH_WG_Y 16

// Mathematical Morphology: Grayscale Dilation (Maximum in neighborhood)
__kernel void kernel_morphology_dilation_gray(
    __global const float* input,
    __global float* output,
    const int width,
    const int height,
    const int radius) {
    
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    if (gx >= width || gy >= height) return;

    float max_val = -LARGE_FLOAT_F;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            float val = read_image_2d_safe(input, gx + dx, gy + dy, width, height, BORDER_CLAMP_EDGE, -LARGE_FLOAT_F);
            max_val = fmax(max_val, val);
        }
    }

    output[(size_t)gy * (size_t)width + (size_t)gx] = max_val;
}

// Mathematical Morphology: Grayscale Erosion (Minimum in neighborhood)
__kernel void kernel_morphology_erosion_gray(
    __global const float* input,
    __global float* output,
    const int width,
    const int height,
    const int radius) {
    
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    if (gx >= width || gy >= height) return;

    float min_val = LARGE_FLOAT_F;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            float val = read_image_2d_safe(input, gx + dx, gy + dy, width, height, BORDER_CLAMP_EDGE, LARGE_FLOAT_F);
            min_val = fmin(min_val, val);
        }
    }

    output[(size_t)gy * (size_t)width + (size_t)gx] = min_val;
}

// Morphological Gradient: Output = Dilation - Erosion
__kernel void kernel_morphology_gradient(
    __global const float* in_dilated,
    __global const float* in_eroded,
    __global float* out_gradient,
    const int width,
    const int height) {
    
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    if (gx >= width || gy >= height) return;

    size_t idx = (size_t)gy * (size_t)width + (size_t)gx;
    out_gradient[idx] = in_dilated[idx] - in_eroded[idx];
}
