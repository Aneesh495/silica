#include "common/math.clh"
#include "common/memory.clh"

#define FILTER_WG_X 16
#define FILTER_WG_Y 16
#define MAX_KERNEL_RADIUS 16

// Separable Horizontal Gaussian Convolution Pass
// Caches row tile with halo margins in local LDS memory.
__kernel void kernel_gaussian_blur_horizontal(
    __global const float* input,
    __global float* output,
    __constant const float* kernel_weights,
    const int width,
    const int height,
    const int radius) {
    
    // Local tile size: (16 + 2 * radius) x 16
    __local float l_tile[FILTER_WG_Y][FILTER_WG_X + 2 * MAX_KERNEL_RADIUS + 1];

    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    int halo = radius;
    int tile_width = FILTER_WG_X + 2 * halo;

    // Load center pixels
    int src_x = gx;
    int src_y = gy;
    float center_val = read_image_2d_safe(input, src_x, src_y, width, height, BORDER_CLAMP_EDGE, 0.0f);
    l_tile[ty][halo + tx] = center_val;

    // Load left halo
    if (tx < halo) {
        int left_x = gx - halo;
        l_tile[ty][tx] = read_image_2d_safe(input, left_x, src_y, width, height, BORDER_CLAMP_EDGE, 0.0f);
    }

    // Load right halo
    if (tx < halo) {
        int right_x = gx + FILTER_WG_X;
        l_tile[ty][halo + FILTER_WG_X + tx] = read_image_2d_safe(input, right_x, src_y, width, height, BORDER_CLAMP_EDGE, 0.0f);
    }

    sync_local();

    if (gx < width && gy < height) {
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            float weight = kernel_weights[k + radius];
            sum += l_tile[ty][halo + tx + k] * weight;
        }
        output[gy * width + gx] = sum;
    }
}

// Separable Vertical Gaussian Convolution Pass
__kernel void kernel_gaussian_blur_vertical(
    __global const float* input,
    __global float* output,
    __constant const float* kernel_weights,
    const int width,
    const int height,
    const int radius) {
    
    __local float l_tile[FILTER_WG_Y + 2 * MAX_KERNEL_RADIUS + 1][FILTER_WG_X + 1];

    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    int halo = radius;

    // Center
    l_tile[halo + ty][tx] = read_image_2d_safe(input, gx, gy, width, height, BORDER_CLAMP_EDGE, 0.0f);

    // Top halo
    if (ty < halo) {
        l_tile[ty][tx] = read_image_2d_safe(input, gx, gy - halo, width, height, BORDER_CLAMP_EDGE, 0.0f);
    }

    // Bottom halo
    if (ty < halo) {
        l_tile[halo + FILTER_WG_Y + ty][tx] = read_image_2d_safe(input, gx, gy + FILTER_WG_Y, width, height, BORDER_CLAMP_EDGE, 0.0f);
    }

    sync_local();

    if (gx < width && gy < height) {
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            float weight = kernel_weights[k + radius];
            sum += l_tile[halo + ty + k][tx] * weight;
        }
        output[gy * width + gx] = sum;
    }
}

// Sobel Gradient Operator: computes Ix, Iy, magnitude, and orientation angle in radians
__kernel void kernel_sobel_gradient(
    __global const float* input,
    __global float* out_grad_x,
    __global float* out_grad_y,
    __global float* out_magnitude,
    __global float* out_angle,
    const int width,
    const int height) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    // 3x3 neighborhood samples
    float p00 = read_image_2d_safe(input, x - 1, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float p10 = read_image_2d_safe(input, x,     y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float p20 = read_image_2d_safe(input, x + 1, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f);

    float p01 = read_image_2d_safe(input, x - 1, y,     width, height, BORDER_CLAMP_EDGE, 0.0f);
    float p21 = read_image_2d_safe(input, x + 1, y,     width, height, BORDER_CLAMP_EDGE, 0.0f);

    float p02 = read_image_2d_safe(input, x - 1, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float p12 = read_image_2d_safe(input, x,     y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float p22 = read_image_2d_safe(input, x + 1, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f);

    // Standard Sobel filter weights
    float gx = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);
    float gy = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);

    size_t idx = (size_t)y * (size_t)width + (size_t)x;

    if (out_grad_x) out_grad_x[idx] = gx;
    if (out_grad_y) out_grad_y[idx] = gy;
    if (out_magnitude) out_magnitude[idx] = sqrt(gx * gx + gy * gy);
    if (out_angle) out_angle[idx] = atan2(gy, gx);
}
