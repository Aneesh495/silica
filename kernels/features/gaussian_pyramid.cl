#include "common/math.clh"
#include "common/memory.clh"

// Downsamples an image by factor of 2x in both dimensions with 4-tap box filter
__kernel void kernel_pyramid_downsample_2x(
    __global const float* in_img,
    __global float* out_downsampled,
    const int in_width,
    const int in_height,
    const int out_width,
    const int out_height) {
    
    int ox = get_global_id(0);
    int oy = get_global_id(1);

    if (ox >= out_width || oy >= out_height) return;

    int ix = ox * 2;
    int iy = oy * 2;

    float p00 = read_image_2d_safe(in_img, ix,     iy,     in_width, in_height, BORDER_CLAMP_EDGE, 0.0f);
    float p10 = read_image_2d_safe(in_img, ix + 1, iy,     in_width, in_height, BORDER_CLAMP_EDGE, 0.0f);
    float p01 = read_image_2d_safe(in_img, ix,     iy + 1, in_width, in_height, BORDER_CLAMP_EDGE, 0.0f);
    float p11 = read_image_2d_safe(in_img, ix + 1, iy + 1, in_width, in_height, BORDER_CLAMP_EDGE, 0.0f);

    out_downsampled[(size_t)oy * (size_t)out_width + (size_t)ox] = 0.25f * (p00 + p10 + p01 + p11);
}

// Computes Difference of Gaussians (DoG): DoG = Gaussian_scale_k - Gaussian_scale_orig
__kernel void kernel_compute_dog(
    __global const float* in_scale_k,
    __global const float* in_scale_orig,
    __global float* out_dog,
    const int width,
    const int height) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t idx = (size_t)y * (size_t)width + (size_t)x;
    out_dog[idx] = in_scale_k[idx] - in_scale_orig[idx];
}

// Upsamples an image by 2x using bilinear interpolation
__kernel void kernel_pyramid_upsample_2x(
    __global const float* in_img,
    __global float* out_upsampled,
    const int in_width,
    const int in_height,
    const int out_width,
    const int out_height) {
    
    int ox = get_global_id(0);
    int oy = get_global_id(1);

    if (ox >= out_width || oy >= out_height) return;

    float ix = ((float)ox + 0.5f) * 0.5f - 0.5f;
    float iy = ((float)oy + 0.5f) * 0.5f - 0.5f;

    int x0 = (int)floor(ix);
    int y0 = (int)floor(iy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = ix - (float)x0;
    float fy = iy - (float)y0;

    float v00 = read_image_2d_safe(in_img, x0, y0, in_width, in_height, BORDER_CLAMP_EDGE, 0.0f);
    float v10 = read_image_2d_safe(in_img, x1, y0, in_width, in_height, BORDER_CLAMP_EDGE, 0.0f);
    float v01 = read_image_2d_safe(in_img, x0, y1, in_width, in_height, BORDER_CLAMP_EDGE, 0.0f);
    float v11 = read_image_2d_safe(in_img, x1, y1, in_width, in_height, BORDER_CLAMP_EDGE, 0.0f);

    out_upsampled[(size_t)oy * (size_t)out_width + (size_t)ox] = bilinear_interp_unit(v00, v10, v01, v11, fx, fy);
}
