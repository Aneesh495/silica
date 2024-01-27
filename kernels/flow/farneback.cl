#include "common/math.clh"
#include "common/memory.clh"

#define FARNEBACK_RADIUS 4

// Farneback Polynomial Expansion: computes local 2nd order surface (A, b, c)
// f(x, y) = r1 + r2*x + r3*y + r4*x^2 + r5*y^2 + r6*x*y
__kernel void kernel_farneback_poly_expand(
    __global const float* img,
    __global float* out_poly_coeffs, // 5 floats per pixel: (A11, A22, A12, b1, b2)
    const int width,
    const int height) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    // Basis moments over 5x5 separable window
    float p00 = read_image_2d_safe(img, x,     y,     width, height, BORDER_CLAMP_EDGE, 0.0f);
    float p10 = read_image_2d_safe(img, x + 1, y,     width, height, BORDER_CLAMP_EDGE, 0.0f);
    float pm10 = read_image_2d_safe(img, x - 1, y,     width, height, BORDER_CLAMP_EDGE, 0.0f);
    float p01 = read_image_2d_safe(img, x,     y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float p0m1 = read_image_2d_safe(img, x,     y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f);

    float p11 = read_image_2d_safe(img, x + 1, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float pm11 = read_image_2d_safe(img, x - 1, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float p1m1 = read_image_2d_safe(img, x + 1, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float pm1m1 = read_image_2d_safe(img, x - 1, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f);

    // 1st derivatives (b1, b2)
    float b1 = 0.5f * (p10 - pm10);
    float b2 = 0.5f * (p01 - p0m1);

    // 2nd derivatives (A11, A22, A12)
    float A11 = 0.5f * (p10 + pm10 - 2.0f * p00);
    float A22 = 0.5f * (p01 + p0m1 - 2.0f * p00);
    float A12 = 0.25f * (p11 - pm11 - p1m1 + pm1m1);

    size_t out_base = ((size_t)y * (size_t)width + (size_t)x) * 5;
    out_poly_coeffs[out_base + 0] = A11;
    out_poly_coeffs[out_base + 1] = A22;
    out_poly_coeffs[out_base + 2] = A12;
    out_poly_coeffs[out_base + 3] = b1;
    out_poly_coeffs[out_base + 4] = b2;
}

// Solves Dense Optical Flow Field from Polynomial Differences
__kernel void kernel_farneback_solve_displacement(
    __global const float* poly1,
    __global const float* poly2,
    __global float2* out_flow,
    const int width,
    const int height) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    // Accumulate smoothed equations over window
    float G11 = 0.0f;
    float G22 = 0.0f;
    float G12 = 0.0f;
    float v1 = 0.0f;
    float v2 = 0.0f;

    for (int dy = -FARNEBACK_RADIUS; dy <= FARNEBACK_RADIUS; ++dy) {
        for (int dx = -FARNEBACK_RADIUS; dx <= FARNEBACK_RADIUS; ++dx) {
            int sx = handle_border_1d(x + dx, width, BORDER_CLAMP_EDGE);
            int sy = handle_border_1d(y + dy, height, BORDER_CLAMP_EDGE);

            size_t base1 = ((size_t)sy * (size_t)width + (size_t)sx) * 5;
            size_t base2 = ((size_t)sy * (size_t)width + (size_t)sx) * 5;

            float A11 = 0.5f * (poly1[base1 + 0] + poly2[base2 + 0]);
            float A22 = 0.5f * (poly1[base1 + 1] + poly2[base2 + 1]);
            float A12 = 0.5f * (poly1[base1 + 2] + poly2[base2 + 2]);

            float db1 = -0.5f * (poly2[base2 + 3] - poly1[base1 + 3]);
            float db2 = -0.5f * (poly2[base2 + 4] - poly1[base1 + 4]);

            float w = exp(-0.5f * (float)(dx * dx + dy * dy) / 4.0f);

            // A^T * A
            G11 += w * (A11 * A11 + A12 * A12);
            G22 += w * (A12 * A12 + A22 * A22);
            G12 += w * (A11 * A12 + A12 * A22);

            // A^T * db
            v1 += w * (A11 * db1 + A12 * db2);
            v2 += w * (A12 * db1 + A22 * db2);
        }
    }

    float det = G11 * G22 - G12 * G12;
    float2 flow = (float2)(0.0f, 0.0f);

    if (fabs(det) > 1.0e-6f) {
        float inv_det = 1.0f / det;
        flow.x = inv_det * (G22 * v1 - G12 * v2);
        flow.y = inv_det * (-G12 * v1 + G11 * v2);
    }

    out_flow[(size_t)y * (size_t)width + (size_t)x] = flow;
}
