#include "common/math.clh"
#include "common/memory.clh"

#define CFA_RGGB 0
#define CFA_BGGR 1
#define CFA_GRBG 2
#define CFA_GBRG 3

// Bilinear Bayer Demosaicing
// Converts single-channel raw CFA mosaiced sensor data into RGB image.
__kernel void kernel_demosaic_bilinear(
    __global const float* raw_cfa,
    __global float* out_rgb,
    const int width,
    const int height,
    const int cfa_pattern) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int r_row = 0, r_col = 0;
    int b_row = 1, b_col = 1;

    if (cfa_pattern == CFA_BGGR) {
        r_row = 1; r_col = 1;
        b_row = 0; b_col = 0;
    } else if (cfa_pattern == CFA_GRBG) {
        r_row = 0; r_col = 1;
        b_row = 1; b_col = 0;
    } else if (cfa_pattern == CFA_GBRG) {
        r_row = 1; r_col = 0;
        b_row = 0; b_col = 1;
    }

    bool is_r_row = ((y & 1) == r_row);
    bool is_r_col = ((x & 1) == r_col);
    bool is_b_row = ((y & 1) == b_row);
    bool is_b_col = ((x & 1) == b_col);

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;

    if (is_r_row && is_r_col) {
        // Pixel is Red
        r = read_image_2d_safe(raw_cfa, x, y, width, height, BORDER_CLAMP_EDGE, 0.0f);
        g = 0.25f * (
            read_image_2d_safe(raw_cfa, x - 1, y, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x + 1, y, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f)
        );
        b = 0.25f * (
            read_image_2d_safe(raw_cfa, x - 1, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x + 1, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x - 1, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x + 1, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f)
        );
    } else if (is_b_row && is_b_col) {
        // Pixel is Blue
        b = read_image_2d_safe(raw_cfa, x, y, width, height, BORDER_CLAMP_EDGE, 0.0f);
        g = 0.25f * (
            read_image_2d_safe(raw_cfa, x - 1, y, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x + 1, y, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f)
        );
        r = 0.25f * (
            read_image_2d_safe(raw_cfa, x - 1, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x + 1, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x - 1, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x + 1, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f)
        );
    } else if (is_r_row && is_b_col) {
        // Pixel is Green on Red row
        g = read_image_2d_safe(raw_cfa, x, y, width, height, BORDER_CLAMP_EDGE, 0.0f);
        r = 0.5f * (
            read_image_2d_safe(raw_cfa, x - 1, y, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x + 1, y, width, height, BORDER_CLAMP_EDGE, 0.0f)
        );
        b = 0.5f * (
            read_image_2d_safe(raw_cfa, x, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f)
        );
    } else {
        // Pixel is Green on Blue row
        g = read_image_2d_safe(raw_cfa, x, y, width, height, BORDER_CLAMP_EDGE, 0.0f);
        b = 0.5f * (
            read_image_2d_safe(raw_cfa, x - 1, y, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x + 1, y, width, height, BORDER_CLAMP_EDGE, 0.0f)
        );
        r = 0.5f * (
            read_image_2d_safe(raw_cfa, x, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f) +
            read_image_2d_safe(raw_cfa, x, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f)
        );
    }

    size_t out_idx = ((size_t)y * (size_t)width + (size_t)x) * 3;
    out_rgb[out_idx + 0] = clamp_f(r, 0.0f, 1.0f);
    out_rgb[out_idx + 1] = clamp_f(g, 0.0f, 1.0f);
    out_rgb[out_idx + 2] = clamp_f(b, 0.0f, 1.0f);
}

// Malvar-He-Cutler (MHC) High-Fidelity Gradient-Corrected Demosaicing
// Uses 5x5 linear FIR kernels with cross-color gradient Laplacian correction.
__kernel void kernel_demosaic_malvar_he_cutler(
    __global const float* raw_cfa,
    __global float* out_rgb,
    const int width,
    const int height,
    const int cfa_pattern) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    // Center pixel value
    float p00 = read_image_2d_safe(raw_cfa, x, y, width, height, BORDER_CLAMP_EDGE, 0.0f);

    // 5x5 cross neighborhood samples
    float p0m2 = read_image_2d_safe(raw_cfa, x, y - 2, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float p0p2 = read_image_2d_safe(raw_cfa, x, y + 2, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float pm20 = read_image_2d_safe(raw_cfa, x - 2, y, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float pp20 = read_image_2d_safe(raw_cfa, x + 2, y, width, height, BORDER_CLAMP_EDGE, 0.0f);

    float p0m1 = read_image_2d_safe(raw_cfa, x, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float p0p1 = read_image_2d_safe(raw_cfa, x, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float pm10 = read_image_2d_safe(raw_cfa, x - 1, y, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float pp10 = read_image_2d_safe(raw_cfa, x + 1, y, width, height, BORDER_CLAMP_EDGE, 0.0f);

    float pm1m1 = read_image_2d_safe(raw_cfa, x - 1, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float pp1m1 = read_image_2d_safe(raw_cfa, x + 1, y - 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float pm1p1 = read_image_2d_safe(raw_cfa, x - 1, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
    float pp1p1 = read_image_2d_safe(raw_cfa, x + 1, y + 1, width, height, BORDER_CLAMP_EDGE, 0.0f);

    int r_row = (cfa_pattern == CFA_BGGR || cfa_pattern == CFA_GBRG) ? 1 : 0;
    int r_col = (cfa_pattern == CFA_BGGR || cfa_pattern == CFA_GRBG) ? 1 : 0;

    bool is_r_row = ((y & 1) == r_row);
    bool is_r_col = ((x & 1) == r_col);

    float r = 0.0f, g = 0.0f, b = 0.0f;

    if (is_r_row && is_r_col) {
        // At Red pixel
        r = p00;
        // G at R: 4*(cardinal) + 2*(center) - (corners/outer) / 8
        g = 0.125f * (2.0f * (pm10 + pp10 + p0m1 + p0p1) + 2.0f * p00 - (pm20 + pp20 + p0m2 + p0p2));
        // B at R: 4*(corners) + 6*(center) - 3/2*(outer cross) / 16
        b = 0.0625f * (4.0f * (pm1m1 + pp1m1 + pm1p1 + pp1p1) + 6.0f * p00 - 1.5f * (pm20 + pp20 + p0m2 + p0p2));
    } else if (!is_r_row && !is_r_col) {
        // At Blue pixel
        b = p00;
        g = 0.125f * (2.0f * (pm10 + pp10 + p0m1 + p0p1) + 2.0f * p00 - (pm20 + pp20 + p0m2 + p0p2));
        r = 0.0625f * (4.0f * (pm1m1 + pp1m1 + pm1p1 + pp1p1) + 6.0f * p00 - 1.5f * (pm20 + pp20 + p0m2 + p0p2));
    } else if (is_r_row && !is_r_col) {
        // At Green pixel on Red row
        g = p00;
        // R at G_R: horizontal neighbors + vertical laplacian
        r = 0.125f * (4.0f * (pm10 + pp10) + 5.0f * p00 - (pm20 + pp20 + 2.0f * (p0m1 + p0p1)) + p0m2 + p0p2);
        // B at G_R: vertical neighbors + horizontal laplacian
        b = 0.125f * (4.0f * (p0m1 + p0p1) + 5.0f * p00 - (p0m2 + p0p2 + 2.0f * (pm10 + pp10)) + pm20 + pp20);
    } else {
        // At Green pixel on Blue row
        g = p00;
        b = 0.125f * (4.0f * (pm10 + pp10) + 5.0f * p00 - (pm20 + pp20 + 2.0f * (p0m1 + p0p1)) + p0m2 + p0p2);
        r = 0.125f * (4.0f * (p0m1 + p0p1) + 5.0f * p00 - (p0m2 + p0p2 + 2.0f * (pm10 + pp10)) + pm20 + pp20);
    }

    size_t out_idx = ((size_t)y * (size_t)width + (size_t)x) * 3;
    out_rgb[out_idx + 0] = clamp_f(r, 0.0f, 1.0f);
    out_rgb[out_idx + 1] = clamp_f(g, 0.0f, 1.0f);
    out_rgb[out_idx + 2] = clamp_f(b, 0.0f, 1.0f);
}
