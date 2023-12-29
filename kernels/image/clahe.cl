#include "common/math.clh"
#include "common/memory.clh"

#define CLAHE_BINS 256

// Stage 1: Build local histogram per grid tile
__kernel void kernel_clahe_tile_histograms(
    __global const uchar* in_gray,
    __global uint* out_tile_histograms,
    const int width,
    const int height,
    const int tile_size_x,
    const int tile_size_y,
    const int grid_tiles_x,
    const int grid_tiles_y) {
    
    int tx = get_global_id(0);
    int ty = get_global_id(1);

    if (tx >= grid_tiles_x || ty >= grid_tiles_y) return;

    int tile_id = ty * grid_tiles_x + tx;
    __global uint* hist = out_tile_histograms + tile_id * CLAHE_BINS;

    for (int b = 0; b < CLAHE_BINS; ++b) {
        hist[b] = 0;
    }

    int start_x = tx * tile_size_x;
    int start_y = ty * tile_size_y;
    int end_x = min(width, start_x + tile_size_x);
    int end_y = min(height, start_y + tile_size_y);

    for (int y = start_y; y < end_y; ++y) {
        for (int x = start_x; x < end_x; ++x) {
            uchar val = in_gray[(size_t)y * (size_t)width + (size_t)x];
            hist[val]++;
        }
    }
}

// Stage 2: Clip histograms and calculate Cumulative Distribution Function (CDF)
__kernel void kernel_clahe_clip_and_cdf(
    __global uint* tile_histograms,
    __global float* tile_cdfs,
    const int num_tiles,
    const uint clip_limit) {
    
    int tile_id = get_global_id(0);
    if (tile_id >= num_tiles) return;

    __global uint* hist = tile_histograms + tile_id * CLAHE_BINS;
    __global float* cdf = tile_cdfs + tile_id * CLAHE_BINS;

    // Calculate total clipped excess
    uint excess = 0;
    for (int b = 0; b < CLAHE_BINS; ++b) {
        if (hist[b] > clip_limit) {
            excess += (hist[b] - clip_limit);
            hist[b] = clip_limit;
        }
    }

    // Redistribute excess evenly
    uint bonus = excess / CLAHE_BINS;
    uint remainder = excess % CLAHE_BINS;
    for (int b = 0; b < CLAHE_BINS; ++b) {
        hist[b] += bonus;
        if ((uint)b < remainder) {
            hist[b]++;
        }
    }

    // Compute normalized cumulative distribution function (CDF)
    uint running_sum = 0;
    for (int b = 0; b < CLAHE_BINS; ++b) {
        running_sum += hist[b];
    }

    float inv_total = (running_sum > 0) ? (255.0f / (float)running_sum) : 0.0f;
    uint accum = 0;
    for (int b = 0; b < CLAHE_BINS; ++b) {
        accum += hist[b];
        cdf[b] = (float)accum * inv_total;
    }
}

// Stage 3: Map pixels using bilinear interpolation of surrounding 4 tile CDFs
__kernel void kernel_clahe_interpolate(
    __global const uchar* in_gray,
    __global uchar* out_gray,
    __global const float* tile_cdfs,
    const int width,
    const int height,
    const int tile_size_x,
    const int tile_size_y,
    const int grid_tiles_x,
    const int grid_tiles_y) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    uchar val = in_gray[(size_t)y * (size_t)width + (size_t)x];

    // Normalized tile grid coordinates
    float gx = ((float)x + 0.5f) / (float)tile_size_x - 0.5f;
    float gy = ((float)y + 0.5f) / (float)tile_size_y - 0.5f;

    int tx0 = clamp_i((int)floor(gx), 0, grid_tiles_x - 1);
    int ty0 = clamp_i((int)floor(gy), 0, grid_tiles_y - 1);
    int tx1 = clamp_i(tx0 + 1, 0, grid_tiles_x - 1);
    int ty1 = clamp_i(ty0 + 1, 0, grid_tiles_y - 1);

    float fx = clamp_f(gx - (float)tx0, 0.0f, 1.0f);
    float fy = clamp_f(gy - (float)ty0, 0.0f, 1.0f);

    float cdf00 = tile_cdfs[(ty0 * grid_tiles_x + tx0) * CLAHE_BINS + val];
    float cdf10 = tile_cdfs[(ty0 * grid_tiles_x + tx1) * CLAHE_BINS + val];
    float cdf01 = tile_cdfs[(ty1 * grid_tiles_x + tx0) * CLAHE_BINS + val];
    float cdf11 = tile_cdfs[(ty1 * grid_tiles_x + tx1) * CLAHE_BINS + val];

    float mapped = bilinear_interp_unit(cdf00, cdf10, cdf01, cdf11, fx, fy);
    out_gray[(size_t)y * (size_t)width + (size_t)x] = (uchar)clamp_f(mapped + 0.5f, 0.0f, 255.0f);
}
