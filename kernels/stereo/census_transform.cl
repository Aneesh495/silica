#include "common/math.clh"
#include "common/memory.clh"

#define CENSUS_W 9
#define CENSUS_H 7
#define CENSUS_RX (CENSUS_W / 2) // 4
#define CENSUS_RY (CENSUS_H / 2) // 3
#define CENSUS_BITS 63

// Modified Census Transform (MCT 9x7)
// Compares 63 neighborhood pixels against local average intensity.
// Stores bit string in 64-bit ulong (2 words).
__kernel void kernel_modified_census_transform(
    __global const float* in_gray,
    __global ulong* out_census,
    const int width,
    const int height) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    if (x < CENSUS_RX || x >= width - CENSUS_RX ||
        y < CENSUS_RY || y >= height - CENSUS_RY) {
        out_census[(size_t)y * (size_t)width + (size_t)x] = 0;
        return;
    }

    // Compute average intensity over 9x7 window
    float sum = 0.0f;
    for (int dy = -CENSUS_RY; dy <= CENSUS_RY; ++dy) {
        for (int dx = -CENSUS_RX; dx <= CENSUS_RX; ++dx) {
            sum += in_gray[(size_t)(y + dy) * (size_t)width + (size_t)(x + dx)];
        }
    }
    float avg = sum / (float)(CENSUS_W * CENSUS_H);

    // Build 63-bit string
    ulong census_bits = 0;
    int bit_idx = 0;

    for (int dy = -CENSUS_RY; dy <= CENSUS_RY; ++dy) {
        for (int dx = -CENSUS_RX; dx <= CENSUS_RX; ++dx) {
            if (dx == 0 && dy == 0) continue; // Skip center
            float val = in_gray[(size_t)(y + dy) * (size_t)width + (size_t)(x + dx)];
            if (val >= avg) {
                census_bits |= (1UL << bit_idx);
            }
            bit_idx++;
        }
    }

    out_census[(size_t)y * (size_t)width + (size_t)x] = census_bits;
}

// Builds 3D Stereo Matching Cost Volume using Census Hamming Distance
// Cost volume size: width x height x max_disparity
__kernel void kernel_compute_census_cost_volume(
    __global const ulong* left_census,
    __global const ulong* right_census,
    __global uchar* out_cost_volume,
    const int width,
    const int height,
    const int max_disparity) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    ulong l_bits = left_census[(size_t)y * (size_t)width + (size_t)x];

    for (int d = 0; d < max_disparity; ++d) {
        int rx = x - d;
        uchar cost = 63; // Max Hamming distance for 63 bits

        if (rx >= 0) {
            ulong r_bits = right_census[(size_t)y * (size_t)width + (size_t)rx];
            ulong xor_val = l_bits ^ r_bits;
            // Popcount 64-bit
            cost = (uchar)(popcount((uint)xor_val) + popcount((uint)(xor_val >> 32)));
        }

        size_t cost_idx = (((size_t)y * (size_t)width + (size_t)x) * (size_t)max_disparity) + (size_t)d;
        out_cost_volume[cost_idx] = cost;
    }
}
