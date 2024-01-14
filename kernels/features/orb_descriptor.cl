#include "common/math.clh"
#include "common/memory.clh"

#define ORB_PATCH_RADIUS 15

// Computes Keypoint Orientation via Rosin's Intensity Centroid
__kernel void kernel_orb_compute_orientation(
    __global const float* in_gray,
    __global const float4* in_keypoints, // (x, y, scale, response)
    __global float* out_angles,
    const int width,
    const int height,
    const uint num_keypoints) {
    
    uint gid = get_global_id(0);
    if (gid >= num_keypoints) return;

    float4 kp = in_keypoints[gid];
    int cx = (int)round(kp.x);
    int cy = (int)round(kp.y);

    if (cx < ORB_PATCH_RADIUS || cx >= width - ORB_PATCH_RADIUS ||
        cy < ORB_PATCH_RADIUS || cy >= height - ORB_PATCH_RADIUS) {
        out_angles[gid] = 0.0f;
        return;
    }

    float m10 = 0.0f;
    float m01 = 0.0f;

    for (int v = -ORB_PATCH_RADIUS; v <= ORB_PATCH_RADIUS; ++v) {
        for (int u = -ORB_PATCH_RADIUS; u <= ORB_PATCH_RADIUS; ++u) {
            if (u * u + v * v <= ORB_PATCH_RADIUS * ORB_PATCH_RADIUS) {
                float intensity = in_gray[(size_t)(cy + v) * (size_t)width + (size_t)(cx + u)];
                m10 += (float)u * intensity;
                m01 += (float)v * intensity;
            }
        }
    }

    out_angles[gid] = atan2(m01, m10);
}

// 256-bit Steered BRIEF Descriptor Extraction
// Evaluates 256 rotated pixel comparison pairs per keypoint.
__kernel void kernel_orb_extract_descriptors(
    __global const float* in_gray,
    __global const float4* in_keypoints,
    __global const float* in_angles,
    __constant const int4* brief_pattern_pairs, // 256 pairs of (x1, y1, x2, y2)
    __global uint* out_descriptors,             // 8 uints (256 bits) per keypoint
    const int width,
    const int height,
    const uint num_keypoints) {
    
    uint gid = get_global_id(0);
    if (gid >= num_keypoints) return;

    float4 kp = in_keypoints[gid];
    float angle = in_angles[gid];

    float cos_t = cos(angle);
    float sin_t = sin(angle);

    float kpx = kp.x;
    float kpy = kp.y;

    uint desc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        desc[i] = 0;
    }

    for (int i = 0; i < 256; ++i) {
        int4 pair = brief_pattern_pairs[i];

        // Rotated sample positions
        float p1x = kpx + (float)pair.x * cos_t - (float)pair.y * sin_t;
        float p1y = kpy + (float)pair.x * sin_t + (float)pair.y * cos_t;

        float p2x = kpx + (float)pair.z * cos_t - (float)pair.w * sin_t;
        float p2y = kpy + (float)pair.z * sin_t + (float)pair.w * cos_t;

        // Bilinear sample at p1
        int x1_0 = (int)floor(p1x);
        int y1_0 = (int)floor(p1y);
        float fx1 = p1x - (float)x1_0;
        float fy1 = p1y - (float)y1_0;
        float v1_00 = read_image_2d_safe(in_gray, x1_0, y1_0, width, height, BORDER_CLAMP_EDGE, 0.0f);
        float v1_10 = read_image_2d_safe(in_gray, x1_0 + 1, y1_0, width, height, BORDER_CLAMP_EDGE, 0.0f);
        float v1_01 = read_image_2d_safe(in_gray, x1_0, y1_0 + 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
        float v1_11 = read_image_2d_safe(in_gray, x1_0 + 1, y1_0 + 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
        float val1 = bilinear_interp_unit(v1_00, v1_10, v1_01, v1_11, fx1, fy1);

        // Bilinear sample at p2
        int x2_0 = (int)floor(p2x);
        int y2_0 = (int)floor(p2y);
        float fx2 = p2x - (float)x2_0;
        float fy2 = p2y - (float)y2_0;
        float v2_00 = read_image_2d_safe(in_gray, x2_0, y2_0, width, height, BORDER_CLAMP_EDGE, 0.0f);
        float v2_10 = read_image_2d_safe(in_gray, x2_0 + 1, y2_0, width, height, BORDER_CLAMP_EDGE, 0.0f);
        float v2_01 = read_image_2d_safe(in_gray, x2_0, y2_0 + 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
        float v2_11 = read_image_2d_safe(in_gray, x2_0 + 1, y2_0 + 1, width, height, BORDER_CLAMP_EDGE, 0.0f);
        float val2 = bilinear_interp_unit(v2_00, v2_10, v2_01, v2_11, fx2, fy2);

        if (val1 < val2) {
            uint word_idx = (uint)i / 32;
            uint bit_idx  = (uint)i % 32;
            desc[word_idx] |= (1u << bit_idx);
        }
    }

    size_t out_base = (size_t)gid * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_descriptors[out_base + i] = desc[i];
    }
}
