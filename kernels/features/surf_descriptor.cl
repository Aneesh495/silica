#include "common/math.clh"
#include "common/memory.clh"

#define SURF_SUBREGIONS 4 // 4x4 subregions
#define SURF_VEC_LEN 64   // 16 subregions * 4 values = 64D

// Evaluates 2D Box Sum on Summed-Area Table (SAT)
inline float query_sat_box(
    __global const float* sat,
    int x1, int y1, int x2, int y2,
    int width, int height) {
    
    x1 = clamp_i(x1, 0, width - 1);
    y1 = clamp_i(y1, 0, height - 1);
    x2 = clamp_i(x2, 0, width - 1);
    y2 = clamp_i(y2, 0, height - 1);

    if (x1 > x2 || y1 > y2) return 0.0f;

    float d = sat[(size_t)y2 * (size_t)width + (size_t)x2];
    float b = (y1 > 0) ? sat[(size_t)(y1 - 1) * (size_t)width + (size_t)x2] : 0.0f;
    float c = (x1 > 0) ? sat[(size_t)y2 * (size_t)width + (size_t)(x1 - 1)] : 0.0f;
    float a = (x1 > 0 && y1 > 0) ? sat[(size_t)(y1 - 1) * (size_t)width + (size_t)(x1 - 1)] : 0.0f;

    return d - b - c + a;
}

// SURF 64-Dimensional Descriptor Extraction
// Computes Haar wavelet responses (dx, dy, |dx|, |dy|) over 4x4 grid of 5x5 subregions
__kernel void kernel_surf_extract_64d(
    __global const float* sat_image,
    __global const float4* in_keypoints, // (x, y, scale, angle)
    __global float* out_descriptors,     // num_keypoints x 64 floats
    const int width,
    const int height,
    const uint num_keypoints) {
    
    uint gid = get_global_id(0);
    if (gid >= num_keypoints) return;

    float4 kp = in_keypoints[gid];
    float kpx = kp.x;
    float kpy = kp.y;
    float scale = fmax(1.0f, kp.z);
    float angle = kp.w;

    float cos_t = cos(angle);
    float sin_t = sin(angle);

    float desc[SURF_VEC_LEN];
    #pragma unroll
    for (int i = 0; i < SURF_VEC_LEN; ++i) desc[i] = 0.0f;

    int haar_size = (int)round(2.0f * scale);
    int half_haar = haar_size / 2;
    if (half_haar < 1) half_haar = 1;

    int desc_idx = 0;

    // Iterate over 4x4 subregions
    for (int sy = -2; sy < 2; ++sy) {
        for (int sx = -2; sx < 2; ++sx) {
            float sum_dx = 0.0f;
            float sum_dy = 0.0f;
            float sum_abs_dx = 0.0f;
            float sum_abs_dy = 0.0f;

            // Sample 5x5 grid of sample points within each subregion
            for (int py = 0; py < 5; ++py) {
                for (int px = 0; px < 5; ++px) {
                    float sample_u = ((float)sx + (float)px * 0.2f) * 5.0f * scale;
                    float sample_v = ((float)sy + (float)py * 0.2f) * 5.0f * scale;

                    // Rotated sample position
                    int rx = (int)round(kpx + sample_u * cos_t - sample_v * sin_t);
                    int ry = (int)round(kpy + sample_u * sin_t + sample_v * cos_t);

                    // Haar wavelet dx: right box minus left box
                    float box_r = query_sat_box(sat_image, rx, ry - half_haar, rx + half_haar, ry + half_haar, width, height);
                    float box_l = query_sat_box(sat_image, rx - half_haar, ry - half_haar, rx, ry + half_haar, width, height);
                    float dx = box_r - box_l;

                    // Haar wavelet dy: bottom box minus top box
                    float box_b = query_sat_box(sat_image, rx - half_haar, ry, rx + half_haar, ry + half_haar, width, height);
                    float box_t = query_sat_box(sat_image, rx - half_haar, ry - half_haar, rx + half_haar, ry, width, height);
                    float dy = box_b - box_t;

                    // Rotate Haar responses to align with dominant keypoint orientation
                    float rot_dx =  dx * cos_t + dy * sin_t;
                    float rot_dy = -dx * sin_t + dy * cos_t;

                    // Gaussian weighting
                    float dist_sq = sample_u * sample_u + sample_v * sample_v;
                    float w = exp(-0.5f * dist_sq / (3.3f * scale * 3.3f * scale));

                    sum_dx += rot_dx * w;
                    sum_dy += rot_dy * w;
                    sum_abs_dx += fabs(rot_dx) * w;
                    sum_abs_dy += fabs(rot_dy) * w;
                }
            }

            desc[desc_idx++] = sum_dx;
            desc[desc_idx++] = sum_dy;
            desc[desc_idx++] = sum_abs_dx;
            desc[desc_idx++] = sum_abs_dy;
        }
    }

    // L2 vector normalization
    float norm_sq = 0.0f;
    for (int i = 0; i < SURF_VEC_LEN; ++i) {
        norm_sq += desc[i] * desc[i];
    }
    float inv_norm = safe_inv_f(sqrt(norm_sq));

    size_t out_base = (size_t)gid * SURF_VEC_LEN;
    for (int i = 0; i < SURF_VEC_LEN; ++i) {
        out_descriptors[out_base + i] = desc[i] * inv_norm;
    }
}
