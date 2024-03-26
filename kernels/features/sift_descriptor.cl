#include "common/math.clh"
#include "common/memory.clh"

#define SIFT_DESCR_WIDTH 4      // 4x4 subregions
#define SIFT_DESCR_HIST_BINS 8  // 8 orientation bins
#define SIFT_DESCR_MAG_THR 0.2f // Clamping threshold for non-linear illumination changes
#define SIFT_INT_DESCR_FCTR 512.0f

// SIFT 128-Dimensional Feature Descriptor Extraction
// Computes 4x4 spatial subregions with 8-bin gradient orientation histograms.
// Incorporates trilinear interpolation across spatial x, spatial y, and orientation angle bins.
__kernel void kernel_sift_extract_128d(
    __global const float* in_grad_mag,
    __global const float* in_grad_angle,
    __global const float4* in_keypoints, // (x, y, scale, angle)
    __global float* out_descriptors,     // num_keypoints x 128 floats
    const int width,
    const int height,
    const uint num_keypoints) {
    
    uint gid = get_global_id(0);
    if (gid >= num_keypoints) return;

    float4 kp = in_keypoints[gid];
    float kpx = kp.x;
    float kpy = kp.y;
    float scale = fmax(1.0f, kp.z);
    float orientation = kp.w;

    float cos_t = cos(orientation);
    float sin_t = sin(orientation);

    // Initialize 128-bin descriptor histogram in thread-private registers
    float hist[SIFT_DESCR_WIDTH][SIFT_DESCR_WIDTH][SIFT_DESCR_HIST_BINS];
    #pragma unroll
    for (int i = 0; i < SIFT_DESCR_WIDTH; ++i) {
        #pragma unroll
        for (int j = 0; j < SIFT_DESCR_WIDTH; ++j) {
            #pragma unroll
            for (int k = 0; k < SIFT_DESCR_HIST_BINS; ++k) {
                hist[i][j][k] = 0.0f;
            }
        }
    }

    // Window radius in scale units
    float hist_width = 3.0f * scale;
    float radius = hist_width * SQRT2_F * (float)(SIFT_DESCR_WIDTH + 1) * 0.5f;
    int int_radius = (int)ceil(radius);

    float exp_scale = -1.0f / (2.0f * (0.5f * (float)SIFT_DESCR_WIDTH) * (0.5f * (float)SIFT_DESCR_WIDTH));

    for (int dy = -int_radius; dy <= int_radius; ++dy) {
        for (int dx = -int_radius; dx <= int_radius; ++dx) {
            // Coordinate in keypoint frame
            float rot_x = ((float)dx * cos_t + (float)dy * sin_t) / hist_width;
            float rot_y = (-(float)dx * sin_t + (float)dy * cos_t) / hist_width;

            float bin_x = rot_x + (float)SIFT_DESCR_WIDTH * 0.5f - 0.5f;
            float bin_y = rot_y + (float)SIFT_DESCR_WIDTH * 0.5f - 0.5f;

            if (bin_x > -1.0f && bin_x < (float)SIFT_DESCR_WIDTH &&
                bin_y > -1.0f && bin_y < (float)SIFT_DESCR_WIDTH) {
                
                int img_x = (int)round(kpx) + dx;
                int img_y = (int)round(kpy) + dy;

                if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
                    size_t pixel_idx = (size_t)img_y * (size_t)width + (size_t)img_x;
                    float mag = in_grad_mag[pixel_idx];
                    float ang = in_grad_angle[pixel_idx] - orientation;

                    // Normalize angle to [0, 2*pi)
                    while (ang < 0.0f) ang += TWO_PI_F;
                    while (ang >= TWO_PI_F) ang -= TWO_PI_F;

                    float bin_angle = ang * (float)SIFT_DESCR_HIST_BINS * INV_TWO_PI_F;

                    // Gaussian spatial weighting
                    float dist_sq = (rot_x * rot_x + rot_y * rot_y);
                    float w = mag * exp(dist_sq * exp_scale);

                    // Trilinear interpolation across (bin_x, bin_y, bin_angle)
                    int bx0 = (int)floor(bin_x);
                    int by0 = (int)floor(bin_y);
                    int ba0 = (int)floor(bin_angle);

                    float fx = bin_x - (float)bx0;
                    float fy = bin_y - (float)by0;
                    float fa = bin_angle - (float)ba0;

                    for (int di = 0; di <= 1; ++di) {
                        int bx = bx0 + di;
                        if (bx >= 0 && bx < SIFT_DESCR_WIDTH) {
                            float wx = w * (di == 0 ? (1.0f - fx) : fx);

                            for (int dj = 0; dj <= 1; ++dj) {
                                int by = by0 + dj;
                                if (by >= 0 && by < SIFT_DESCR_WIDTH) {
                                    float wxy = wx * (dj == 0 ? (1.0f - fy) : fy);

                                    for (int dk = 0; dk <= 1; ++dk) {
                                        int ba = (ba0 + dk) % SIFT_DESCR_HIST_BINS;
                                        float wxya = wxy * (dk == 0 ? (1.0f - fa) : fa);

                                        hist[bx][by][ba] += wxya;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // L2 norm computation
    float norm_sq = 0.0f;
    for (int i = 0; i < SIFT_DESCR_WIDTH; ++i) {
        for (int j = 0; j < SIFT_DESCR_WIDTH; ++j) {
            for (int k = 0; k < SIFT_DESCR_HIST_BINS; ++k) {
                norm_sq += hist[i][j][k] * hist[i][j][k];
            }
        }
    }

    float inv_norm = safe_inv_f(sqrt(norm_sq));

    // Clamp at 0.2 to reduce non-linear illumination sensitivity
    float re_norm_sq = 0.0f;
    for (int i = 0; i < SIFT_DESCR_WIDTH; ++i) {
        for (int j = 0; j < SIFT_DESCR_WIDTH; ++j) {
            for (int k = 0; k < SIFT_DESCR_HIST_BINS; ++k) {
                float v = fmin(hist[i][j][k] * inv_norm, SIFT_DESCR_MAG_THR);
                hist[i][j][k] = v;
                re_norm_sq += v * v;
            }
        }
    }

    float inv_re_norm = safe_inv_f(sqrt(re_norm_sq));

    // Write final 128 normalized float values
    size_t out_base = (size_t)gid * 128;
    int idx = 0;
    for (int i = 0; i < SIFT_DESCR_WIDTH; ++i) {
        for (int j = 0; j < SIFT_DESCR_WIDTH; ++j) {
            for (int k = 0; k < SIFT_DESCR_HIST_BINS; ++k) {
                out_descriptors[out_base + idx++] = hist[i][j][k] * inv_re_norm;
            }
        }
    }
}
