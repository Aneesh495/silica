#include "common/math.clh"
#include "common/memory.clh"

#define FILTER_RAMLAK     0
#define FILTER_SHEPP_LOGAN 1
#define FILTER_COSINE     2
#define FILTER_HANN       3

// Evaluates Tomographic Frequency Filter Weight at normalized frequency w in [0, 1]
inline float evaluate_filter_window(float w, int filter_type) {
    if (w <= 0.0f || w > 1.0f) return 0.0f;

    float ramp = w;

    switch (filter_type) {
        case FILTER_SHEPP_LOGAN: {
            float sinc_val = sin(0.5f * PI_F * w) / (0.5f * PI_F * w);
            return ramp * sinc_val;
        }
        case FILTER_COSINE:
            return ramp * cos(0.5f * PI_F * w);
        case FILTER_HANN:
            return ramp * 0.5f * (1.0f + cos(PI_F * w));
        case FILTER_RAMLAK:
        default:
            return ramp;
    }
}

// Frequency-Domain 1D Ramp Filter Multiplication on FFT-Transformed Sinogram Rows
// Complex input/output: [num_angles x num_freq_bins] of float2 (real, imag)
__kernel void kernel_apply_frequency_ramp_filter(
    __global const float2* freq_sinogram,
    __global float2* out_filtered_freq_sinogram,
    const int num_angles,
    const int num_freq_bins,
    const int filter_type) {
    
    int freq_idx = get_global_id(0);
    int angle_idx = get_global_id(1);

    if (freq_idx >= num_freq_bins || angle_idx >= num_angles) return;

    // Symmetric frequency distance from DC component (center of spectrum)
    int half_bins = num_freq_bins / 2;
    int k = freq_idx - half_bins;
    float norm_w = fabs((float)k) / (float)half_bins;

    float filter_w = evaluate_filter_window(norm_w, filter_type);

    size_t idx = (size_t)angle_idx * (size_t)num_freq_bins + (size_t)freq_idx;
    float2 val = freq_sinogram[idx];

    out_filtered_freq_sinogram[idx] = (float2)(val.x * filter_w, val.y * filter_w);
}

// Spatial-domain Discrete FIR Ramp Filter Convolution
// Uses bandlimited discrete Ram-Lak kernel: h[0] = 1/(4*d^2), h[odd] = -1/(pi^2 * n^2 * d^2), h[even] = 0
__kernel void kernel_spatial_ramlak_filter_row(
    __global const float* sinogram,
    __global float* out_filtered_sinogram,
    const int num_angles,
    const int num_detectors,
    const int kernel_radius,
    const float detector_spacing) {
    
    int det_idx = get_global_id(0);
    int angle_idx = get_global_id(1);

    if (det_idx >= num_detectors || angle_idx >= num_angles) return;

    size_t row_base = (size_t)angle_idx * (size_t)num_detectors;
    float d2 = detector_spacing * detector_spacing;
    float sum = 0.0f;

    for (int k = -kernel_radius; k <= kernel_radius; ++k) {
        int neighbor_idx = det_idx + k;
        if (neighbor_idx >= 0 && neighbor_idx < num_detectors) {
            float h_k = 0.0f;
            if (k == 0) {
                h_k = 0.25f / d2;
            } else if ((k & 1) != 0) { // odd k
                float fk = (float)k;
                h_k = -1.0f / (PI_F * PI_F * fk * fk * d2);
            }
            sum += sinogram[row_base + neighbor_idx] * h_k;
        }
    }

    out_filtered_sinogram[row_base + det_idx] = sum * detector_spacing;
}
