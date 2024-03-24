#include "common/math.clh"

// Generates 1D Transfer Function Color Table from Multi-Peak Gaussian Parameters
// Each peak defines (center, width, red, green, blue, alpha)
__kernel void kernel_generate_gaussian_transfer_function(
    __global float4* out_tf_lut,
    __constant const float* peak_params, // [num_peaks x 6]
    const int num_peaks,
    const int lut_size) {
    
    int gid = get_global_id(0);
    if (gid >= lut_size) return;

    float norm_density = (float)gid / (float)(lut_size - 1);

    float3 color_accum = (float3)(0.0f, 0.0f, 0.0f);
    float alpha_accum = 0.0f;

    for (int p = 0; p < num_peaks; ++p) {
        int base = p * 6;
        float center = peak_params[base + 0];
        float width  = peak_params[base + 1];
        float r      = peak_params[base + 2];
        float g      = peak_params[base + 3];
        float b      = peak_params[base + 4];
        float a      = peak_params[base + 5];

        float dist = (norm_density - center) / (width + EPSILON_F);
        float weight = exp(-0.5f * dist * dist);

        color_accum += (float3)(r, g, b) * (weight * a);
        alpha_accum = fmax(alpha_accum, weight * a);
    }

    out_tf_lut[gid] = (float4)(
        clamp_f(color_accum.x, 0.0f, 1.0f),
        clamp_f(color_accum.y, 0.0f, 1.0f),
        clamp_f(color_accum.z, 0.0f, 1.0f),
        clamp_f(alpha_accum, 0.0f, 1.0f)
    );
}
