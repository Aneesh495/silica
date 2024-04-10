#include "common/math.clh"
#include "common/memory.clh"

// Calibrated Multi-Light Photometric Stereo with Normal and Albedo Decoupling

// Linear Least-Squares Surface Normal Estimation: I = rho * (L . n)
__kernel void kernel_photometric_stereo_least_squares(
    __global const float* observed_intensities, // [num_lights x num_pixels]
    __constant const float* light_pseudoinverse, // [3 x num_lights]
    __global float4* out_surface_normals,
    __global float* out_albedo,
    const int num_lights,
    const uint total_pixels) {
    uint pid = get_global_id(0);
    if (pid >= total_pixels) return;
    float3 g = (float3)(0.0f, 0.0f, 0.0f);
    for (int l = 0; l < num_lights; ++l) {
        float intensity = observed_intensities[(size_t)l * (size_t)total_pixels + pid];
        float3 pinv_row = (float3)(light_pseudoinverse[0 * num_lights + l],
                                  light_pseudoinverse[1 * num_lights + l],
                                  light_pseudoinverse[2 * num_lights + l]);
        g += pinv_row * intensity;
    }
    float albedo = length(g);
    float3 n = (albedo > 1.0e-6f) ? (g / albedo) : (float3)(0.0f, 0.0f, 1.0f);
    out_surface_normals[pid] = (float4)(n.x, n.y, n.z, 0.0f);
    out_albedo[pid] = albedo;
}
