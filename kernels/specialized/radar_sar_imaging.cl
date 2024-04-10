#include "common/math.clh"
#include "common/memory.clh"

// ============================================================================
// Synthetic Aperture Radar (SAR) Polar Format Algorithm and Polarimetry
// ============================================================================

// 1. Polar Format Algorithm (PFA) Keystone Interpolator (Polar to Cartesian Grid)
__kernel void kernel_sar_pfa_keystone_transform(
    __global const float2* polar_data, // [num_pulses x range_bins]
    __global float2* cartesian_kspace,  // [ky_bins x kx_bins]
    const float delta_theta, const float k_center, const float delta_kr,
    const int num_pulses, const int range_bins, const int kx_bins, const int ky_bins) {
    int kx_idx = get_global_id(0); int ky_idx = get_global_id(1);
    if (kx_idx >= kx_bins || ky_idx >= ky_bins) return;
    float kx = ((float)kx_idx - 0.5f * (float)kx_bins) * delta_kr;
    float ky = k_center + ((float)ky_idx - 0.5f * (float)ky_bins) * delta_kr;
    float r = sqrt(kx * kx + ky * ky); float theta = atan2(kx, ky);
    float pulse_f = (theta / delta_theta) + 0.5f * (float)num_pulses;
    float range_f = ((r - k_center) / delta_kr) + 0.5f * (float)range_bins;
    int p0 = clamp_i((int)floor(pulse_f), 0, num_pulses - 2);
    int r0 = clamp_i((int)floor(range_f), 0, range_bins - 2);
    float fp = pulse_f - (float)p0; float fr = range_f - (float)r0;
    size_t stride = (size_t)range_bins;
    float2 v00 = polar_data[(size_t)p0 * stride + r0];
    float2 v10 = polar_data[(size_t)p0 * stride + (r0 + 1)];
    float2 v01 = polar_data[(size_t)(p0 + 1) * stride + r0];
    float2 v11 = polar_data[(size_t)(p0 + 1) * stride + (r0 + 1)];
    float2 interp = mix(mix(v00, v10, fr), mix(v01, v11, fr), fp);
    cartesian_kspace[(size_t)ky_idx * (size_t)kx_bins + (size_t)kx_idx] = interp;
}}


// 2. Freeman-Durden 3-Component Polarimetric Decomposition
__kernel void kernel_polsar_freeman_durden_decomposition(
    __global const float2* C11, __global const float2* C12, __global const float2* C13,
    __global const float2* C22, __global const float2* C23, __global const float2* C33,
    __global float4* out_scattering_powers, // (P_surface, P_double, P_volume, 0)
    const uint total_pixels) {
    uint gid = get_global_id(0); if (gid >= total_pixels) return;
    // Volume scattering power: Pv = 8/3 * <|HV|^2> = 8/3 * C22
    float Pv = (8.0f / 3.0f) * C22[gid].x;
    float Ps = max(0.0f, C11[gid].x - Pv * 0.5f);
    float Pd = max(0.0f, C33[gid].x - Pv * 0.5f);
    out_scattering_powers[gid] = (float4)(Ps, Pd, Pv, 0.0f);
}}


// ============================================================================
// Advanced SAR Range Migration (RMA/Omega-K) and Chirp Scaling (CSA) Solvers
// ============================================================================

// 1. Range Migration Algorithm (RMA / Omega-K) 2D Stolt Resampling
__kernel void kernel_sar_omega_k_stolt_resampling(
    __global const float2* in_kx_kr_spectrum, __global float2* out_kx_ky_spectrum,
    const float k_carrier, const float delta_kx, const float delta_ky,
    const int num_kx, const int num_ky) {
    int kx_idx = get_global_id(0); int ky_idx = get_global_id(1);
    if (kx_idx >= num_kx || ky_idx >= num_ky) return;
    float kx = ((float)kx_idx - 0.5f * (float)num_kx) * delta_kx;
    float ky = ((float)ky_idx - 0.5f * (float)num_ky) * delta_ky + k_carrier;
    // Stolt dispersion relation: k_r = sqrt(kx^2 + ky^2) - k_carrier
    float kr = sqrt(kx * kx + ky * ky) - k_carrier;
    float kr_pos = (kr / delta_ky) + 0.5f * (float)num_ky;
    int r0 = clamp_i((int)floor(kr_pos), 0, num_ky - 2);
    float frac = kr_pos - (float)r0;
    size_t row0 = (size_t)r0 * (size_t)num_kx + (size_t)kx_idx;
    size_t row1 = (size_t)(r0 + 1) * (size_t)num_kx + (size_t)kx_idx;
    float2 v0 = in_kx_kr_spectrum[row0]; float2 v1 = in_kx_kr_spectrum[row1];
    float2 val = mix(v0, v1, frac);
    // Stolt Jacobian factor: |d(kr)/d(ky)| = ky / sqrt(kx^2 + ky^2)
    float jacobian = ky / sqrt(kx * kx + ky * ky + 1.0e-7f);
    out_kx_ky_spectrum[(size_t)ky_idx * (size_t)num_kx + (size_t)kx_idx] = val * jacobian;
}}


// 2. Chirp Scaling Phase Multiplication Step
__kernel void kernel_sar_chirp_scaling_phase_multiply(
    __global float2* range_doppler_data, const float cs_factor,
    const float carrier_freq, const float chirp_rate,
    const int num_azimuth, const int num_range) {
    int r_idx = get_global_id(0); int a_idx = get_global_id(1);
    if (r_idx >= num_range || a_idx >= num_azimuth) return;
    float tau = ((float)r_idx - 0.5f * (float)num_range) / carrier_freq;
    float phase = PI_F * chirp_rate * cs_factor * tau * tau;
    float cos_p = cos(phase); float sin_p = sin(phase);
    size_t idx = (size_t)a_idx * (size_t)num_range + (size_t)r_idx;
    float2 val = range_doppler_data[idx];
    range_doppler_data[idx] = (float2)(val.x * cos_p - val.y * sin_p, val.x * sin_p + val.y * cos_p);
}}
