#include "common/math.clh"
#include "common/memory.clh"

// ============================================================================
// 3D Time-of-Flight (TOF) PET/SPECT Ordered Subsets Expectation Maximization
// ============================================================================

// 1. 3D TOF Gaussian Line-of-Response (LOR) Projector
__kernel void kernel_pet_tof_lor_forward_projector(
    __global const float* activity_volume,
    __global const float4* lor_endpoints_a, // (x, y, z, 0)
    __global const float4* lor_endpoints_b,
    __global const float* tof_time_delays,  // Delta t in meters
    __global float* out_simulated_prompts,
    const float tof_sigma, const float step_size,
    const int nx, const int ny, const int nz, const float voxel_size,
    const uint num_lors) {
    uint lid = get_global_id(0); if (lid >= num_lors) return;
    float4 pA = lor_endpoints_a[lid]; float4 pB = lor_endpoints_b[lid];
    float delta_t = tof_time_delays ? tof_time_delays[lid] : 0.0f;
    float3 dir = (float3)(pB.x - pA.x, pB.y - pA.y, pB.z - pA.z);
    float lor_len = length(dir);
    if (lor_len < 1.0e-5f) return;
    dir /= lor_len;
    float3 center = 0.5f * (float3)(pA.x + pB.x, pA.y + pB.y, pA.z + pB.z);
    float3 tof_peak = center + dir * (delta_t * 0.5f);
    float sum = 0.0f; int num_steps = (int)ceil(lor_len / step_size);
    size_t slice = (size_t)nx * (size_t)ny;
    float inv_2sigma_sq = 1.0f / (2.0f * tof_sigma * tof_sigma + 1.0e-7f);
    for (int s = 0; s < num_steps; ++s) {
        float dist = (float)s * step_size;
        float3 pos = (float3)(pA.x, pA.y, pA.z) + dir * dist;
        int vx = clamp_i((int)floor((pos.x / voxel_size) + 0.5f * (float)nx), 0, nx - 1);
        int vy = clamp_i((int)floor((pos.y / voxel_size) + 0.5f * (float)ny), 0, ny - 1);
        int vz = clamp_i((int)floor((pos.z / voxel_size) + 0.5f * (float)nz), 0, nz - 1);
        float3 dp = pos - tof_peak; float d_sq = dot(dp, dp);
        float tof_weight = exp(-d_sq * inv_2sigma_sq);
        float act = activity_volume[(size_t)vz * slice + (size_t)vy * (size_t)nx + (size_t)vx];
        sum += act * tof_weight * step_size;
    }
    out_simulated_prompts[lid] = sum;
}}


// 2. Ordered Subsets EM (OSEM) Multiplicative Image Update
__kernel void kernel_pet_osem_subsets_update(
    __global float* activity_image, __global const float* backprojected_ratio,
    __global const float* subset_sensitivity, const float epsilon, const uint total_voxels) {
    uint gid = get_global_id(0); if (gid >= total_voxels) return;
    float sens = subset_sensitivity[gid];
    if (sens > epsilon) {
        activity_image[gid] *= (backprojected_ratio[gid] / sens);
    }
}}


// ============================================================================
// Maximum A Posteriori (MAP-EM) Huber Prior Regularization Solvers
// ============================================================================

// 3D Huber Prior Penalty Gradient for PET/SPECT Image Reconstruction
__kernel void kernel_pet_huber_prior_gradient_3d(
    __global const float* activity_image, __global float* out_prior_gradient,
    const float huber_delta, const float beta_penalty,
    const int nx, const int ny, const int nz) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1 || z <= 0 || z >= nz - 1) return;
    size_t slice = (size_t)nx * (size_t)ny; size_t idx = (size_t)z * slice + (size_t)y * (size_t)nx + (size_t)x;
    float center_val = activity_image[idx]; float grad_sum = 0.0f;
    // 26-neighborhood spatial interaction
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                float dist = sqrt((float)(dx*dx + dy*dy + dz*dz));
                float w_dist = 1.0f / dist;
                size_t n_idx = (size_t)(z + dz) * slice + (size_t)(y + dy) * (size_t)nx + (size_t)(x + dx);
                float diff = center_val - activity_image[n_idx];
                // Huber derivative: diff if |diff| <= delta, else delta * sgn(diff)
                float huber_deriv = (fabs(diff) <= huber_delta) ? diff : (huber_delta * ((diff > 0.0f) ? 1.0f : -1.0f));
                grad_sum += w_dist * huber_deriv;
            }
        }
    }
    out_prior_gradient[idx] = beta_penalty * grad_sum;
}
