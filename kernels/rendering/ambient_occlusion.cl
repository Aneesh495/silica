#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

#define AO_NUM_SAMPLES 8

// Precomputed hemisphere sample directions oriented towards +Z
__constant float3 AO_HEMISPHERE_SAMPLES[8] = {
    (float3)( 0.0f,       0.0f,       1.0f),
    (float3)( 0.57735f,   0.57735f,   0.57735f),
    (float3)(-0.57735f,   0.57735f,   0.57735f),
    (float3)( 0.57735f,  -0.57735f,   0.57735f),
    (float3)(-0.57735f,  -0.57735f,   0.57735f),
    (float3)( 0.7071f,    0.0f,       0.7071f),
    (float3)(-0.7071f,    0.0f,       0.7071f),
    (float3)( 0.0f,       0.7071f,    0.7071f)
};

// Computes Ambient Occlusion Accessibility Factor for 3D Volumetric Points
__kernel void kernel_compute_volumetric_ao(
    __global const float* volume_data,
    __global const float4* surface_normals,
    __global float* out_ao_factor,
    const int3 volume_dims,
    const float3 voxel_size,
    const float ao_radius,
    const float ao_intensity) {
    
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if (ix >= volume_dims.x || iy >= volume_dims.y || iz >= volume_dims.z) return;

    size_t vol_idx = (((size_t)iz * (size_t)volume_dims.y + (size_t)iy) * (size_t)volume_dims.x) + (size_t)ix;

    float density = volume_data[vol_idx];
    if (density < 0.05f) {
        out_ao_factor[vol_idx] = 1.0f; // Free space is fully illuminated
        return;
    }

    float4 norm4 = surface_normals[vol_idx];
    float3 n = normalize((float3)(norm4.x, norm4.y, norm4.z));

    // Construct local tangent basis (T, B, N)
    float3 up = (fabs(n.z) < 0.999f) ? (float3)(0.0f, 0.0f, 1.0f) : (float3)(1.0f, 0.0f, 0.0f);
    float3 tangent = normalize(cross(up, n));
    float3 bitangent = cross(n, tangent);

    float occlusion_sum = 0.0f;

    for (int i = 0; i < AO_NUM_SAMPLES; ++i) {
        float3 hs = AO_HEMISPHERE_SAMPLES[i];
        // Rotate hemisphere sample to align with surface normal n
        float3 dir = hs.x * tangent + hs.y * bitangent + hs.z * n;

        float ray_occ = 0.0f;
        for (float r = 1.0f; r <= ao_radius; r += 1.0f) {
            float3 sample_p = (float3)((float)ix, (float)iy, (float)iz) + r * dir;

            float val = read_volume_3d_safe(
                volume_data,
                (int)sample_p.x, (int)sample_p.y, (int)sample_p.z,
                volume_dims.x, volume_dims.y, volume_dims.z,
                BORDER_CLAMP_EDGE, 0.0f);

            ray_occ += val / r; // Attenuation over distance
        }
        occlusion_sum += ray_occ;
    }

    float avg_occ = occlusion_sum / (float)AO_NUM_SAMPLES;
    float accessibility = clamp_f(1.0f - avg_occ * ao_intensity, 0.0f, 1.0f);

    out_ao_factor[vol_idx] = accessibility;
}
