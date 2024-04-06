#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

// Blinn-Phong and GGX Microfacet Shading for Volumetric Surfaces
inline float3 cook_torrance_ggx_brdf(
    float3 N,
    float3 V,
    float3 L,
    float3 albedo,
    float roughness,
    float metallic) {
    
    float3 H = normalize(V + L);

    float NdotL = fmax(0.0f, dot(N, L));
    float NdotV = fmax(1.0e-5f, dot(N, V));
    float NdotH = fmax(0.0f, dot(N, H));
    float VdotH = fmax(0.0f, dot(V, H));

    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    // GGX / Trowbridge-Reitz normal distribution function D
    float d_denom = (NdotH * NdotH * (alpha2 - 1.0f) + 1.0f);
    float D = alpha2 / (PI_F * d_denom * d_denom + EPSILON_F);

    // Fresnel Schlick approximation F
    float3 F0 = lerp_f(0.04f, 1.0f, metallic) * (float3)(1.0f, 1.0f, 1.0f);
    float3 F = F0 + (1.0f - F0) * pow(clamp_f(1.0f - VdotH, 0.0f, 1.0f), 5.0f);

    // Smith geometric shadowing/masking function G
    float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
    float G_L = NdotL / (NdotL * (1.0f - k) + k);
    float G_V = NdotV / (NdotV * (1.0f - k) + k);
    float G = G_L * G_V;

    // Specular Cook-Torrance term
    float3 specular = (D * G * F) / (4.0f * NdotL * NdotV + EPSILON_F);

    // Lambertian diffuse term
    float3 kD = ((float3)(1.0f, 1.0f, 1.0f) - F) * (1.0f - metallic);
    float3 diffuse = kD * albedo * INV_PI_F;

    return (diffuse + specular) * NdotL;
}

// Applies Cook-Torrance Microfacet Shading to Normal and Depth Buffers
__kernel void kernel_shade_volumetric_surface(
    __global const float4* in_normals, // (nx, ny, nz, valid)
    __global const float* in_depth,
    __global float4* out_shaded_image,
    const int width,
    const int height,
    const float3 light_dir_world,
    const float3 camera_pos_world,
    const float3 albedo,
    const float roughness,
    const float metallic) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t idx = (size_t)y * (size_t)width + (size_t)x;

    float4 norm4 = in_normals[idx];
    float depth = in_depth[idx];

    if (norm4.w <= 0.0f || depth <= 0.0f) {
        out_shaded_image[idx] = (float4)(0.05f, 0.05f, 0.08f, 1.0f); // Dark background
        return;
    }

    float3 N = normalize((float3)(norm4.x, norm4.y, norm4.z));
    float3 L = normalize(light_dir_world);
    float3 V = (float3)(0.0f, 0.0f, 1.0f); // Camera look direction

    float3 shaded = cook_torrance_ggx_brdf(N, V, L, albedo, roughness, metallic);
    // Add ambient term
    shaded += albedo * 0.1f;

    out_shaded_image[idx] = (float4)(clamp_f(shaded.x, 0.0f, 1.0f),
                                     clamp_f(shaded.y, 0.0f, 1.0f),
                                     clamp_f(shaded.z, 0.0f, 1.0f),
                                     1.0f);
}
