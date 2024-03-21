#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

// 1D Transfer Function Lookup: maps density in [0.0, 1.0] to RGBA color and opacity
inline float4 sample_transfer_function(
    __global const float4* tf_lut,
    float density,
    int lut_size) {
    
    float norm_d = clamp_f(density, 0.0f, 1.0f);
    float pos = norm_d * (float)(lut_size - 1);
    int idx0 = (int)floor(pos);
    int idx1 = min(lut_size - 1, idx0 + 1);
    float frac = pos - (float)idx0;

    float4 c0 = tf_lut[idx0];
    float4 c1 = tf_lut[idx1];

    return (float4)(
        lerp_f(c0.x, c1.x, frac),
        lerp_f(c0.y, c1.y, frac),
        lerp_f(c0.z, c1.z, frac),
        lerp_f(c0.w, c1.w, frac)
    );
}

// Computes 3D Volume Gradient for Surface Normal
inline float3 compute_volume_gradient(
    __global const float* volume,
    float3 p_voxel,
    int3 dims) {
    
    float step = 1.0f;
    float dx = read_volume_3d_safe(volume, (int)(p_voxel.x + step), (int)p_voxel.y, (int)p_voxel.z, dims.x, dims.y, dims.z, BORDER_CLAMP_EDGE, 0.0f) -
               read_volume_3d_safe(volume, (int)(p_voxel.x - step), (int)p_voxel.y, (int)p_voxel.z, dims.x, dims.y, dims.z, BORDER_CLAMP_EDGE, 0.0f);
    float dy = read_volume_3d_safe(volume, (int)p_voxel.x, (int)(p_voxel.y + step), (int)p_voxel.z, dims.x, dims.y, dims.z, BORDER_CLAMP_EDGE, 0.0f) -
               read_volume_3d_safe(volume, (int)p_voxel.x, (int)(p_voxel.y - step), (int)p_voxel.z, dims.x, dims.y, dims.z, BORDER_CLAMP_EDGE, 0.0f);
    float dz = read_volume_3d_safe(volume, (int)p_voxel.x, (int)p_voxel.y, (int)(p_voxel.z + step), dims.x, dims.y, dims.z, BORDER_CLAMP_EDGE, 0.0f) -
               read_volume_3d_safe(volume, (int)p_voxel.x, (int)p_voxel.y, (int)(p_voxel.z - step), dims.x, dims.y, dims.z, BORDER_CLAMP_EDGE, 0.0f);

    float3 g = (float3)(dx, dy, dz);
    float len = length(g);
    return len > 1.0e-5f ? (g / len) : (float3)(0.0f, 0.0f, 1.0f);
}

// Direct Volume Raycasting (DVR) with Emission-Absorption Optical Model
__kernel void kernel_volume_raycast_dvr(
    __global const float* volume_data,
    __global const float4* transfer_function_lut,
    __global float4* out_framebuffer, // RGBA
    __constant const float* c2w_matrix,
    const int img_w,
    const int img_h,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const int3 volume_dims,
    const float3 voxel_size,
    const float step_size,
    const int lut_size,
    const float3 light_dir_world) {
    
    int u = get_global_id(0);
    int v = get_global_id(1);

    if (u >= img_w || v >= img_h) return;

    size_t pixel_idx = (size_t)v * (size_t)img_w + (size_t)u;

    // Ray setup
    float3 ray_dir_cam = normalize((float3)(((float)u - cx) / fx, ((float)v - cy) / fy, 1.0f));
    float3 ray_orig_world = mat4_transform_point(c2w_matrix, (float3)(0.0f, 0.0f, 0.0f));
    float3 ray_dir_world  = mat4_transform_direction(c2w_matrix, ray_dir_cam);

    Ray3D ray = make_ray3d(ray_orig_world, ray_dir_world, 0.01f, 100.0f);

    float3 box_min = (float3)(
        -0.5f * (float)(volume_dims.x - 1) * voxel_size.x,
        -0.5f * (float)(volume_dims.y - 1) * voxel_size.y,
        -0.5f * (float)(volume_dims.z - 1) * voxel_size.z
    );
    float3 box_max = -box_min;

    float t_near = 0.0f, t_far = 0.0f;
    if (!intersect_aabb(&ray, box_min, box_max, &t_near, &t_far)) {
        out_framebuffer[pixel_idx] = (float4)(0.0f, 0.0f, 0.0f, 0.0f); // Background
        return;
    }

    float3 accum_color = (float3)(0.0f, 0.0f, 0.0f);
    float accum_alpha = 0.0f;
    float3 L = normalize(light_dir_world);

    // Front-to-back ray compositing
    for (float t = t_near; t <= t_far; t += step_size) {
        float3 p_world = ray.origin + t * ray.direction;
        float3 p_voxel = (p_world - box_min) / voxel_size;

        float density = read_volume_3d_safe(
            volume_data,
            (int)p_voxel.x, (int)p_voxel.y, (int)p_voxel.z,
            volume_dims.x, volume_dims.y, volume_dims.z,
            BORDER_CLAMP_EDGE, 0.0f);

        if (density > 0.01f) {
            float4 sample_color = sample_transfer_function(transfer_function_lut, density, lut_size);
            float alpha = sample_color.w;

            if (alpha > 0.005f) {
                // Surface normal from gradient for diffuse lighting
                float3 normal = compute_volume_gradient(volume_data, p_voxel, volume_dims);
                float NdotL = fmax(0.15f, dot(normal, L)); // Diffuse + ambient term

                float3 shaded_rgb = (float3)(sample_color.x, sample_color.y, sample_color.z) * NdotL;

                // Front-to-back opacity accumulation
                float weight = (1.0f - accum_alpha) * alpha;
                accum_color += shaded_rgb * weight;
                accum_alpha += weight;

                // Early ray termination
                if (accum_alpha >= 0.99f) {
                    accum_alpha = 1.0f;
                    break;
                }
            }
        }
    }

    out_framebuffer[pixel_idx] = (float4)(accum_color.x, accum_color.y, accum_color.z, accum_alpha);
}
