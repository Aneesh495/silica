#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

#define PROJ_MIP   0
#define PROJ_MINIP 1
#define PROJ_AIP   2

// Intensity Projections: MIP (Max), MinIP (Min), and AIP (Average)
// Used in CT angiography, pulmonary assessment, and vascular radiology.
__kernel void kernel_intensity_projection_raycast(
    __global const float* volume_data,
    __global float* out_projection_image,
    __global float* out_depth_image,
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
    const int projection_mode) {
    
    int u = get_global_id(0);
    int v = get_global_id(1);

    if (u >= img_w || v >= img_h) return;

    size_t pixel_idx = (size_t)v * (size_t)img_w + (size_t)u;

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
        out_projection_image[pixel_idx] = 0.0f;
        if (out_depth_image) out_depth_image[pixel_idx] = 0.0f;
        return;
    }

    float max_intensity = -LARGE_FLOAT_F;
    float min_intensity =  LARGE_FLOAT_F;
    float sum_intensity =  0.0f;
    int step_count = 0;

    float best_t = t_near;

    for (float t = t_near; t <= t_far; t += step_size) {
        float3 p_world = ray.origin + t * ray.direction;
        float3 p_voxel = (p_world - box_min) / voxel_size;

        float density = read_volume_3d_safe(
            volume_data,
            (int)p_voxel.x, (int)p_voxel.y, (int)p_voxel.z,
            volume_dims.x, volume_dims.y, volume_dims.z,
            BORDER_CLAMP_EDGE, 0.0f);

        if (density > max_intensity) {
            max_intensity = density;
            if (projection_mode == PROJ_MIP) best_t = t;
        }

        if (density < min_intensity) {
            min_intensity = density;
            if (projection_mode == PROJ_MINIP) best_t = t;
        }

        sum_intensity += density;
        step_count++;
    }

    float result = 0.0f;
    if (projection_mode == PROJ_MIP) {
        result = (max_intensity > -LARGE_FLOAT_F) ? max_intensity : 0.0f;
    } else if (projection_mode == PROJ_MINIP) {
        result = (min_intensity < LARGE_FLOAT_F) ? min_intensity : 0.0f;
    } else {
        result = (step_count > 0) ? (sum_intensity / (float)step_count) : 0.0f;
    }

    out_projection_image[pixel_idx] = result;
    if (out_depth_image) {
        out_depth_image[pixel_idx] = best_t;
    }
}
