#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

// Trilinear TSDF Grid Interpolation at Continuous Voxel Position
inline float sample_tsdf_trilinear(
    __global const float* tsdf_grid,
    float3 p_voxel,
    int3 dims) {
    
    int x0 = (int)floor(p_voxel.x);
    int y0 = (int)floor(p_voxel.y);
    int z0 = (int)floor(p_voxel.z);

    if (x0 < 0 || x0 >= dims.x - 1 ||
        y0 < 0 || y0 >= dims.y - 1 ||
        z0 < 0 || z0 >= dims.z - 1) {
        return 1.0f;
    }

    float fx = p_voxel.x - (float)x0;
    float fy = p_voxel.y - (float)y0;
    float fz = p_voxel.z - (float)z0;

    int sx = dims.x;
    int sy = dims.y;

    float c000 = tsdf_grid[((size_t)z0 * sy + y0) * sx + x0];
    float c100 = tsdf_grid[((size_t)z0 * sy + y0) * sx + (x0 + 1)];
    float c010 = tsdf_grid[((size_t)z0 * sy + (y0 + 1)) * sx + x0];
    float c110 = tsdf_grid[((size_t)z0 * sy + (y0 + 1)) * sx + (x0 + 1)];

    float c001 = tsdf_grid[((size_t)(z0 + 1) * sy + y0) * sx + x0];
    float c101 = tsdf_grid[((size_t)(z0 + 1) * sy + y0) * sx + (x0 + 1)];
    float c011 = tsdf_grid[((size_t)(z0 + 1) * sy + (y0 + 1)) * sx + x0];
    float c111 = tsdf_grid[((size_t)(z0 + 1) * sy + (y0 + 1)) * sx + (x0 + 1)];

    return trilinear_interp_unit(c000, c100, c010, c110, c001, c101, c011, c111, fx, fy, fz);
}

// Computes Analytical Surface Normal via Central Difference TSDF Gradient
inline float3 compute_tsdf_gradient(
    __global const float* tsdf_grid,
    float3 p_voxel,
    int3 dims) {
    
    float step = 1.0f;
    float dx = sample_tsdf_trilinear(tsdf_grid, p_voxel + (float3)(step, 0.0f, 0.0f), dims) -
               sample_tsdf_trilinear(tsdf_grid, p_voxel - (float3)(step, 0.0f, 0.0f), dims);
    float dy = sample_tsdf_trilinear(tsdf_grid, p_voxel + (float3)(0.0f, step, 0.0f), dims) -
               sample_tsdf_trilinear(tsdf_grid, p_voxel - (float3)(0.0f, step, 0.0f), dims);
    float dz = sample_tsdf_trilinear(tsdf_grid, p_voxel + (float3)(0.0f, 0.0f, step), dims) -
               sample_tsdf_trilinear(tsdf_grid, p_voxel - (float3)(0.0f, 0.0f, step), dims);

    float3 grad = (float3)(dx, dy, dz);
    float len = length(grad);
    return len > 1.0e-5f ? (grad / len) : (float3)(0.0f, 0.0f, 1.0f);
}

// TSDF Volumetric Raycaster and Sphere Tracer
__kernel void kernel_tsdf_raycast(
    __global const float* tsdf_grid,
    __global float* out_depth_image,
    __global float4* out_normal_image,
    __constant const float* c2w_matrix,
    const int img_w,
    const int img_h,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const int vx,
    const int vy,
    const int vz,
    const float voxel_size,
    const float truncation_mu) {
    
    int u = get_global_id(0);
    int v = get_global_id(1);

    if (u >= img_w || v >= img_h) return;

    size_t out_pixel = (size_t)v * (size_t)img_w + (size_t)u;

    // Generate camera ray in world space
    float3 ray_dir_cam = normalize((float3)(((float)u - cx) / fx, ((float)v - cy) / fy, 1.0f));
    float3 ray_orig_world = mat4_transform_point_const(c2w_matrix, (float3)(0.0f, 0.0f, 0.0f));
    float3 ray_dir_world  = mat4_transform_direction_const(c2w_matrix, ray_dir_cam);

    Ray3D ray = make_ray3d(ray_orig_world, ray_dir_world, 0.1f, 10.0f);

    float3 box_min = (float3)(
        -0.5f * (float)(vx - 1) * voxel_size,
        -0.5f * (float)(vy - 1) * voxel_size,
        -0.5f * (float)(vz - 1) * voxel_size
    );
    float3 box_max = -box_min;

    float t_near = 0.0f;
    float t_far  = 0.0f;

    if (!intersect_aabb(&ray, box_min, box_max, &t_near, &t_far)) {
        out_depth_image[out_pixel] = 0.0f;
        out_normal_image[out_pixel] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    int3 dims = (int3)(vx, vy, vz);
    float ray_step = voxel_size * 0.8f;
    float prev_sdf = 1.0f;
    float prev_t = t_near;
    bool hit = false;
    float hit_t = 0.0f;

    for (float t = t_near; t <= t_far; t += ray_step) {
        float3 p_world = ray.origin + t * ray.direction;
        float3 p_voxel = (p_world - box_min) / voxel_size;

        float sdf = sample_tsdf_trilinear(tsdf_grid, p_voxel, dims);

        // Detect zero crossing from positive to negative
        if (prev_sdf > 0.0f && sdf < 0.0f) {
            // Secant root bisection
            hit_t = prev_t + (prev_sdf / (prev_sdf - sdf + EPSILON_F)) * (t - prev_t);
            hit = true;
            break;
        }

        prev_sdf = sdf;
        prev_t = t;
    }

    if (hit) {
        out_depth_image[out_pixel] = hit_t;

        float3 hit_world = ray.origin + hit_t * ray.direction;
        float3 hit_voxel = (hit_world - box_min) / voxel_size;
        float3 normal = compute_tsdf_gradient(tsdf_grid, hit_voxel, dims);

        out_normal_image[out_pixel] = (float4)(normal.x, normal.y, normal.z, 1.0f);
    } else {
        out_depth_image[out_pixel] = 0.0f;
        out_normal_image[out_pixel] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
}
