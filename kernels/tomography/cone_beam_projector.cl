#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

// Forward 3D Cone-Beam Ray-Tracing Projector
// Projects 3D voxel volume onto 2D flat panel detector for gantry angle beta.
__kernel void kernel_forward_cone_beam_projector(
    __global const float* volume_grid,     // [vz x vy x vx]
    __global float* out_projection_panel,  // [det_h x det_w]
    const int vx,
    const int vy,
    const int vz,
    const int det_w,
    const int det_h,
    const float beta,                      // Gantry rotation angle in radians
    const float source_to_isocenter_dist,  // Dso
    const float source_to_detector_dist,   // Dsd
    const float voxel_size,
    const float det_pixel_size) {
    
    int u = get_global_id(0);
    int v = get_global_id(1);

    if (u >= det_w || v >= det_h) return;

    size_t out_pixel = (size_t)v * (size_t)det_w + (size_t)u;

    float Dso = source_to_isocenter_dist;
    float Dsd = source_to_detector_dist;
    float Ddo = Dsd - Dso; // Isocenter to detector

    float cos_b = cos(beta);
    float sin_b = sin(beta);

    // X-ray source position in 3D world space
    float3 source_pos = (float3)(-Dso * sin_b, Dso * cos_b, 0.0f);

    // Detector coordinate in detector local frame
    float half_u = 0.5f * (float)(det_w - 1);
    float half_v = 0.5f * (float)(det_h - 1);
    float det_u = ((float)u - half_u) * det_pixel_size;
    float det_v = ((float)v - half_v) * det_pixel_size;

    // Detector pixel position in 3D world space
    float3 det_pos = (float3)(
        Ddo * sin_b + det_u * cos_b,
        -Ddo * cos_b + det_u * sin_b,
        det_v
    );

    // Ray from source through detector pixel
    float3 ray_dir = normalize(det_pos - source_pos);
    Ray3D ray = make_ray3d(source_pos, ray_dir, 0.1f, Dsd * 2.0f);

    float3 box_min = (float3)(
        -0.5f * (float)(vx - 1) * voxel_size,
        -0.5f * (float)(vy - 1) * voxel_size,
        -0.5f * (float)(vz - 1) * voxel_size
    );
    float3 box_max = -box_min;

    float t_near = 0.0f, t_far = 0.0f;
    if (!intersect_aabb(&ray, box_min, box_max, &t_near, &t_far)) {
        out_projection_panel[out_pixel] = 0.0f;
        return;
    }

    // Line integration through voxel grid
    float step = voxel_size * 0.5f;
    float line_integral = 0.0f;

    for (float t = t_near; t <= t_far; t += step) {
        float3 p_world = ray.origin + t * ray.direction;
        float3 p_voxel = (p_world - box_min) / voxel_size;

        int ix = (int)floor(p_voxel.x);
        int iy = (int)floor(p_voxel.y);
        int iz = (int)floor(p_voxel.z);

        if (ix >= 0 && ix < vx && iy >= 0 && iy < vy && iz >= 0 && iz < vz) {
            float val = volume_grid[(((size_t)iz * (size_t)vy + (size_t)iy) * (size_t)vx) + (size_t)ix];
            line_integral += val;
        }
    }

    out_projection_panel[out_pixel] = line_integral * step;
}
