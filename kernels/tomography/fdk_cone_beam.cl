#include "common/math.clh"
#include "common/memory.clh"

// FDK Stage 1: Pre-weight 2D cone-beam projection frame by source-distance cosine
__kernel void kernel_fdk_cosine_weight_projections(
    __global const float* in_projections,
    __global float* out_weighted_projections,
    const int det_w,
    const int det_h,
    const int num_projections,
    const float source_to_detector_dist,
    const float det_pixel_size_u,
    const float det_pixel_size_v) {
    
    int u = get_global_id(0);
    int v = get_global_id(1);
    int proj_idx = get_global_id(2);

    if (u >= det_w || v >= det_h || proj_idx >= num_projections) return;

    float half_u = 0.5f * (float)(det_w - 1);
    float half_v = 0.5f * (float)(det_h - 1);

    float pu = ((float)u - half_u) * det_pixel_size_u;
    float pv = ((float)v - half_v) * det_pixel_size_v;
    float D = source_to_detector_dist;

    float cosine_weight = D / sqrt(D * D + pu * pu + pv * pv);

    size_t idx = (((size_t)proj_idx * (size_t)det_h + (size_t)v) * (size_t)det_w) + (size_t)u;
    out_weighted_projections[idx] = in_projections[idx] * cosine_weight;
}

// FDK Stage 2: 3D Cone-Beam Back-Projection into Volumetric Voxel Grid
// Voxel grid dimension: vx x vy x vz
__kernel void kernel_fdk_cone_beam_backproject_3d(
    __global const float* filtered_projections, // [num_proj x det_h x det_w]
    __global float* out_volume_grid,            // [vz x vy x vx]
    const int vx,
    const int vy,
    const int vz,
    const int det_w,
    const int det_h,
    const int num_projections,
    const float source_to_origin_dist,   // Dso
    const float source_to_detector_dist, // Dsd
    const float voxel_size,
    const float det_pixel_size) {
    
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if (ix >= vx || iy >= vy || iz >= vz) return;

    // Voxel coordinates centered at origin (0, 0, 0)
    float x = ((float)ix - 0.5f * (float)(vx - 1)) * voxel_size;
    float y = ((float)iy - 0.5f * (float)(vy - 1)) * voxel_size;
    float z = ((float)iz - 0.5f * (float)(vz - 1)) * voxel_size;

    float Dso = source_to_origin_dist;
    float Dsd = source_to_detector_dist;
    float d_beta = TWO_PI_F / (float)num_projections;

    float half_u = 0.5f * (float)(det_w - 1);
    float half_v = 0.5f * (float)(det_h - 1);
    float inv_det_size = 1.0f / det_pixel_size;

    float voxel_sum = 0.0f;

    for (int p = 0; p < num_projections; ++p) {
        float beta = (float)p * d_beta;
        float cos_b = cos(beta);
        float sin_b = sin(beta);

        // Distance along central ray
        float U = (Dso - x * sin_b + y * cos_b) / Dso;
        if (U <= 0.05f) continue; // Behind source or too close

        float inv_U = 1.0f / U;

        // Perspective projection onto detector coordinates
        float det_u_world = (Dsd / Dso) * (x * cos_b + y * sin_b) * inv_U;
        float det_v_world = (Dsd / Dso) * z * inv_U;

        float u_px = det_u_world * inv_det_size + half_u;
        float v_px = det_v_world * inv_det_size + half_v;

        int u0 = (int)floor(u_px);
        int v0 = (int)floor(v_px);
        int u1 = u0 + 1;
        int v1 = v0 + 1;

        if (u0 >= 0 && u1 < det_w && v0 >= 0 && v1 < det_h) {
            float fu = u_px - (float)u0;
            float fv = v_px - (float)v0;

            size_t frame_base = (size_t)p * (size_t)det_h * (size_t)det_w;
            float val00 = filtered_projections[frame_base + (size_t)v0 * (size_t)det_w + (size_t)u0];
            float val10 = filtered_projections[frame_base + (size_t)v0 * (size_t)det_w + (size_t)u1];
            float val01 = filtered_projections[frame_base + (size_t)v1 * (size_t)det_w + (size_t)u0];
            float val11 = filtered_projections[frame_base + (size_t)v1 * (size_t)det_w + (size_t)u1];

            float interp = bilinear_interp_unit(val00, val10, val01, val11, fu, fv);
            voxel_sum += interp * (inv_U * inv_U);
        }
    }

    size_t vol_idx = (((size_t)iz * (size_t)vy + (size_t)iy) * (size_t)vx) + (size_t)ix;
    out_volume_grid[vol_idx] = voxel_sum * d_beta;
}
