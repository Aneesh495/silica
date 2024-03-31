#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

// Helical 180-Degree Linear Interpolation (180LI)
// Synthesizes planar slice projections from helical scanning trajectory with table pitch.
__kernel void kernel_helical_180li_interpolate(
    __global const float* helical_projections, // [total_turns x num_views x det_w]
    __global float* out_planar_sinogram,       // [num_views x det_w] for target z_slice
    const int num_views,
    const int det_w,
    const float target_z,
    const float table_feed_per_rotation,
    const float total_helical_turns) {
    
    int u = get_global_id(0);
    int v = get_global_id(1); // view index: 0 .. num_views - 1

    if (u >= det_w || v >= num_views) return;

    float d_beta = TWO_PI_F / (float)num_views;
    float beta = (float)v * d_beta;

    // Find turns where gantry angle matches beta or beta + pi (opposite ray)
    float z_per_turn = table_feed_per_rotation;
    float float_turn = (target_z - (beta / TWO_PI_F) * z_per_turn) / z_per_turn;

    int turn0 = (int)floor(float_turn);
    int turn1 = turn0 + 1;
    float frac = float_turn - (float)turn0;

    int total_t = (int)total_helical_turns;
    float val = 0.0f;

    if (turn0 >= 0 && turn1 < total_t) {
        size_t idx0 = (((size_t)turn0 * (size_t)num_views + (size_t)v) * (size_t)det_w) + (size_t)u;
        size_t idx1 = (((size_t)turn1 * (size_t)num_views + (size_t)v) * (size_t)det_w) + (size_t)u;

        val = lerp_f(helical_projections[idx0], helical_projections[idx1], frac);
    } else if (turn0 >= 0 && turn0 < total_t) {
        size_t idx0 = (((size_t)turn0 * (size_t)num_views + (size_t)v) * (size_t)det_w) + (size_t)u;
        val = helical_projections[idx0];
    }

    size_t out_idx = (size_t)v * (size_t)det_w + (size_t)u;
    out_planar_sinogram[out_idx] = val;
}
