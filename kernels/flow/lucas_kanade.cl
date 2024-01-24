#include "common/math.clh"
#include "common/memory.clh"

#define LK_WINDOW_RADIUS 5
#define LK_MAX_ITERATIONS 10
#define LK_CONVERGENCE_EPS 0.03f

// Pyramidal Lucas-Kanade Sparse Optical Flow Tracker
// Tracks 2D feature coordinates from previous frame to current frame.
__kernel void kernel_lucas_kanade_track(
    __global const float* img_prev,
    __global const float* img_curr,
    __global const float2* in_pts_prev,
    __global float2* in_out_pts_curr,
    __global uchar* out_status, // 1 = tracked successfully, 0 = lost
    const int width,
    const int height,
    const uint num_points) {
    
    uint gid = get_global_id(0);
    if (gid >= num_points) return;

    float2 p_prev = in_pts_prev[gid];
    float2 p_curr = in_out_pts_curr[gid];

    int ix = (int)floor(p_prev.x);
    int iy = (int)floor(p_prev.y);

    if (ix < LK_WINDOW_RADIUS || ix >= width - LK_WINDOW_RADIUS - 1 ||
        iy < LK_WINDOW_RADIUS || iy >= height - LK_WINDOW_RADIUS - 1) {
        out_status[gid] = 0;
        return;
    }

    // Precompute spatial gradients and spatial Hessian matrix G on prev image
    float Gxx = 0.0f;
    float Gyy = 0.0f;
    float Gxy = 0.0f;

    for (int wy = -LK_WINDOW_RADIUS; wy <= LK_WINDOW_RADIUS; ++wy) {
        for (int wx = -LK_WINDOW_RADIUS; wx <= LK_WINDOW_RADIUS; ++wx) {
            int sx = ix + wx;
            int sy = iy + wy;

            float ix_val = 0.5f * (read_image_2d_safe(img_prev, sx + 1, sy, width, height, BORDER_CLAMP_EDGE, 0.0f) -
                                  read_image_2d_safe(img_prev, sx - 1, sy, width, height, BORDER_CLAMP_EDGE, 0.0f));
            float iy_val = 0.5f * (read_image_2d_safe(img_prev, sx, sy + 1, width, height, BORDER_CLAMP_EDGE, 0.0f) -
                                  read_image_2d_safe(img_prev, sx, sy - 1, width, height, BORDER_CLAMP_EDGE, 0.0f));

            Gxx += ix_val * ix_val;
            Gyy += iy_val * iy_val;
            Gxy += ix_val * iy_val;
        }
    }

    float det = Gxx * Gyy - Gxy * Gxy;
    if (det < 1.0e-5f) {
        out_status[gid] = 0; // Singular / aperture problem
        return;
    }

    float inv_det = 1.0f / det;

    // Iterative Newton-Raphson optical flow update
    float2 flow = p_curr - p_prev;

    bool converged = false;
    for (int iter = 0; iter < LK_MAX_ITERATIONS; ++iter) {
        float bx = 0.0f;
        float by = 0.0f;

        for (int wy = -LK_WINDOW_RADIUS; wy <= LK_WINDOW_RADIUS; ++wy) {
            for (int wx = -LK_WINDOW_RADIUS; wx <= LK_WINDOW_RADIUS; ++wx) {
                float sx_prev = p_prev.x + (float)wx;
                float sy_prev = p_prev.y + (float)wy;

                float sx_curr = p_prev.x + flow.x + (float)wx;
                float sy_curr = p_prev.y + flow.y + (float)wy;

                // Subpixel sample from prev
                int px0 = (int)floor(sx_prev);
                int py0 = (int)floor(sy_prev);
                float fpx = sx_prev - (float)px0;
                float fpy = sy_prev - (float)py0;
                float I_prev = bilinear_interp_unit(
                    read_image_2d_safe(img_prev, px0, py0, width, height, BORDER_CLAMP_EDGE, 0.0f),
                    read_image_2d_safe(img_prev, px0 + 1, py0, width, height, BORDER_CLAMP_EDGE, 0.0f),
                    read_image_2d_safe(img_prev, px0, py0 + 1, width, height, BORDER_CLAMP_EDGE, 0.0f),
                    read_image_2d_safe(img_prev, px0 + 1, py0 + 1, width, height, BORDER_CLAMP_EDGE, 0.0f),
                    fpx, fpy);

                // Subpixel sample from curr
                int cx0 = (int)floor(sx_curr);
                int cy0 = (int)floor(sy_curr);
                float fcx = sx_curr - (float)cx0;
                float fcy = sy_curr - (float)cy0;
                float I_curr = bilinear_interp_unit(
                    read_image_2d_safe(img_curr, cx0, cy0, width, height, BORDER_CLAMP_EDGE, 0.0f),
                    read_image_2d_safe(img_curr, cx0 + 1, cy0, width, height, BORDER_CLAMP_EDGE, 0.0f),
                    read_image_2d_safe(img_curr, cx0, cy0 + 1, width, height, BORDER_CLAMP_EDGE, 0.0f),
                    read_image_2d_safe(img_curr, cx0 + 1, cy0 + 1, width, height, BORDER_CLAMP_EDGE, 0.0f),
                    fcx, fcy);

                float it_val = I_prev - I_curr;

                float ix_val = 0.5f * (read_image_2d_safe(img_prev, px0 + 1, py0, width, height, BORDER_CLAMP_EDGE, 0.0f) -
                                      read_image_2d_safe(img_prev, px0 - 1, py0, width, height, BORDER_CLAMP_EDGE, 0.0f));
                float iy_val = 0.5f * (read_image_2d_safe(img_prev, px0, py0 + 1, width, height, BORDER_CLAMP_EDGE, 0.0f) -
                                      read_image_2d_safe(img_prev, px0, py0 - 1, width, height, BORDER_CLAMP_EDGE, 0.0f));

                bx += it_val * ix_val;
                by += it_val * iy_val;
            }
        }

        // Solve delta_flow = G^-1 * b
        float delta_u = inv_det * (Gyy * bx - Gxy * by);
        float delta_v = inv_det * (-Gxy * bx + Gxx * by);

        flow.x += delta_u;
        flow.y += delta_v;

        if (delta_u * delta_u + delta_v * delta_v < LK_CONVERGENCE_EPS * LK_CONVERGENCE_EPS) {
            converged = true;
            break;
        }
    }

    float2 final_p = p_prev + flow;
    if (final_p.x < 0.0f || final_p.x >= (float)width ||
        final_p.y < 0.0f || final_p.y >= (float)height) {
        out_status[gid] = 0;
    } else {
        out_status[gid] = 1;
        in_out_pts_curr[gid] = final_p;
    }
}
