#include "common/math.clh"
#include "common/memory.clh"

#define SUBPIX_RADIUS 3
#define SUBPIX_MAX_ITERS 15
#define SUBPIX_EPS 0.005f

// Subpixel Corner Refinement via Orthogonal Gradient Dot Products
__kernel void kernel_corner_subpixel_refine(
    __global const float* img,
    __global const float2* in_coarse_corners,
    __global float2* out_refined_corners,
    const int width,
    const int height,
    const uint num_corners) {
    
    uint gid = get_global_id(0);
    if (gid >= num_corners) return;

    float2 c = in_coarse_corners[gid];

    for (int iter = 0; iter < SUBPIX_MAX_ITERS; ++iter) {
        float A = 0.0f;
        float B = 0.0f;
        float C = 0.0f;
        float bx = 0.0f;
        float by = 0.0f;

        for (int dy = -SUBPIX_RADIUS; dy <= SUBPIX_RADIUS; ++dy) {
            for (int dx = -SUBPIX_RADIUS; dx <= SUBPIX_RADIUS; ++dx) {
                float px = c.x + (float)dx;
                float py = c.y + (float)dy;

                int ix = (int)round(px);
                int iy = (int)round(py);

                if (ix >= 1 && ix < width - 1 && iy >= 1 && iy < height - 1) {
                    float gx = 0.5f * (img[(size_t)iy * width + (ix + 1)] - img[(size_t)iy * width + (ix - 1)]);
                    float gy = 0.5f * (img[(size_t)(iy + 1) * width + ix] - img[(size_t)(iy - 1) * width + ix]);

                    A += gx * gx;
                    B += gy * gy;
                    C += gx * gy;

                    bx += (gx * gx * px + gx * gy * py);
                    by += (gx * gy * px + gy * gy * py);
                }
            }
        }

        float det = A * B - C * C;
        if (fabs(det) < 1.0e-5f) break;

        float inv_det = 1.0f / det;
        float next_cx = inv_det * (B * bx - C * by);
        float next_cy = inv_det * (-C * bx + A * by);

        float shift_x = next_cx - c.x;
        float shift_y = next_cy - c.y;

        c.x = next_cx;
        c.y = next_cy;

        if (shift_x * shift_x + shift_y * shift_y < SUBPIX_EPS * SUBPIX_EPS) {
            break;
        }
    }

    out_refined_corners[gid] = c;
}
