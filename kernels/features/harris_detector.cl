#include "common/math.clh"
#include "common/memory.clh"

#define HARRIS_RADIUS 2

// Harris and Shi-Tomasi Structure Tensor Corner Response
__kernel void kernel_harris_response(
    __global const float* grad_x,
    __global const float* grad_y,
    __global float* out_response,
    const int width,
    const int height,
    const float k_param,
    const int method) { // 0 = Harris, 1 = Shi-Tomasi, 2 = Noble
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < HARRIS_RADIUS || x >= width - HARRIS_RADIUS ||
        y < HARRIS_RADIUS || y >= height - HARRIS_RADIUS) {
        if (x < width && y < height) {
            out_response[(size_t)y * (size_t)width + (size_t)x] = 0.0f;
        }
        return;
    }

    // Accumulate structure tensor moments over window W
    float A = 0.0f; // Ix^2
    float B = 0.0f; // Iy^2
    float C = 0.0f; // Ix * Iy

    for (int dy = -HARRIS_RADIUS; dy <= HARRIS_RADIUS; ++dy) {
        for (int dx = -HARRIS_RADIUS; dx <= HARRIS_RADIUS; ++dx) {
            size_t idx = (size_t)(y + dy) * (size_t)width + (size_t)(x + dx);
            float gx = grad_x[idx];
            float gy = grad_y[idx];

            A += gx * gx;
            B += gy * gy;
            C += gx * gy;
        }
    }

    float det = A * B - C * C;
    float trace = A + B;
    float response = 0.0f;

    if (method == 0) {
        // Harris: det - k * trace^2
        response = det - k_param * (trace * trace);
    } else if (method == 1) {
        // Shi-Tomasi: minimum eigenvalue lambda_2
        // Characteristic equation: lambda^2 - trace*lambda + det = 0
        float disc = sqrt(fmax(0.0f, (A - B) * (A - B) + 4.0f * C * C));
        response = 0.5f * (trace - disc);
    } else {
        // Noble: det / (trace + eps)
        response = det / (trace + EPSILON_F);
    }

    out_response[(size_t)y * (size_t)width + (size_t)x] = fmax(0.0f, response);
}
