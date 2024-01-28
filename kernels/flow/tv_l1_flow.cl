#include "common/math.clh"
#include "common/memory.clh"

// Primal-Dual TV-L1 Optical Flow Step: Updates Dual Variables p = (p1, p2)
// p^{n+1} = (p^n + sigma * grad(u)) / max(1, |p^n + sigma * grad(u)|)
__kernel void kernel_tv_l1_dual_step(
    __global const float2* u_flow,
    __global float4* p_dual, // (p1_x, p1_y, p2_x, p2_y) for u.x and u.y
    const int width,
    const int height,
    const float sigma) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t idx = (size_t)y * (size_t)width + (size_t)x;

    float2 u_curr = u_flow[idx];
    float2 u_right = (x + 1 < width) ? u_flow[idx + 1] : u_curr;
    float2 u_down  = (y + 1 < height) ? u_flow[idx + width] : u_curr;

    // Spatial gradients
    float2 du_dx = u_right - u_curr;
    float2 du_dy = u_down - u_curr;

    float4 p = p_dual[idx];

    // Dual step for u.x
    float px_x = p.x + sigma * du_dx.x;
    float px_y = p.y + sigma * du_dy.x;
    float norm_x = sqrt(px_x * px_x + px_y * px_y);
    float scale_x = fmax(1.0f, norm_x);
    p.x = px_x / scale_x;
    p.y = px_y / scale_x;

    // Dual step for u.y
    float py_x = p.z + sigma * du_dx.y;
    float py_y = p.w + sigma * du_dy.y;
    float norm_y = sqrt(py_x * py_x + py_y * py_y);
    float scale_y = fmax(1.0f, norm_y);
    p.z = py_x / scale_y;
    p.w = py_y / scale_y;

    p_dual[idx] = p;
}

// Primal-Dual TV-L1 Optical Flow Step: Updates Primal Variables u = (u, v)
// Computes divergence of dual vector p, evaluates residual rho(u), and applies soft-shrinkage.
__kernel void kernel_tv_l1_primal_step(
    __global const float* grad_x,
    __global const float* grad_y,
    __global const float* grad_t,
    __global const float4* p_dual,
    __global float2* u_flow,
    const int width,
    const int height,
    const float tau,
    const float lambda) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t idx = (size_t)y * (size_t)width + (size_t)x;

    // Compute divergence: div(p) = d/dx(p_x) + d/dy(p_y) using backward differences
    float4 p_curr = p_dual[idx];
    float4 p_left = (x > 0) ? p_dual[idx - 1] : (float4)(0.0f, p_curr.y, 0.0f, p_curr.w);
    float4 p_up   = (y > 0) ? p_dual[idx - width] : (float4)(p_curr.x, 0.0f, p_curr.z, 0.0f);

    float div_ux = (p_curr.x - p_left.x) + (p_curr.y - p_up.y);
    float div_uy = (p_curr.z - p_left.z) + (p_curr.w - p_up.w);

    float2 u = u_flow[idx];
    float2 v_step = u + tau * (float2)(div_ux, div_uy);

    // Data fidelity gradient and temporal residual rho
    float gx = grad_x[idx];
    float gy = grad_y[idx];
    float gt = grad_t[idx];

    float rho = gt + gx * v_step.x + gy * v_step.y;
    float grad_sq = gx * gx + gy * gy;

    // Pointwise thresholding (soft shrinkage)
    float2 delta_u = (float2)(0.0f, 0.0f);
    if (grad_sq > EPSILON_F) {
        if (rho < -tau * lambda * grad_sq) {
            delta_u = tau * lambda * (float2)(gx, gy);
        } else if (rho > tau * lambda * grad_sq) {
            delta_u = -tau * lambda * (float2)(gx, gy);
        } else {
            delta_u = -(rho / grad_sq) * (float2)(gx, gy);
        }
    }

    u_flow[idx] = v_step + delta_u;
}
