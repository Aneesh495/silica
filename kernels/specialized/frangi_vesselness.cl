#include "common/math.clh"
#include "common/memory.clh"

// Multiscale 3D Frangi Vesselness and Tubular Structure Analysis

// Analytical 3x3 Symmetric Matrix Eigenvalues (|lambda_1| <= |lambda_2| <= |lambda_3|)
inline float3 compute_symmetric_eigenvalues_3x3(float a11, float a12, float a13, float a22, float a23, float a33) {
    float p1 = a12*a12 + a13*a13 + a23*a23;
    if (p1 < 1.0e-12f) {
        // Diagonal matrix
        float l1 = a11; float l2 = a22; float l3 = a33;
        if (fabs(l1) > fabs(l2)) { float t = l1; l1 = l2; l2 = t; }
        if (fabs(l2) > fabs(l3)) { float t = l2; l2 = l3; l3 = t; }
        if (fabs(l1) > fabs(l2)) { float t = l1; l1 = l2; l2 = t; }
        return (float3)(l1, l2, l3);
    }
    float q = (a11 + a22 + a33) * (1.0f / 3.0f);
    float p2 = (a11 - q)*(a11 - q) + (a22 - q)*(a22 - q) + (a33 - q)*(a33 - q) + 2.0f * p1;
    float p = sqrt(p2 * (1.0f / 6.0f));
    float inv_p = 1.0f / p;
    // B = (1/p) * (A - q * I)
    float b11 = (a11 - q) * inv_p; float b12 = a12 * inv_p; float b13 = a13 * inv_p;
    float b22 = (a22 - q) * inv_p; float b23 = a23 * inv_p; float b33 = (a33 - q) * inv_p;
    float r = 0.5f * (b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b13 * b23) + b13 * (b12 * b23 - b13 * b22));
    float phi = (r <= -1.0f) ? (PI_F / 3.0f) : ((r >= 1.0f) ? 0.0f : (acos(r) * (1.0f / 3.0f)));
    float eig1 = q + 2.0f * p * cos(phi);
    float eig3 = q + 2.0f * p * cos(phi + (2.0f * PI_F / 3.0f));
    float eig2 = 3.0f * q - eig1 - eig3;
    // Sort by absolute magnitude
    if (fabs(eig1) > fabs(eig2)) { float t = eig1; eig1 = eig2; eig2 = t; }
    if (fabs(eig2) > fabs(eig3)) { float t = eig2; eig2 = eig3; eig3 = t; }
    if (fabs(eig1) > fabs(eig2)) { float t = eig1; eig1 = eig2; eig2 = t; }
    return (float3)(eig1, eig2, eig3);
}

// 3D Frangi Vesselness Filter Computation from Hessian components
__kernel void kernel_frangi_vesselness_3d(
    __global const float* Dxx, __global const float* Dyy, __global const float* Dzz,
    __global const float* Dxy, __global const float* Dxz, __global const float* Dyz,
    __global float* out_vesselness,
    const float alpha, const float beta, const float c_val,
    const uint total_voxels) {
    uint gid = get_global_id(0);
    if (gid >= total_voxels) return;
    float3 eigs = compute_symmetric_eigenvalues_3x3(Dxx[gid], Dxy[gid], Dxz[gid], Dyy[gid], Dyz[gid], Dzz[gid]);
    float l1 = eigs.x; float l2 = eigs.y; float l3 = eigs.z;
    // Bright tubular structures in 3D require negative l2 and l3
    if (l2 >= 0.0f || l3 >= 0.0f) {
        out_vesselness[gid] = 0.0f;
        return;
    }
    float Ra = fabs(l2) / (fabs(l3) + 1.0e-6f);
    float Rb = fabs(l1) / (sqrt(fabs(l2 * l3)) + 1.0e-6f);
    float S  = sqrt(l1 * l1 + l2 * l2 + l3 * l3);
    float term_a = 1.0f - exp(- (Ra * Ra) / (2.0f * alpha * alpha));
    float term_b = exp(- (Rb * Rb) / (2.0f * beta * beta));
    float term_c = 1.0f - exp(- (S * S) / (2.0f * c_val * c_val));
    out_vesselness[gid] = term_a * term_b * term_c;
}
