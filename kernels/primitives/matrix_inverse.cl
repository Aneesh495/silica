#include "common/math.clh"

// Batched 3x3 Matrix Inversion via Analytical Adjugate and Determinant
__kernel void kernel_batched_matrix_inverse_3x3(
    __global const float* in_matrices, // [num_matrices x 9]
    __global float* out_inverses,      // [num_matrices x 9]
    __global uchar* out_is_invertible,
    const uint num_matrices) {
    
    uint gid = get_global_id(0);
    if (gid >= num_matrices) return;

    size_t base = (size_t)gid * 9;

    float m00 = in_matrices[base + 0], m01 = in_matrices[base + 1], m02 = in_matrices[base + 2];
    float m10 = in_matrices[base + 3], m11 = in_matrices[base + 4], m12 = in_matrices[base + 5];
    float m20 = in_matrices[base + 6], m21 = in_matrices[base + 7], m22 = in_matrices[base + 8];

    // Compute co-factors
    float c00 = m11 * m22 - m12 * m21;
    float c01 = m12 * m20 - m10 * m22;
    float c02 = m10 * m21 - m11 * m20;

    float c10 = m02 * m21 - m01 * m22;
    float c11 = m00 * m22 - m02 * m20;
    float c12 = m01 * m20 - m00 * m21;

    float c20 = m01 * m12 - m02 * m11;
    float c21 = m02 * m10 - m00 * m12;
    float c22 = m00 * m11 - m01 * m10;

    float det = m00 * c00 + m01 * c01 + m02 * c02;

    if (fabs(det) < 1.0e-7f) {
        out_is_invertible[gid] = 0;
        for (int i = 0; i < 9; ++i) out_inverses[base + i] = 0.0f;
        return;
    }

    float inv_det = 1.0f / det;
    out_is_invertible[gid] = 1;

    // Transposed co-factor matrix scaled by 1/det
    out_inverses[base + 0] = c00 * inv_det;
    out_inverses[base + 1] = c10 * inv_det;
    out_inverses[base + 2] = c20 * inv_det;

    out_inverses[base + 3] = c01 * inv_det;
    out_inverses[base + 4] = c11 * inv_det;
    out_inverses[base + 5] = c21 * inv_det;

    out_inverses[base + 6] = c02 * inv_det;
    out_inverses[base + 7] = c12 * inv_det;
    out_inverses[base + 8] = c22 * inv_det;
}

// Batched 4x4 Matrix Inversion via Laplace Expansion
__kernel void kernel_batched_matrix_inverse_4x4(
    __global const float* in_matrices, // [num_matrices x 16]
    __global float* out_inverses,      // [num_matrices x 16]
    __global uchar* out_is_invertible,
    const uint num_matrices) {
    
    uint gid = get_global_id(0);
    if (gid >= num_matrices) return;

    size_t base = (size_t)gid * 16;
    const float* m = in_matrices + base;
    float* inv = out_inverses + base;

    float s0 = m[0] * m[5] - m[1] * m[4];
    float s1 = m[0] * m[6] - m[2] * m[4];
    float s2 = m[0] * m[7] - m[3] * m[4];
    float s3 = m[1] * m[6] - m[2] * m[5];
    float s4 = m[1] * m[7] - m[3] * m[5];
    float s5 = m[2] * m[7] - m[3] * m[6];

    float c5 = m[10] * m[15] - m[11] * m[14];
    float c4 = m[9]  * m[15] - m[11] * m[13];
    float c3 = m[9]  * m[14] - m[10] * m[13];
    float c2 = m[8]  * m[15] - m[11] * m[12];
    float c1 = m[8]  * m[14] - m[10] * m[12];
    float c0 = m[8]  * m[13] - m[9]  * m[12];

    float det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;

    if (fabs(det) < 1.0e-7f) {
        out_is_invertible[gid] = 0;
        for (int i = 0; i < 16; ++i) inv[i] = 0.0f;
        return;
    }

    float inv_det = 1.0f / det;
    out_is_invertible[gid] = 1;

    inv[0]  = ( m[5] * c5 - m[6] * c4 + m[7] * c3) * inv_det;
    inv[1]  = (-m[1] * c5 + m[2] * c4 - m[3] * c3) * inv_det;
    inv[2]  = ( m[13]* s5 - m[14]* s4 + m[15]* s3) * inv_det;
    inv[3]  = (-m[9] * s5 + m[10]* s4 - m[11]* s3) * inv_det;

    inv[4]  = (-m[4] * c5 + m[6] * c2 - m[7] * c1) * inv_det;
    inv[5]  = ( m[0] * c5 - m[2] * c2 + m[3] * c1) * inv_det;
    inv[6]  = (-m[12]* s5 + m[14]* s2 - m[15]* s1) * inv_det;
    inv[7]  = ( m[8] * s5 - m[10]* s2 + m[11]* s1) * inv_det;

    inv[8]  = ( m[4] * c4 - m[5] * c2 + m[7] * c0) * inv_det;
    inv[9]  = (-m[0] * c4 + m[1] * c2 - m[3] * c0) * inv_det;
    inv[10] = ( m[12]* s4 - m[13]* s2 + m[15]* s0) * inv_det;
    inv[11] = (-m[8] * s4 + m[9] * s2 - m[11]* s0) * inv_det;

    inv[12] = (-m[4] * c3 + m[5] * c1 - m[6] * c0) * inv_det;
    inv[13] = ( m[0] * c3 - m[1] * c1 + m[2] * c0) * inv_det;
    inv[14] = (-m[12]* s3 + m[13]* s1 - m[14]* s0) * inv_det;
    inv[15] = ( m[8] * s3 - m[9] * s1 + m[10]* s0) * inv_det;
}
