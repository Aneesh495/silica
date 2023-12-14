#include "common/math.clh"

// Batched 3x3 Symmetric Eigenvalue and Eigenvector Solver via Cyclic Jacobi Rotations
// Used for PCA covariance analysis, normal estimation, and structure tensor analysis.
// Solves: A * v_i = lambda_i * v_i for symmetric 3x3 matrix A.
__kernel void kernel_batched_eigen_3x3_symm(
    __global const float* in_cov_matrices, // 9 floats per system (symmetric)
    __global float* out_eigenvalues,       // 3 floats per system (ascending order)
    __global float* out_eigenvectors,      // 9 floats per system (columns are eigenvectors)
    const uint total_systems) {
    
    uint gid = get_global_id(0);
    if (gid >= total_systems) return;

    uint in_idx = gid * 9;
    uint val_idx = gid * 3;
    uint vec_idx = gid * 9;

    // Load 3x3 matrix into thread-private registers
    float A[3][3];
    A[0][0] = in_cov_matrices[in_idx + 0];
    A[0][1] = in_cov_matrices[in_idx + 1];
    A[0][2] = in_cov_matrices[in_idx + 2];
    A[1][0] = in_cov_matrices[in_idx + 3];
    A[1][1] = in_cov_matrices[in_idx + 4];
    A[1][2] = in_cov_matrices[in_idx + 5];
    A[2][0] = in_cov_matrices[in_idx + 6];
    A[2][1] = in_cov_matrices[in_idx + 7];
    A[2][2] = in_cov_matrices[in_idx + 8];

    // Initialize V as 3x3 Identity matrix
    float V[3][3] = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };

    // Up to 15 sweeps of cyclic Jacobi rotations
    const int MAX_SWEEPS = 15;
    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        float off_diag_norm = fabs(A[0][1]) + fabs(A[0][2]) + fabs(A[1][2]);
        if (off_diag_norm < 1.0e-6f) {
            break;
        }

        // Rotate pairs (p, q): (0,1), (0,2), (1,2)
        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                float app = A[p][p];
                float aqq = A[q][q];
                float apq = A[p][q];

                if (fabs(apq) < 1.0e-7f) continue;

                float tau = (aqq - app) / (2.0f * apq);
                float t;
                if (tau >= 0.0f) {
                    t = 1.0f / (tau + sqrt(1.0f + tau * tau));
                } else {
                    t = -1.0f / (-tau + sqrt(1.0f + tau * tau));
                }

                float c = 1.0f / sqrt(1.0f + t * t);
                float s = t * c;
                float theta_h = t * apq;

                A[p][p] = app - theta_h;
                A[q][q] = aqq + theta_h;
                A[p][q] = 0.0f;
                A[q][p] = 0.0f;

                // Update remaining off-diagonals
                for (int r = 0; r < 3; ++r) {
                    if (r != p && r != q) {
                        float arp = A[r][p];
                        float arq = A[r][q];
                        A[r][p] = c * arp - s * arq;
                        A[p][r] = A[r][p];
                        A[r][q] = s * arp + c * arq;
                        A[q][r] = A[r][q];
                    }
                }

                // Accumulate eigenvectors in V
                for (int r = 0; r < 3; ++r) {
                    float vrp = V[r][p];
                    float vrq = V[r][q];
                    V[r][p] = c * vrp - s * vrq;
                    V[r][q] = s * vrp + c * vrq;
                }
            }
        }
    }

    // Sort eigenvalues and corresponding eigenvector columns ascending
    float evals[3] = {A[0][0], A[1][1], A[2][2]};
    int order[3] = {0, 1, 2};

    if (evals[order[0]] > evals[order[1]]) { int tmp = order[0]; order[0] = order[1]; order[1] = tmp; }
    if (evals[order[1]] > evals[order[2]]) { int tmp = order[1]; order[1] = order[2]; order[2] = tmp; }
    if (evals[order[0]] > evals[order[1]]) { int tmp = order[0]; order[0] = order[1]; order[1] = tmp; }

    out_eigenvalues[val_idx + 0] = evals[order[0]];
    out_eigenvalues[val_idx + 1] = evals[order[1]];
    out_eigenvalues[val_idx + 2] = evals[order[2]];

    for (int r = 0; r < 3; ++r) {
        out_eigenvectors[vec_idx + r * 3 + 0] = V[r][order[0]];
        out_eigenvectors[vec_idx + r * 3 + 1] = V[r][order[1]];
        out_eigenvectors[vec_idx + r * 3 + 2] = V[r][order[2]];
    }
}

// Batched 6x6 Cholesky Solver: A * x = b
// Where A is symmetric positive-definite (JT * J) and b is JT * r from Gauss-Newton ICP.
__kernel void kernel_batched_cholesky_solve_6x6(
    __global const float* in_A_matrices, // 36 floats per system
    __global const float* in_b_vectors,  // 6 floats per system
    __global float* out_x_vectors,       // 6 floats per system
    const uint total_systems) {
    
    uint gid = get_global_id(0);
    if (gid >= total_systems) return;

    uint a_base = gid * 36;
    uint b_base = gid * 6;

    float L[6][6];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            L[i][j] = 0.0f;
        }
    }

    // Cholesky decomposition: L * L^T = A
    bool success = true;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }

            if (i == j) {
                float diag = in_A_matrices[a_base + i * 6 + i] - sum;
                if (diag <= 1.0e-7f) {
                    diag = 1.0e-7f; // Regularization
                }
                L[i][j] = sqrt(diag);
            } else {
                L[i][j] = (in_A_matrices[a_base + i * 6 + j] - sum) / (L[j][j] + EPSILON_F);
            }
        }
    }

    // Forward substitution: L * y = b
    float y[6];
    for (int i = 0; i < 6; ++i) {
        float sum = 0.0f;
        for (int k = 0; k < i; ++k) {
            sum += L[i][k] * y[k];
        }
        y[i] = (in_b_vectors[b_base + i] - sum) / (L[i][i] + EPSILON_F);
    }

    // Back substitution: L^T * x = y
    float x[6];
    for (int i = 5; i >= 0; --i) {
        float sum = 0.0f;
        for (int k = i + 1; k < 6; ++k) {
            sum += L[k][i] * x[k];
        }
        x[i] = (y[i] - sum) / (L[i][i] + EPSILON_F);
        out_x_vectors[b_base + i] = x[i];
    }
}
