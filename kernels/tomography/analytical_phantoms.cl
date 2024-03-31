#include "common/math.clh"
#include "common/vector.clh"
#include "common/memory.clh"

// Analytical Ellipsoid Descriptor for 3D Shepp-Logan Phantom
typedef struct {
    float3 center;
    float3 radius;
    float3 euler_angles; // (phi, theta, psi) in radians
    float density;
} Ellipsoid3D;

// 10 Standard Analytical Ellipsoids forming the 3D Shepp-Logan Phantom
__constant Ellipsoid3D SHEPP_LOGAN_ELLIPSOIDS[10] = {
    // 0: Outer Skull
    {(float3)( 0.0f,     0.0f,      0.0f),    (float3)(0.6900f, 0.920f, 0.900f), (float3)(0.0f, 0.0f, 0.0f),  2.0f},
    // 1: Inner Skull / Brain Parenchyma
    {(float3)( 0.0f,    -0.0184f,   0.0f),    (float3)(0.6624f, 0.874f, 0.880f), (float3)(0.0f, 0.0f, 0.0f), -0.98f},
    // 2: Left Ventricle (oblique)
    {(float3)( 0.22f,    0.0f,     -0.25f),   (float3)(0.4100f, 0.160f, 0.210f), (float3)(0.0f, 0.0f, -1.885f), -0.02f},
    // 3: Right Ventricle (oblique)
    {(float3)(-0.22f,    0.0f,     -0.25f),   (float3)(0.3100f, 0.110f, 0.220f), (float3)(0.0f, 0.0f,  1.885f), -0.02f},
    // 4: Central Brain Soft Tissue
    {(float3)( 0.0f,     0.35f,    -0.25f),   (float3)(0.2100f, 0.250f, 0.350f), (float3)(0.0f, 0.0f,  0.0f),    0.01f},
    // 5: Small Tumor 1
    {(float3)( 0.0f,     0.10f,    -0.25f),   (float3)(0.0460f, 0.046f, 0.046f), (float3)(0.0f, 0.0f,  0.0f),    0.01f},
    // 6: Small Tumor 2
    {(float3)( 0.0f,    -0.10f,    -0.25f),   (float3)(0.0460f, 0.046f, 0.046f), (float3)(0.0f, 0.0f,  0.0f),    0.01f},
    // 7: Tumor cluster A
    {(float3)(-0.08f,   -0.605f,    0.0f),    (float3)(0.0460f, 0.023f, 0.020f), (float3)(0.0f, 0.0f,  0.0f),    0.01f},
    // 8: Tumor cluster B
    {(float3)( 0.0f,    -0.605f,    0.0f),    (float3)(0.0230f, 0.023f, 0.020f), (float3)(0.0f, 0.0f,  0.0f),    0.01f},
    // 9: Tumor cluster C
    {(float3)( 0.06f,   -0.605f,    0.0f),    (float3)(0.0460f, 0.023f, 0.020f), (float3)(0.0f, 0.0f,  0.0f),    0.01f}
};

// Generates 3D Voxel Grid of Analytical Shepp-Logan Phantom
__kernel void kernel_generate_shepp_logan_3d(
    __global float* out_volume_grid,
    const int vx,
    const int vy,
    const int vz,
    const float voxel_size) {
    
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if (ix >= vx || iy >= vy || iz >= vz) return;

    // Normalized physical coordinates in [-1, 1]
    float half_vx = 0.5f * (float)(vx - 1);
    float half_vy = 0.5f * (float)(vy - 1);
    float half_vz = 0.5f * (float)(vz - 1);

    float px = ((float)ix - half_vx) / half_vx;
    float py = ((float)iy - half_vy) / half_vy;
    float pz = ((float)iz - half_vz) / half_vz;

    float3 pt = (float3)(px, py, pz);
    float total_density = 0.0f;

    for (int e = 0; e < 10; ++e) {
        Ellipsoid3D ell = SHEPP_LOGAN_ELLIPSOIDS[e];

        float3 d = pt - ell.center;

        // Rotate point by inverse Euler rotation around Z
        float cos_psi = cos(-ell.euler_angles.z);
        float sin_psi = sin(-ell.euler_angles.z);
        float rx = d.x * cos_psi - d.y * sin_psi;
        float ry = d.x * sin_psi + d.y * cos_psi;
        float rz = d.z;

        // Quadratic ellipsoid equation: (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1.0
        float val = (rx * rx) / (ell.radius.x * ell.radius.x) +
                    (ry * ry) / (ell.radius.y * ell.radius.y) +
                    (rz * rz) / (ell.radius.z * ell.radius.z);

        if (val <= 1.0f) {
            total_density += ell.density;
        }
    }

    size_t vol_idx = (((size_t)iz * (size_t)vy + (size_t)iy) * (size_t)vx) + (size_t)ix;
    out_volume_grid[vol_idx] = total_density;
}

// Generates Exact Analytical 2D Sinogram from 2D Shepp-Logan Ellipses
// Closed form: integral of ray through ellipse is 2 * rho * a * b / d^2 * sqrt(d^2 - s^2)
__kernel void kernel_generate_analytical_sinogram(
    __global float* out_sinogram,
    const int num_angles,
    const int num_detectors,
    const float detector_spacing) {
    
    int det_idx = get_global_id(0);
    int angle_idx = get_global_id(1);

    if (det_idx >= num_detectors || angle_idx >= num_angles) return;

    float theta = (float)angle_idx * PI_F / (float)num_angles;
    float s = ((float)det_idx - 0.5f * (float)(num_detectors - 1)) * detector_spacing;

    float sum_proj = 0.0f;

    for (int e = 0; e < 10; ++e) {
        Ellipsoid3D ell = SHEPP_LOGAN_ELLIPSOIDS[e];
        float a = ell.radius.x;
        float b = ell.radius.y;
        float x0 = ell.center.x;
        float y0 = ell.center.y;
        float alpha = ell.euler_angles.z;

        float theta_rel = theta - alpha;
        float cos_rel = cos(theta_rel);
        float sin_rel = sin(theta_rel);

        float a2 = a * a;
        float b2 = b * b;
        float s_center = x0 * cos(theta) + y0 * sin(theta);
        float s_diff = s - s_center;

        float a_sq_cos = a2 * cos_rel * cos_rel;
        float b_sq_sin = b2 * sin_rel * sin_rel;
        float a2_prime = a_sq_cos + b_sq_sin;

        if (s_diff * s_diff <= a2_prime) {
            float factor = (2.0f * ell.density * a * b) / a2_prime;
            float line_integral = factor * sqrt(a2_prime - s_diff * s_diff);
            sum_proj += line_integral;
        }
    }

    size_t out_idx = (size_t)angle_idx * (size_t)num_detectors + (size_t)det_idx;
    out_sinogram[out_idx] = sum_proj;
}
