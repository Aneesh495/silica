#include "test_framework.hpp"
#include "silica/silica.hpp"
#include <cmath>

using namespace silica;

SILICA_TEST(Spatial, TSDFVolumeReset) {
    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    spatial::TSDFVolumeConfig cfg;
    cfg.vx = 32; cfg.vy = 32; cfg.vz = 32;
    spatial::SpatialEngine engine(context, queue, cfg);

    auto tsdf_buf = engine.tsdf_buffer();
    ASSERT_TRUE(tsdf_buf != nullptr);

    std::vector<float> tsdf_data(32 * 32 * 32);
    tsdf_buf->read_vector(queue, tsdf_data);

    // Initial TSDF values should be 1.0 (empty space)
    for (size_t i = 0; i < 100; ++i) {
        ASSERT_NEAR(tsdf_data[i], 1.0f, 1e-5f);
    }
}

SILICA_TEST(Spatial, MarchingCubesSphereExtraction) {
    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    spatial::TSDFVolumeConfig cfg;
    cfg.vx = 32; cfg.vy = 32; cfg.vz = 32;
    cfg.voxel_size = 0.05f;
    spatial::SpatialEngine engine(context, queue, cfg);

    // Create analytical sphere SDF: f(x, y, z) = sqrt(x^2 + y^2 + z^2) - radius
    float radius = 0.4f;
    size_t num_voxels = 32 * 32 * 32;
    std::vector<float> sphere_sdf(num_voxels);

    float half_grid = 0.5f * 31.0f * cfg.voxel_size;
    for (int z = 0; z < 32; ++z) {
        float pz = static_cast<float>(z) * cfg.voxel_size - half_grid;
        for (int y = 0; y < 32; ++y) {
            float py = static_cast<float>(y) * cfg.voxel_size - half_grid;
            for (int x = 0; x < 32; ++x) {
                float px = static_cast<float>(x) * cfg.voxel_size - half_grid;
                float dist = std::sqrt(px * px + py * py + pz * pz) - radius;
                size_t idx = static_cast<size_t>((z * 32 + y) * 32 + x);
                sphere_sdf[idx] = dist;
            }
        }
    }

    engine.tsdf_buffer()->write(queue, sphere_sdf.data(), num_voxels * sizeof(float));

    // Extract Marching Cubes mesh
    auto mesh = engine.extract_mesh(0.0f);
    ASSERT_TRUE(mesh.triangle_count > 0);
    ASSERT_EQ(mesh.vertices.size(), mesh.triangle_count * 3);
}
