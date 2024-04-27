#include "test_framework.hpp"
#include "silica/silica.hpp"
#include <cmath>

using namespace silica;

SILICA_TEST(PointCloud, MortonCodeOrdering) {
    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    pointcloud::PointCloudEngine engine(context, queue);

    std::vector<cl_float4> points = {
        {0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f}
    };

    auto pt_buf = core::Buffer::create_from_host<cl_float4>(context, queue, points);
    std::array<float, 3> b_min = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> b_max = {1.0f, 1.0f, 1.0f};

    auto morton_buf = engine.compute_morton_codes(pt_buf, points.size(), b_min, b_max);
    ASSERT_TRUE(morton_buf != nullptr);

    std::vector<uint32_t> codes(points.size());
    morton_buf->read_vector(queue, codes);

    // Origin (0,0,0) should have Morton code 0
    ASSERT_EQ(codes[0], 0u);
    // (1,1,1) should have highest Morton code
    ASSERT_TRUE(codes[3] > codes[0]);
}

SILICA_TEST(PointCloud, VoxelGridDownsampling) {
    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    pointcloud::PointCloudEngine engine(context, queue);

    // 100 points clustered inside a 0.01 box
    std::vector<cl_float4> points;
    for (int i = 0; i < 100; ++i) {
        float f = static_cast<float>(i) * 0.0001f;
        points.push_back({f, f, f, 1.0f});
    }

    auto in_buf = core::Buffer::create_from_host<cl_float4>(context, queue, points);
    size_t out_count = 0;
    std::array<float, 3> origin = {-1.0f, -1.0f, -1.0f};
    std::array<int, 3> dims = {64, 64, 64};

    auto downsampled = engine.downsample_voxel_grid(in_buf, points.size(), 0.1f, origin, dims, out_count);
    ASSERT_TRUE(downsampled != nullptr);

    // All points fall into the same voxel cell, so out_count should be exactly 1
    ASSERT_EQ(out_count, 1u);
}

SILICA_TEST(PointCloud, NormalEstimationPCA) {
    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    pointcloud::PointCloudEngine engine(context, queue);

    // Flat horizontal plane at z = 0
    std::vector<cl_float4> plane_pts;
    for (int y = -5; y <= 5; ++y) {
        for (int x = -5; x <= 5; ++x) {
            plane_pts.push_back({static_cast<float>(x), static_cast<float>(y), 0.0f, 1.0f});
        }
    }

    auto in_buf = core::Buffer::create_from_host<cl_float4>(context, queue, plane_pts);
    auto normals_buf = engine.estimate_normals(in_buf, plane_pts.size(), {0.0f, 0.0f, 10.0f}, 16);
    ASSERT_TRUE(normals_buf != nullptr);

    std::vector<cl_float4> normals(plane_pts.size());
    normals_buf->read_vector(queue, normals);

    // Normal at center should point predominantly in +Z direction
    size_t center_idx = plane_pts.size() / 2;
    ASSERT_TRUE(std::abs(normals[center_idx].s[2]) > 0.5f);
}
