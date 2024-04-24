#pragma once

#include "silica/core/context.hpp"
#include "silica/core/queue.hpp"
#include "silica/core/buffer.hpp"
#include "silica/core/program.hpp"
#include "silica/core/kernel.hpp"
#include <vector>
#include <memory>
#include <array>

namespace silica::pointcloud {

struct PointCloud {
    std::vector<std::array<float, 4>> points;
    std::vector<std::array<float, 4>> normals;
};

struct ICPResult {
    std::array<float, 16> transform{};
    float fitness_score{0.0f};
    int iterations{0};
    bool converged{false};
};

class PointCloudEngine {
public:
    PointCloudEngine(std::shared_ptr<core::Context> context,
                     std::shared_ptr<core::CommandQueue> queue);
    ~PointCloudEngine() = default;

    // Morton Codes Generation (Z-order curve)
    std::shared_ptr<core::Buffer> compute_morton_codes(
        const std::shared_ptr<core::Buffer>& points,
        size_t num_points,
        const std::array<float, 3>& bbox_min,
        const std::array<float, 3>& bbox_max);

    // Voxel Grid Centroid Downsampling
    std::shared_ptr<core::Buffer> downsample_voxel_grid(
        const std::shared_ptr<core::Buffer>& in_points,
        size_t num_points,
        float voxel_size,
        const std::array<float, 3>& origin,
        const std::array<int, 3>& grid_dims,
        size_t& out_downsampled_count);

    // Surface Normal and Curvature Estimation (PCA)
    std::shared_ptr<core::Buffer> estimate_normals(
        const std::shared_ptr<core::Buffer>& points,
        size_t num_points,
        const std::array<float, 3>& viewpoint = {0.0f, 0.0f, 0.0f},
        int search_window = 16);

    // Point-to-Plane ICP Rigid Registration (calculates R, t)
    ICPResult register_icp_point_to_plane(
        const std::shared_ptr<core::Buffer>& src_points,
        const std::shared_ptr<core::Buffer>& dst_points,
        const std::shared_ptr<core::Buffer>& dst_normals,
        size_t num_points,
        int max_iterations = 20,
        float max_corr_dist = 0.5f);

private:
    std::shared_ptr<core::Context> context_;
    std::shared_ptr<core::CommandQueue> queue_;

    std::shared_ptr<core::Program> morton_program_;
    std::shared_ptr<core::Program> voxel_program_;
    std::shared_ptr<core::Program> normal_program_;
    std::shared_ptr<core::Program> icp_program_;
    std::shared_ptr<core::Program> sor_program_;
};

} // namespace silica::pointcloud
