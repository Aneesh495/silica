#pragma once

#include "silica/core/context.hpp"
#include "silica/core/queue.hpp"
#include "silica/core/buffer.hpp"
#include "silica/core/program.hpp"
#include "silica/core/kernel.hpp"
#include <vector>
#include <memory>
#include <array>

namespace silica::spatial {

struct TSDFVolumeConfig {
    int vx{128};
    int vy{128};
    int vz{128};
    float voxel_size{0.01f};        // 1cm per voxel
    float truncation_mu{0.04f};     // 4cm truncation distance
    float max_weight{50.0f};
};

struct CameraIntrinsics {
    int width{640};
    int height{480};
    float fx{525.0f};
    float fy{525.0f};
    float cx{319.5f};
    float cy{239.5f};
};

struct TriangleMesh {
    std::vector<std::array<float, 4>> vertices;
    std::vector<std::array<float, 4>> normals;
    size_t triangle_count{0};
};

class SpatialEngine {
public:
    SpatialEngine(std::shared_ptr<core::Context> context,
                  std::shared_ptr<core::CommandQueue> queue,
                  TSDFVolumeConfig config = {});
    ~SpatialEngine() = default;

    // Reset TSDF volume to empty distance field
    void reset();

    // Integrate incoming depth frame with camera extrinsic pose
    void integrate_frame(
        const std::shared_ptr<core::Buffer>& depth_image,
        const std::array<float, 16>& c2w_matrix,
        const CameraIntrinsics& intrinsics);

    // Direct Volume Raymarching and Sphere Tracing
    std::shared_ptr<core::Buffer> raycast(
        const std::array<float, 16>& c2w_matrix,
        const CameraIntrinsics& intrinsics);

    // Marching Cubes Isosurface Extraction
    TriangleMesh extract_mesh(float iso_level = 0.0f);

    [[nodiscard]] const TSDFVolumeConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::shared_ptr<core::Buffer> tsdf_buffer() const noexcept { return tsdf_buf_; }
    [[nodiscard]] std::shared_ptr<core::Buffer> weight_buffer() const noexcept { return weight_buf_; }

private:
    std::shared_ptr<core::Context> context_;
    std::shared_ptr<core::CommandQueue> queue_;
    TSDFVolumeConfig config_;

    std::shared_ptr<core::Buffer> tsdf_buf_;
    std::shared_ptr<core::Buffer> weight_buf_;
    std::shared_ptr<core::Buffer> color_buf_;

    std::shared_ptr<core::Program> integrate_program_;
    std::shared_ptr<core::Program> raycast_program_;
    std::shared_ptr<core::Program> mc_program_;
};

} // namespace silica::spatial
