#include "silica/spatial/spatial.hpp"
#include "silica/core/error.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace silica::spatial {

static std::array<float, 16> invert_rigid_transform(const std::array<float, 16>& c2w) {
    // c2w is column-major or row-major 4x4. Assuming row-major [R | t; 0 0 0 1]
    // R is upper-left 3x3, t is (c2w[3], c2w[7], c2w[11])
    std::array<float, 16> w2c{};
    // R^T
    w2c[0]  = c2w[0]; w2c[1]  = c2w[4]; w2c[2]  = c2w[8];
    w2c[4]  = c2w[1]; w2c[5]  = c2w[5]; w2c[6]  = c2w[9];
    w2c[8]  = c2w[2]; w2c[9]  = c2w[6]; w2c[10] = c2w[10];

    // -R^T * t
    float tx = c2w[3], ty = c2w[7], tz = c2w[11];
    w2c[3]  = -(w2c[0] * tx + w2c[1] * ty + w2c[2] * tz);
    w2c[7]  = -(w2c[4] * tx + w2c[5] * ty + w2c[6] * tz);
    w2c[11] = -(w2c[8] * tx + w2c[9] * ty + w2c[10] * tz);

    w2c[12] = 0.0f; w2c[13] = 0.0f; w2c[14] = 0.0f; w2c[15] = 1.0f;
    return w2c;
}

SpatialEngine::SpatialEngine(std::shared_ptr<core::Context> context,
                             std::shared_ptr<core::CommandQueue> queue,
                             TSDFVolumeConfig config)
    : context_(std::move(context)), queue_(std::move(queue)), config_(config) {
    SILICA_CHECK_THROW(context_ != nullptr, "Context cannot be null");
    SILICA_CHECK_THROW(queue_ != nullptr, "Queue cannot be null");

    // Allocate 3D volume grids
    size_t num_voxels = static_cast<size_t>(config_.vx * config_.vy * config_.vz);
    tsdf_buf_ = core::Buffer::create(context_, num_voxels * sizeof(float), core::MemoryAccess::ReadWrite);
    weight_buf_ = core::Buffer::create(context_, num_voxels * sizeof(float), core::MemoryAccess::ReadWrite);
    color_buf_ = core::Buffer::create(context_, num_voxels * sizeof(uint32_t), core::MemoryAccess::ReadWrite);

    // Load OpenCL programs
    std::string root = SILICA_KERNEL_ROOT;
    integrate_program_ = core::Program::create_from_file(context_, root + "/tsdf/tsdf_integrate.cl");
    raycast_program_ = core::Program::create_from_file(context_, root + "/tsdf/sphere_tracer.cl");
    mc_program_ = core::Program::create_from_file(context_, root + "/tsdf/marching_cubes.cl");

    reset();
}

void SpatialEngine::reset() {
    size_t num_voxels = static_cast<size_t>(config_.vx * config_.vy * config_.vz);
    std::vector<float> init_tsdf(num_voxels, 1.0f);
    std::vector<float> init_weights(num_voxels, 0.0f);
    std::vector<uint32_t> init_colors(num_voxels, 0);

    tsdf_buf_->write(queue_, init_tsdf.data(), num_voxels * sizeof(float));
    weight_buf_->write(queue_, init_weights.data(), num_voxels * sizeof(float));
    color_buf_->write(queue_, init_colors.data(), num_voxels * sizeof(uint32_t));
}

void SpatialEngine::integrate_frame(
    const std::shared_ptr<core::Buffer>& depth_image,
    const std::array<float, 16>& c2w_matrix,
    const CameraIntrinsics& intrinsics) {
    
    SILICA_CHECK_THROW(depth_image != nullptr, "Depth buffer cannot be null");

    auto w2c_matrix = invert_rigid_transform(c2w_matrix);
    auto c2w_buf = core::Buffer::create_from_host<float>(context_, queue_, c2w_matrix);
    auto w2c_buf = core::Buffer::create_from_host<float>(context_, queue_, w2c_matrix);

    auto kernel = core::Kernel::create(integrate_program_, "kernel_tsdf_integrate_frame");

    kernel->set_args(
        *tsdf_buf_,
        *weight_buf_,
        *depth_image,
        *depth_image, // dummy rgb reference
        *color_buf_,
        *c2w_buf,
        *w2c_buf,
        config_.vx,
        config_.vy,
        config_.vz,
        config_.voxel_size,
        intrinsics.width,
        intrinsics.height,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
        config_.truncation_mu,
        config_.max_weight
    );

    core::WorkSize ws(
        {static_cast<size_t>(config_.vx), static_cast<size_t>(config_.vy), static_cast<size_t>(config_.vz)},
        {8, 8, 4}
    );

    kernel->dispatch(queue_, ws);
}

std::shared_ptr<core::Buffer> SpatialEngine::raycast(
    const std::array<float, 16>& c2w_matrix,
    const CameraIntrinsics& intrinsics) {
    
    size_t img_pixels = static_cast<size_t>(intrinsics.width * intrinsics.height);
    auto out_depth = core::Buffer::create(context_, img_pixels * sizeof(float), core::MemoryAccess::ReadWrite);
    auto out_normals = core::Buffer::create(context_, img_pixels * sizeof(float) * 4, core::MemoryAccess::ReadWrite);
    auto c2w_buf = core::Buffer::create_from_host<float>(context_, queue_, c2w_matrix);

    auto kernel = core::Kernel::create(raycast_program_, "kernel_tsdf_raycast");

    kernel->set_args(
        *tsdf_buf_,
        *out_depth,
        *out_normals,
        *c2w_buf,
        intrinsics.width,
        intrinsics.height,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
        config_.vx,
        config_.vy,
        config_.vz,
        config_.voxel_size,
        config_.truncation_mu
    );

    core::WorkSize ws(
        {static_cast<size_t>(intrinsics.width), static_cast<size_t>(intrinsics.height)},
        {16, 16}
    );

    kernel->dispatch(queue_, ws);
    return out_normals;
}

TriangleMesh SpatialEngine::extract_mesh(float iso_level) {
    TriangleMesh mesh;
    size_t num_cells = static_cast<size_t>((config_.vx - 1) * (config_.vy - 1) * (config_.vz - 1));
    auto cube_indices = core::Buffer::create(context_, num_cells * sizeof(uint32_t), core::MemoryAccess::ReadWrite);
    auto edge_counts = core::Buffer::create(context_, num_cells * sizeof(uint32_t), core::MemoryAccess::ReadWrite);

    // Phase 1: Classify cells
    auto classify_kernel = core::Kernel::create(mc_program_, "kernel_mc_classify_voxels");
    classify_kernel->set_args(
        *tsdf_buf_,
        *cube_indices,
        *edge_counts,
        config_.vx,
        config_.vy,
        config_.vz,
        iso_level
    );

    size_t gx = ((static_cast<size_t>(config_.vx - 1) + 7) / 8) * 8;
    size_t gy = ((static_cast<size_t>(config_.vy - 1) + 7) / 8) * 8;
    size_t gz = ((static_cast<size_t>(config_.vz - 1) + 3) / 4) * 4;

    core::WorkSize classify_ws(
        {gx, gy, gz},
        {8, 8, 4}
    );
    classify_kernel->dispatch(queue_, classify_ws);

    // Phase 2: Allocate vertex buffers (capacity up to 1M vertices)
    uint32_t max_vertices = 1000000;
    auto vert_buf = core::Buffer::create(context_, static_cast<size_t>(max_vertices) * sizeof(float) * 4, core::MemoryAccess::ReadWrite);
    auto norm_buf = core::Buffer::create(context_, static_cast<size_t>(max_vertices) * sizeof(float) * 4, core::MemoryAccess::ReadWrite);

    std::vector<uint32_t> zero_count(1, 0);
    auto counter_buf = core::Buffer::create_from_host<uint32_t>(context_, queue_, zero_count);

    // Phase 3: Extract mesh
    auto extract_kernel = core::Kernel::create(mc_program_, "kernel_mc_extract_mesh");
    extract_kernel->set_args(
        *tsdf_buf_,
        *cube_indices,
        *vert_buf,
        *norm_buf,
        *counter_buf,
        config_.vx,
        config_.vy,
        config_.vz,
        config_.voxel_size,
        iso_level,
        max_vertices
    );

    extract_kernel->dispatch(queue_, classify_ws);

    // Read back vertex count and mesh geometry
    std::vector<uint32_t> total_verts(1, 0);
    counter_buf->read(queue_, total_verts.data(), sizeof(uint32_t));

    uint32_t count = std::min(total_verts[0], max_vertices);
    mesh.triangle_count = count / 3;

    if (count > 0) {
        mesh.vertices.resize(count);
        mesh.normals.resize(count);
        vert_buf->read(queue_, mesh.vertices.data(), count * sizeof(float) * 4);
        norm_buf->read(queue_, mesh.normals.data(), count * sizeof(float) * 4);
    }

    return mesh;
}

} // namespace silica::spatial
