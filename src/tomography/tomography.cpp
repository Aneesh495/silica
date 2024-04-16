#include "silica/tomography/tomography.hpp"
#include "silica/core/error.hpp"
#include <cmath>
#include <iostream>

namespace silica::tomography {

TomographyEngine::TomographyEngine(std::shared_ptr<core::Context> context,
                                   std::shared_ptr<core::CommandQueue> queue)
    : context_(std::move(context)), queue_(std::move(queue)) {
    SILICA_CHECK_THROW(context_ != nullptr, "Context cannot be null");
    SILICA_CHECK_THROW(queue_ != nullptr, "Queue cannot be null");

    // Load and build OpenCL programs
    std::string kernel_root = SILICA_KERNEL_ROOT;
    radon_program_ = core::Program::create_from_file(context_, kernel_root + "/tomography/forward_radon.cl");
    filter_program_ = core::Program::create_from_file(context_, kernel_root + "/tomography/ramp_filter.cl");
    fbp_program_ = core::Program::create_from_file(context_, kernel_root + "/tomography/filtered_backprojection.cl");
    sirt_program_ = core::Program::create_from_file(context_, kernel_root + "/tomography/sirt_solver.cl");
    fdk_program_ = core::Program::create_from_file(context_, kernel_root + "/tomography/fdk_cone_beam.cl");
}

std::vector<float> TomographyEngine::generate_shepp_logan_2d(int size) const {
    SILICA_CHECK_THROW(size > 0, "Image size must be positive");
    std::vector<float> img(static_cast<size_t>(size * size), 0.0f);

    // Standard Shepp-Logan 10-ellipse configuration
    const std::vector<PhantomEllipse> ellipses = {
        { 0.0f,      0.0f,      0.69f,   0.92f,   0.0f,               1.0f  }, // Skull
        { 0.0f,     -0.0184f,   0.6624f, 0.874f,  0.0f,              -0.8f  }, // Brain boundary
        { 0.22f,     0.0f,      0.11f,   0.31f,  -18.0f * (3.14159f / 180.0f), -0.2f  }, // Left ventricle
        {-0.22f,     0.0f,      0.16f,   0.41f,   18.0f * (3.14159f / 180.0f), -0.2f  }, // Right ventricle
        { 0.0f,      0.35f,     0.21f,   0.25f,   0.0f,               0.1f  }, // Upper ventricle
        { 0.0f,      0.1f,      0.046f,  0.046f,  0.0f,               0.1f  }, // Central lesion
        { 0.0f,     -0.1f,      0.046f,  0.046f,  0.0f,               0.1f  }, // Mid lesion
        {-0.08f,    -0.605f,    0.046f,  0.023f,  0.0f,               0.1f  }, // Small lower lesion 1
        { 0.0f,     -0.606f,    0.023f,  0.023f,  0.0f,               0.1f  }, // Small lower lesion 2
        { 0.06f,    -0.605f,    0.046f,  0.023f,  0.0f,               0.1f  }  // Small lower lesion 3
    };

    float half_s = 0.5f * static_cast<float>(size - 1);
    for (int y = 0; y < size; ++y) {
        float ny = (static_cast<float>(y) - half_s) / half_s;
        for (int x = 0; x < size; ++x) {
            float nx = (static_cast<float>(x) - half_s) / half_s;
            float val = 0.0f;

            for (const auto& e : ellipses) {
                float dx = nx - e.x0;
                float dy = ny - e.y0;
                float cos_a = std::cos(e.theta);
                float sin_a = std::sin(e.theta);
                float rx =  dx * cos_a + dy * sin_a;
                float ry = -dx * sin_a + dy * cos_a;

                if ((rx * rx) / (e.a * e.a) + (ry * ry) / (e.b * e.b) <= 1.0f) {
                    val += e.gray_val;
                }
            }
            img[static_cast<size_t>(y * size + x)] = std::max(0.0f, val);
        }
    }

    return img;
}

std::shared_ptr<core::Buffer> TomographyEngine::forward_radon_2d(
    const std::shared_ptr<core::Buffer>& image_buffer,
    int img_width,
    int img_height,
    const ProjectionGeometry& geometry) {
    
    SILICA_CHECK_THROW(image_buffer != nullptr, "Image buffer is null");
    SILICA_CHECK_THROW(img_width == img_height, "Forward Radon requires square image");

    size_t sinogram_bytes = static_cast<size_t>(geometry.num_angles * geometry.num_detectors) * sizeof(float);
    auto sinogram_buf = core::Buffer::create(context_, sinogram_bytes, core::MemoryAccess::ReadWrite);

    auto kernel = core::Kernel::create(radon_program_, "kernel_forward_radon_parallel");
    float pixel_size = 1.0f;

    kernel->set_args(
        *image_buffer,
        *sinogram_buf,
        img_width,
        geometry.num_angles,
        geometry.num_detectors,
        pixel_size,
        geometry.detector_spacing
    );

    core::WorkSize ws(
        {static_cast<size_t>(geometry.num_detectors), static_cast<size_t>(geometry.num_angles)},
        {16, 16}
    );

    kernel->dispatch(queue_, ws);
    return sinogram_buf;
}

std::shared_ptr<core::Buffer> TomographyEngine::apply_ramp_filter(
    const std::shared_ptr<core::Buffer>& sinogram_buffer,
    const ProjectionGeometry& geometry,
    RampFilterType /*filter_type*/) {
    
    SILICA_CHECK_THROW(sinogram_buffer != nullptr, "Sinogram buffer is null");

    size_t sinogram_bytes = static_cast<size_t>(geometry.num_angles * geometry.num_detectors) * sizeof(float);
    auto filtered_buf = core::Buffer::create(context_, sinogram_bytes, core::MemoryAccess::ReadWrite);

    auto kernel = core::Kernel::create(filter_program_, "kernel_spatial_ramlak_filter_row");
    int kernel_radius = std::min(64, geometry.num_detectors / 4);

    kernel->set_args(
        *sinogram_buffer,
        *filtered_buf,
        geometry.num_angles,
        geometry.num_detectors,
        kernel_radius,
        geometry.detector_spacing
    );

    core::WorkSize ws(
        {static_cast<size_t>(geometry.num_detectors), static_cast<size_t>(geometry.num_angles)},
        {16, 16}
    );

    kernel->dispatch(queue_, ws);
    return filtered_buf;
}

std::shared_ptr<core::Buffer> TomographyEngine::reconstruct_fbp_2d(
    const std::shared_ptr<core::Buffer>& filtered_sinogram,
    int out_width,
    int out_height,
    const ProjectionGeometry& geometry) {
    
    SILICA_CHECK_THROW(filtered_sinogram != nullptr, "Filtered sinogram buffer is null");
    SILICA_CHECK_THROW(out_width == out_height, "Output image must be square");

    size_t recon_bytes = static_cast<size_t>(out_width * out_height) * sizeof(float);
    auto recon_buf = core::Buffer::create(context_, recon_bytes, core::MemoryAccess::ReadWrite);

    auto kernel = core::Kernel::create(fbp_program_, "kernel_filtered_backprojection_2d");
    float pixel_size = 1.0f;

    kernel->set_args(
        *filtered_sinogram,
        *recon_buf,
        out_width,
        geometry.num_angles,
        geometry.num_detectors,
        pixel_size,
        geometry.detector_spacing
    );

    core::WorkSize ws(
        {static_cast<size_t>(out_width), static_cast<size_t>(out_height)},
        {16, 16}
    );

    kernel->dispatch(queue_, ws);
    return recon_buf;
}

std::shared_ptr<core::Buffer> TomographyEngine::solve_sirt(
    const std::shared_ptr<core::Buffer>& sinogram_buffer,
    int img_width,
    int img_height,
    const ProjectionGeometry& geometry,
    int iterations,
    float relaxation) {
    
    SILICA_CHECK_THROW(sinogram_buffer != nullptr, "Sinogram buffer is null");
    SILICA_CHECK_THROW(img_width == img_height, "SIRT requires square image");

    size_t sino_bytes = static_cast<size_t>(geometry.num_angles * geometry.num_detectors) * sizeof(float);

    // Initial guess x_0 = 0
    std::vector<float> zero_img(static_cast<size_t>(img_width * img_height), 0.0f);
    auto image_buf = core::Buffer::create_from_host<float>(context_, queue_, zero_img);
    auto residual_buf = core::Buffer::create(context_, sino_bytes, core::MemoryAccess::ReadWrite);

    auto sim_kernel = core::Kernel::create(radon_program_, "kernel_forward_radon_parallel");
    auto res_kernel = core::Kernel::create(sirt_program_, "kernel_sirt_compute_residual");
    auto update_kernel = core::Kernel::create(sirt_program_, "kernel_sirt_update_image_grid");

    float pixel_size = 1.0f;
    uint32_t total_bins = static_cast<uint32_t>(geometry.num_angles * geometry.num_detectors);

    core::WorkSize sim_ws(
        {static_cast<size_t>(geometry.num_detectors), static_cast<size_t>(geometry.num_angles)},
        {16, 16}
    );
    core::WorkSize res_ws({static_cast<size_t>(total_bins)}, {256});
    core::WorkSize update_ws(
        {static_cast<size_t>(img_width), static_cast<size_t>(img_height)},
        {16, 16}
    );

    auto sim_sino_buf = core::Buffer::create(context_, sino_bytes, core::MemoryAccess::ReadWrite);

    for (int iter = 0; iter < iterations; ++iter) {
        // Forward project current estimate
        sim_kernel->set_args(
            *image_buf,
            *sim_sino_buf,
            img_width,
            geometry.num_angles,
            geometry.num_detectors,
            pixel_size,
            geometry.detector_spacing
        );
        sim_kernel->dispatch(queue_, sim_ws);

        // Residual r = measured - simulated
        res_kernel->set_args(
            *sinogram_buffer,
            *sim_sino_buf,
            *residual_buf,
            total_bins
        );
        res_kernel->dispatch(queue_, res_ws);

        // Backproject and update
        update_kernel->set_args(
            *image_buf,
            *residual_buf,
            img_width,
            geometry.num_angles,
            geometry.num_detectors,
            pixel_size,
            geometry.detector_spacing,
            relaxation
        );
        update_kernel->dispatch(queue_, update_ws);
    }

    return image_buf;
}

std::shared_ptr<core::Buffer> TomographyEngine::reconstruct_fdk_3d(
    const std::shared_ptr<core::Buffer>& cone_projections,
    int vol_nx, int vol_ny, int vol_nz,
    float voxel_size_mm,
    const ProjectionGeometry& geometry) {
    
    SILICA_CHECK_THROW(cone_projections != nullptr, "Cone projections buffer is null");

    size_t vol_bytes = static_cast<size_t>(vol_nx * vol_ny * vol_nz) * sizeof(float);
    auto volume_buf = core::Buffer::create(context_, vol_bytes, core::MemoryAccess::ReadWrite);

    auto kernel = core::Kernel::create(fdk_program_, "kernel_fdk_cone_beam_backproject_3d");

    kernel->set_args(
        *cone_projections,
        *volume_buf,
        vol_nx,
        vol_ny,
        vol_nz,
        geometry.num_detectors,
        geometry.num_detectors,
        geometry.num_angles,
        geometry.source_to_isocenter,
        geometry.source_to_isocenter + geometry.isocenter_to_detector,
        voxel_size_mm,
        geometry.detector_spacing
    );

    core::WorkSize ws(
        {static_cast<size_t>(vol_nx), static_cast<size_t>(vol_ny), static_cast<size_t>(vol_nz)},
        {8, 8, 4}
    );

    kernel->dispatch(queue_, ws);
    return volume_buf;
}

} // namespace silica::tomography
