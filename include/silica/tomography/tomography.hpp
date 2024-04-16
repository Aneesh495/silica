#pragma once

#include "silica/core/context.hpp"
#include "silica/core/queue.hpp"
#include "silica/core/buffer.hpp"
#include "silica/core/program.hpp"
#include "silica/core/kernel.hpp"
#include <vector>
#include <memory>
#include <cmath>

namespace silica::tomography {

struct PhantomEllipse {
    float x0;       // Center x normalized [-1, 1]
    float y0;       // Center y normalized [-1, 1]
    float a;        // Semi-major axis
    float b;        // Semi-minor axis
    float theta;    // Rotation angle in radians
    float gray_val; // Attenuation density value
};

enum class RampFilterType {
    RamLak,
    SheppLogan,
    Cosine,
    Hann
};

struct ProjectionGeometry {
    int num_detectors{512};
    int num_angles{360};
    float detector_spacing{1.0f};
    float angular_range_rad{3.141592653589793f};
    float source_to_isocenter{1000.0f};
    float isocenter_to_detector{500.0f};
};

class TomographyEngine {
public:
    TomographyEngine(std::shared_ptr<core::Context> context,
                     std::shared_ptr<core::CommandQueue> queue);
    ~TomographyEngine() = default;

    // Generate analytical Shepp-Logan phantom
    [[nodiscard]] std::vector<float> generate_shepp_logan_2d(int size) const;

    // Forward Radon Projection (Analytical or raytraced)
    std::shared_ptr<core::Buffer> forward_radon_2d(
        const std::shared_ptr<core::Buffer>& image_buffer,
        int img_width,
        int img_height,
        const ProjectionGeometry& geometry);

    // Apply frequency ramp filter to sinogram
    std::shared_ptr<core::Buffer> apply_ramp_filter(
        const std::shared_ptr<core::Buffer>& sinogram_buffer,
        const ProjectionGeometry& geometry,
        RampFilterType filter_type = RampFilterType::SheppLogan);

    // Filtered Back-Projection (FBP) reconstruction
    std::shared_ptr<core::Buffer> reconstruct_fbp_2d(
        const std::shared_ptr<core::Buffer>& filtered_sinogram,
        int out_width,
        int out_height,
        const ProjectionGeometry& geometry);

    // Simultaneous Iterative Reconstruction Technique (SIRT) solver
    std::shared_ptr<core::Buffer> solve_sirt(
        const std::shared_ptr<core::Buffer>& sinogram_buffer,
        int img_width,
        int img_height,
        const ProjectionGeometry& geometry,
        int iterations = 20,
        float relaxation = 1.0f);

    // 3D Feldkamp-Davis-Kress (FDK) Cone-Beam CT Reconstruction
    std::shared_ptr<core::Buffer> reconstruct_fdk_3d(
        const std::shared_ptr<core::Buffer>& cone_projections,
        int vol_nx, int vol_ny, int vol_nz,
        float voxel_size_mm,
        const ProjectionGeometry& geometry);

private:
    std::shared_ptr<core::Context> context_;
    std::shared_ptr<core::CommandQueue> queue_;
    std::shared_ptr<core::Program> radon_program_;
    std::shared_ptr<core::Program> filter_program_;
    std::shared_ptr<core::Program> fbp_program_;
    std::shared_ptr<core::Program> sirt_program_;
    std::shared_ptr<core::Program> fdk_program_;
};

} // namespace silica::tomography
