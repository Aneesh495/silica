#pragma once

#include "silica/core/context.hpp"
#include "silica/core/queue.hpp"
#include "silica/core/buffer.hpp"
#include "silica/core/program.hpp"
#include "silica/core/kernel.hpp"
#include <vector>
#include <memory>
#include <array>

namespace silica::vision {

struct StereoParameters {
    int max_disparity{64};
    uint16_t p1{10};
    uint16_t p2{120};
    float baseline_m{0.1f};
    float fx{525.0f};
    float fy{525.0f};
    float cx{319.5f};
    float cy{239.5f};
};

struct Keypoint {
    float x{0.0f};
    float y{0.0f};
    float score{0.0f};
    float response{0.0f};
    int octave{0};
};

class VisionEngine {
public:
    VisionEngine(std::shared_ptr<core::Context> context,
                 std::shared_ptr<core::CommandQueue> queue);
    ~VisionEngine() = default;

    // 2D Gaussian Filtering
    std::shared_ptr<core::Buffer> gaussian_blur_2d(
        const std::shared_ptr<core::Buffer>& input_image,
        int width,
        int height,
        int radius = 3,
        float sigma = 1.5f);

    // Edge-Preserving Bilateral Filter
    std::shared_ptr<core::Buffer> bilateral_filter(
        const std::shared_ptr<core::Buffer>& input_image,
        int width,
        int height,
        int radius = 5,
        float sigma_spatial = 3.0f,
        float sigma_range = 0.1f);

    // FAST-9 Corner Detector
    std::shared_ptr<core::Buffer> detect_fast_corners(
        const std::shared_ptr<core::Buffer>& gray_image,
        int width,
        int height,
        float threshold = 20.0f);

    // Harris & Shi-Tomasi Corner Detector
    std::shared_ptr<core::Buffer> detect_harris_corners(
        const std::shared_ptr<core::Buffer>& gray_image,
        int width,
        int height,
        float k_param = 0.04f,
        float threshold = 1000.0f);

    // Semi-Global Matching (SGM) Dense Stereo Disparity
    std::shared_ptr<core::Buffer> compute_stereo_disparity(
        const std::shared_ptr<core::Buffer>& left_gray,
        const std::shared_ptr<core::Buffer>& right_gray,
        int width,
        int height,
        const StereoParameters& params = {});

    // Disparity to 3D Point Cloud Reprojection
    std::shared_ptr<core::Buffer> reproject_disparity_to_depth(
        const std::shared_ptr<core::Buffer>& disparity_image,
        int width,
        int height,
        const StereoParameters& params = {});

private:
    std::shared_ptr<core::Context> context_;
    std::shared_ptr<core::CommandQueue> queue_;

    std::shared_ptr<core::Program> spatial_filter_program_;
    std::shared_ptr<core::Program> bilateral_program_;
    std::shared_ptr<core::Program> fast_program_;
    std::shared_ptr<core::Program> harris_program_;
    std::shared_ptr<core::Program> census_program_;
    std::shared_ptr<core::Program> sgm_program_;
    std::shared_ptr<core::Program> reproject_program_;
};

} // namespace silica::vision
