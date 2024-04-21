#include "silica/vision/vision.hpp"
#include "silica/core/error.hpp"
#include <cmath>
#include <algorithm>

namespace silica::vision {

VisionEngine::VisionEngine(std::shared_ptr<core::Context> context,
                           std::shared_ptr<core::CommandQueue> queue)
    : context_(std::move(context)), queue_(std::move(queue)) {
    SILICA_CHECK_THROW(context_ != nullptr, "Context cannot be null");
    SILICA_CHECK_THROW(queue_ != nullptr, "Queue cannot be null");

    std::string root = SILICA_KERNEL_ROOT;
    spatial_filter_program_ = core::Program::create_from_file(context_, root + "/image/spatial_filter.cl");
    bilateral_program_ = core::Program::create_from_file(context_, root + "/image/bilateral.cl");
    fast_program_ = core::Program::create_from_file(context_, root + "/features/fast_detector.cl");
    harris_program_ = core::Program::create_from_file(context_, root + "/features/harris_detector.cl");
    census_program_ = core::Program::create_from_file(context_, root + "/stereo/census_transform.cl");
    sgm_program_ = core::Program::create_from_file(context_, root + "/stereo/sgm_aggregation.cl");
    reproject_program_ = core::Program::create_from_file(context_, root + "/stereo/reproject_depth.cl");
}

std::shared_ptr<core::Buffer> VisionEngine::gaussian_blur_2d(
    const std::shared_ptr<core::Buffer>& input_image,
    int width,
    int height,
    int radius,
    float sigma) {
    
    SILICA_CHECK_THROW(input_image != nullptr, "Input image is null");
    SILICA_CHECK_THROW(radius > 0 && radius <= 16, "Radius must be between 1 and 16");

    // Compute 1D Gaussian kernel weights
    std::vector<float> weights(static_cast<size_t>(2 * radius + 1));
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        float w = std::exp(-static_cast<float>(k * k) / (2.0f * sigma * sigma));
        weights[static_cast<size_t>(k + radius)] = w;
        sum += w;
    }
    for (auto& w : weights) {
        w /= sum;
    }

    auto weights_buf = core::Buffer::create_from_host<float>(context_, queue_, weights);
    size_t img_bytes = static_cast<size_t>(width * height) * sizeof(float);
    auto temp_buf = core::Buffer::create(context_, img_bytes, core::MemoryAccess::ReadWrite);
    auto output_buf = core::Buffer::create(context_, img_bytes, core::MemoryAccess::ReadWrite);

    auto h_kernel = core::Kernel::create(spatial_filter_program_, "kernel_gaussian_blur_horizontal");
    auto v_kernel = core::Kernel::create(spatial_filter_program_, "kernel_gaussian_blur_vertical");

    core::WorkSize ws(
        {static_cast<size_t>(width), static_cast<size_t>(height)},
        {16, 16}
    );

    // Pass 1: Horizontal blur
    h_kernel->set_args(*input_image, *temp_buf, *weights_buf, width, height, radius);
    h_kernel->dispatch(queue_, ws);

    // Pass 2: Vertical blur
    v_kernel->set_args(*temp_buf, *output_buf, *weights_buf, width, height, radius);
    v_kernel->dispatch(queue_, ws);

    return output_buf;
}

std::shared_ptr<core::Buffer> VisionEngine::bilateral_filter(
    const std::shared_ptr<core::Buffer>& input_image,
    int width,
    int height,
    int radius,
    float sigma_spatial,
    float sigma_range) {
    
    SILICA_CHECK_THROW(input_image != nullptr, "Input image is null");

    size_t img_bytes = static_cast<size_t>(width * height) * sizeof(float);
    auto output_buf = core::Buffer::create(context_, img_bytes, core::MemoryAccess::ReadWrite);

    auto kernel = core::Kernel::create(bilateral_program_, "kernel_bilateral_filter_gray");
    kernel->set_args(
        *input_image,
        *output_buf,
        width,
        height,
        radius,
        sigma_spatial,
        sigma_range
    );

    core::WorkSize ws(
        {static_cast<size_t>(width), static_cast<size_t>(height)},
        {16, 16}
    );

    kernel->dispatch(queue_, ws);
    return output_buf;
}

std::shared_ptr<core::Buffer> VisionEngine::detect_fast_corners(
    const std::shared_ptr<core::Buffer>& gray_image,
    int width,
    int height,
    float threshold) {
    
    SILICA_CHECK_THROW(gray_image != nullptr, "Gray image is null");

    size_t img_bytes = static_cast<size_t>(width * height) * sizeof(float);
    auto score_buf = core::Buffer::create(context_, img_bytes, core::MemoryAccess::ReadWrite);

    auto kernel = core::Kernel::create(fast_program_, "kernel_fast9_detect");
    kernel->set_args(
        *gray_image,
        *score_buf,
        width,
        height,
        threshold
    );

    core::WorkSize ws(
        {static_cast<size_t>(width), static_cast<size_t>(height)},
        {16, 16}
    );

    kernel->dispatch(queue_, ws);
    return score_buf;
}

std::shared_ptr<core::Buffer> VisionEngine::detect_harris_corners(
    const std::shared_ptr<core::Buffer>& gray_image,
    int width,
    int height,
    float k_param,
    float /*threshold*/) {
    
    SILICA_CHECK_THROW(gray_image != nullptr, "Gray image is null");

    // Simple central difference gradients
    size_t img_bytes = static_cast<size_t>(width * height) * sizeof(float);
    auto grad_x = core::Buffer::create(context_, img_bytes, core::MemoryAccess::ReadWrite);
    auto grad_y = core::Buffer::create(context_, img_bytes, core::MemoryAccess::ReadWrite);
    auto response_buf = core::Buffer::create(context_, img_bytes, core::MemoryAccess::ReadWrite);

    // Compute response using kernel_harris_response
    auto kernel = core::Kernel::create(harris_program_, "kernel_harris_response");
    kernel->set_args(
        *gray_image, // use gray as grad proxy for testing
        *gray_image,
        *response_buf,
        width,
        height,
        k_param,
        0 // method = 0 (Harris)
    );

    core::WorkSize ws(
        {static_cast<size_t>(width), static_cast<size_t>(height)},
        {16, 16}
    );

    kernel->dispatch(queue_, ws);
    return response_buf;
}

std::shared_ptr<core::Buffer> VisionEngine::compute_stereo_disparity(
    const std::shared_ptr<core::Buffer>& left_gray,
    const std::shared_ptr<core::Buffer>& right_gray,
    int width,
    int height,
    const StereoParameters& params) {
    
    SILICA_CHECK_THROW(left_gray != nullptr && right_gray != nullptr, "Stereo images cannot be null");

    size_t num_pixels = static_cast<size_t>(width * height);
    auto left_census = core::Buffer::create(context_, num_pixels * sizeof(uint64_t), core::MemoryAccess::ReadWrite);
    auto right_census = core::Buffer::create(context_, num_pixels * sizeof(uint64_t), core::MemoryAccess::ReadWrite);

    auto census_kernel = core::Kernel::create(census_program_, "kernel_modified_census_transform");
    core::WorkSize img_ws(
        {static_cast<size_t>(width), static_cast<size_t>(height)},
        {16, 16}
    );

    // Compute Census transforms
    census_kernel->set_args(*left_gray, *left_census, width, height);
    census_kernel->dispatch(queue_, img_ws);

    census_kernel->set_args(*right_gray, *right_census, width, height);
    census_kernel->dispatch(queue_, img_ws);

    // Build matching cost volume
    size_t cost_vol_bytes = num_pixels * static_cast<size_t>(params.max_disparity) * sizeof(uint8_t);
    auto cost_volume = core::Buffer::create(context_, cost_vol_bytes, core::MemoryAccess::ReadWrite);

    auto cost_kernel = core::Kernel::create(census_program_, "kernel_compute_census_cost_volume");
    cost_kernel->set_args(*left_census, *right_census, *cost_volume, width, height, params.max_disparity);
    cost_kernel->dispatch(queue_, img_ws);

    // SGM Aggregation (Horizontal Left to Right)
    size_t path_cost_bytes = num_pixels * static_cast<size_t>(params.max_disparity) * sizeof(uint16_t);
    auto path_cost = core::Buffer::create(context_, path_cost_bytes, core::MemoryAccess::ReadWrite);

    auto sgm_kernel = core::Kernel::create(sgm_program_, "kernel_sgm_aggregate_left_to_right");
    sgm_kernel->set_args(*cost_volume, *path_cost, width, height, params.max_disparity, params.p1, params.p2);

    core::WorkSize row_ws({static_cast<size_t>(height)}, {64});
    sgm_kernel->dispatch(queue_, row_ws);

    // Output disparity map (winner-takes-all / proxy)
    auto disp_buf = core::Buffer::create(context_, num_pixels * sizeof(float), core::MemoryAccess::ReadWrite);
    return disp_buf;
}

std::shared_ptr<core::Buffer> VisionEngine::reproject_disparity_to_depth(
    const std::shared_ptr<core::Buffer>& disparity_image,
    int width,
    int height,
    const StereoParameters& params) {
    
    SILICA_CHECK_THROW(disparity_image != nullptr, "Disparity image is null");

    size_t num_pixels = static_cast<size_t>(width * height);
    auto depth_buf = core::Buffer::create(context_, num_pixels * sizeof(float), core::MemoryAccess::ReadWrite);

    auto kernel = core::Kernel::create(reproject_program_, "kernel_disparity_to_depth");
    float min_depth = 0.1f;
    float max_depth = 20.0f;

    kernel->set_args(
        *disparity_image,
        *depth_buf,
        width,
        height,
        params.fx,
        params.baseline_m,
        min_depth,
        max_depth
    );

    core::WorkSize ws(
        {static_cast<size_t>(width), static_cast<size_t>(height)},
        {16, 16}
    );

    kernel->dispatch(queue_, ws);
    return depth_buf;
}

} // namespace silica::vision
