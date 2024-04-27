#include "test_framework.hpp"
#include "silica/silica.hpp"
#include <numeric>

using namespace silica;

SILICA_TEST(Vision, GaussianBlurSmoothing) {
    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    vision::VisionEngine engine(context, queue);
    int width = 64;
    int height = 64;

    // Step function: left half = 0.0, right half = 1.0
    std::vector<float> step_img(static_cast<size_t>(width * height), 0.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = width / 2; x < width; ++x) {
            step_img[static_cast<size_t>(y * width + x)] = 1.0f;
        }
    }

    auto in_buf = core::Buffer::create_from_host<float>(context, queue, step_img);
    auto blurred_buf = engine.gaussian_blur_2d(in_buf, width, height, 3, 1.5f);
    ASSERT_TRUE(blurred_buf != nullptr);

    std::vector<float> out_data(static_cast<size_t>(width * height));
    blurred_buf->read_vector(queue, out_data);

    // Pixel at the boundary should be smoothed to ~0.5
    size_t edge_idx = static_cast<size_t>((height / 2) * width + (width / 2));
    ASSERT_NEAR(out_data[edge_idx], 0.5f, 0.15f);
}

SILICA_TEST(Vision, BilateralFilterEdgePreservation) {
    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    vision::VisionEngine engine(context, queue);
    int width = 64;
    int height = 64;

    std::vector<float> img(static_cast<size_t>(width * height), 0.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = width / 2; x < width; ++x) {
            img[static_cast<size_t>(y * width + x)] = 1.0f;
        }
    }

    auto in_buf = core::Buffer::create_from_host<float>(context, queue, img);
    auto bi_buf = engine.bilateral_filter(in_buf, width, height, 5, 3.0f, 0.1f);
    ASSERT_TRUE(bi_buf != nullptr);

    std::vector<float> out_data(static_cast<size_t>(width * height));
    bi_buf->read_vector(queue, out_data);

    // Flat region on left should stay close to 0.0, flat region on right should stay close to 1.0
    size_t left_idx = static_cast<size_t>((height / 2) * width + 10);
    size_t right_idx = static_cast<size_t>((height / 2) * width + width - 10);
    ASSERT_NEAR(out_data[left_idx], 0.0f, 0.05f);
    ASSERT_NEAR(out_data[right_idx], 1.0f, 0.05f);
}

SILICA_TEST(Vision, FastCornerDetector) {
    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    vision::VisionEngine engine(context, queue);
    int width = 64;
    int height = 64;

    std::vector<float> img(static_cast<size_t>(width * height), 0.0f);
    // Draw a sharp white square (20x20) in the center
    for (int y = 20; y < 40; ++y) {
        for (int x = 20; x < 40; ++x) {
            img[static_cast<size_t>(y * width + x)] = 100.0f;
        }
    }

    auto in_buf = core::Buffer::create_from_host<float>(context, queue, img);
    auto corners_buf = engine.detect_fast_corners(in_buf, width, height, 30.0f);
    ASSERT_TRUE(corners_buf != nullptr);

    std::vector<float> scores(static_cast<size_t>(width * height));
    corners_buf->read_vector(queue, scores);

    // Sum of corner scores should be positive
    float total_score = std::accumulate(scores.begin(), scores.end(), 0.0f);
    ASSERT_TRUE(total_score > 0.0f);
}
