#include "test_framework.hpp"
#include "silica/silica.hpp"
#include <numeric>

using namespace silica;

SILICA_TEST(Tomography, SheppLoganPhantomGeneration) {
    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    tomography::TomographyEngine engine(context, queue);
    int size = 128;

    auto phantom = engine.generate_shepp_logan_2d(size);
    ASSERT_EQ(phantom.size(), static_cast<size_t>(size * size));

    // Center of phantom should be non-zero
    size_t center_idx = static_cast<size_t>((size / 2) * size + (size / 2));
    ASSERT_TRUE(phantom[center_idx] > 0.0f);

    // Sum of intensities should be positive
    float total = std::accumulate(phantom.begin(), phantom.end(), 0.0f);
    ASSERT_TRUE(total > 100.0f);
}

SILICA_TEST(Tomography, ForwardRadonAndFBPReconstruction) {
    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    tomography::TomographyEngine engine(context, queue);
    int size = 128;

    auto phantom = engine.generate_shepp_logan_2d(size);
    auto img_buf = core::Buffer::create_from_host<float>(context, queue, phantom);

    tomography::ProjectionGeometry geom;
    geom.num_detectors = 128;
    geom.num_angles = 180;
    geom.detector_spacing = 1.0f;

    // 1. Forward Radon Projection
    auto sinogram = engine.forward_radon_2d(img_buf, size, size, geom);
    ASSERT_TRUE(sinogram != nullptr);
    ASSERT_EQ(sinogram->size_bytes(), static_cast<size_t>(geom.num_angles * geom.num_detectors) * sizeof(float));

    // 2. Frequency Ramp Filter
    auto filtered_sino = engine.apply_ramp_filter(sinogram, geom, tomography::RampFilterType::SheppLogan);
    ASSERT_TRUE(filtered_sino != nullptr);

    // 3. FBP Reconstruction
    auto recon = engine.reconstruct_fbp_2d(filtered_sino, size, size, geom);
    ASSERT_TRUE(recon != nullptr);

    std::vector<float> recon_img(static_cast<size_t>(size * size));
    recon->read_vector(queue, recon_img);

    // Verify reconstruction contains positive energy in central regions
    float recon_sum = std::accumulate(recon_img.begin(), recon_img.end(), 0.0f);
    ASSERT_TRUE(recon_sum > 0.0f);
}

SILICA_TEST(Tomography, SIRTIterativeSolver) {
    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    tomography::TomographyEngine engine(context, queue);
    int size = 64;

    auto phantom = engine.generate_shepp_logan_2d(size);
    auto img_buf = core::Buffer::create_from_host<float>(context, queue, phantom);

    tomography::ProjectionGeometry geom;
    geom.num_detectors = 64;
    geom.num_angles = 90;
    geom.detector_spacing = 1.0f;

    auto sinogram = engine.forward_radon_2d(img_buf, size, size, geom);
    ASSERT_TRUE(sinogram != nullptr);

    auto sirt_recon = engine.solve_sirt(sinogram, size, size, geom, 5, 0.5f);
    ASSERT_TRUE(sirt_recon != nullptr);

    std::vector<float> sirt_img(static_cast<size_t>(size * size));
    sirt_recon->read_vector(queue, sirt_img);

    float sirt_sum = std::accumulate(sirt_img.begin(), sirt_img.end(), 0.0f);
    ASSERT_TRUE(sirt_sum > 0.0f);
}
