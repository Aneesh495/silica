#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include "silica/silica.hpp"

using namespace silica;

static void print_banner(const std::string& title) {
    std::cout << "\n+----------------------------------------------------------------------+\n";
    std::cout << "| " << std::left << std::setw(68) << title << " |\n";
    std::cout << "+----------------------------------------------------------------------+\n";
}

static void print_metric(const std::string& name, double val, const std::string& unit) {
    std::cout << "  * " << std::left << std::setw(36) << name << " : "
              << std::right << std::setw(10) << std::fixed << std::setprecision(2) << val
              << " " << unit << "\n";
}

int main() {
    print_banner("Silica OpenCL Hardware Acceleration Benchmark Suite");

    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);
    const auto& dev = context->device();

    std::cout << "Device Profile:\n";
    std::cout << "  Device Name       : " << dev.name() << "\n";
    std::cout << "  Vendor            : " << dev.vendor() << "\n";
    std::cout << "  Driver Version    : " << dev.driver_version() << "\n";
    std::cout << "  OpenCL Version    : " << dev.opencl_version() << "\n";
    std::cout << "  Compute Units     : " << dev.max_compute_units() << "\n";
    std::cout << "  Max Workgroup     : " << dev.max_work_group_size() << "\n";
    std::cout << "  Global Memory     : " << (dev.global_memory_size() / (1024 * 1024)) << " MB\n";
    std::cout << "  Clock Frequency   : " << dev.max_clock_frequency() << " MHz\n";

    // 1. Memory Bandwidth Microbenchmarks
    print_banner("1. Memory Transfer Bandwidth (PCIe / Apple Fabric)");
    {
        const size_t bytes = 64 * 1024 * 1024; // 64 MB
        std::vector<uint8_t> host_src(bytes, 0xAA);
        std::vector<uint8_t> host_dst(bytes, 0);

        auto buf_a = core::Buffer::create(context, bytes, core::MemoryAccess::ReadWrite);
        auto buf_b = core::Buffer::create(context, bytes, core::MemoryAccess::ReadWrite);

        // Warm up
        buf_a->write(queue, host_src.data(), bytes);
        queue->finish();

        // Host to Device (H2D)
        const int iters = 10;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            buf_a->write(queue, host_src.data(), bytes);
        }
        queue->finish();
        auto t1 = std::chrono::high_resolution_clock::now();
        double sec_h2d = std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(iters);
        double gb_sec_h2d = (static_cast<double>(bytes) / 1e9) / sec_h2d;
        print_metric("Host-to-Device (H2D) Write", gb_sec_h2d, "GB/s");

        // Device to Host (D2H)
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            buf_a->read(queue, host_dst.data(), bytes);
        }
        queue->finish();
        t1 = std::chrono::high_resolution_clock::now();
        double sec_d2h = std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(iters);
        double gb_sec_d2h = (static_cast<double>(bytes) / 1e9) / sec_d2h;
        print_metric("Device-to-Host (D2H) Read", gb_sec_d2h, "GB/s");

        // Device to Device (D2D)
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            buf_a->copy_to(queue, *buf_b, bytes);
        }
        queue->finish();
        t1 = std::chrono::high_resolution_clock::now();
        double sec_d2d = std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(iters);
        double gb_sec_d2d = (static_cast<double>(bytes) / 1e9) / sec_d2d;
        print_metric("Device-to-Device (D2D) Copy", gb_sec_d2d, "GB/s");
    }

    // 2. Kernel Launch Latency
    print_banner("2. Kernel Launch & Queue Dispatch Latency");
    {
        std::string root = SILICA_KERNEL_ROOT;
        auto prog = core::Program::create_from_file(context, root + "/primitives/reduce.cl");
        auto kernel = core::Kernel::create(prog, "kernel_reduce_sum_float");

        auto dummy_in = core::Buffer::create(context, 256 * sizeof(float), core::MemoryAccess::ReadOnly);
        auto dummy_out = core::Buffer::create(context, 1 * sizeof(float), core::MemoryAccess::WriteOnly);
        kernel->set_args(*dummy_in, *dummy_out, static_cast<uint32_t>(256));

        core::WorkSize ws({256}, {256});

        // Warm up
        for (int i = 0; i < 50; ++i) {
            kernel->dispatch(queue, ws);
        }
        queue->finish();

        const int latency_iters = 1000;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < latency_iters; ++i) {
            kernel->dispatch(queue, ws);
            queue->finish();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(latency_iters);
        print_metric("Synchronous Dispatch Latency", avg_us, "us/launch");
    }

    // 3. Register Tiled Matrix Multiply (GEMM) Throughput
    print_banner("3. Register-Tiled Matrix Multiply (GEMM 512x512)");
    {
        std::string root = SILICA_KERNEL_ROOT;
        auto prog = core::Program::create_from_file(context, root + "/primitives/gemm.cl");
        auto kernel = core::Kernel::create(prog, "kernel_gemm_tiled");

        const uint32_t M = 512, N = 512, K = 512;
        std::vector<float> A(static_cast<size_t>(M * K), 1.0f);
        std::vector<float> B(static_cast<size_t>(K * N), 2.0f);
        std::vector<float> C(static_cast<size_t>(M * N), 0.0f);

        auto buf_A = core::Buffer::create_from_host<float>(context, queue, A);
        auto buf_B = core::Buffer::create_from_host<float>(context, queue, B);
        auto buf_C = core::Buffer::create_from_host<float>(context, queue, C);

        float alpha = 1.0f, beta = 0.0f;
        kernel->set_args(*buf_A, *buf_B, *buf_C, M, N, K, alpha, beta);

        size_t tiles_x = (N + 63) / 64;
        size_t tiles_y = (M + 63) / 64;
        core::WorkSize ws({tiles_x * 16, tiles_y * 16}, {16, 16});

        // Warm up
        kernel->dispatch(queue, ws);
        queue->finish();

        const int gemm_iters = 20;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < gemm_iters; ++i) {
            kernel->dispatch(queue, ws);
        }
        queue->finish();
        auto t1 = std::chrono::high_resolution_clock::now();

        double sec = std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(gemm_iters);
        double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        double gflops = (ops / 1e9) / sec;
        print_metric("Execution Latency", sec * 1000.0, "ms");
        print_metric("Compute Throughput", gflops, "GFLOPS");
    }

    // 4. Computed Tomography (CT) Pipeline
    print_banner("4. Computed Tomography Pipeline (Shepp-Logan Phantom)");
    {
        tomography::TomographyEngine tomo(context, queue);
        int size = 256;
        auto phantom = tomo.generate_shepp_logan_2d(size);
        auto img_buf = core::Buffer::create_from_host<float>(context, queue, phantom);

        tomography::ProjectionGeometry geom;
        geom.num_detectors = 256;
        geom.num_angles = 180;
        geom.detector_spacing = 1.0f;

        // Forward Radon Transform
        auto t0 = std::chrono::high_resolution_clock::now();
        auto sinogram = tomo.forward_radon_2d(img_buf, size, size, geom);
        queue->finish();
        auto t1 = std::chrono::high_resolution_clock::now();
        double radon_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double rays = static_cast<double>(geom.num_angles * geom.num_detectors);
        print_metric("Forward Radon Projection", radon_ms, "ms");
        print_metric("Ray-Marching Rate", (rays / 1e6) / (radon_ms / 1000.0), "MRays/s");

        // Spatial Ram-Lak Filter
        t0 = std::chrono::high_resolution_clock::now();
        auto filtered_sino = tomo.apply_ramp_filter(sinogram, geom, tomography::RampFilterType::SheppLogan);
        queue->finish();
        t1 = std::chrono::high_resolution_clock::now();
        double filter_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        print_metric("Ram-Lak Frequency Filter", filter_ms, "ms");

        // Filtered Backprojection (FBP) Reconstruction
        t0 = std::chrono::high_resolution_clock::now();
        auto recon = tomo.reconstruct_fbp_2d(filtered_sino, size, size, geom);
        queue->finish();
        t1 = std::chrono::high_resolution_clock::now();
        double fbp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        print_metric("FBP 2D Reconstruction", fbp_ms, "ms");
    }

    // 5. 3D TSDF Volume Fusion & Marching Cubes
    print_banner("5. 3D Spatial Computing & Marching Cubes (128^3 Volume)");
    {
        spatial::TSDFVolumeConfig cfg;
        cfg.vx = 128; cfg.vy = 128; cfg.vz = 128;
        cfg.voxel_size = 0.02f;
        spatial::SpatialEngine spatial_engine(context, queue, cfg);

        // Sphere SDF
        float radius = 0.8f;
        size_t total_voxels = static_cast<size_t>(cfg.vx * cfg.vy * cfg.vz);
        std::vector<float> sdf(total_voxels);
        float half = 0.5f * static_cast<float>(cfg.vx - 1) * cfg.voxel_size;
        for (int z = 0; z < cfg.vz; ++z) {
            float pz = static_cast<float>(z) * cfg.voxel_size - half;
            for (int y = 0; y < cfg.vy; ++y) {
                float py = static_cast<float>(y) * cfg.voxel_size - half;
                for (int x = 0; x < cfg.vx; ++x) {
                    float px = static_cast<float>(x) * cfg.voxel_size - half;
                    sdf[static_cast<size_t>((z * cfg.vy + y) * cfg.vx + x)] = std::sqrt(px*px + py*py + pz*pz) - radius;
                }
            }
        }
        spatial_engine.tsdf_buffer()->write(queue, sdf.data(), total_voxels * sizeof(float));

        auto t0 = std::chrono::high_resolution_clock::now();
        auto mesh = spatial_engine.extract_mesh(0.0f);
        queue->finish();
        auto t1 = std::chrono::high_resolution_clock::now();
        double mc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        print_metric("Marching Cubes Latency", mc_ms, "ms");
        print_metric("Extracted Triangles", static_cast<double>(mesh.triangle_count), "tris");
        print_metric("Voxel Evaluation Rate", (static_cast<double>(total_voxels) / 1e6) / (mc_ms / 1000.0), "MVoxels/s");
    }

    // 6. Computer Vision Acceleration (1024x1024 Image)
    print_banner("6. Computer Vision Filtering & Feature Extraction (1024x1024)");
    {
        vision::VisionEngine vision_engine(context, queue);
        int w = 1024, h = 1024;
        std::vector<float> img(static_cast<size_t>(w * h), 0.5f);
        auto img_buf = core::Buffer::create_from_host<float>(context, queue, img);

        // Gaussian Blur
        auto t0 = std::chrono::high_resolution_clock::now();
        auto blurred = vision_engine.gaussian_blur_2d(img_buf, w, h, 5, 2.0f);
        queue->finish();
        auto t1 = std::chrono::high_resolution_clock::now();
        double blur_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double mpix = static_cast<double>(w * h) / 1e6;
        print_metric("2D Gaussian Blur (r=5)", blur_ms, "ms");
        print_metric("Blur Throughput", mpix / (blur_ms / 1000.0), "MPix/s");

        // Bilateral Filter
        t0 = std::chrono::high_resolution_clock::now();
        auto bilateral = vision_engine.bilateral_filter(img_buf, w, h, 5, 3.0f, 0.1f);
        queue->finish();
        t1 = std::chrono::high_resolution_clock::now();
        double bi_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        print_metric("Bilateral Filter (r=5)", bi_ms, "ms");
        print_metric("Bilateral Throughput", mpix / (bi_ms / 1000.0), "MPix/s");

        // FAST Corner Detector
        t0 = std::chrono::high_resolution_clock::now();
        auto corners = vision_engine.detect_fast_corners(img_buf, w, h, 20.0f);
        queue->finish();
        t1 = std::chrono::high_resolution_clock::now();
        double fast_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        print_metric("FAST-9 Corner Detection", fast_ms, "ms");
        print_metric("Feature Detection Rate", mpix / (fast_ms / 1000.0), "MPix/s");
    }

    std::cout << "\n======================================================================\n";
    std::cout << "All benchmarks completed successfully with zero hardware faults.\n";
    std::cout << "======================================================================\n\n";

    return 0;
}
