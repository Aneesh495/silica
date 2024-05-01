#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <numeric>
#include "silica/silica.hpp"

using namespace silica;

static void print_ascii_art() {
    std::cout << R"(
   _____ _____ _      _____ _____          
  / ____|_   _| |    |_   _/ ____/\        
 | (___   | | | |      | || |   /  \       
  \___ \  | | | |      | || |  / /\ \      
  ____) |_| |_| |____ _| || |____/ ____ \  
 |_____/|_____|______|_____\_____/_/    \_\ 
  High-Performance Heterogeneous Computing Engine
  Apple Silicon & Discrete OpenCL Hardware Acceleration
)" << "\n";
}

static void print_help(const char* prog_name) {
    print_ascii_art();
    std::cout << "Usage: " << prog_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --help              Display this help manual\n";
    std::cout << "  --info              Display hardware accelerator and OpenCL telemetry\n";
    std::cout << "  --bench             Run full hardware latency, memory bandwidth, and GFLOPS suite\n";
    std::cout << "  --demo-tomography   Run CT forward Radon projection and FBP reconstruction\n";
    std::cout << "  --demo-tsdf         Run 3D TSDF fusion and Marching Cubes mesh extraction\n";
    std::cout << "  --demo-vision       Run 2D bilateral filtering and FAST-9 corner detection\n";
    std::cout << "  --demo-pointcloud   Run Morton code sorting and voxel grid downsampling\n\n";
}

static void cmd_info() {
    print_ascii_art();
    std::cout << "Hardware Discovery & System Telemetry:\n";
    std::cout << "----------------------------------------------------------------------\n";
    auto context = core::Context::create();
    const auto& dev = context->device();

    std::cout << "  Device Name          : " << dev.name() << "\n";
    std::cout << "  Vendor               : " << dev.vendor() << "\n";
    std::cout << "  Driver Version       : " << dev.driver_version() << "\n";
    std::cout << "  OpenCL C Version     : " << dev.opencl_version() << "\n";
    std::cout << "  Device Type          : " << (dev.is_gpu() ? "Dedicated/Integrated GPU" : "Host CPU") << "\n";
    std::cout << "  Unified Memory       : " << (dev.is_unified_memory() ? "Yes (Zero-Copy Shared Memory)" : "No (Discrete VRAM)") << "\n";
    std::cout << "  Compute Units        : " << dev.max_compute_units() << "\n";
    std::cout << "  Max Clock Frequency  : " << dev.max_clock_frequency() << " MHz\n";
    std::cout << "  Max Workgroup Size   : " << dev.max_work_group_size() << "\n";
    auto item_sizes = dev.max_work_item_sizes();
    std::cout << "  Max Work Item Sizes  : (" << item_sizes[0] << ", " << item_sizes[1] << ", " << item_sizes[2] << ")\n";
    std::cout << "  Global Memory Size   : " << (dev.global_memory_size() / (1024 * 1024)) << " MB\n";
    std::cout << "  Local Memory Size    : " << (dev.local_memory_size() / 1024) << " KB\n";
    std::cout << "----------------------------------------------------------------------\n\n";
}

static void cmd_demo_tomography() {
    std::cout << "\n[Tomography Demo] Shepp-Logan Phantom CT Reconstruction Pipeline\n";
    std::cout << "----------------------------------------------------------------------\n";

    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);
    tomography::TomographyEngine tomo(context, queue);

    const int img_size = 128;
    std::cout << "1. Generating analytical 2D Shepp-Logan phantom (" << img_size << "x" << img_size << ")...\n";
    auto phantom = tomo.generate_shepp_logan_2d(img_size);
    auto img_buf = core::Buffer::create_from_host<float>(context, queue, phantom);

    tomography::ProjectionGeometry geom;
    geom.num_detectors = 128;
    geom.num_angles = 180;
    geom.detector_spacing = 1.0f;

    std::cout << "2. Forward Radon Projection: ray-marching " << geom.num_angles << " projection angles x "
              << geom.num_detectors << " detectors on OpenCL device...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    auto sinogram = tomo.forward_radon_2d(img_buf, img_size, img_size, geom);
    queue->finish();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "   Completed in " << std::fixed << std::setprecision(2)
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    std::cout << "3. Applying Ram-Lak bandlimited spatial FIR filter to sinogram rows...\n";
    auto filtered_sino = tomo.apply_ramp_filter(sinogram, geom, tomography::RampFilterType::SheppLogan);
    queue->finish();

    std::cout << "4. Reconstructing spatial density via Filtered Back-Projection (FBP)...\n";
    t0 = std::chrono::high_resolution_clock::now();
    auto recon = tomo.reconstruct_fbp_2d(filtered_sino, img_size, img_size, geom);
    queue->finish();
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "   Completed in " << std::fixed << std::setprecision(2)
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    std::vector<float> recon_img(static_cast<size_t>(img_size * img_size));
    recon->read_vector(queue, recon_img);

    float min_val = *std::min_element(recon_img.begin(), recon_img.end());
    float max_val = *std::max_element(recon_img.begin(), recon_img.end());
    std::cout << "\nReconstructed Image Dynamic Range: [" << min_val << ", " << max_val << "]\n";

    std::cout << "\nASCII Preview of Reconstructed Phantom Center Slice (32x16 downsampled):\n";
    const char shades[] = " .:-=+*#%@";
    for (int y = 32; y < 96; y += 4) {
        std::cout << "  ";
        for (int x = 32; x < 96; x += 2) {
            float v = recon_img[static_cast<size_t>(y * img_size + x)];
            float norm = std::clamp((v - min_val) / std::max(1e-5f, max_val - min_val), 0.0f, 1.0f);
            size_t s_idx = static_cast<size_t>(norm * 9.0f);
            std::cout << shades[s_idx];
        }
        std::cout << "\n";
    }
    std::cout << "----------------------------------------------------------------------\n\n";
}

static void cmd_demo_tsdf() {
    std::cout << "\n[Spatial Demo] Truncated Signed Distance Field (TSDF) & Marching Cubes\n";
    std::cout << "----------------------------------------------------------------------\n";

    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    spatial::TSDFVolumeConfig cfg;
    cfg.vx = 64; cfg.vy = 64; cfg.vz = 64;
    cfg.voxel_size = 0.03f;
    cfg.truncation_mu = 0.09f;

    std::cout << "1. Allocating 3D TSDF Volume (" << cfg.vx << "x" << cfg.vy << "x" << cfg.vz
              << " voxels = " << (cfg.vx * cfg.vy * cfg.vz) << " total cells)...\n";
    spatial::SpatialEngine spatial_engine(context, queue, cfg);

    float sphere_radius = 0.6f;
    std::cout << "2. Synthesizing sphere isosurface SDF (radius = " << sphere_radius << " m)...\n";
    size_t num_voxels = static_cast<size_t>(cfg.vx * cfg.vy * cfg.vz);
    std::vector<float> sdf(num_voxels);
    float half = 0.5f * static_cast<float>(cfg.vx - 1) * cfg.voxel_size;

    for (int z = 0; z < cfg.vz; ++z) {
        float pz = static_cast<float>(z) * cfg.voxel_size - half;
        for (int y = 0; y < cfg.vy; ++y) {
            float py = static_cast<float>(y) * cfg.voxel_size - half;
            for (int x = 0; x < cfg.vx; ++x) {
                float px = static_cast<float>(x) * cfg.voxel_size - half;
                sdf[static_cast<size_t>((z * cfg.vy + y) * cfg.vx + x)] = std::sqrt(px*px + py*py + pz*pz) - sphere_radius;
            }
        }
    }
    spatial_engine.tsdf_buffer()->write(queue, sdf.data(), num_voxels * sizeof(float));

    std::cout << "3. Extracting polygonal mesh via 3D Marching Cubes on OpenCL GPU...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    auto mesh = spatial_engine.extract_mesh(0.0f);
    queue->finish();
    auto t1 = std::chrono::high_resolution_clock::now();

    double mc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "   Completed in " << std::fixed << std::setprecision(2) << mc_ms << " ms\n";
    std::cout << "   Extracted Triangles : " << mesh.triangle_count << "\n";
    std::cout << "   Extracted Vertices  : " << mesh.vertices.size() << "\n";

    if (!mesh.vertices.empty()) {
        std::cout << "   Sample Vertex 0     : ("
                  << mesh.vertices[0][0] << ", "
                  << mesh.vertices[0][1] << ", "
                  << mesh.vertices[0][2] << ")\n";
        std::cout << "   Sample Normal 0     : ("
                  << mesh.normals[0][0] << ", "
                  << mesh.normals[0][1] << ", "
                  << mesh.normals[0][2] << ")\n";
    }
    std::cout << "----------------------------------------------------------------------\n\n";
}

static void cmd_demo_vision() {
    std::cout << "\n[Vision Demo] Edge-Preserving Bilateral Filtering & FAST-9 Feature Points\n";
    std::cout << "----------------------------------------------------------------------\n";

    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);
    vision::VisionEngine vision_engine(context, queue);

    const int width = 512, height = 512;
    std::cout << "1. Generating synthetic test pattern with step discontinuities and corners ("
              << width << "x" << height << ")...\n";

    std::vector<float> img(static_cast<size_t>(width * height), 10.0f);
    // Draw 3 distinct bright rectangular objects
    for (int y = 50; y < 200; ++y) {
        for (int x = 50; x < 200; ++x) {
            img[static_cast<size_t>(y * width + x)] = 200.0f;
        }
    }
    for (int y = 300; y < 450; ++y) {
        for (int x = 100; x < 250; ++x) {
            img[static_cast<size_t>(y * width + x)] = 150.0f;
        }
    }

    auto in_buf = core::Buffer::create_from_host<float>(context, queue, img);

    std::cout << "2. Dispatching 2D Bilateral Filter (spatial sigma = 3.0, range sigma = 0.1)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    auto filtered_buf = vision_engine.bilateral_filter(in_buf, width, height, 5, 3.0f, 0.1f);
    queue->finish();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "   Bilateral filter executed in " << std::fixed << std::setprecision(2)
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    std::cout << "3. Dispatching FAST-9 Corner Detector on GPU...\n";
    t0 = std::chrono::high_resolution_clock::now();
    auto corners_buf = vision_engine.detect_fast_corners(in_buf, width, height, 25.0f);
    queue->finish();
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "   FAST-9 detector executed in " << std::fixed << std::setprecision(2)
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    std::vector<float> scores(static_cast<size_t>(width * height));
    corners_buf->read_vector(queue, scores);

    int corner_count = 0;
    for (float s : scores) {
        if (s > 0.0f) corner_count++;
    }
    std::cout << "   Total Detected Corner Features: " << corner_count << " keypoints\n";
    std::cout << "----------------------------------------------------------------------\n\n";
}

static void cmd_demo_pointcloud() {
    std::cout << "\n[PointCloud Demo] Morton Octree Codes & Voxel Grid Downsampling\n";
    std::cout << "----------------------------------------------------------------------\n";

    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);
    pointcloud::PointCloudEngine pc_engine(context, queue);

    const size_t num_points = 100000;
    std::cout << "1. Generating synthetic point cloud with " << num_points << " 3D points...\n";
    std::vector<float> points_data(num_points * 4);
    for (size_t i = 0; i < num_points; ++i) {
        float theta = static_cast<float>(i) * 0.01f;
        float r = 1.0f + 0.1f * std::sin(theta * 5.0f);
        points_data[i * 4 + 0] = r * std::cos(theta);
        points_data[i * 4 + 1] = r * std::sin(theta);
        points_data[i * 4 + 2] = static_cast<float>(i) * (2.0f / static_cast<float>(num_points)) - 1.0f;
        points_data[i * 4 + 3] = 1.0f;
    }

    auto pts_buf = core::Buffer::create_from_host<float>(context, queue, points_data);

    std::cout << "2. Computing 30-bit Morton z-order spatial curve codes...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    auto morton_buf = pc_engine.compute_morton_codes(pts_buf, num_points, {-2.0f, -2.0f, -2.0f}, {2.0f, 2.0f, 2.0f});
    queue->finish();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "   Morton codes computed in " << std::fixed << std::setprecision(2)
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    std::cout << "3. Downsampling via 3D Voxel Grid Centroid Aggregation (leaf size = 0.05 m)...\n";
    size_t out_count = 0;
    t0 = std::chrono::high_resolution_clock::now();
    auto downsampled = pc_engine.downsample_voxel_grid(
        pts_buf,
        num_points,
        0.05f,
        {-2.0f, -2.0f, -2.0f},
        {80, 80, 80},
        out_count
    );
    queue->finish();
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "   Downsampling executed in " << std::fixed << std::setprecision(2)
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
    std::cout << "   Reduced from " << num_points << " points to " << out_count << " voxel centroids\n";
    std::cout << "----------------------------------------------------------------------\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_help(argv[0]);
        return 0;
    }

    std::string arg = argv[1];

    if (arg == "--help" || arg == "-h") {
        print_help(argv[0]);
    } else if (arg == "--info") {
        cmd_info();
    } else if (arg == "--demo-tomography") {
        cmd_demo_tomography();
    } else if (arg == "--demo-tsdf") {
        cmd_demo_tsdf();
    } else if (arg == "--demo-vision") {
        cmd_demo_vision();
    } else if (arg == "--demo-pointcloud") {
        cmd_demo_pointcloud();
    } else if (arg == "--bench") {
        std::cout << "Launching full hardware benchmark harness...\n";
        cmd_info();
        cmd_demo_tomography();
        cmd_demo_tsdf();
        cmd_demo_vision();
        cmd_demo_pointcloud();
    } else {
        std::cerr << "Unknown argument: " << arg << "\n";
        print_help(argv[0]);
        return 1;
    }

    return 0;
}
