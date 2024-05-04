# Silica

[![OpenCL 1.2](https://img.shields.io/badge/OpenCL-1.2-blue.svg)](https://www.khronos.org/opencl/)
[![C++20](https://img.shields.io/badge/Standard-C%2B%2B20-red.svg)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: macOS | Linux | Windows](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-green.svg)]()
[![Tests: 18/18 Passed](https://img.shields.io/badge/Tests-18%2F18%20Passed-brightgreen.svg)]()
[![Kernels: 100 Files | 30k+ LOC](https://img.shields.io/badge/OpenCL%20Kernels-30%2C272%20LOC-orange.svg)]()

Silica is a high-performance heterogeneous computing engine and hardware acceleration framework written from scratch in modern C++20 and OpenCL C. Designed as a unified computational platform for volumetric tomography, 3D spatial computing, real-time computer vision, point cloud analysis, and high-throughput linear algebra primitives, Silica demonstrates bare-metal hardware efficiency across Apple Silicon unified memory GPUs, AMD RDNA/CDNA architectures, and NVIDIA CUDA compute devices.

At the heart of Silica is a library of over **30,000 lines of hand-crafted OpenCL C kernels** spanning 100 dedicated source files. Each kernel is engineered with register-level tiling, Local Data Share (LDS) banking conflict mitigation, coalesced global memory access patterns, and mathematical optimizations.

---

## Architectural Overview

```
+---------------------------------------------------------------------------------------+
|                                    Client Application                                 |
|                     (CLI Workbench / Interactive Pipelines / Python Bindings)         |
+---------------------------------------------------------------------------------------+
                                           |
+---------------------------------------------------------------------------------------+
|                                    Silica Master API                                  |
|                            (#include "silica/silica.hpp")                             |
+---------------------------------------------------------------------------------------+
        |                     |                     |                    |
+---------------+     +---------------+     +---------------+    +---------------+
|  Tomography   |     |    Spatial    |     |    Vision     |    |  PointCloud   |
|    Engine     |     |    Engine     |     |    Engine     |    |    Engine     |
| (Radon/FBP)   |     | (TSDF/March)  |     | (Bilateral)   |    | (Morton/ICP)  |
+---------------+     +---------------+     +---------------+    +---------------+
        \                     \                     /                    /
         +--------------------------------------------------------------+
         |              Pipeline Graph & Execution Scheduler            |
         |         (DAG Task Graph, Event Barrier Synchronization)       |
         +--------------------------------------------------------------+
                                           |
+---------------------------------------------------------------------------------------+
|                                Core Host Runtime (C++20)                              |
|   * RAII Resource Managers (Context, CommandQueue, Program, Kernel, Buffer, Event)    |
|   * Lock-Free Memory Sub-Allocator & Buffer Recycling Pool (MemoryPool)               |
|   * Automatic Workgroup Sizing & SIMD Subgroup Multiple Dispatcher                   |
+---------------------------------------------------------------------------------------+
                                           |
+---------------------------------------------------------------------------------------+
|                            OpenCL 1.2 Driver & Hardware HAL                           |
|       Apple Silicon Unified Fabric  |  PCIe Discrete AMD/NVIDIA  |  Intel Iris/ARC    |
+---------------------------------------------------------------------------------------+
                                           |
+---------------------------------------------------------------------------------------+
|                         OpenCL C Kernel Compute Fabric (30,272 LOC)                   |
|   * Tomography  : 3D Cone-Beam FDK, Helical CT, ART, SIRT, CG-Krylov, Spatial Ram-Lak |
|   * Spatial     : Truncated Signed Distance Fields (TSDF), Sphere Tracing, Marching   |
|   * Vision      : Bilateral Filter, Gaussian Blur, Sobel, FAST-9, Harris, SGM Stereo  |
|   * PointCloud  : Morton 3D Codes, Voxel Grid Downsampling, PCA Normal Estimation     |
|   * Primitives  : Register-Tiled GEMM, Warp Reduction, Bitonic Sort, SpMV CSR         |
+---------------------------------------------------------------------------------------+
```

---

## Key Engineering Highlights

- **30,272 Lines of Production OpenCL C**: Hand-written computational kernels with zero synthetic boilerplate, featuring mathematical implementations of tomographic projection, raymarching, spatial octree partitioning, and feature detection.
- **Zero-Overhead Modern C++20 Runtime**: Idiomatic host wrapper utilizing `std::span`, concept-constrained templates, `std::shared_ptr` RAII lifetime management, and strong type safety.
- **Unified Memory Architecture Awareness**: Zero-copy host pointer mappings and unified memory support tailored specifically for Apple Silicon M-series unified memory fabric.
- **LDS Bank Conflict Mitigation**: Hardware LDS tile padding rules (`LDS_STRIDE_PAD`) that eliminate 32-way and 16-way memory bank conflicts during 2D/3D neighborhood processing.
- **Dynamic Workgroup Padding**: Mathematical dispatch alignment that guarantees global work sizes remain multiples of workgroup dimensions, avoiding invalid workgroup errors across different GPU architectures.
- **Zero Third-Party Runtime Dependencies**: Pure standard C++20 and Khronos OpenCL 1.2 headers. Compiles cleanly with `-Wall -Wextra -Wpedantic` on macOS, Linux, and Windows.

---

## Subsystem Architecture

### 1. Computed Tomography Engine (`silica::tomography`)
- **Forward Radon Transform**: Parallel-beam ray-marching line integration over arbitrary angular discretizations.
- **Frequency-Domain & Spatial Ram-Lak Filtering**: Bandlimited ramp filters including Ram-Lak, Shepp-Logan sinc, Cosine, and Hann windowing functions.
- **Filtered Back-Projection (FBP)**: Exact 2D back-projection reconstruction with sub-pixel bilinear interpolation.
- **Iterative Solvers (SIRT / ART)**: Algebraic Reconstruction Technique and Simultaneous Iterative Reconstruction Technique for sparse and noisy projection sets.
- **3D Cone-Beam FDK**: Feldkamp-Davis-Kress circular trajectory volumetric reconstruction.

### 2. 3D Spatial & Volumetric TSDF Engine (`silica::spatial`)
- **Real-Time TSDF Fusion**: Point-to-camera projection and truncated projective signed distance accumulation with running weight normalization.
- **Volumetric Sphere Tracer**: Multi-scale ray-marching through voxel volumes with early ray termination and normal gradient calculation.
- **3D Marching Cubes**: Parallel voxel classification, edge intersection lookup tables, and atomic triangle vertex emission.

### 3. Real-Time Vision & Feature Engine (`silica::vision`)
- **Edge-Preserving Bilateral Filter**: Joint spatial-range Gaussian weighting caching tiles into Local Data Share (LDS).
- **Separable Gaussian Smoothing**: Two-pass horizontal and vertical convolutions caching halo boundary margins.
- **Feature Keypoint Extraction**: FAST-9 circular arc early-rejection detector and Harris corner response operator with non-maximum suppression.
- **Dense Stereo Disparity**: Semi-Global Matching (SGM) multi-path cost aggregation and Census transform.

### 4. Point Cloud Processing Engine (`silica::pointcloud`)
- **Morton Code Encoding**: 30-bit 3D Morton space-filling curves for spatial locality and linear octree hierarchy construction.
- **Voxel Grid Downsampling**: High-speed centroid accumulation using atomic floating-point arithmetic.
- **PCA Normal Estimation**: Local neighborhood covariance matrix calculation and 3x3 Jacobi eigenvalue decomposition.
- **Iterative Closest Point (ICP)**: Point-to-plane rigid registration with 6x6 normal equations linear system reduction.

### 5. High-Throughput Linear Algebra & Primitives (`silica::linalg`)
- **Register-Tiled GEMM**: Multi-level register tiling (4x4 register blocking per thread) with LDS double buffering.
- **Multi-Stage Parallel Reductions**: Grid-stride accumulation and intra-workgroup logarithmic reduction trees.
- **Bitonic Sorting**: Fully parallel sorting networks handling power-of-two and padded buffers.
- **Sparse Matrix-Vector Multiply (SpMV)**: Compressed Sparse Row (CSR) format matrix multiplication.

---

## Microbenchmark Performance

Measured directly on Apple Silicon M2 GPU (10 Compute Units, 12 GB Unified Memory, macOS 15.3):

| Benchmark Test Case | Problem Size / Configuration | Latency | Measured Throughput |
|:---|:---|:---:|:---:|
| **Host-to-Device (H2D) Bandwidth** | 64 MB Transfer Buffer | 2.93 ms | **21.83 GB/s** |
| **Device-to-Host (D2H) Bandwidth** | 64 MB Transfer Buffer | 3.00 ms | **21.34 GB/s** |
| **Device-to-Device (D2D) Bandwidth** | 64 MB Copy Buffer | 2.53 ms | **25.33 GB/s** |
| **Kernel Launch Latency** | Synchronous Enqueue & Finish | 0.17 ms | **178 us / launch** |
| **Register-Tiled GEMM** | Matrix Dimensions 512 x 512 x 512 | 0.73 ms | **368.32 GFLOPS** |
| **Forward Radon Projection** | 256x256 Phantom, 180 Angles x 256 Det | 3.44 ms | **13.41 MRays/s** |
| **Filtered Backprojection (FBP)** | 256x256 Image Reconstruction | 1.39 ms | **47.19 MRecon/s** |
| **3D Marching Cubes** | 128x128x128 Volume (40,192 Triangles) | 5.29 ms | **396.21 MVoxels/s** |
| **2D Gaussian Blur (Radius=5)** | 1024 x 1024 Grayscale Image | 0.93 ms | **1,125.74 MPixels/s** |
| **2D Bilateral Filter (Radius=5)** | 1024 x 1024 Grayscale Image | 5.63 ms | **186.12 MPixels/s** |
| **FAST-9 Feature Detector** | 1024 x 1024 Grayscale Image | 0.60 ms | **1,733.54 MPixels/s** |

---

## Quickstart & Build Instructions

### Prerequisites
- Modern C++20 compiler: Clang (>= 14.0), GCC (>= 11.0), or AppleClang (>= 14.0).
- CMake (>= 3.20).
- OpenCL 1.2 runtime:
  - **macOS**: Built-in system OpenCL Framework.
  - **Linux**: NVIDIA CUDA Toolkit, AMD ROCm, or Intel oneAPI OpenCL driver (`sudo apt install ocl-icd-opencl-dev opencl-headers`).
  - **Windows**: OpenCL SDK or vendor GPU driver.

### Building from Source

```bash
# Clone the repository
git clone https://github.com/Aneesh495/silica.git
cd silica

# Configure CMake with Release optimizations
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Compile all targets (libsilica_core, silica CLI, silica_tests, silica_bench)
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

### Running the Test Suite
Silica includes an integrated unit test runner verifying all 6 core subsystems:

```bash
./build/silica_tests
```

Output:
```
======================================================================
                  Silica OpenCL Test Suite Runner                     
======================================================================

[ RUN      ] Core.DeviceDiscovery ... [PASS] (186.41 ms)
[ RUN      ] Core.ContextAndQueueCreation ... [PASS] (0.60 ms)
[ RUN      ] Core.BufferHostReadWrite ... [PASS] (5.05 ms)
[ RUN      ] Core.MemoryPoolLeaseAndRecycle ... [PASS] (0.31 ms)
[ RUN      ] Core.ProgramCompilationAndExecution ... [PASS] (6.57 ms)
[ RUN      ] PointCloud.MortonCodeOrdering ... [PASS] (19.24 ms)
[ RUN      ] PointCloud.VoxelGridDownsampling ... [PASS] (20.70 ms)
[ RUN      ] PointCloud.NormalEstimationPCA ... [PASS] (16.66 ms)
[ RUN      ] Primitives.ParallelReductionSum ... [PASS] (4.35 ms)
[ RUN      ] Primitives.RegisterTiledGEMM ... [PASS] (5.28 ms)
[ RUN      ] Spatial.TSDFVolumeReset ... [PASS] (10.48 ms)
[ RUN      ] Spatial.MarchingCubesSphereExtraction ... [PASS] (13.80 ms)
[ RUN      ] Tomography.SheppLoganPhantomGeneration ... [PASS] (13.78 ms)
[ RUN      ] Tomography.ForwardRadonAndFBPReconstruction ... [PASS] (14.84 ms)
[ RUN      ] Tomography.SIRTIterativeSolver ... [PASS] (11.35 ms)
[ RUN      ] Vision.GaussianBlurSmoothing ... [PASS] (15.41 ms)
[ RUN      ] Vision.BilateralFilterEdgePreservation ... [PASS] (12.70 ms)
[ RUN      ] Vision.FastCornerDetector ... [PASS] (11.29 ms)

======================================================================
TEST RESULTS: 18 Passed, 0 Failed in 369 ms
======================================================================
```

### Running the Benchmark Suite
```bash
./build/silica_bench
```

### CLI Workbench Usage
```bash
# Display hardware discovery telemetry
./build/silica --info

# Run Computed Tomography forward Radon projection & FBP reconstruction with ASCII slice viewer
./build/silica --demo-tomography

# Run 3D TSDF sphere isosurface fusion & Marching Cubes mesh extraction
./build/silica --demo-tsdf

# Run 2D Bilateral filtering & FAST-9 corner keypoint extraction
./build/silica --demo-vision

# Run Morton space-filling curve encoding & 3D voxel grid downsampling
./build/silica --demo-pointcloud
```

---

## Repository Structure

```
silica/
├── CMakeLists.txt              # CMake build orchestration (C++20, OpenCL)
├── include/
│   └── silica/
│       ├── silica.hpp          # Unified master library header
│       ├── core/               # Host runtime (Context, Queue, Buffer, Kernel)
│       ├── tomography/         # CT reconstruction (Radon, FBP, SIRT, FDK)
│       ├── spatial/            # 3D TSDF volume & Marching Cubes
│       ├── vision/             # Computer vision & feature detection
│       ├── pointcloud/         # Octrees, Morton curves, ICP registration
│       └── pipeline/           # Asynchronous execution graphs & profilers
├── kernels/                    # OpenCL C kernel implementations (30,272 LOC)
│   ├── common/                 # Vector algebra, math constants, memory helpers
│   ├── primitives/             # Reductions, GEMM, bitonic sort, SpMV
│   ├── tomography/             # Radon transform, Ram-Lak filter, FBP, SIRT
│   ├── tsdf/                   # Volume integrate, raymarcher, marching cubes
│   ├── image/                  # Bilateral, Gaussian, Canny, anisotropic diffusion
│   ├── features/               # FAST-9, Harris, ORB, SIFT, AKAZE
│   ├── stereo/                 # Census transform, SGM cost aggregation
│   ├── pointcloud/             # Morton codes, voxel grid downsampling, PCA
│   └── specialized/            # Unrolled GEMM, Krylov solvers, clinical imaging
├── src/                        # C++20 engine source implementations
│   ├── core/                   # OpenCL driver abstractions & memory pools
│   ├── tomography/             # CT reconstruction host engine
│   ├── spatial/                # TSDF fusion & mesh extraction host engine
│   ├── vision/                 # Image processing host engine
│   ├── pointcloud/             # Point cloud analysis host engine
│   └── app/                    # Silica CLI workbench main application
├── benchmarks/                 # Latency, bandwidth, and GFLOPS microbenchmark harness
├── tests/                      # Comprehensive unit and integration test suite
└── docs/                       # Technical documentation
    ├── ARCHITECTURE.md         # Detailed runtime and queuing model
    ├── KERNELS.md              # Mathematical derivations and kernel complexity
    └── REPRODUCIBILITY.md      # Clean build and reproduction guide
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
