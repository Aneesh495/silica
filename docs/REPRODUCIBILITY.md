# Silica Build & Verification Reproducibility Guide

This guide provides instructions for cloning, compiling, verifying, and benchmarking Silica from source across supported platforms and GPU architectures.

---

## Supported Environments

| Operating System | Recommended Compiler | OpenCL Backend | Verified Hardware |
|:---|:---|:---|:---|
| **macOS (Apple Silicon)** | AppleClang >= 14.0 | System OpenCL Framework | Apple M1 / M2 / M3 / M4 |
| **Linux (Ubuntu / RHEL / Arch)** | GCC >= 11.0 / Clang >= 14.0 | NVIDIA CUDA / AMD ROCm / Intel oneAPI / PoCL | NVIDIA RTX / Tesla, AMD Radeon, Intel ARC |
| **Windows 10 / 11** | MSVC 2022 / Clang-CL | Vendor OpenCL ICD Driver | NVIDIA GeForce / AMD Radeon |

---

## 1. Prerequisites Installation

### macOS
Install Xcode Command Line Tools and CMake:
```bash
xcode-select --install
brew install cmake ninja
```
The OpenCL Framework (`OpenCL.framework`) is pre-installed with macOS.

### Linux (Debian / Ubuntu)
Install modern build tools and OpenCL developer headers:
```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build clang \
                    ocl-icd-opencl-dev opencl-headers clinfo
```

To verify that your GPU OpenCL driver is functional:
```bash
clinfo
```

---

## 2. Cloning and Configuring

```bash
# Clone the repository
git clone https://github.com/Aneesh495/silica.git
cd silica

# Configure CMake with Release optimization flags
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

---

## 3. Compilation

Compile all targets in parallel:
```bash
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

Target artifacts generated in `build/`:
- `libsilica_core.a` : Core C++20 static runtime library
- `silica`           : CLI workbench application
- `silica_tests`     : Automated unit and integration test runner
- `silica_bench`     : Hardware latency, bandwidth, and throughput benchmark suite

---

## 4. Running the Test Suite

Run the full automated test suite:
```bash
./build/silica_tests
```

All 18 tests should report `[PASS]` with zero failures:
```
======================================================================
                  Silica OpenCL Test Suite Runner                     
======================================================================

[ RUN      ] Core.DeviceDiscovery ... [PASS]
[ RUN      ] Core.ContextAndQueueCreation ... [PASS]
[ RUN      ] Core.BufferHostReadWrite ... [PASS]
[ RUN      ] Core.MemoryPoolLeaseAndRecycle ... [PASS]
[ RUN      ] Core.ProgramCompilationAndExecution ... [PASS]
[ RUN      ] PointCloud.MortonCodeOrdering ... [PASS]
[ RUN      ] PointCloud.VoxelGridDownsampling ... [PASS]
[ RUN      ] PointCloud.NormalEstimationPCA ... [PASS]
[ RUN      ] Primitives.ParallelReductionSum ... [PASS]
[ RUN      ] Primitives.RegisterTiledGEMM ... [PASS]
[ RUN      ] Spatial.TSDFVolumeReset ... [PASS]
[ RUN      ] Spatial.MarchingCubesSphereExtraction ... [PASS]
[ RUN      ] Tomography.SheppLoganPhantomGeneration ... [PASS]
[ RUN      ] Tomography.ForwardRadonAndFBPReconstruction ... [PASS]
[ RUN      ] Tomography.SIRTIterativeSolver ... [PASS]
[ RUN      ] Vision.GaussianBlurSmoothing ... [PASS]
[ RUN      ] Vision.BilateralFilterEdgePreservation ... [PASS]
[ RUN      ] Vision.FastCornerDetector ... [PASS]

======================================================================
TEST RESULTS: 18 Passed, 0 Failed
======================================================================
```

---

## 5. Running the Microbenchmark Suite

Measure bare-metal memory transfer bandwidth, dispatch latency, and compute throughput:
```bash
./build/silica_bench
```

Expected outputs include:
- Host-to-Device (H2D) and Device-to-Host (D2H) memory bandwidth (> 20 GB/s on unified memory).
- Kernel launch dispatch latency (< 200 us).
- Register-tiled GEMM compute throughput (> 300 GFLOPS on modern integrated GPUs).
- Tomography ray-marching throughput (> 10 MRays/s).
- 3D Marching Cubes isosurface extraction rate (> 300 MVoxels/s).
- 2D Gaussian blur and Bilateral filter throughput (> 1,000 MPixels/s).

---

## 6. Running Interactive CLI Demonstrations

```bash
# Query accelerator details and device limits
./build/silica --info

# Computed Tomography forward Radon projection & FBP reconstruction
./build/silica --demo-tomography

# 3D TSDF sphere isosurface fusion & Marching Cubes mesh extraction
./build/silica --demo-tsdf

# 2D Bilateral filtering & FAST-9 feature detection
./build/silica --demo-vision

# 30-bit Morton code spatial curves & point cloud voxel downsampling
./build/silica --demo-pointcloud
```

---

## 7. Verifying Code Standards & Line Counts

To verify that the project strictly contains zero em-dash and en-dash characters:
```bash
python3 scripts/verify_dashes.py
```

To display line-of-code statistics across OpenCL kernels and C++ runtime:
```bash
python3 scripts/count_loc.py
```
