#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <array>
#include <vector>

#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace silica::core {

enum class DeviceType {
    Default,
    GPU,
    CPU,
    All
};

enum class MemoryAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite
};

enum class MemoryLocation {
    Device,
    HostPinned,
    HostUnified
};

struct Dim2 {
    size_t x{0};
    size_t y{0};

    [[nodiscard]] constexpr size_t total() const noexcept {
        return x * y;
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return x == 0 || y == 0;
    }
};

struct Dim3 {
    size_t x{0};
    size_t y{0};
    size_t z{0};

    [[nodiscard]] constexpr size_t total() const noexcept {
        return x * y * z;
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return x == 0 || y == 0 || z == 0;
    }
};

struct WorkSize {
    Dim3 global{1, 1, 1};
    Dim3 local{0, 0, 0};
    size_t dimensions{1};

    WorkSize() = default;

    WorkSize(std::initializer_list<size_t> g, std::initializer_list<size_t> l = {}) {
        dimensions = g.size();
        auto git = g.begin();
        if (dimensions >= 1) global.x = *git++;
        if (dimensions >= 2) global.y = *git++;
        if (dimensions >= 3) global.z = *git++;

        auto lit = l.begin();
        size_t l_size = l.size();
        if (l_size >= 1) local.x = *lit++;
        if (l_size >= 2) local.y = *lit++;
        if (l_size >= 3) local.z = *lit++;
    }

    static WorkSize make_1d(size_t global_x, size_t local_x = 0) {
        WorkSize ws;
        ws.dimensions = 1;
        ws.global = {global_x, 1, 1};
        ws.local = {local_x, 0, 0};
        return ws;
    }

    static WorkSize make_2d(size_t gx, size_t gy, size_t lx = 0, size_t ly = 0) {
        WorkSize ws;
        ws.dimensions = 2;
        ws.global = {gx, gy, 1};
        ws.local = {lx, ly, 0};
        return ws;
    }

    static WorkSize make_3d(size_t gx, size_t gy, size_t gz, size_t lx = 0, size_t ly = 0, size_t lz = 0) {
        WorkSize ws;
        ws.dimensions = 3;
        ws.global = {gx, gy, gz};
        ws.local = {lx, ly, lz};
        return ws;
    }
};

enum class PixelFormat {
    Gray8,
    Gray16,
    Gray32F,
    RGB8,
    RGBA8,
    RGBA32F,
    BayerRGGB8,
    BayerBGGR8
};

struct ImageDesc {
    size_t width{0};
    size_t height{0};
    size_t channels{1};
    size_t bytes_per_pixel{1};
    PixelFormat format{PixelFormat::Gray8};

    [[nodiscard]] size_t row_pitch() const noexcept {
        return width * channels * bytes_per_pixel;
    }

    [[nodiscard]] size_t total_bytes() const noexcept {
        return row_pitch() * height;
    }
};

struct VolumeDesc {
    size_t width{0};
    size_t height{0};
    size_t depth{0};
    float voxel_size_x{1.0f};
    float voxel_size_y{1.0f};
    float voxel_size_z{1.0f};

    [[nodiscard]] size_t total_voxels() const noexcept {
        return width * height * depth;
    }

    [[nodiscard]] size_t total_bytes(size_t bytes_per_voxel = sizeof(float)) const noexcept {
        return total_voxels() * bytes_per_voxel;
    }
};

struct CameraIntrinsics {
    float fx{0.0f};
    float fy{0.0f};
    float cx{0.0f};
    float cy{0.0f};
    float k1{0.0f};
    float k2{0.0f};
    float p1{0.0f};
    float p2{0.0f};
};

struct Matrix3x3 {
    std::array<float, 9> m{
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
};

struct Matrix4x4 {
    std::array<float, 16> m{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
};

} // namespace silica::core
