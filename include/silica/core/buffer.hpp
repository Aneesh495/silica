#pragma once

#include "silica/core/types.hpp"
#include "silica/core/context.hpp"
#include "silica/core/queue.hpp"
#include <memory>
#include <vector>
#include <span>

namespace silica::core {

class Event;

class Buffer {
public:
    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    static std::shared_ptr<Buffer> create(const std::shared_ptr<Context>& context,
                                          size_t size_bytes,
                                          MemoryAccess access = MemoryAccess::ReadWrite,
                                          MemoryLocation location = MemoryLocation::Device,
                                          void* host_ptr = nullptr);

    template<typename T>
    static std::shared_ptr<Buffer> create_from_host(const std::shared_ptr<Context>& context,
                                                    const std::shared_ptr<CommandQueue>& queue,
                                                    std::span<const T> data,
                                                    MemoryAccess access = MemoryAccess::ReadWrite) {
        auto buf = create(context, data.size_bytes(), access, MemoryLocation::Device);
        buf->write(queue, data.data(), data.size_bytes());
        return buf;
    }

    [[nodiscard]] cl_mem raw_mem() const noexcept { return mem_; }
    [[nodiscard]] size_t size_bytes() const noexcept { return size_bytes_; }
    [[nodiscard]] MemoryAccess access() const noexcept { return access_; }
    [[nodiscard]] MemoryLocation location() const noexcept { return location_; }

    void write(const std::shared_ptr<CommandQueue>& queue,
               const void* host_ptr,
               size_t bytes,
               size_t offset = 0,
               bool blocking = true) const;

    void read(const std::shared_ptr<CommandQueue>& queue,
              void* host_ptr,
              size_t bytes,
              size_t offset = 0,
              bool blocking = true) const;

    template<typename T>
    void read_vector(const std::shared_ptr<CommandQueue>& queue,
                     std::vector<T>& out_vec) const {
        out_vec.resize(size_bytes_ / sizeof(T));
        read(queue, out_vec.data(), size_bytes_, 0, true);
    }

    void copy_to(const std::shared_ptr<CommandQueue>& queue,
                 const Buffer& dest,
                 size_t bytes,
                 size_t src_offset = 0,
                 size_t dst_offset = 0) const;

    void fill_pattern(const std::shared_ptr<CommandQueue>& queue,
                      const void* pattern,
                      size_t pattern_size,
                      size_t offset = 0,
                      size_t size = 0) const;

    void* map(const std::shared_ptr<CommandQueue>& queue,
              cl_map_flags map_flags,
              size_t offset = 0,
              size_t bytes = 0,
              bool blocking = true) const;

    void unmap(const std::shared_ptr<CommandQueue>& queue,
               void* mapped_ptr) const;

private:
    Buffer() = default;

    std::shared_ptr<Context> context_;
    cl_mem mem_{nullptr};
    size_t size_bytes_{0};
    MemoryAccess access_{MemoryAccess::ReadWrite};
    MemoryLocation location_{MemoryLocation::Device};
};

} // namespace silica::core
