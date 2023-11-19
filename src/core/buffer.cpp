#include "silica/core/buffer.hpp"
#include "silica/core/error.hpp"

namespace silica::core {

Buffer::~Buffer() {
    if (mem_) {
        clReleaseMemObject(mem_);
        mem_ = nullptr;
    }
}

Buffer::Buffer(Buffer&& other) noexcept
    : context_(std::move(other.context_)),
      mem_(other.mem_),
      size_bytes_(other.size_bytes_),
      access_(other.access_),
      location_(other.location_) {
    other.mem_ = nullptr;
    other.size_bytes_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        if (mem_) {
            clReleaseMemObject(mem_);
        }
        context_ = std::move(other.context_);
        mem_ = other.mem_;
        size_bytes_ = other.size_bytes_;
        access_ = other.access_;
        location_ = other.location_;
        other.mem_ = nullptr;
        other.size_bytes_ = 0;
    }
    return *this;
}

std::shared_ptr<Buffer> Buffer::create(const std::shared_ptr<Context>& context,
                                      size_t size_bytes,
                                      MemoryAccess access,
                                      MemoryLocation location,
                                      void* host_ptr) {
    SILICA_CHECK_THROW(context != nullptr, "Context cannot be null");
    SILICA_CHECK_THROW(size_bytes > 0, "Buffer size must be greater than zero");

    cl_mem_flags flags = 0;
    switch (access) {
        case MemoryAccess::ReadOnly:  flags |= CL_MEM_READ_ONLY; break;
        case MemoryAccess::WriteOnly: flags |= CL_MEM_WRITE_ONLY; break;
        case MemoryAccess::ReadWrite: flags |= CL_MEM_READ_WRITE; break;
    }

    switch (location) {
        case MemoryLocation::Device:
            break;
        case MemoryLocation::HostPinned:
            flags |= CL_MEM_ALLOC_HOST_PTR;
            break;
        case MemoryLocation::HostUnified:
            SILICA_CHECK_THROW(host_ptr != nullptr, "Host pointer required for HostUnified memory");
            flags |= CL_MEM_USE_HOST_PTR;
            break;
    }

    cl_int err = CL_SUCCESS;
    cl_mem raw = clCreateBuffer(context->raw_context(), flags, size_bytes, host_ptr, &err);
    SILICA_CHECK_CL(err);

    auto buf = std::shared_ptr<Buffer>(new Buffer());
    buf->context_ = context;
    buf->mem_ = raw;
    buf->size_bytes_ = size_bytes;
    buf->access_ = access;
    buf->location_ = location;

    return buf;
}

void Buffer::write(const std::shared_ptr<CommandQueue>& queue,
                   const void* host_ptr,
                   size_t bytes,
                   size_t offset,
                   bool blocking) const {
    SILICA_CHECK_THROW(queue != nullptr, "Command queue cannot be null");
    SILICA_CHECK_THROW(host_ptr != nullptr, "Host pointer cannot be null");
    SILICA_CHECK_THROW(offset + bytes <= size_bytes_, "Buffer write exceeds bounds");

    SILICA_CHECK_CL(clEnqueueWriteBuffer(
        queue->raw_queue(),
        mem_,
        blocking ? CL_TRUE : CL_FALSE,
        offset,
        bytes,
        host_ptr,
        0,
        nullptr,
        nullptr
    ));
}

void Buffer::read(const std::shared_ptr<CommandQueue>& queue,
                  void* host_ptr,
                  size_t bytes,
                  size_t offset,
                  bool blocking) const {
    SILICA_CHECK_THROW(queue != nullptr, "Command queue cannot be null");
    SILICA_CHECK_THROW(host_ptr != nullptr, "Host pointer cannot be null");
    SILICA_CHECK_THROW(offset + bytes <= size_bytes_, "Buffer read exceeds bounds");

    SILICA_CHECK_CL(clEnqueueReadBuffer(
        queue->raw_queue(),
        mem_,
        blocking ? CL_TRUE : CL_FALSE,
        offset,
        bytes,
        host_ptr,
        0,
        nullptr,
        nullptr
    ));
}

void Buffer::copy_to(const std::shared_ptr<CommandQueue>& queue,
                     const Buffer& dest,
                     size_t bytes,
                     size_t src_offset,
                     size_t dst_offset) const {
    SILICA_CHECK_THROW(queue != nullptr, "Command queue cannot be null");
    SILICA_CHECK_THROW(src_offset + bytes <= size_bytes_, "Buffer copy src out of bounds");
    SILICA_CHECK_THROW(dst_offset + bytes <= dest.size_bytes_, "Buffer copy dst out of bounds");

    SILICA_CHECK_CL(clEnqueueCopyBuffer(
        queue->raw_queue(),
        mem_,
        dest.mem_,
        src_offset,
        dst_offset,
        bytes,
        0,
        nullptr,
        nullptr
    ));
}

void Buffer::fill_pattern(const std::shared_ptr<CommandQueue>& queue,
                          const void* pattern,
                          size_t pattern_size,
                          size_t offset,
                          size_t size) const {
    SILICA_CHECK_THROW(queue != nullptr, "Command queue cannot be null");
    SILICA_CHECK_THROW(pattern != nullptr, "Pattern cannot be null");
    if (size == 0) {
        size = size_bytes_ - offset;
    }
    SILICA_CHECK_THROW(offset + size <= size_bytes_, "Buffer fill out of bounds");

    SILICA_CHECK_CL(clEnqueueFillBuffer(
        queue->raw_queue(),
        mem_,
        pattern,
        pattern_size,
        offset,
        size,
        0,
        nullptr,
        nullptr
    ));
}

void* Buffer::map(const std::shared_ptr<CommandQueue>& queue,
                  cl_map_flags map_flags,
                  size_t offset,
                  size_t bytes,
                  bool blocking) const {
    SILICA_CHECK_THROW(queue != nullptr, "Command queue cannot be null");
    if (bytes == 0) {
        bytes = size_bytes_ - offset;
    }
    SILICA_CHECK_THROW(offset + bytes <= size_bytes_, "Buffer map out of bounds");

    cl_int err = CL_SUCCESS;
    void* ptr = clEnqueueMapBuffer(
        queue->raw_queue(),
        mem_,
        blocking ? CL_TRUE : CL_FALSE,
        map_flags,
        offset,
        bytes,
        0,
        nullptr,
        nullptr,
        &err
    );
    SILICA_CHECK_CL(err);
    return ptr;
}

void Buffer::unmap(const std::shared_ptr<CommandQueue>& queue,
                   void* mapped_ptr) const {
    SILICA_CHECK_THROW(queue != nullptr, "Command queue cannot be null");
    SILICA_CHECK_THROW(mapped_ptr != nullptr, "Mapped pointer cannot be null");

    SILICA_CHECK_CL(clEnqueueUnmapMemObject(
        queue->raw_queue(),
        mem_,
        mapped_ptr,
        0,
        nullptr,
        nullptr
    ));
}

} // namespace silica::core
