#pragma once

#include "silica/core/types.hpp"
#include "silica/core/buffer.hpp"
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace silica::core {

struct PoolStats {
    size_t total_allocated_bytes{0};
    size_t active_leased_bytes{0};
    size_t cached_buffers_count{0};
    size_t allocations_count{0};
    size_t reuses_count{0};
};

class MemoryPool {
public:
    explicit MemoryPool(std::shared_ptr<Context> context);
    ~MemoryPool();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    std::shared_ptr<Buffer> acquire(size_t size_bytes,
                                   MemoryAccess access = MemoryAccess::ReadWrite,
                                   MemoryLocation location = MemoryLocation::Device);

    void release(std::shared_ptr<Buffer> buffer);

    void trim();

    [[nodiscard]] PoolStats stats() const;

private:
    static size_t round_to_bucket(size_t size);

    std::shared_ptr<Context> context_;
    mutable std::mutex mutex_;
    std::unordered_map<size_t, std::vector<std::shared_ptr<Buffer>>> pool_;
    PoolStats stats_;
};

} // namespace silica::core
