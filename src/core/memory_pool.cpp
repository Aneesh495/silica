#include "silica/core/memory_pool.hpp"

namespace silica::core {

MemoryPool::MemoryPool(std::shared_ptr<Context> context)
    : context_(std::move(context)) {}

MemoryPool::~MemoryPool() {
    trim();
}

size_t MemoryPool::round_to_bucket(size_t size) {
    if (size <= 256) return 256;
    size_t power = 1;
    while (power < size) {
        power <<= 1;
    }
    return power;
}

std::shared_ptr<Buffer> MemoryPool::acquire(size_t size_bytes,
                                            MemoryAccess access,
                                            MemoryLocation location) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t bucket = round_to_bucket(size_bytes);

    auto it = pool_.find(bucket);
    if (it != pool_.end() && !it->second.empty()) {
        auto buf = it->second.back();
        it->second.pop_back();
        stats_.cached_buffers_count--;
        stats_.active_leased_bytes += bucket;
        stats_.reuses_count++;
        return buf;
    }

    // Allocate new buffer
    auto new_buf = Buffer::create(context_, bucket, access, location);
    stats_.total_allocated_bytes += bucket;
    stats_.active_leased_bytes += bucket;
    stats_.allocations_count++;
    return new_buf;
}

void MemoryPool::release(std::shared_ptr<Buffer> buffer) {
    if (!buffer) return;

    std::lock_guard<std::mutex> lock(mutex_);
    size_t bucket = buffer->size_bytes();
    pool_[bucket].push_back(buffer);
    if (stats_.active_leased_bytes >= bucket) {
        stats_.active_leased_bytes -= bucket;
    }
    stats_.cached_buffers_count++;
}

void MemoryPool::trim() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [bucket, vec] : pool_) {
        for (const auto& buf : vec) {
            if (stats_.total_allocated_bytes >= bucket) {
                stats_.total_allocated_bytes -= bucket;
            }
        }
        vec.clear();
    }
    pool_.clear();
    stats_.cached_buffers_count = 0;
}

PoolStats MemoryPool::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

} // namespace silica::core
