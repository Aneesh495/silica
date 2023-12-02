#pragma once

#include "silica/pipeline/pipeline.hpp"
#include "silica/core/queue.hpp"
#include "silica/core/memory_pool.hpp"
#include <unordered_map>
#include <string>
#include <memory>

namespace silica::pipeline {

struct ExecutionReport {
    std::unordered_map<std::string, double> node_durations_ms;
    double total_wall_time_ms{0.0};
};

class Scheduler {
public:
    Scheduler(std::shared_ptr<core::CommandQueue> queue,
              std::shared_ptr<core::MemoryPool> memory_pool);

    ExecutionReport execute(const Pipeline& pipeline);

private:
    std::shared_ptr<core::CommandQueue> queue_;
    std::shared_ptr<core::MemoryPool> memory_pool_;
};

} // namespace silica::pipeline
