#include "silica/pipeline/scheduler.hpp"
#include "silica/core/error.hpp"
#include <chrono>

namespace silica::pipeline {

Scheduler::Scheduler(std::shared_ptr<core::CommandQueue> queue,
                     std::shared_ptr<core::MemoryPool> memory_pool)
    : queue_(std::move(queue)),
      memory_pool_(std::move(memory_pool)) {
    SILICA_CHECK_THROW(queue_ != nullptr, "Command queue cannot be null");
    SILICA_CHECK_THROW(memory_pool_ != nullptr, "Memory pool cannot be null");
}

ExecutionReport Scheduler::execute(const Pipeline& pipeline) {
    auto sorted_nodes = pipeline.topological_order();

    std::unordered_map<std::string, std::string> output_to_node;
    for (const auto& node : sorted_nodes) {
        for (const auto& out : node->outputs()) {
            output_to_node[out] = node->name();
        }
    }

    std::unordered_map<std::string, core::Event> node_events;
    ExecutionReport report;

    auto start_wall = std::chrono::high_resolution_clock::now();

    for (const auto& node : sorted_nodes) {
        std::vector<core::Event> dependencies;
        for (const auto& in : node->inputs()) {
            auto it = output_to_node.find(in);
            if (it != output_to_node.end()) {
                const std::string& parent_name = it->second;
                auto ev_it = node_events.find(parent_name);
                if (ev_it != node_events.end() && ev_it->second.valid()) {
                    dependencies.push_back(std::move(ev_it->second));
                }
            }
        }

        core::Event node_ev = node->execute(queue_, *memory_pool_, dependencies);
        if (node_ev.valid()) {
            node_ev.wait();
            report.node_durations_ms[node->name()] = node_ev.duration_ms();
            node_events[node->name()] = std::move(node_ev);
        } else {
            report.node_durations_ms[node->name()] = 0.0;
        }
    }

    queue_->finish();

    auto end_wall = std::chrono::high_resolution_clock::now();
    report.total_wall_time_ms = std::chrono::duration<double, std::milli>(end_wall - start_wall).count();

    return report;
}

} // namespace silica::pipeline
