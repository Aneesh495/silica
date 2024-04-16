#include "silica/pipeline/pipeline.hpp"
#include "silica/core/error.hpp"
#include <queue>
#include <unordered_set>

namespace silica::pipeline {

void Pipeline::add_node(std::shared_ptr<PipelineNode> node) {
    SILICA_CHECK_THROW(node != nullptr, "Pipeline node cannot be null");
    SILICA_CHECK_THROW(node_map_.find(node->name()) == node_map_.end(),
                       "Duplicate node name in pipeline: " + node->name());

    nodes_.push_back(node);
    node_map_[node->name()] = node;
}

std::shared_ptr<PipelineNode> Pipeline::find_node(const std::string& name) const {
    auto it = node_map_.find(name);
    if (it != node_map_.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<std::shared_ptr<PipelineNode>> Pipeline::topological_order() const {
    // Map output buffer names to producer node
    std::unordered_map<std::string, std::string> output_producer;
    for (const auto& node : nodes_) {
        for (const auto& out : node->outputs()) {
            output_producer[out] = node->name();
        }
    }

    // Build in-degree and adjacency
    std::unordered_map<std::string, size_t> in_degree;
    std::unordered_map<std::string, std::vector<std::string>> adj;

    for (const auto& node : nodes_) {
        in_degree[node->name()] = 0;
    }

    for (const auto& node : nodes_) {
        for (const auto& in : node->inputs()) {
            auto it = output_producer.find(in);
            if (it != output_producer.end()) {
                const std::string& parent = it->second;
                if (parent != node->name()) {
                    adj[parent].push_back(node->name());
                    in_degree[node->name()]++;
                }
            }
        }
    }

    std::queue<std::string> ready;
    for (const auto& [name, deg] : in_degree) {
        if (deg == 0) {
            ready.push(name);
        }
    }

    std::vector<std::shared_ptr<PipelineNode>> sorted;
    sorted.reserve(nodes_.size());

    while (!ready.empty()) {
        std::string curr = ready.front();
        ready.pop();

        sorted.push_back(node_map_.at(curr));

        for (const auto& child : adj[curr]) {
            if (--in_degree[child] == 0) {
                ready.push(child);
            }
        }
    }

    SILICA_CHECK_THROW(sorted.size() == nodes_.size(),
                       "Cycle detected in pipeline DAG dependency graph");

    return sorted;
}

void Pipeline::validate_acyclic() const {
    (void)topological_order();
}

} // namespace silica::pipeline
