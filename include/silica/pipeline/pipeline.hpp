#pragma once

#include "silica/pipeline/node.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace silica::pipeline {

class Pipeline {
public:
    Pipeline() = default;
    ~Pipeline() = default;

    void add_node(std::shared_ptr<PipelineNode> node);
    [[nodiscard]] const std::vector<std::shared_ptr<PipelineNode>>& nodes() const noexcept { return nodes_; }
    [[nodiscard]] std::shared_ptr<PipelineNode> find_node(const std::string& name) const;

    [[nodiscard]] std::vector<std::shared_ptr<PipelineNode>> topological_order() const;

    void validate_acyclic() const;

private:
    std::vector<std::shared_ptr<PipelineNode>> nodes_;
    std::unordered_map<std::string, std::shared_ptr<PipelineNode>> node_map_;
};

} // namespace silica::pipeline
