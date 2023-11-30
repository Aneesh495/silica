#pragma once

#include "silica/core/types.hpp"
#include "silica/core/queue.hpp"
#include "silica/core/buffer.hpp"
#include "silica/core/event.hpp"
#include "silica/core/memory_pool.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace silica::pipeline {

class PipelineNode {
public:
    virtual ~PipelineNode() = default;

    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] const std::vector<std::string>& inputs() const noexcept { return inputs_; }
    [[nodiscard]] const std::vector<std::string>& outputs() const noexcept { return outputs_; }

    virtual core::Event execute(
        const std::shared_ptr<core::CommandQueue>& queue,
        core::MemoryPool& pool,
        const std::vector<core::Event>& dependency_events) = 0;

protected:
    explicit PipelineNode(std::string name,
                          std::vector<std::string> inputs = {},
                          std::vector<std::string> outputs = {})
        : name_(std::move(name)),
          inputs_(std::move(inputs)),
          outputs_(std::move(outputs)) {}

    std::string name_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
};

} // namespace silica::pipeline
