#pragma once

#include "silica/core/types.hpp"
#include "silica/core/context.hpp"
#include <memory>

namespace silica::core {

class CommandQueue {
public:
    ~CommandQueue();
    CommandQueue(const CommandQueue&) = delete;
    CommandQueue& operator=(const CommandQueue&) = delete;
    CommandQueue(CommandQueue&& other) noexcept;
    CommandQueue& operator=(CommandQueue&& other) noexcept;

    static std::shared_ptr<CommandQueue> create(const std::shared_ptr<Context>& context,
                                                bool enable_profiling = true,
                                                bool out_of_order = false);

    [[nodiscard]] cl_command_queue raw_queue() const noexcept { return queue_; }
    [[nodiscard]] const std::shared_ptr<Context>& context() const noexcept { return context_; }

    void finish() const;
    void flush() const;
    void barrier() const;

private:
    CommandQueue() = default;

    std::shared_ptr<Context> context_;
    cl_command_queue queue_{nullptr};
    bool profiling_enabled_{true};
};

} // namespace silica::core
