#include "silica/core/queue.hpp"
#include "silica/core/error.hpp"

namespace silica::core {

CommandQueue::~CommandQueue() {
    if (queue_) {
        clReleaseCommandQueue(queue_);
        queue_ = nullptr;
    }
}

CommandQueue::CommandQueue(CommandQueue&& other) noexcept
    : context_(std::move(other.context_)),
      queue_(other.queue_),
      profiling_enabled_(other.profiling_enabled_) {
    other.queue_ = nullptr;
}

CommandQueue& CommandQueue::operator=(CommandQueue&& other) noexcept {
    if (this != &other) {
        if (queue_) {
            clReleaseCommandQueue(queue_);
        }
        context_ = std::move(other.context_);
        queue_ = other.queue_;
        profiling_enabled_ = other.profiling_enabled_;
        other.queue_ = nullptr;
    }
    return *this;
}

std::shared_ptr<CommandQueue> CommandQueue::create(const std::shared_ptr<Context>& context,
                                                   bool enable_profiling,
                                                   bool out_of_order) {
    SILICA_CHECK_THROW(context != nullptr, "Context cannot be null");

    auto queue_obj = std::shared_ptr<CommandQueue>(new CommandQueue());
    queue_obj->context_ = context;
    queue_obj->profiling_enabled_ = enable_profiling;

    cl_command_queue_properties props = 0;
    if (enable_profiling) {
        props |= CL_QUEUE_PROFILING_ENABLE;
    }
    if (out_of_order) {
        props |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    }

    cl_int err = CL_SUCCESS;
    queue_obj->queue_ = clCreateCommandQueue(
        context->raw_context(),
        context->device().raw_id(),
        props,
        &err
    );

    // If out of order failed, fallback to in-order queue
    if (err != CL_SUCCESS && out_of_order) {
        props &= ~static_cast<cl_command_queue_properties>(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
        queue_obj->queue_ = clCreateCommandQueue(
            context->raw_context(),
            context->device().raw_id(),
            props,
            &err
        );
    }
    SILICA_CHECK_CL(err);

    return queue_obj;
}

void CommandQueue::finish() const {
    SILICA_CHECK_CL(clFinish(queue_));
}

void CommandQueue::flush() const {
    SILICA_CHECK_CL(clFlush(queue_));
}

void CommandQueue::barrier() const {
    SILICA_CHECK_CL(clEnqueueBarrier(queue_));
}

} // namespace silica::core
