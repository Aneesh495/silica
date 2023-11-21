#pragma once

#include "silica/core/types.hpp"
#include "silica/core/program.hpp"
#include "silica/core/queue.hpp"
#include "silica/core/buffer.hpp"
#include "silica/core/event.hpp"
#include "silica/core/error.hpp"
#include <memory>
#include <string>
#include <vector>
#include <type_traits>

namespace silica::core {

class Kernel {
public:
    ~Kernel();
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    Kernel(Kernel&& other) noexcept;
    Kernel& operator=(Kernel&& other) noexcept;

    static std::shared_ptr<Kernel> create(const std::shared_ptr<Program>& program,
                                          const std::string& kernel_name);

    [[nodiscard]] cl_kernel raw_kernel() const noexcept { return kernel_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    void set_arg_buffer(cl_uint index, const Buffer& buffer);
    void set_arg_local(cl_uint index, size_t size_bytes);

    template<typename T>
    void set_arg_pod(cl_uint index, const T& value) {
        static_assert(!std::is_pointer_v<T>, "Pointer types must use set_arg_buffer or set_arg_local");
        SILICA_CHECK_CL(clSetKernelArg(kernel_, index, sizeof(T), &value));
    }

    template<typename T>
    void set_arg(cl_uint index, const T& arg) {
        if constexpr (std::is_same_v<std::decay_t<T>, Buffer>) {
            set_arg_buffer(index, arg);
        } else if constexpr (std::is_same_v<std::decay_t<T>, std::shared_ptr<Buffer>>) {
            SILICA_CHECK_THROW(arg != nullptr, "Buffer shared pointer cannot be null");
            set_arg_buffer(index, *arg);
        } else {
            set_arg_pod(index, arg);
        }
    }

    template<typename... Args>
    void set_args(Args&&... args) {
        cl_uint idx = 0;
        (set_arg(idx++, std::forward<Args>(args)), ...);
    }

    Event dispatch(const std::shared_ptr<CommandQueue>& queue,
                   const WorkSize& work_size,
                   const std::vector<cl_event>& wait_list = {}) const;

    [[nodiscard]] size_t max_work_group_size(const Device& device) const;
    [[nodiscard]] size_t preferred_multiple(const Device& device) const;

    WorkSize compute_optimal_worksize_1d(const Device& device, size_t global_elements) const;
    WorkSize compute_optimal_worksize_2d(const Device& device, size_t width, size_t height) const;

private:
    Kernel() = default;

    std::shared_ptr<Program> program_;
    cl_kernel kernel_{nullptr};
    std::string name_;
};

} // namespace silica::core
