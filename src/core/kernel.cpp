#include "silica/core/kernel.hpp"
#include <algorithm>

namespace silica::core {

Kernel::~Kernel() {
    if (kernel_) {
        clReleaseKernel(kernel_);
        kernel_ = nullptr;
    }
}

Kernel::Kernel(Kernel&& other) noexcept
    : program_(std::move(other.program_)),
      kernel_(other.kernel_),
      name_(std::move(other.name_)) {
    other.kernel_ = nullptr;
}

Kernel& Kernel::operator=(Kernel&& other) noexcept {
    if (this != &other) {
        if (kernel_) {
            clReleaseKernel(kernel_);
        }
        program_ = std::move(other.program_);
        kernel_ = other.kernel_;
        name_ = std::move(other.name_);
        other.kernel_ = nullptr;
    }
    return *this;
}

std::shared_ptr<Kernel> Kernel::create(const std::shared_ptr<Program>& program,
                                       const std::string& kernel_name) {
    SILICA_CHECK_THROW(program != nullptr, "Program cannot be null");
    SILICA_CHECK_THROW(!kernel_name.empty(), "Kernel name cannot be empty");

    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(program->raw_program(), kernel_name.c_str(), &err);
    SILICA_CHECK_CL(err);

    auto kernel_obj = std::shared_ptr<Kernel>(new Kernel());
    kernel_obj->program_ = program;
    kernel_obj->kernel_ = k;
    kernel_obj->name_ = kernel_name;

    return kernel_obj;
}

void Kernel::set_arg_buffer(cl_uint index, const Buffer& buffer) {
    cl_mem raw = buffer.raw_mem();
    SILICA_CHECK_CL(clSetKernelArg(kernel_, index, sizeof(cl_mem), &raw));
}

void Kernel::set_arg_local(cl_uint index, size_t size_bytes) {
    SILICA_CHECK_CL(clSetKernelArg(kernel_, index, size_bytes, nullptr));
}

Event Kernel::dispatch(const std::shared_ptr<CommandQueue>& queue,
                       const WorkSize& work_size,
                       const std::vector<cl_event>& wait_list) const {
    SILICA_CHECK_THROW(queue != nullptr, "Command queue cannot be null");

    size_t gws[3] = {work_size.global.x, work_size.global.y, work_size.global.z};
    size_t lws[3] = {work_size.local.x, work_size.local.y, work_size.local.z};

    const size_t* local_work_size = nullptr;
    if (work_size.local.x > 0 && work_size.local.y > 0 && work_size.local.z > 0) {
        local_work_size = lws;
    }

    cl_event out_event = nullptr;
    cl_uint num_waits = static_cast<cl_uint>(wait_list.size());
    const cl_event* wait_ptr = num_waits > 0 ? wait_list.data() : nullptr;

    SILICA_CHECK_CL(clEnqueueNDRangeKernel(
        queue->raw_queue(),
        kernel_,
        static_cast<cl_uint>(work_size.dimensions),
        nullptr,
        gws,
        local_work_size,
        num_waits,
        wait_ptr,
        &out_event
    ));

    return Event(out_event);
}

size_t Kernel::max_work_group_size(const Device& device) const {
    size_t size = 0;
    SILICA_CHECK_CL(clGetKernelWorkGroupInfo(
        kernel_,
        device.raw_id(),
        CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(size),
        &size,
        nullptr
    ));
    return size;
}

size_t Kernel::preferred_multiple(const Device& device) const {
    size_t size = 0;
    SILICA_CHECK_CL(clGetKernelWorkGroupInfo(
        kernel_,
        device.raw_id(),
        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(size),
        &size,
        nullptr
    ));
    return size > 0 ? size : 32;
}

WorkSize Kernel::compute_optimal_worksize_1d(const Device& device, size_t global_elements) const {
    size_t max_wg = max_work_group_size(device);
    size_t pref_mul = preferred_multiple(device);

    size_t local = std::min(max_wg, size_t{256});
    if (pref_mul > 0) {
        local = (local / pref_mul) * pref_mul;
        if (local == 0) local = pref_mul;
    }

    size_t global = ((global_elements + local - 1) / local) * local;
    return WorkSize::make_1d(global, local);
}

WorkSize Kernel::compute_optimal_worksize_2d(const Device& device, size_t width, size_t height) const {
    size_t max_wg = max_work_group_size(device);
    
    // Choose 16x16 = 256 or 16x8 = 128 depending on limit
    size_t lx = 16;
    size_t ly = 16;
    while (lx * ly > max_wg && ly > 1) {
        ly /= 2;
    }

    size_t gx = ((width + lx - 1) / lx) * lx;
    size_t gy = ((height + ly - 1) / ly) * ly;
    return WorkSize::make_2d(gx, gy, lx, ly);
}

} // namespace silica::core
