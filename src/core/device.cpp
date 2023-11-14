#include "silica/core/device.hpp"
#include "silica/core/error.hpp"
#include <iostream>
#include <vector>

namespace silica::core {

Device::Device(cl_device_id device_id) : device_id_(device_id) {
    if (device_id_) {
        query_device_info();
    }
}

void Device::query_device_info() {
    auto query_string = [this](cl_device_info param) -> std::string {
        size_t size = 0;
        clGetDeviceInfo(device_id_, param, 0, nullptr, &size);
        if (size <= 1) return "";
        std::vector<char> buf(size);
        clGetDeviceInfo(device_id_, param, size, buf.data(), nullptr);
        return std::string(buf.data());
    };

    name_ = query_string(CL_DEVICE_NAME);
    vendor_ = query_string(CL_DEVICE_VENDOR);
    driver_version_ = query_string(CL_DRIVER_VERSION);
    opencl_version_ = query_string(CL_DEVICE_VERSION);

    clGetDeviceInfo(device_id_, CL_DEVICE_TYPE, sizeof(device_type_), &device_type_, nullptr);
    clGetDeviceInfo(device_id_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_), &max_compute_units_, nullptr);
    clGetDeviceInfo(device_id_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size_), &max_work_group_size_, nullptr);
    clGetDeviceInfo(device_id_, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes_), max_work_item_sizes_.data(), nullptr);
    clGetDeviceInfo(device_id_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_memory_size_), &global_memory_size_, nullptr);
    clGetDeviceInfo(device_id_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_memory_size_), &local_memory_size_, nullptr);
    clGetDeviceInfo(device_id_, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_clock_frequency_), &max_clock_frequency_, nullptr);

    cl_bool unified = CL_FALSE;
    clGetDeviceInfo(device_id_, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, nullptr);
    host_unified_memory_ = (unified == CL_TRUE);
}

size_t Device::preferred_work_group_size_multiple(cl_kernel kernel) const {
    size_t size = 0;
    cl_int status = clGetKernelWorkGroupInfo(kernel, device_id_, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size), &size, nullptr);
    if (status != CL_SUCCESS) {
        return 32; // Default reasonable warp/wavefront multiple fallback
    }
    return size > 0 ? size : 32;
}

void Device::print_summary(std::ostream& os) const {
    os << "Device: " << name_ << "\n"
       << "  Vendor: " << vendor_ << "\n"
       << "  OpenCL Version: " << opencl_version_ << "\n"
       << "  Driver Version: " << driver_version_ << "\n"
       << "  Compute Units: " << max_compute_units_ << "\n"
       << "  Max Work Group: " << max_work_group_size_ << "\n"
       << "  Local Memory: " << (local_memory_size_ / 1024) << " KB\n"
       << "  Global Memory: " << (global_memory_size_ / (1024 * 1024)) << " MB\n"
       << "  Unified Memory Architecture: " << (host_unified_memory_ ? "Yes" : "No") << "\n";
}

} // namespace silica::core
