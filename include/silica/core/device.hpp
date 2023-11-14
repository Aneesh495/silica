#pragma once

#include "silica/core/types.hpp"
#include <string>
#include <vector>
#include <array>
#include <ostream>

namespace silica::core {

class Device {
public:
    Device() noexcept = default;
    explicit Device(cl_device_id device_id);

    [[nodiscard]] cl_device_id raw_id() const noexcept { return device_id_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] const std::string& vendor() const noexcept { return vendor_; }
    [[nodiscard]] const std::string& driver_version() const noexcept { return driver_version_; }
    [[nodiscard]] const std::string& opencl_version() const noexcept { return opencl_version_; }
    
    [[nodiscard]] cl_uint max_compute_units() const noexcept { return max_compute_units_; }
    [[nodiscard]] size_t max_work_group_size() const noexcept { return max_work_group_size_; }
    [[nodiscard]] const std::array<size_t, 3>& max_work_item_sizes() const noexcept { return max_work_item_sizes_; }
    [[nodiscard]] cl_ulong global_memory_size() const noexcept { return global_memory_size_; }
    [[nodiscard]] cl_ulong local_memory_size() const noexcept { return local_memory_size_; }
    [[nodiscard]] cl_uint max_clock_frequency() const noexcept { return max_clock_frequency_; }
    [[nodiscard]] bool is_unified_memory() const noexcept { return host_unified_memory_; }
    [[nodiscard]] bool is_gpu() const noexcept { return device_type_ & CL_DEVICE_TYPE_GPU; }
    [[nodiscard]] bool is_cpu() const noexcept { return device_type_ & CL_DEVICE_TYPE_CPU; }
    
    [[nodiscard]] size_t preferred_work_group_size_multiple(cl_kernel kernel) const;

    void print_summary(std::ostream& os) const;

private:
    void query_device_info();

    cl_device_id device_id_{nullptr};
    std::string name_;
    std::string vendor_;
    std::string driver_version_;
    std::string opencl_version_;
    cl_device_type device_type_{0};
    cl_uint max_compute_units_{0};
    size_t max_work_group_size_{0};
    std::array<size_t, 3> max_work_item_sizes_{0, 0, 0};
    cl_ulong global_memory_size_{0};
    cl_ulong local_memory_size_{0};
    cl_uint max_clock_frequency_{0};
    bool host_unified_memory_{false};
};

} // namespace silica::core
