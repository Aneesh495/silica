#pragma once

#include "silica/core/types.hpp"
#include "silica/core/device.hpp"
#include <memory>
#include <vector>
#include <string>

namespace silica::core {

class Context {
public:
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&& other) noexcept;
    Context& operator=(Context&& other) noexcept;

    static std::shared_ptr<Context> create(DeviceType type = DeviceType::Default);

    [[nodiscard]] cl_context raw_context() const noexcept { return context_; }
    [[nodiscard]] cl_platform_id raw_platform() const noexcept { return platform_id_; }
    [[nodiscard]] const Device& device() const noexcept { return active_device_; }
    [[nodiscard]] const std::vector<Device>& all_devices() const noexcept { return devices_; }
    [[nodiscard]] const std::string& platform_name() const noexcept { return platform_name_; }

    void synchronize() const;

private:
    Context() = default;

    cl_platform_id platform_id_{nullptr};
    cl_context context_{nullptr};
    std::string platform_name_;
    std::vector<Device> devices_;
    Device active_device_;
};

} // namespace silica::core
