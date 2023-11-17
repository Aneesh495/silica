#include "silica/core/context.hpp"
#include "silica/core/error.hpp"
#include <iostream>

namespace silica::core {

Context::~Context() {
    if (context_) {
        clReleaseContext(context_);
        context_ = nullptr;
    }
}

Context::Context(Context&& other) noexcept
    : platform_id_(other.platform_id_),
      context_(other.context_),
      platform_name_(std::move(other.platform_name_)),
      devices_(std::move(other.devices_)),
      active_device_(std::move(other.active_device_)) {
    other.context_ = nullptr;
    other.platform_id_ = nullptr;
}

Context& Context::operator=(Context&& other) noexcept {
    if (this != &other) {
        if (context_) {
            clReleaseContext(context_);
        }
        platform_id_ = other.platform_id_;
        context_ = other.context_;
        platform_name_ = std::move(other.platform_name_);
        devices_ = std::move(other.devices_);
        active_device_ = std::move(other.active_device_);
        other.context_ = nullptr;
        other.platform_id_ = nullptr;
    }
    return *this;
}

std::shared_ptr<Context> Context::create(DeviceType type) {
    auto ctx = std::shared_ptr<Context>(new Context());

    cl_uint num_platforms = 0;
    SILICA_CHECK_CL(clGetPlatformIDs(0, nullptr, &num_platforms));
    SILICA_CHECK_THROW(num_platforms > 0, "No OpenCL platforms discovered on system");

    std::vector<cl_platform_id> platforms(num_platforms);
    SILICA_CHECK_CL(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    ctx->platform_id_ = platforms[0];
    char pbuf[256];
    if (clGetPlatformInfo(ctx->platform_id_, CL_PLATFORM_NAME, sizeof(pbuf), pbuf, nullptr) == CL_SUCCESS) {
        ctx->platform_name_ = pbuf;
    }

    cl_device_type cl_dtype = CL_DEVICE_TYPE_DEFAULT;
    switch (type) {
        case DeviceType::GPU: cl_dtype = CL_DEVICE_TYPE_GPU; break;
        case DeviceType::CPU: cl_dtype = CL_DEVICE_TYPE_CPU; break;
        case DeviceType::All: cl_dtype = CL_DEVICE_TYPE_ALL; break;
        default: cl_dtype = CL_DEVICE_TYPE_DEFAULT; break;
    }

    cl_uint num_devices = 0;
    cl_int dev_status = clGetDeviceIDs(ctx->platform_id_, cl_dtype, 0, nullptr, &num_devices);
    if (dev_status != CL_SUCCESS || num_devices == 0) {
        // Fallback to CL_DEVICE_TYPE_ALL if requested type wasn't available
        cl_dtype = CL_DEVICE_TYPE_ALL;
        SILICA_CHECK_CL(clGetDeviceIDs(ctx->platform_id_, cl_dtype, 0, nullptr, &num_devices));
        SILICA_CHECK_THROW(num_devices > 0, "No OpenCL devices found for platform");
    }

    std::vector<cl_device_id> raw_devices(num_devices);
    SILICA_CHECK_CL(clGetDeviceIDs(ctx->platform_id_, cl_dtype, num_devices, raw_devices.data(), nullptr));

    for (cl_device_id did : raw_devices) {
        ctx->devices_.emplace_back(did);
    }
    ctx->active_device_ = ctx->devices_.front();

    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(ctx->platform_id_),
        0
    };

    cl_int err = CL_SUCCESS;
    ctx->context_ = clCreateContext(properties, 1, &raw_devices[0], nullptr, nullptr, &err);
    SILICA_CHECK_CL(err);

    return ctx;
}

void Context::synchronize() const {
    // Basic barrier across context queues
}

} // namespace silica::core
