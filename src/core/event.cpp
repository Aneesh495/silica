#include "silica/core/event.hpp"
#include "silica/core/error.hpp"

namespace silica::core {

Event::~Event() {
    if (event_) {
        clReleaseEvent(event_);
        event_ = nullptr;
    }
}

Event::Event(Event&& other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
}

Event& Event::operator=(Event&& other) noexcept {
    if (this != &other) {
        if (event_) {
            clReleaseEvent(event_);
        }
        event_ = other.event_;
        other.event_ = nullptr;
    }
    return *this;
}

void Event::wait() const {
    if (event_) {
        SILICA_CHECK_CL(clWaitForEvents(1, &event_));
    }
}

cl_ulong Event::queued_time_ns() const {
    cl_ulong time = 0;
    SILICA_CHECK_CL(clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_QUEUED, sizeof(time), &time, nullptr));
    return time;
}

cl_ulong Event::submit_time_ns() const {
    cl_ulong time = 0;
    SILICA_CHECK_CL(clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_SUBMIT, sizeof(time), &time, nullptr));
    return time;
}

cl_ulong Event::start_time_ns() const {
    cl_ulong time = 0;
    SILICA_CHECK_CL(clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_START, sizeof(time), &time, nullptr));
    return time;
}

cl_ulong Event::end_time_ns() const {
    cl_ulong time = 0;
    SILICA_CHECK_CL(clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_END, sizeof(time), &time, nullptr));
    return time;
}

double Event::duration_ms() const {
    if (!event_) return 0.0;
    cl_ulong start = start_time_ns();
    cl_ulong end = end_time_ns();
    if (end > start) {
        return static_cast<double>(end - start) * 1e-6;
    }
    return 0.0;
}

void Event::wait_all(const std::vector<Event>& events) {
    if (events.empty()) return;
    std::vector<cl_event> raw;
    raw.reserve(events.size());
    for (const auto& ev : events) {
        if (ev.valid()) {
            raw.push_back(ev.raw_event());
        }
    }
    if (!raw.empty()) {
        SILICA_CHECK_CL(clWaitForEvents(static_cast<cl_uint>(raw.size()), raw.data()));
    }
}

} // namespace silica::core
