#pragma once

#include "silica/core/types.hpp"
#include <vector>

namespace silica::core {

class Event {
public:
    Event() noexcept = default;
    explicit Event(cl_event raw_event) noexcept : event_(raw_event) {}
    ~Event();

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) noexcept;

    [[nodiscard]] cl_event raw_event() const noexcept { return event_; }
    [[nodiscard]] cl_event* raw_ptr() noexcept { return &event_; }
    [[nodiscard]] bool valid() const noexcept { return event_ != nullptr; }

    void wait() const;

    [[nodiscard]] cl_ulong queued_time_ns() const;
    [[nodiscard]] cl_ulong submit_time_ns() const;
    [[nodiscard]] cl_ulong start_time_ns() const;
    [[nodiscard]] cl_ulong end_time_ns() const;
    [[nodiscard]] double duration_ms() const;

    static void wait_all(const std::vector<Event>& events);

private:
    cl_event event_{nullptr};
};

} // namespace silica::core
