#pragma once

#include "silica/core/types.hpp"
#include "silica/core/event.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <ostream>

namespace silica::pipeline {

struct ProfileRecord {
    std::string stage_name;
    double duration_ms{0.0};
    size_t bytes_processed{0};
    size_t operations_count{0};
    double bandwidth_gb_s{0.0};
    double gflops{0.0};
};

class Profiler {
public:
    Profiler() = default;

    void record_event(const std::string& stage_name,
                      const core::Event& event,
                      size_t bytes_processed = 0,
                      size_t operations_count = 0);

    void record_manual(const std::string& stage_name,
                       double duration_ms,
                       size_t bytes_processed = 0,
                       size_t operations_count = 0);

    void reset();

    [[nodiscard]] const std::vector<ProfileRecord>& records() const noexcept { return records_; }
    [[nodiscard]] double total_kernel_time_ms() const noexcept;

    void print_summary(std::ostream& os) const;

private:
    std::vector<ProfileRecord> records_;
};

} // namespace silica::pipeline
