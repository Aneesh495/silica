#include "silica/pipeline/profiler.hpp"
#include <iomanip>
#include <numeric>

namespace silica::pipeline {

void Profiler::record_event(const std::string& stage_name,
                            const core::Event& event,
                            size_t bytes_processed,
                            size_t operations_count) {
    if (!event.valid()) return;

    double ms = event.duration_ms();
    record_manual(stage_name, ms, bytes_processed, operations_count);
}

void Profiler::record_manual(const std::string& stage_name,
                             double duration_ms,
                             size_t bytes_processed,
                             size_t operations_count) {
    ProfileRecord rec;
    rec.stage_name = stage_name;
    rec.duration_ms = duration_ms;
    rec.bytes_processed = bytes_processed;
    rec.operations_count = operations_count;

    if (duration_ms > 0.0) {
        double seconds = duration_ms * 1e-3;
        if (bytes_processed > 0) {
            rec.bandwidth_gb_s = (static_cast<double>(bytes_processed) / (1024.0 * 1024.0 * 1024.0)) / seconds;
        }
        if (operations_count > 0) {
            rec.gflops = (static_cast<double>(operations_count) * 1e-9) / seconds;
        }
    }

    records_.push_back(rec);
}

void Profiler::reset() {
    records_.clear();
}

double Profiler::total_kernel_time_ms() const noexcept {
    double total = 0.0;
    for (const auto& r : records_) {
        total += r.duration_ms;
    }
    return total;
}

void Profiler::print_summary(std::ostream& os) const {
    double total_ms = total_kernel_time_ms();

    os << "\n" << std::string(80, '=') << "\n";
    os << "SILICA PIPELINE PROFILING REPORT\n";
    os << std::string(80, '=') << "\n";
    os << std::left << std::setw(32) << "Stage"
       << std::right << std::setw(12) << "Time (ms)"
       << std::setw(10) << "% Total"
       << std::setw(14) << "Bandwidth"
       << std::setw(12) << "GFLOPS" << "\n";
    os << std::string(80, '-') << "\n";

    for (const auto& rec : records_) {
        double pct = (total_ms > 0.0) ? (rec.duration_ms / total_ms * 100.0) : 0.0;

        os << std::left << std::setw(32) << rec.stage_name
           << std::right << std::fixed << std::setprecision(3)
           << std::setw(12) << rec.duration_ms
           << std::setprecision(1) << std::setw(9) << pct << "%";

        if (rec.bandwidth_gb_s > 0.0) {
            os << std::setprecision(2) << std::setw(10) << rec.bandwidth_gb_s << " GB/s";
        } else {
            os << std::setw(14) << "N/A";
        }

        if (rec.gflops > 0.0) {
            os << std::setprecision(2) << std::setw(10) << rec.gflops << " GFLOPS";
        } else {
            os << std::setw(12) << "N/A";
        }
        os << "\n";
    }

    os << std::string(80, '=') << "\n";
    os << std::left << std::setw(32) << "Total Kernel Time"
       << std::right << std::fixed << std::setprecision(3)
       << std::setw(12) << total_ms << " ms\n";
    os << std::string(80, '=') << "\n\n";
}

} // namespace silica::pipeline
