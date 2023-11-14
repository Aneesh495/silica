#pragma once

#include "silica/core/types.hpp"
#include <stdexcept>
#include <string>
#include <string_view>
#include <sstream>

namespace silica::core {

[[nodiscard]] const char* cl_error_string(cl_int error_code) noexcept;

class OpenCLException : public std::runtime_error {
public:
    OpenCLException(cl_int error_code,
                    std::string_view message,
                    std::string_view file,
                    int line)
        : std::runtime_error(format_message(error_code, message, file, line)),
          error_code_(error_code),
          file_(file),
          line_(line) {}

    [[nodiscard]] cl_int error_code() const noexcept { return error_code_; }
    [[nodiscard]] const std::string& file() const noexcept { return file_; }
    [[nodiscard]] int line() const noexcept { return line_; }

private:
    static std::string format_message(cl_int error_code,
                                      std::string_view message,
                                      std::string_view file,
                                      int line) {
        std::ostringstream ss;
        ss << "[OpenCL Error " << error_code << ": " << cl_error_string(error_code) << "] "
           << message << " at " << file << ":" << line;
        return ss.str();
    }

    cl_int error_code_;
    std::string file_;
    int line_;
};

class SilicaException : public std::runtime_error {
public:
    SilicaException(std::string_view message,
                    std::string_view file,
                    int line)
        : std::runtime_error(format_message(message, file, line)),
          file_(file),
          line_(line) {}

    [[nodiscard]] const std::string& file() const noexcept { return file_; }
    [[nodiscard]] int line() const noexcept { return line_; }

private:
    static std::string format_message(std::string_view message,
                                      std::string_view file,
                                      int line) {
        std::ostringstream ss;
        ss << "[Silica Runtime Error] " << message << " at " << file << ":" << line;
        return ss.str();
    }

    std::string file_;
    int line_;
};

#define SILICA_CHECK_CL(expr)                                                 \
    do {                                                                      \
        cl_int status_code_##__LINE__ = (expr);                               \
        if (status_code_##__LINE__ != CL_SUCCESS) {                          \
            throw ::silica::core::OpenCLException(                           \
                status_code_##__LINE__, #expr, __FILE__, __LINE__);           \
        }                                                                     \
    } while (false)

#define SILICA_CHECK_THROW(condition, msg)                                    \
    do {                                                                      \
        if (!(condition)) {                                                   \
            throw ::silica::core::SilicaException(msg, __FILE__, __LINE__);   \
        }                                                                     \
    } while (false)

} // namespace silica::core
