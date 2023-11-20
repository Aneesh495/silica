#pragma once

#include "silica/core/types.hpp"
#include "silica/core/context.hpp"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>

namespace silica::core {

class Program {
public:
    ~Program();
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&& other) noexcept;
    Program& operator=(Program&& other) noexcept;

    static std::shared_ptr<Program> create_from_source(
        const std::shared_ptr<Context>& context,
        const std::string& source,
        const std::string& build_options = "",
        const std::vector<std::string>& include_paths = {});

    static std::shared_ptr<Program> create_from_file(
        const std::shared_ptr<Context>& context,
        const std::filesystem::path& file_path,
        const std::string& build_options = "",
        const std::vector<std::string>& include_paths = {});

    [[nodiscard]] cl_program raw_program() const noexcept { return program_; }
    [[nodiscard]] const std::string& source() const noexcept { return source_; }
    [[nodiscard]] const std::string& build_options() const noexcept { return build_options_; }
    [[nodiscard]] const std::string& build_log() const noexcept { return build_log_; }

private:
    Program() = default;

    static std::string preprocess_includes(
        const std::string& source,
        const std::vector<std::string>& include_paths,
        std::unordered_map<std::string, bool>& visited);

    std::shared_ptr<Context> context_;
    cl_program program_{nullptr};
    std::string source_;
    std::string build_options_;
    std::string build_log_;
};

} // namespace silica::core
