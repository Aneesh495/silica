#include "silica/core/program.hpp"
#include "silica/core/error.hpp"
#include <fstream>
#include <sstream>
#include <regex>
#include <iostream>

namespace silica::core {

Program::~Program() {
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }
}

Program::Program(Program&& other) noexcept
    : context_(std::move(other.context_)),
      program_(other.program_),
      source_(std::move(other.source_)),
      build_options_(std::move(other.build_options_)),
      build_log_(std::move(other.build_log_)) {
    other.program_ = nullptr;
}

Program& Program::operator=(Program&& other) noexcept {
    if (this != &other) {
        if (program_) {
            clReleaseProgram(program_);
        }
        context_ = std::move(other.context_);
        program_ = other.program_;
        source_ = std::move(other.source_);
        build_options_ = std::move(other.build_options_);
        build_log_ = std::move(other.build_log_);
        other.program_ = nullptr;
    }
    return *this;
}

std::string Program::preprocess_includes(
    const std::string& source,
    const std::vector<std::string>& include_paths,
    std::unordered_map<std::string, bool>& visited) {
    
    std::istringstream stream(source);
    std::ostringstream out;
    std::string line;
    std::regex include_regex(R"(^\s*#include\s+["<]([^">]+)[">])");

    while (std::getline(stream, line)) {
        std::smatch match;
        if (std::regex_search(line, match, include_regex)) {
            std::string header_name = match[1].str();
            bool resolved = false;

            // Search in include paths
            for (const auto& dir : include_paths) {
                std::filesystem::path full_path = std::filesystem::path(dir) / header_name;
                if (std::filesystem::exists(full_path)) {
                    std::string canonical = std::filesystem::canonical(full_path).string();
                    if (!visited[canonical]) {
                        visited[canonical] = true;
                        std::ifstream header_file(canonical);
                        if (header_file.is_open()) {
                            std::stringstream header_content;
                            header_content << header_file.rdbuf();
                            out << "// Begin Include: " << header_name << "\n";
                            out << preprocess_includes(header_content.str(), include_paths, visited);
                            out << "\n// End Include: " << header_name << "\n";
                            resolved = true;
                            break;
                        }
                    } else {
                        // Already included
                        resolved = true;
                        break;
                    }
                }
            }

            if (!resolved) {
                // If not found in custom include paths, keep line as-is for OpenCL compiler
                out << line << "\n";
            }
        } else {
            out << line << "\n";
        }
    }

    return out.str();
}

std::shared_ptr<Program> Program::create_from_source(
    const std::shared_ptr<Context>& context,
    const std::string& raw_source,
    const std::string& build_options,
    const std::vector<std::string>& include_paths) {
    
    SILICA_CHECK_THROW(context != nullptr, "Context cannot be null");
    SILICA_CHECK_THROW(!raw_source.empty(), "Source cannot be empty");

    std::vector<std::string> search_paths = include_paths;
#ifdef SILICA_KERNEL_ROOT
    search_paths.emplace_back(SILICA_KERNEL_ROOT);
    search_paths.emplace_back(std::string(SILICA_KERNEL_ROOT) + "/common");
#endif

    std::unordered_map<std::string, bool> visited;
    std::string processed_source = preprocess_includes(raw_source, search_paths, visited);

    auto prog = std::shared_ptr<Program>(new Program());
    prog->context_ = context;
    prog->source_ = processed_source;
    prog->build_options_ = build_options;

    const char* src_ptr = processed_source.c_str();
    size_t src_len = processed_source.size();

    cl_int err = CL_SUCCESS;
    prog->program_ = clCreateProgramWithSource(
        context->raw_context(),
        1,
        &src_ptr,
        &src_len,
        &err
    );
    SILICA_CHECK_CL(err);

    cl_device_id dev = context->device().raw_id();
    cl_int build_err = clBuildProgram(prog->program_, 1, &dev, build_options.c_str(), nullptr, nullptr);

    // Retrieve build log
    size_t log_size = 0;
    clGetProgramBuildInfo(prog->program_, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    if (log_size > 1) {
        std::vector<char> log_buf(log_size);
        clGetProgramBuildInfo(prog->program_, dev, CL_PROGRAM_BUILD_LOG, log_size, log_buf.data(), nullptr);
        prog->build_log_ = std::string(log_buf.data());
    }

    if (build_err != CL_SUCCESS) {
        std::string err_msg = "OpenCL program build failed:\n" + prog->build_log_;
        throw SilicaException(err_msg, __FILE__, __LINE__);
    }

    return prog;
}

std::shared_ptr<Program> Program::create_from_file(
    const std::shared_ptr<Context>& context,
    const std::filesystem::path& file_path,
    const std::string& build_options,
    const std::vector<std::string>& include_paths) {
    
    SILICA_CHECK_THROW(std::filesystem::exists(file_path),
                         "Kernel file does not exist: " + file_path.string());

    std::ifstream file(file_path);
    SILICA_CHECK_THROW(file.is_open(),
                         "Could not open kernel file: " + file_path.string());

    std::stringstream buffer;
    buffer << file.rdbuf();

    std::vector<std::string> paths = include_paths;
    paths.push_back(file_path.parent_path().string());

    return create_from_source(context, buffer.str(), build_options, paths);
}

} // namespace silica::core
