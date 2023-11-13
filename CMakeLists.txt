cmake_minimum_required(VERSION 3.20)
project(silica VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for language servers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization and warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -O3
    )
    if(APPLE)
        # Suppress deprecation warnings for OpenCL on macOS
        add_compile_options(-Wno-deprecated-declarations)
    endif()
endif()

# OpenCL dependency discovery
if(APPLE)
    find_library(OPENCL_LIBRARY OpenCL REQUIRED)
    set(OPENCL_LIBRARIES ${OPENCL_LIBRARY})
    set(OPENCL_INCLUDE_DIRS "")
    message(STATUS "Found OpenCL Framework on Apple: ${OPENCL_LIBRARY}")
else()
    find_package(OpenCL REQUIRED)
    message(STATUS "Found OpenCL on Linux/Windows: ${OpenCL_LIBRARIES}")
    set(OPENCL_LIBRARIES OpenCL::OpenCL)
endif()

# Root paths
set(SILICA_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
add_compile_definitions(SILICA_KERNEL_ROOT="${SILICA_KERNEL_DIR}")

# Include directory
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OPENCL_INCLUDE_DIRS}
)

# Core library sources
file(GLOB_RECURSE SILICA_SRC_FILES
    "src/core/*.cpp"
    "src/pipeline/*.cpp"
    "src/linalg/*.cpp"
    "src/image/*.cpp"
    "src/features/*.cpp"
    "src/flow/*.cpp"
    "src/stereo/*.cpp"
    "src/tomography/*.cpp"
    "src/spatial/*.cpp"
    "src/pointcloud/*.cpp"
    "src/rendering/*.cpp"
)

add_library(silica_core STATIC ${SILICA_SRC_FILES})
target_link_libraries(silica_core PUBLIC ${OPENCL_LIBRARIES})
target_include_directories(silica_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# CLI Workbench Executable
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/app/main.cpp")
    add_executable(silica src/app/main.cpp)
    target_link_libraries(silica PRIVATE silica_core)
endif()

# Test Suite Executable
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
if(TEST_SOURCES)
    add_executable(silica_tests ${TEST_SOURCES})
    target_link_libraries(silica_tests PRIVATE silica_core)
endif()

# Benchmark Suite Executable
file(GLOB_RECURSE BENCH_SOURCES "benchmarks/*.cpp")
if(BENCH_SOURCES)
    add_executable(silica_bench ${BENCH_SOURCES})
    target_link_libraries(silica_bench PRIVATE silica_core)
endif()
