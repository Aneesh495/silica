#pragma once

// Silica: High-Throughput Heterogeneous OpenCL Compute Engine
// Top-Level Unified Public Header

// Core OpenCL Abstractions
#include "silica/core/types.hpp"
#include "silica/core/error.hpp"
#include "silica/core/device.hpp"
#include "silica/core/context.hpp"
#include "silica/core/queue.hpp"
#include "silica/core/buffer.hpp"
#include "silica/core/program.hpp"
#include "silica/core/kernel.hpp"
#include "silica/core/event.hpp"
#include "silica/core/memory_pool.hpp"

// DAG Pipeline Orchestration
#include "silica/pipeline/node.hpp"
#include "silica/pipeline/pipeline.hpp"
#include "silica/pipeline/scheduler.hpp"
#include "silica/pipeline/profiler.hpp"

// Domain Engines
#include "silica/tomography/tomography.hpp"
#include "silica/spatial/spatial.hpp"
#include "silica/vision/vision.hpp"
#include "silica/pointcloud/pointcloud.hpp"

namespace silica {

inline std::string version() noexcept {
    return "1.0.0";
}

} // namespace silica
