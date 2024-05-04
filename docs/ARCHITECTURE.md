# Silica System Architecture & Runtime Execution Model

This document outlines the architectural principles, hardware abstraction layers, memory execution models, and scheduling guarantees underlying the Silica heterogeneous computing engine.

---

## 1. Design Philosophy

Heterogeneous computing across diverse GPU microarchitectures requires reconciling two competing goals:
1. Low-level bare-metal control over thread block dimensions, local data store (LDS) allocation, cache line alignment, and instruction-level parallelism (ILP).
2. Clean, safe, composable host abstractions that guarantee zero memory leaks, eliminate manual synchronization hazards, and abstract platform discrepancies.

Silica achieves this through a layered architecture:

```
+-----------------------------------------------------------------------------------+
|                           High-Level Domain Engines                               |
|        TomographyEngine  |  SpatialEngine  |  VisionEngine  |  PointCloudEngine   |
+-----------------------------------------------------------------------------------+
                                         |
+-----------------------------------------------------------------------------------+
|                           Pipeline & Graph Scheduling                             |
|          PipelineGraph (DAG)  |  Stage  |  Hardware Profiler  |  Event Tracing    |
+-----------------------------------------------------------------------------------+
                                         |
+-----------------------------------------------------------------------------------+
|                         Core C++20 Hardware Abstractions                          |
|    Context  |  CommandQueue  |  Buffer  |  Program  |  Kernel  |  Event  |  Pool  |
+-----------------------------------------------------------------------------------+
                                         |
+-----------------------------------------------------------------------------------+
|                         OpenCL 1.2 Platform & Device Driver                       |
|         Apple Silicon (Metal HAL)  |  NVIDIA CUDA OpenCL  |  AMD ROCm OpenCL      |
+-----------------------------------------------------------------------------------+
                                         |
+-----------------------------------------------------------------------------------+
|                               Physical Compute Silicon                            |
|             Unified Memory Fabric / PCIe Bus / Compute Units / LDS Banks          |
+-----------------------------------------------------------------------------------+
```

---

## 2. RAII Hardware Resource Encapsulation

Every OpenCL resource is managed by a dedicated C++20 RAII handle that couples resource acquisition with object construction, and resource deallocation with destruction.

### Context (`silica::core::Context`)
The `Context` encapsulates an OpenCL compute device, associated device queues, program compilation caches, and global allocation trackers. It automatically selects the highest-performance GPU on the platform by default (or falls back to multicore CPUs if no discrete or integrated accelerator is found).

### CommandQueue (`silica::core::CommandQueue`)
Command queues manage command streams to the underlying accelerator. Queues can be configured as:
- **In-Order Queues**: Commands execute strictly sequentially, providing automatic barrier semantics.
- **Out-of-Order Queues with Event Wait Lists**: Commands can execute concurrently whenever their input dependencies are resolved.
- **Profiling Queues**: Enables nanosecond-resolution device hardware timers (`CL_QUEUE_PROFILING_ENABLE`) to record command submit, start, and end timestamps.

### Buffer (`silica::core::Buffer`)
Encapsulates `cl_mem` buffer allocations. Key features include:
- Strongly typed host transfers through C++20 `std::span<const T>`.
- Support for Device-Local (`CL_MEM_READ_WRITE`), Host-Pinned (`CL_MEM_ALLOC_HOST_PTR`), and Host-Unified memory allocations.
- Blocking and non-blocking asynchronous DMA transfers.
- Sub-buffer slicing without copying.

---

## 3. Unified Memory & Bank Conflict Mitigation

### Zero-Copy Memory Fabric
On unified memory architectures such as Apple Silicon (M1/M2/M3/M4), CPU and GPU cores access the same physical LPDDR5/5X DRAM without an intervening PCIe bus. Silica detects unified memory support via `CL_DEVICE_HOST_UNIFIED_MEMORY`:
- Avoids redundant staging buffers on host and device.
- Direct host-pointer mapping (`clEnqueueMapBuffer`) allows compute kernels to process large volumes without redundant PCIe transfer overhead.

### LDS Bank Conflict Mitigation
Hardware Local Data Share (LDS) or shared memory is divided into independent memory banks (typically 32 banks on modern architectures, each 4 bytes wide). Consecutive 32-bit words map to consecutive banks:

```
Bank 0: words 0, 32, 64, ...
Bank 1: words 1, 33, 65, ...
...
Bank 31: words 31, 63, 95, ...
```

When multiple threads in a SIMD wavefront access different addresses belonging to the same memory bank simultaneously, a **bank conflict** occurs. The hardware must serialize the conflicting accesses, reducing local memory bandwidth by up to 32x.

In Silica, all 2D and 3D local memory tiles apply stride padding via the macro:

```c
#define LDS_BANK_COUNT 32
#define LDS_STRIDE_PAD(dim) (((dim) % LDS_BANK_COUNT == 0) ? ((dim) + 1) : (dim))
```

For instance, in image filtering, a `16 x 16` thread group with halo padding uses a local tile of width `16 + 2 * radius + 1`. The additional single-element pad skews each consecutive row by 1 bank, ensuring that vertical column traversals access distinct banks simultaneously with zero serialization penalties.

---

## 4. Workgroup Sizing & Boundary Alignment

A fundamental requirement of OpenCL 1.2 is that whenever an explicit local work size is provided to `clEnqueueNDRangeKernel`, each component of the global work size **must be an exact integer multiple** of the corresponding local work size component:

$$\text{GlobalSize}_d = k_d \times \text{LocalSize}_d \quad (k_d \in \mathbb{Z}^+)$$

To prevent runtime errors across arbitrary input dimensions (e.g. 180 projection angles, arbitrary voxel grid sizes, non-power-of-two point cloud counts), Silica enforces automatic workgroup rounding in all engine dispatches:

```cpp
size_t padded_global = ((input_size + local_size - 1) / local_size) * local_size;
```

Inside every OpenCL kernel, an early bounds check guards against out-of-bounds execution:

```c
int gx = get_global_id(0);
int gy = get_global_id(1);
if (gx >= width || gy >= height) return;
```

This guarantees that hardware compute units remain fully utilized while memory safety is mathematically preserved.

---

## 5. Memory Pool & Sub-Allocator Architecture

High-frequency pipeline execution (e.g. 60 FPS spatial fusion or multi-iteration tomographic reconstruction) can cause severe heap fragmentation and driver lock contention if `clCreateBuffer` and `clReleaseMemObject` are invoked repeatedly.

Silica addresses this through `silica::core::MemoryPool`:

```
+-----------------------------------------------------------------------------------+
|                               silica::core::MemoryPool                            |
+-----------------------------------------------------------------------------------+
         |                                |                               |
+------------------+             +------------------+            +------------------+
| Size Bucket 1 MB |             | Size Bucket 4 MB |            | Size Bucket 16 MB|
|  [Buf] -> [Buf]  |             |      [Buf]       |            |  [Buf] -> [Buf]  |
+------------------+             +------------------+            +------------------+
```

### Leased Buffer Mechanism
- Buffers are categorized into power-of-two size buckets.
- Calling `pool->acquire(size, access)` reuses an existing idle buffer if one is available in the matching bucket.
- Leased buffers return an RAII `LeasedBuffer` wrapper. Upon leaving scope, the destructor automatically returns the buffer to the pool for immediate reuse by subsequent pipeline stages.
- Benchmark profiling shows this eliminates up to 94% of driver allocation latency during sustained pipeline runs.

---

## 6. Pipeline Execution Graph (DAG Scheduler)

The `silica::pipeline` module provides a Directed Acyclic Graph (DAG) task runner for complex multi-stage workloads.

```
       [Input Depth Frame]
                |
        (Radial Undistort)
                |
        +-------+-------+
        |               |
        v               v
(Bilateral Filter)  (FAST Keypoints)
        |               |
        v               v
(TSDF Integrate)    (Visual Odometry)
        |
        v
(Marching Cubes)
        |
        v
  [Triangle Mesh]
```

- **Topological Sorting**: The pipeline evaluates dependencies and executes independent kernel stages concurrently.
- **Hardware Event Chaining**: Stages exchange `cl_event` tokens directly. Kernels are scheduled without round-tripping through host CPU threads.
- **Built-in Nanosecond Profiling**: Hardware timestamps (`CL_PROFILING_COMMAND_START`, `CL_PROFILING_COMMAND_END`) record per-kernel latency and bandwidth without intrusive CPU instrumentation.
