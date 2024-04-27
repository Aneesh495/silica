#include "test_framework.hpp"
#include "silica/silica.hpp"
#include <numeric>

using namespace silica;

SILICA_TEST(Core, DeviceDiscovery) {
    auto context = core::Context::create();
    ASSERT_TRUE(context != nullptr);
    
    auto dev = context->device();
    ASSERT_TRUE(!dev.name().empty());
    ASSERT_TRUE(dev.global_memory_size() > 0);
    ASSERT_TRUE(dev.max_compute_units() > 0);
}

SILICA_TEST(Core, ContextAndQueueCreation) {
    auto context = core::Context::create();
    ASSERT_TRUE(context != nullptr);
    ASSERT_TRUE(context->raw_context() != nullptr);

    auto queue = core::CommandQueue::create(context, true);
    ASSERT_TRUE(queue != nullptr);
    ASSERT_TRUE(queue->raw_queue() != nullptr);
}

SILICA_TEST(Core, BufferHostReadWrite) {
    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    const size_t N = 1024;
    std::vector<float> input(N);
    std::iota(input.begin(), input.end(), 1.0f);

    auto buf = core::Buffer::create_from_host<float>(context, queue, input);
    ASSERT_TRUE(buf != nullptr);
    ASSERT_EQ(buf->size_bytes(), N * sizeof(float));

    std::vector<float> output(N, 0.0f);
    buf->read_vector(queue, output);

    for (size_t i = 0; i < N; ++i) {
        ASSERT_EQ(input[i], output[i]);
    }
}

SILICA_TEST(Core, MemoryPoolLeaseAndRecycle) {
    auto context = core::Context::create();

    core::MemoryPool pool(context);
    size_t requested_bytes = 4096;

    auto buf1 = pool.acquire(requested_bytes);
    ASSERT_TRUE(buf1 != nullptr);
    ASSERT_TRUE(buf1->size_bytes() >= requested_bytes);

    pool.release(buf1);
    ASSERT_EQ(pool.stats().cached_buffers_count, 1u);

    auto buf2 = pool.acquire(requested_bytes);
    ASSERT_TRUE(buf2 != nullptr);
    ASSERT_EQ(pool.stats().reuses_count, 1u);

    pool.release(buf2);
    pool.trim();
    ASSERT_EQ(pool.stats().cached_buffers_count, 0u);
}

SILICA_TEST(Core, ProgramCompilationAndExecution) {
    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    const std::string kernel_src = R"(
        __kernel void kernel_vector_scale(
            __global const float* in,
            __global float* out,
            const float factor,
            const uint n) {
            uint gid = get_global_id(0);
            if (gid < n) {
                out[gid] = in[gid] * factor;
            }
        }
    )";

    auto program = core::Program::create_from_source(context, kernel_src);
    ASSERT_TRUE(program != nullptr);

    auto kernel = core::Kernel::create(program, "kernel_vector_scale");
    ASSERT_TRUE(kernel != nullptr);

    const size_t N = 512;
    std::vector<float> in_data(N, 2.5f);
    auto in_buf = core::Buffer::create_from_host<float>(context, queue, in_data);
    auto out_buf = core::Buffer::create(context, N * sizeof(float), core::MemoryAccess::ReadWrite);

    float factor = 4.0f;
    kernel->set_args(*in_buf, *out_buf, factor, static_cast<uint32_t>(N));

    core::WorkSize ws({N}, {64});
    auto event = kernel->dispatch(queue, ws);
    event.wait();

    std::vector<float> out_data(N);
    out_buf->read_vector(queue, out_data);

    for (size_t i = 0; i < N; ++i) {
        ASSERT_NEAR(out_data[i], 10.0f, 1e-5f);
    }
}
