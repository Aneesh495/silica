#include "test_framework.hpp"
#include "silica/silica.hpp"
#include <numeric>

using namespace silica;

SILICA_TEST(Primitives, ParallelReductionSum) {
    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    std::string root = SILICA_KERNEL_ROOT;
    auto program = core::Program::create_from_file(context, root + "/primitives/reduce.cl");
    auto kernel = core::Kernel::create(program, "kernel_reduce_sum_float");

    const size_t N = 1024;
    std::vector<float> data(N, 1.0f);
    auto in_buf = core::Buffer::create_from_host<float>(context, queue, data);

    size_t wg_size = 256;
    size_t num_groups = (N + wg_size - 1) / wg_size;
    auto out_buf = core::Buffer::create(context, num_groups * sizeof(float), core::MemoryAccess::ReadWrite);

    kernel->set_args(*in_buf, *out_buf, static_cast<uint32_t>(N));

    core::WorkSize ws({num_groups * wg_size}, {wg_size});
    kernel->dispatch(queue, ws);

    std::vector<float> partial_sums(num_groups);
    out_buf->read_vector(queue, partial_sums);

    float total_sum = std::accumulate(partial_sums.begin(), partial_sums.end(), 0.0f);
    ASSERT_NEAR(total_sum, static_cast<float>(N), 1e-4f);
}

SILICA_TEST(Primitives, RegisterTiledGEMM) {
    auto context = core::Context::create();
    auto queue = core::CommandQueue::create(context);

    std::string root = SILICA_KERNEL_ROOT;
    auto program = core::Program::create_from_file(context, root + "/primitives/gemm.cl");
    auto kernel = core::Kernel::create(program, "kernel_gemm_tiled");

    const uint32_t M = 64;
    const uint32_t N = 64;
    const uint32_t K = 64;

    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 2.0f);
    std::vector<float> C(M * N, 0.0f);

    auto buf_A = core::Buffer::create_from_host<float>(context, queue, A);
    auto buf_B = core::Buffer::create_from_host<float>(context, queue, B);
    auto buf_C = core::Buffer::create_from_host<float>(context, queue, C);

    float alpha = 1.0f;
    float beta = 0.0f;

    kernel->set_args(*buf_A, *buf_B, *buf_C, M, N, K, alpha, beta);

    size_t num_tiles_x = (N + 63) / 64;
    size_t num_tiles_y = (M + 63) / 64;
    core::WorkSize ws({num_tiles_x * 16, num_tiles_y * 16}, {16, 16});
    kernel->dispatch(queue, ws);

    std::vector<float> result(M * N);
    buf_C->read_vector(queue, result);

    // Each element should be sum_{k=0..63} (1.0 * 2.0) = 128.0
    for (size_t i = 0; i < M * N; ++i) {
        ASSERT_NEAR(result[i], 128.0f, 1e-3f);
    }
}
