#include "common/math.clh"
#include "common/memory.clh"

#define HIST_BINS 256
#define HIST_SUB_BANKS 4
#define HIST_WG_SIZE 128

// Multi-Bank Local Memory 256-Bin Histogram
// Privatizes histogram into 4 sub-banks to minimize atomic conflicts across threads.
__kernel void kernel_histogram_256_privatized(
    __global const uchar* input_data,
    __global uint* out_global_histogram,
    const uint total_elements) {
    
    __local uint l_sub_hist[HIST_SUB_BANKS][HIST_BINS];

    uint tid = get_local_id(0);
    uint gid = get_global_id(0);
    uint stride = get_global_size(0);

    // Initialize sub-banks to zero
    for (int b = 0; b < HIST_SUB_BANKS; ++b) {
        for (int i = tid; i < HIST_BINS; i += HIST_WG_SIZE) {
            l_sub_hist[b][i] = 0;
        }
    }
    sync_local();

    // Distribute threads across 4 sub-banks
    uint bank_id = tid % HIST_SUB_BANKS;

    for (uint idx = gid; idx < total_elements; idx += stride) {
        uchar val = input_data[idx];
        atomic_inc(&l_sub_hist[bank_id][val]);
    }
    sync_local();

    // Reduce sub-banks into bank 0
    for (int i = tid; i < HIST_BINS; i += HIST_WG_SIZE) {
        uint total_bin = l_sub_hist[0][i] + l_sub_hist[1][i] +
                         l_sub_hist[2][i] + l_sub_hist[3][i];
        if (total_bin > 0) {
            atomic_add(&out_global_histogram[i], total_bin);
        }
    }
}
