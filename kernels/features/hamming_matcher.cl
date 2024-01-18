#include "common/math.clh"
#include "common/memory.clh"

// Computes 256-bit Hamming distance between two binary descriptors
inline uint compute_hamming_distance_256(
    __global const uint* desc_a,
    __global const uint* desc_b) {
    
    uint dist = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        dist += popcount(desc_a[i] ^ desc_b[i]);
    }
    return dist;
}

// k-NN (k=2) Hamming Distance Matcher with Lowe's Ratio Test
// For each query descriptor in Q, finds 1st and 2nd nearest neighbors in T.
__kernel void kernel_match_descriptors_knn2(
    __global const uint* query_descriptors, // num_queries x 8 uints
    __global const uint* train_descriptors, // num_train x 8 uints
    __global int* out_best_train_idx,       // Best matching train index (-1 if rejected)
    __global uint* out_best_distance,       // Best matching Hamming distance
    const uint num_queries,
    const uint num_train,
    const float ratio_threshold,            // Lowe's ratio (e.g. 0.75f)
    const uint max_distance_threshold) {    // Maximum allowed Hamming distance (e.g. 64)
    
    uint qid = get_global_id(0);
    if (qid >= num_queries) return;

    __global const uint* q_desc = query_descriptors + (size_t)qid * 8;

    uint best_dist = 256;
    uint second_dist = 256;
    int best_idx = -1;

    for (uint tid = 0; tid < num_train; ++tid) {
        __global const uint* t_desc = train_descriptors + (size_t)tid * 8;
        uint dist = compute_hamming_distance_256(q_desc, t_desc);

        if (dist < best_dist) {
            second_dist = best_dist;
            best_dist = dist;
            best_idx = (int)tid;
        } else if (dist < second_dist) {
            second_dist = dist;
        }
    }

    // Apply Lowe's ratio test and distance threshold
    bool passed = false;
    if (best_idx >= 0 && best_dist <= max_distance_threshold) {
        if ((float)best_dist < ratio_threshold * (float)second_dist) {
            passed = true;
        }
    }

    out_best_train_idx[qid] = passed ? best_idx : -1;
    out_best_distance[qid]  = passed ? best_dist : 256;
}
