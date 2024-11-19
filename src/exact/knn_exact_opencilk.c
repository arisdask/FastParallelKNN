#include "../../include/exact/knn_exact_opencilk.h"

void knn_exact_opencilk(const float* corpus, const float* query, int k, int* indices, float* distances, int corpus_length, int query_length, int d, int num_of_threads) {
    int q_chunk_length = 0;
    int q_start = 0;

    while (q_start < query_length) {
        // Calculate max_chunk_length based on available memory
        long max_chunk_length = (0.9 * get_usable_memory() / num_of_threads - 2 * corpus_length * sizeof(float) - k * sizeof(size_t)) / ((corpus_length + 1) * sizeof(float));

        // Ensure max_chunk_length is valid
        if (max_chunk_length <= 0) {
            max_chunk_length = 1;
            fprintf(stderr, "knn_exact_opencilk: Run out of usable memory (usable memory has a margin, so the program may not fail)\n");
        }

        // Determine chunk length for this iteration
        q_chunk_length = (q_start + max_chunk_length < query_length) ? max_chunk_length : (query_length - q_start);

        // Allocate query chunk pointer
        const float* query_chunk = &query[q_start * d];
        int* chunk_indices = &indices[q_start * k];
        float* chunk_distances = &distances[q_start * k];

        // Use OpenCilk's cilk_spawn to parallelize the k-NN search for this chunk
        cilk_spawn knn_exact_serial_core(corpus, query_chunk, k, chunk_indices, chunk_distances, corpus_length, q_chunk_length, d);

        // Move to the next chunk
        q_start += q_chunk_length;
    }

    // Wait for all spawned tasks to complete
    cilk_sync;
}