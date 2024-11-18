#include "../../include/exact/knn_exact_serial.h"

void knn_exact_serial_core(const float* corpus, const float* query, int k, int* indices, float* distances, int corpus_length, int query_length, int d) {
    // Allocate memory for the distance matrix D
    float* D = (float*)malloc(corpus_length * query_length * sizeof(float));
    if (!D) {
        fprintf(stderr, "knn_exact_serial_core: Failed to allocate memory for the distance matrix D\n");
        return;
    }

    // Calculate the distance matrix D (squared Euclidean distances)
    distance_square_matrix(corpus, query, D, corpus_length, query_length, d);

    // Temporary arrays for finding the k nearest neighbors
    float* tmp_distances = (float*)malloc(corpus_length * sizeof(float));
    size_t* tmp_indices  = (size_t*)malloc(k * sizeof(size_t));

    if (!tmp_distances || !tmp_indices) {
        fprintf(stderr, "knn_exact_serial_core: Failed to allocate temporary arrays for k-NN\n");
        free(D);
        return;
    }

    // For each query, find the top-k nearest neighbors using GSL's gsl_sort_smallest
    // Its a QuickSelect implementation, so it does't sort the whole array
    for (int q = 0; q < query_length; q++) {
        // Copy the distances of the current query
        memcpy(tmp_distances, &D[q * corpus_length], corpus_length * sizeof(float));

        // Use GSL to find the indices of the k smallest distances
        gsl_sort_float_smallest_index(tmp_indices, k, tmp_distances, 1, corpus_length);

        // Collect the top-k nearest neighbors (sorted)
        for (int i = 0; i < k; ++i) {
            indices[q * k + i] = (int)tmp_indices[i];
            distances[q * k + i] = (float)sqrt( tmp_distances[tmp_indices[i]] );
        }
    }

    free(D);
    free(tmp_distances);
    free(tmp_indices);
}


void knn_exact_serial(const float* corpus, const float* query, int k, int* indices, float* distances, int corpus_length, int query_length, int d, int num_of_blocks) {
    long    max_chunk_length       =   0;
    int     q_start                =   0;
    int     q_chunk_length         =   0;
    float*  query_chunk            =   NULL;
    // Process query chunks/blocks iteratively
    while (q_start < query_length) {
        // Update usable memory status and evaluate max_chunk_length 
        // based on all the memory allocations needs to be done inside knn_exact_serial_core
        max_chunk_length = (get_usable_memory() - 2*corpus_length*sizeof(float) - k*sizeof(size_t)) / ((corpus_length + 1)*sizeof(float));
        
        // Check if max_chunk_length and num_of_blocks are valid:
        if (max_chunk_length == 0) {
            max_chunk_length = 1;
            fprintf(stderr, "knn_exact_serial: Run out of usable memory (usable memory has a margin, so the program may not fail)");
        }
        else if (num_of_blocks > 0 && query_length / num_of_blocks <= max_chunk_length) {
            max_chunk_length = (long)query_length / num_of_blocks;
            max_chunk_length = (max_chunk_length == 0) ? 1 : max_chunk_length;
        }
        // printf("%ld\n", max_chunk_length);

        q_chunk_length = (q_start + max_chunk_length < query_length) ? max_chunk_length : (query_length - q_start);

        // Allocate and load the query chunk
        query_chunk = (float*)&query[q_start * d];

        // The core function fills the indices and distances, block by block:
        knn_exact_serial_core(corpus, query_chunk, k, (indices + q_start * k), (distances + q_start * k), corpus_length, q_chunk_length, d);

        query_chunk = NULL;
        q_start += max_chunk_length;
    }
}
