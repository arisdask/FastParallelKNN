#include "../../include/exact/knn_exact_openmp.h"

/**
 * Wrapper function to perform k-nearest neighbor search using an OpenMP-based parallel approach,
 * handling memory constraints by processing in blocks (if necessary).
 * 
 * @param corpus            Pointer to the corpus matrix (reference data points)
 * @param query             Pointer to the query matrix (data points to compare)
 * @param k                 Number of nearest neighbors to find
 * @param indices           Pre-allocated array to store indices of the k-nearest neighbors for each query (length `query_length x k`)
 * @param distances         Pre-allocated array to store the Euclidean distances to the k-nearest neighbors (length `query_length x k`)
 * @param corpus_length     Number of rows (data points) in the corpus
 * @param query_length      Number of rows (data points) in the query
 * @param d                 Dimensionality of each data point (number of columns in corpus/query)
 * @param num_of_threads    Number of threads to run the parallel search.
 * 
 * @return                  None (results are stored in the pre-allocated arrays `indices` and `distances`)
 */
void knn_exact_openmp(const float* corpus, const float* query, int k, int* indices, float* distances, int corpus_length, int query_length, int d, int num_of_threads) {
    // Calculate the maximum chunk length based on available memory and other constraints
    long max_chunk_length = (0.9 * get_usable_memory() / num_of_threads - 2 * corpus_length * sizeof(float) - k * sizeof(size_t)) / ((corpus_length + 1) * sizeof(float));

    // Ensure max_chunk_length is valid
    if (max_chunk_length <= 0) {
        max_chunk_length = 1;
        fprintf(stderr, "knn_exact_openmp: Run out of usable memory (usable memory has a margin, so the program may not fail)\n");
    }

    // Parallelize the loop using OpenMP
    #pragma omp parallel for num_threads(num_of_threads) schedule(dynamic) shared(corpus, query, indices, distances)
    for (int q_start = 0; q_start < query_length; q_start += max_chunk_length) {
        // Determine chunk length for this iteration
        int q_chunk_length = (q_start + max_chunk_length < query_length) ? max_chunk_length : (query_length - q_start);

        // Allocate query chunk pointer
        const float* query_chunk = &query[q_start * d];
        int* chunk_indices = &indices[q_start * k];
        float* chunk_distances = &distances[q_start * k];

        // Compute k-NN for this chunk
        knn_exact_serial_core(corpus, query_chunk, k, chunk_indices, chunk_distances, corpus_length, q_chunk_length, d);
    }
}