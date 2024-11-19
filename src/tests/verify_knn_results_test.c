#include "../../include/tests/verify_knn_results_test.h"

int verify_knn_results_test(knn_func_t knnsearch, const char* data_path, const char* corpus_name, const char* query_name, const char* neighbors_name, const char* distances_name, int num_of_threads) {
    int k, corpus_length, query_length, d;

    // Open neighbors set to find the value of k:
    float* neighbors = load_hdf5(data_path, neighbors_name, &query_length, &k);
    if (neighbors == NULL) {
        fprintf(stderr, "verify_knn_results_test: Failed to load the %s data of %s.\n", neighbors_name, data_path);
        return -1;
    }
    free(neighbors); // We only needed to extract the value of k

    // Load the corpus (training) set
    float* corpus = load_hdf5(data_path, corpus_name, &corpus_length, &d);
    if (corpus == NULL) {
        fprintf(stderr, "verify_knn_results_test: Failed to load the %s data of %s.\n", corpus_name, data_path);
        return -1;
    }

    // Load the query (test) set
    float* query = load_hdf5(data_path, query_name, &query_length, &d);
    if (query == NULL) {
        fprintf(stderr, "verify_knn_results_test: Failed to load the %s data of %s.\n", query_name, data_path);
        free(corpus);
        return -1;
    }

    // Allocate memory for the k-NN results (indices and distances)
    int* idx = (int*)malloc(query_length * k * sizeof(int));
    float* dst = (float*)malloc(query_length * k * sizeof(float));
    if (idx == NULL || dst == NULL) {
        fprintf(stderr, "verify_knn_results_test: Memory allocation failed for k-NN results.\n");
        free(corpus);
        free(query);
        return -1;
    }

    // Timing the k-NN function
    struct timeval start, end;
    gettimeofday(&start, NULL);
    knnsearch(corpus, query, k, idx, dst, corpus_length, query_length, d, num_of_threads);
    gettimeofday(&end, NULL);

    // Calculate elapsed time in seconds
    double time_taken = (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) / 1e6);
    // printf("verify_knn_results_test: k-NN search elapsed time is %lf seconds\n", time_taken);
    printf("%lf seconds: ", time_taken);

    // Cleanup test and train datasets
    free(corpus);
    free(query);

    // Load ground truth neighbors and distances for comparison
    float* ground_truth_neighbors = load_hdf5(data_path, neighbors_name, &query_length, &k);
    if (ground_truth_neighbors == NULL) {
        fprintf(stderr, "verify_knn_results_test: Failed to load the %s data.\n", neighbors_name);
        free(idx);
        free(dst);
        return -1;
    }

    float* ground_truth_distances = load_hdf5(data_path, distances_name, &query_length, &k);
    if (ground_truth_distances == NULL) {
        fprintf(stderr, "verify_knn_results_test: Failed to load the %s data.\n", distances_name);
        free(idx);
        free(dst);
        free(ground_truth_neighbors);
        return -1;
    }

    // Verify neighbors (indices) and distances
    int neighbor_errors = 0, distance_errors = 0;
    for (int i = 0; i < query_length; i++) {
        for (int j = 0; j < k; j++) {
            int idx_pos = i * k + j;
            
            // Check neighbors (integer comparison)
            if (abs(idx[idx_pos] - (int)ground_truth_neighbors[idx_pos]) > ZERO) {
                neighbor_errors++;
            }

            // Check distances (floating-point comparison)
            if (fabs(dst[idx_pos] - ground_truth_distances[idx_pos]) > ZERO) {
                distance_errors++;
            }
        }
    }

    // Output mismatch percentages
    printf("Neighbors Mismatch Percentage: %f %%, ", 100 * neighbor_errors / (float)(query_length * k));
    printf("Distances Mismatch Percentage: %f %%\n", 100 * distance_errors / (float)(query_length * k));

    // Cleanup
    free(idx);
    free(dst);
    free(ground_truth_neighbors);
    free(ground_truth_distances);

    return 0;
}
