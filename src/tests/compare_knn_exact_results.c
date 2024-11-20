#include "../../include/tests/tests.h"

int compare_knn_exact_results(const char* data_path1, const char* neighbors_name1, const char* distances_name1, const char* data_path2, const char* neighbors_name2, const char* distances_name2) {
    int k, query_length;

    // Load neighbors and distances for comparison
    float* idx1 = load_hdf5(data_path1, neighbors_name1, &query_length, &k);
    if (idx1 == NULL) {
        fprintf(stderr, "compare_knn_exact_results: Failed to load the %s data.\n", neighbors_name1);
        return -1;
    }

    float* dst1 = load_hdf5(data_path1, distances_name1, &query_length, &k);
    if (dst1 == NULL) {
        fprintf(stderr, "compare_knn_exact_results: Failed to load the %s data.\n", distances_name1);
        free(idx1);
        return -1;
    }

    float* idx2 = load_hdf5(data_path2, neighbors_name2, &query_length, &k);
    if (idx2 == NULL) {
        fprintf(stderr, "compare_knn_exact_results: Failed to load the %s data.\n", neighbors_name2);
        free(idx1);
        free(dst1);
        return -1;
    }

    float* dst2 = load_hdf5(data_path2, distances_name2, &query_length, &k);
    if (dst2 == NULL) {
        fprintf(stderr, "compare_knn_exact_results: Failed to load the %s data.\n", distances_name2);
        free(idx1);
        free(idx2);
        free(dst1);
        return -1;
    }

    // Exact compare neighbors (indices) and distances
    int neighbor_errors = 0, distance_errors = 0;
    for (int i = 0; i < query_length; i++) {
        for (int j = 0; j < k; j++) {
            int idx_pos = i * k + j;
            
            // Check neighbors (integer comparison)
            if (abs((int)idx2[idx_pos] - (int)idx1[idx_pos]) > ZERO) {
                neighbor_errors++;
            }

            // Check distances (floating-point comparison)
            if (fabs(dst2[idx_pos] - dst1[idx_pos]) > ZERO) {
                distance_errors++;
            }
        }
    }

    // Output mismatch percentages
    printf("(Exact) Neighbors Mismatch Percentage: %f%%, ", 100 * neighbor_errors / (float)(query_length * k));
    printf("(Exact) Distances Mismatch Percentage: %f%%\n", 100 * distance_errors / (float)(query_length * k));

    // Cleanup
    free(idx1);
    free(dst1);
    free(idx2);
    free(dst2);

    return 0;
}
