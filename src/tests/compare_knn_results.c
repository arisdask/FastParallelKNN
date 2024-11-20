#include "../../include/tests/tests.h"

int compare_knn_exact_results(const char* data_path1, const char* neighbors_name1, const char* distances_name1, const char* data_path2, const char* neighbors_name2, const char* distances_name2) {
    int k1, query_length1, k2, query_length2;

    // Load ground truth neighbors and distances
    float* idx1 = load_hdf5(data_path1, neighbors_name1, &query_length1, &k1);
    if (idx1 == NULL) {
        fprintf(stderr, "compare_knn_approx_results: Failed to load the %s data from %s.\n", neighbors_name1, data_path1);
        return -1;
    }

    float* dst1 = load_hdf5(data_path1, distances_name1, &query_length1, &k1);
    if (dst1 == NULL) {
        fprintf(stderr, "compare_knn_approx_results: Failed to load the %s data from %s.\n", distances_name1, data_path1);
        free(idx1);
        return -1;
    }

    // Load approximate neighbors and distances
    float* idx2 = load_hdf5(data_path2, neighbors_name2, &query_length2, &k2);
    if (idx2 == NULL) {
        fprintf(stderr, "compare_knn_approx_results: Failed to load the %s data from %s.\n", neighbors_name2, data_path2);
        free(idx1);
        free(dst1);
        return -1;
    }

    float* dst2 = load_hdf5(data_path2, distances_name2, &query_length2, &k2);
    if (dst2 == NULL) {
        fprintf(stderr, "compare_knn_approx_results: Failed to load the %s data from %s.\n", distances_name2, data_path2);
        free(idx1);
        free(dst1);
        free(idx2);
        return -1;
    }

    // Ensure that datasets are compatible
    if (query_length1 != query_length2 || k1 != k2) {
        fprintf(stderr, "compare_knn_approx_results: Dataset dimensions do not match: k1=%d, k2=%d, query_length1=%d, query_length2=%d\n", 
                    k1, k2, query_length1, query_length2);
        free(idx1);
        free(dst1);
        free(idx2);
        free(dst2);
        return -1;
    }

    long int query_length = query_length1;
    int k = k1;

    // Exact compare neighbors (indices) and distances:
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


int compare_knn_approx_results(const char* data_path1, const char* neighbors_name1, const char* distances_name1, 
                                const char* data_path2, const char* neighbors_name2, const char* distances_name2) {
    int k1, query_length1, k2, query_length2;

    // Load ground truth neighbors and distances
    float* idx1 = load_hdf5(data_path1, neighbors_name1, &query_length1, &k1);
    if (idx1 == NULL) {
        fprintf(stderr, "compare_knn_approx_results: Failed to load the %s data from %s.\n", neighbors_name1, data_path1);
        return -1;
    }

    float* dst1 = load_hdf5(data_path1, distances_name1, &query_length1, &k1);
    if (dst1 == NULL) {
        fprintf(stderr, "compare_knn_approx_results: Failed to load the %s data from %s.\n", distances_name1, data_path1);
        free(idx1);
        return -1;
    }

    // Load approximate neighbors and distances
    float* idx2 = load_hdf5(data_path2, neighbors_name2, &query_length2, &k2);
    if (idx2 == NULL) {
        fprintf(stderr, "compare_knn_approx_results: Failed to load the %s data from %s.\n", neighbors_name2, data_path2);
        free(idx1);
        free(dst1);
        return -1;
    }

    float* dst2 = load_hdf5(data_path2, distances_name2, &query_length2, &k2);
    if (dst2 == NULL) {
        fprintf(stderr, "compare_knn_approx_results: Failed to load the %s data from %s.\n", distances_name2, data_path2);
        free(idx1);
        free(dst1);
        free(idx2);
        return -1;
    }

    // Ensure that datasets are compatible
    if (query_length1 != query_length2 || k1 != k2) {
        fprintf(stderr, "compare_knn_approx_results: Dataset dimensions do not match: k1=%d, k2=%d, query_length1=%d, query_length2=%d\n", 
                    k1, k2, query_length1, query_length2);
        free(idx1);
        free(dst1);
        free(idx2);
        free(dst2);
        return -1;
    }

    int query_length = query_length1;
    int k = k1;

    long int total_neighbors = query_length * k;
    long int neighbor_hits = 0;

    for (int i = 0; i < query_length; i++) {
        for (int j = 0; j < k; j++) {
            int idx1_pos = i * k + j;
            int neighbor_found = 0;

            // Check if idx1[i][j] is present in idx2[i][:]
            for (int l = 0; l < k; l++) {
                int idx2_pos = i * k + l;
                if ((int)idx1[idx1_pos] == (int)idx2[idx2_pos]) {
                    neighbor_found = 1;
                    break;
                }
            }

            if (neighbor_found) {
                neighbor_hits++;
            }
        }
    }

    // Output approximate match statistics
    printf("(Approximate) Neighbors Hit Rate: %f%%\n", 100 * neighbor_hits / (float)total_neighbors);

    // Cleanup
    free(idx1);
    free(dst1);
    free(idx2);
    free(dst2);

    return 0;
}
