#include <omp.h>
#include "../../include/approximate/knn_approx_serial.h"

void knn_approx_openmp(const float* dataset, int k, int* indices, float* distances, 
                       int dataset_length, int d, int num_of_threads) {
    // Initialize distances and indices arrays
    #pragma omp parallel for num_threads(num_of_threads)
    for (int i = 0; i < dataset_length * k; i++) {
        distances[i] = FLT_MAX; // Initialize distances to large values
        indices[i] = -1;        // Initialize indices to invalid values
    }

    // Parallel processing with OpenMP
    #pragma omp parallel num_threads(num_of_threads)
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        int block_size = (dataset_length + total_threads - 1) / total_threads;

        int start = thread_id * block_size;
        int end = (start + block_size > dataset_length) ? dataset_length : (start + block_size);

        // Allocate memory for the subset of the dataset
        int subset_count = end - start;
        float* subset_data = (float*)malloc(subset_count * d * sizeof(float));
        int* subset_knn_indices = (int*)malloc(subset_count * k * sizeof(int));
        float* subset_knn_distances = (float*)malloc(subset_count * k * sizeof(float));

        if (!subset_data || !subset_knn_indices || !subset_knn_distances) {
            fprintf(stderr, "Memory allocation failed for thread %d.\n", thread_id);
            free(subset_data);
            free(subset_knn_indices);
            free(subset_knn_distances);
        }

        // Copy subset data
        for (int i = 0; i < subset_count; i++) {
            memcpy(&subset_data[i * d], &dataset[(start + i) * d], d * sizeof(float));
        }

        // Perform exact k-NN search for the subset
        knn_exact_serial(subset_data, subset_data, 
                         k, subset_knn_indices, subset_knn_distances, 
                         subset_count, subset_count, d, 1);

        // Write results back to the global distances and indices arrays
        for (int i = 0; i < subset_count; i++) {
            int original_idx = start + i;
            for (int j = 0; j < k; j++) {
                int offset = original_idx * k + j;
                #pragma omp critical
                {
                    if (subset_knn_distances[i * k + j] < distances[offset]) {
                        distances[offset] = subset_knn_distances[i * k + j];
                        indices[offset] = start + subset_knn_indices[i * k + j];
                    }
                }
            }
        }

        // Cleanup
        free(subset_data);
        free(subset_knn_indices);
        free(subset_knn_distances);
    }
}
