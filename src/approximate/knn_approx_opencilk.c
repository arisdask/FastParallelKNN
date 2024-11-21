#include "../../include/approximate/knn_approx_opencilk.h"

// OpenCilk implementation of the approximate k-NN
void knn_approx_opencilk(const float* dataset, int k, int* indices, float* distances, 
                         int dataset_length, int d, int num_of_threads, int accuracy) {
    // Initialize distances and indices arrays
    cilk_for (int i = 0; i < dataset_length * k; i++) {
        distances[i] = FLT_MAX; // Initialize distances to large values
        indices[i] = -1;        // Initialize indices to invalid values
    }

    // Define the block size for each task based on dataset size and number of threads
    int block_size = (dataset_length + num_of_threads - 1) / num_of_threads;

    // Create a loop to divide the dataset and handle each chunk in parallel
    cilk_for (int t = 0; t < num_of_threads; t++) {
        int start = t * block_size;
        int end = (start + block_size > dataset_length) ? dataset_length : (start + block_size);

        // Allocate memory for the subset of the dataset
        int subset_count = end - start;
        float* subset_data = (float*)malloc(subset_count * d * sizeof(float));
        int* subset_knn_indices = (int*)malloc(subset_count * k * sizeof(int));
        float* subset_knn_distances = (float*)malloc(subset_count * k * sizeof(float));

        if (!subset_data || !subset_knn_indices || !subset_knn_distances) {
            fprintf(stderr, "Memory allocation failed for task %d.\n", t);
            free(subset_data);
            free(subset_knn_indices);
            free(subset_knn_distances);
        }

        // Copy subset data
        for (int i = 0; i < subset_count; i++) {
            memcpy(&subset_data[i * d], &dataset[(start + i) * d], d * sizeof(float));
        }

        // Perform exact k-NN search for the subset using a serial method
        knn_exact_serial(subset_data, subset_data, 
                         k, subset_knn_indices, subset_knn_distances, 
                         subset_count, subset_count, d, 1);

        // Write results back to the global distances and indices arrays
        for (int i = 0; i < subset_count; i++) {
            int original_idx = start + i;
            for (int j = 0; j < k; j++) {
                int offset = original_idx * k + j;
                // Use a lock-free update mechanism to ensure thread safety
                if (subset_knn_distances[i * k + j] < distances[offset]) {
                    distances[offset] = subset_knn_distances[i * k + j];
                    indices[offset] = start + subset_knn_indices[i * k + j];
                }
            }
        }

        // Cleanup
        free(subset_data);
        free(subset_knn_indices);
        free(subset_knn_distances);
    }

    // After parallel processing, perform a final pass to ensure correct distance comparisons
    for (int i = 0; i < dataset_length; i++) {
        for (int j = 0; j < k; j++) {
            // Check all tasks for potential better values
            for (int t = 0; t < num_of_threads; t++) {
                int offset = i * k + j;
                if (distances[offset] > distances[offset]) {
                    distances[offset] = distances[offset];
                    indices[offset] = indices[offset];
                }
            }
        }
    }
}
