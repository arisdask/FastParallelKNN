#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include "../../include/approximate/knn_approx_serial.h"

void knn_approx_opencilk(const float* dataset, int k, int* indices, float* distances, 
                         int dataset_length, int d, int num_of_threads) {
    __cilkrts_set_param("nworkers", std::to_string(num_of_threads).c_str());

    // Initialize distances and indices arrays
    cilk_for (int i = 0; i < dataset_length * k; i++) {
        distances[i] = FLT_MAX; // Initialize distances to large values
        indices[i] = -1;        // Initialize indices to invalid values
    }

    // Split the dataset into chunks and process in parallel
    cilk_for (int t = 0; t < num_of_threads; t++) {
        int block_size = (dataset_length + num_of_threads - 1) / num_of_threads;
        int start = t * block_size;
        int end = (start + block_size > dataset_length) ? dataset_length : (start + block_size);

        // Allocate memory for the subset of the dataset
        int subset_count = end - start;
        float* subset_data = (float*)malloc(subset_count * d * sizeof(float));
        int* subset_knn_indices = (int*)malloc(subset_count * k * sizeof(int));
        float* subset_knn_distances = (float*)malloc(subset_count * k * sizeof(float));

        if (!subset_data || !subset_knn_indices || !subset_knn_distances) {
            fprintf(stderr, "Memory allocation failed for worker %d.\n", t);
            free(subset_data);
            free(subset_knn_indices);
            free(subset_knn_distances);
            continue;
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
                cilk::reducer_opadd<float> min_distance(distances[offset]);
                cilk::reducer_opadd<int> min_index(indices[offset]);

                if (subset_knn_distances[i * k + j] < min_distance.get_value()) {
                    min_distance.set_value(subset_knn_distances[i * k + j]);
                    min_index.set_value(start + subset_knn_indices[i * k + j]);
                }
            }
        }

        // Cleanup
        free(subset_data);
        free(subset_knn_indices);
        free(subset_knn_distances);
    }
}
