#include "../../include/approximate/knn_approx_pthread.h"

typedef struct {
    const float* dataset;
    int* indices;
    float* distances;
    int* subset_indices;
    int subset_count;
    int k;
    int d;
} knn_approx_thread_args_t;

void* knn_approx_thread(void* args) {
    knn_approx_thread_args_t* thread_args = (knn_approx_thread_args_t*)args;

    // Allocate space for reduced dataset
    float* subset_data = (float*)malloc(thread_args->subset_count * thread_args->d * sizeof(float));
    if (!subset_data) {
        fprintf(stderr, "Memory allocation failed for subset data.\n");
        pthread_exit(NULL);
    }

    // Copy subset data
    for (int i = 0; i < thread_args->subset_count; i++) {
        memcpy(&subset_data[i * thread_args->d], 
               &thread_args->dataset[thread_args->subset_indices[i] * thread_args->d], 
               thread_args->d * sizeof(float));
    }

    // Allocate memory for k-NN results for this subset
    int* subset_knn_indices = (int*)malloc(thread_args->subset_count * thread_args->k * sizeof(int));
    float* subset_knn_distances = (float*)malloc(thread_args->subset_count * thread_args->k * sizeof(float));
    if (!subset_knn_indices || !subset_knn_distances) {
        fprintf(stderr, "Memory allocation failed for k-NN results.\n");
        free(subset_data);
        pthread_exit(NULL);
    }

    // Initialize distances to large values
    for (int i = 0; i < thread_args->subset_count * thread_args->k; i++) {
        subset_knn_distances[i] = FLT_MAX;
        subset_knn_indices[i] = -1;
    }

    // Perform exact k-NN search for the subset
    knn_exact_serial(subset_data, subset_data, 
                     thread_args->k, subset_knn_indices, subset_knn_distances, 
                     thread_args->subset_count, thread_args->subset_count, 
                     thread_args->d, 1); // Use serial computation for each thread

    // Map the subset results back to global indices and distances
    for (int i = 0; i < thread_args->subset_count; i++) {
        int original_idx = thread_args->subset_indices[i];
        for (int j = 0; j < thread_args->k; j++) {
            thread_args->indices[original_idx * thread_args->k + j] = 
                thread_args->subset_indices[subset_knn_indices[i * thread_args->k + j]];
            thread_args->distances[original_idx * thread_args->k + j] = 
                subset_knn_distances[i * thread_args->k + j];
        }
    }

    // Cleanup
    free(subset_data);
    free(subset_knn_indices);
    free(subset_knn_distances);
    pthread_exit(NULL);
}

void knn_approx_pthread(const float* dataset, int k, int* indices, float* distances, 
                        int dataset_length, int d, int num_of_threads, int accuracy) {
    // Allocate memory for distances and indices arrays
    for (int i = 0; i < dataset_length * k; i++) {
        distances[i] = FLT_MAX; // Initialize distances to large values
        indices[i] = -1;        // Initialize indices to invalid values
    }

    // Split dataset into subsets for threads
    int block_size = (dataset_length + num_of_threads - 1) / num_of_threads;
    pthread_t* threads = (pthread_t*)malloc(num_of_threads * sizeof(pthread_t));
    knn_approx_thread_args_t* thread_args = (knn_approx_thread_args_t*)malloc(num_of_threads * sizeof(knn_approx_thread_args_t));

    for (int t = 0; t < num_of_threads; t++) {
        int start = t * block_size;
        int end = (start + block_size > dataset_length) ? dataset_length : (start + block_size);

        thread_args[t].dataset = dataset;
        thread_args[t].indices = indices;
        thread_args[t].distances = distances;
        thread_args[t].subset_count = end - start;
        thread_args[t].subset_indices = (int*)malloc(thread_args[t].subset_count * sizeof(int));
        thread_args[t].k = k;
        thread_args[t].d = d;

        for (int i = 0; i < thread_args[t].subset_count; i++) {
            thread_args[t].subset_indices[i] = start + i;
        }

        pthread_create(&threads[t], NULL, knn_approx_thread, &thread_args[t]);
    }

    // Wait for all threads to complete
    for (int t = 0; t < num_of_threads; t++) {
        pthread_join(threads[t], NULL);
        free(thread_args[t].subset_indices); // Free the subset indices
    }

    // Verification step (single-threaded)
    for (int i = 0; i < dataset_length; i++) {
        for (int j = 0; j < k; j++) {
            // Ensure correct distance comparisons
            for (int t = 0; t < num_of_threads; t++) {
                int offset = i * k + j;
                if (thread_args[t].distances[offset] < distances[offset]) {
                    distances[offset] = thread_args[t].distances[offset];
                    indices[offset] = thread_args[t].indices[offset];
                }
            }
        }
    }

    // Cleanup
    free(threads);
    free(thread_args);
}