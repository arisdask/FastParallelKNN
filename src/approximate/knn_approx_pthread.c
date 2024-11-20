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






// Dead Code.... RIP :/


// #include "../../include/approximate/knn_approx_pthread.h"

// // Structure to hold thread arguments
// typedef struct {
//     const float* dataset;
//     int dataset_length;
//     int d;
//     int k;
//     int* indices;
//     float* distances;
//     int num_of_threads;
//     int accuracy;
//     int thread_id;
// } knn_approx_thread_args_t;

// // Function to recursively handle subproblems
// void* knn_approx_pthread_core(void* args) {
//     knn_approx_thread_args_t* thread_args = (knn_approx_thread_args_t*)args;

//     if (thread_args->num_of_threads <= 1 || thread_args->dataset_length < thread_args->accuracy) {
//         // If there is only one thread or the subset is small, run serially
//         knn_approx_serial(thread_args->dataset, thread_args->k, thread_args->indices,
//                           thread_args->distances, thread_args->dataset_length,
//                           thread_args->d, 1, thread_args->accuracy);
//         pthread_exit(NULL);
//     }

//     // Step 1: Split dataset into three parts
//     float* distances_from_hyperplane = (float*)malloc(thread_args->dataset_length * sizeof(float));
//     if (!distances_from_hyperplane) {
//         fprintf(stderr, "knn_approx_pthread: Memory allocation failed for distances_from_hyperplane.\n");
//         pthread_exit(NULL);
//     }

//     float n_norm = 0;
//     split_dataset(thread_args->dataset, distances_from_hyperplane, thread_args->dataset_length,
//                   thread_args->d, thread_args->num_of_threads, thread_args->accuracy, &n_norm);

//     // Partition dataset
//     int* part1_indices = (int*)malloc(thread_args->dataset_length * sizeof(int));
//     int* part2_indices = (int*)malloc(thread_args->dataset_length * sizeof(int));
//     int* part3_indices = (int*)malloc(thread_args->dataset_length * sizeof(int));
//     int part1_count = 0, part2_count = 0, part3_count = 0;

//     for (int i = 0; i < thread_args->dataset_length; i++) {
//         if (distances_from_hyperplane[i] < -n_norm) {
//             part1_indices[part1_count++] = i;
//         }
//         if (distances_from_hyperplane[i] > n_norm) {
//             part2_indices[part2_count++] = i;
//         }
//         if (fabs(distances_from_hyperplane[i]) <= n_norm) {
//             part3_indices[part3_count++] = i;
//         }
//     }

//     // Allocate reduced datasets
//     float* part1_data = (float*)malloc(part1_count * thread_args->d * sizeof(float));
//     float* part2_data = (float*)malloc(part2_count * thread_args->d * sizeof(float));
//     float* part3_data = (float*)malloc(part3_count * thread_args->d * sizeof(float));

//     for (int i = 0; i < part1_count; i++) {
//         memcpy(&part1_data[i * thread_args->d], &thread_args->dataset[part1_indices[i] * thread_args->d],
//                thread_args->d * sizeof(float));
//     }
//     for (int i = 0; i < part2_count; i++) {
//         memcpy(&part2_data[i * thread_args->d], &thread_args->dataset[part2_indices[i] * thread_args->d],
//                thread_args->d * sizeof(float));
//     }
//     for (int i = 0; i < part3_count; i++) {
//         memcpy(&part3_data[i * thread_args->d], &thread_args->dataset[part3_indices[i] * thread_args->d],
//                thread_args->d * sizeof(float));
//     }

//     // Step 2: Parallelize the processing of parts using threads
//     pthread_t threads[3];
//     knn_approx_thread_args_t thread_args_sub[3];

//     for (int i = 0; i < 3; i++) {
//         thread_args_sub[i] = *thread_args; // Copy common parameters
//         thread_args_sub[i].num_of_threads = thread_args->num_of_threads / 3;

//         if (i == 0) { // Part 1
//             thread_args_sub[i].dataset = part1_data;
//             thread_args_sub[i].dataset_length = part1_count;
//             thread_args_sub[i].indices = thread_args->indices;
//             thread_args_sub[i].distances = thread_args->distances;
//         } else if (i == 1) { // Part 2
//             thread_args_sub[i].dataset = part2_data;
//             thread_args_sub[i].dataset_length = part2_count;
//             thread_args_sub[i].indices = thread_args->indices;
//             thread_args_sub[i].distances = thread_args->distances;
//         } else { // Part 3
//             thread_args_sub[i].dataset = part3_data;
//             thread_args_sub[i].dataset_length = part3_count;
//             thread_args_sub[i].indices = thread_args->indices;
//             thread_args_sub[i].distances = thread_args->distances;
//         }

//         pthread_create(&threads[i], NULL, knn_approx_pthread_core, &thread_args_sub[i]);
//     }

//     for (int i = 0; i < 3; i++) {
//         pthread_join(threads[i], NULL);
//     }

//     // Cleanup
//     free(part1_indices);
//     free(part2_indices);
//     free(part3_indices);
//     free(part1_data);
//     free(part2_data);
//     free(part3_data);
//     free(distances_from_hyperplane);

//     pthread_exit(NULL);
// }

// // Main knn_approx_pthread function
// void knn_approx_pthread(const float* dataset, int k, int* indices, float* distances,
//                         int dataset_length, int d, int num_of_threads, int accuracy) {
//     knn_approx_thread_args_t args = {dataset, dataset_length, d, k, indices, distances,
//                               num_of_threads, accuracy, 0};

//     knn_approx_pthread_core(&args);
// }
