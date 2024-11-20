#include "../../include/approximate/knn_approx_serial.h"

void split_dataset(const float* dataset, float* distances_from_hyperplane, 
                            int dataset_length, int d, int num_of_threads, int accuracy, float* _norm_) {
    // Allocate space for midpoints
    float* mean_points = (float*)malloc(3 * d * sizeof(float));
    if (!mean_points) {
        fprintf(stderr, "Memory allocation failed for mean_points.\n");
        return;
    }

    // Compute the vector between the two helper midpoints
    float* v = (float*)malloc(d * sizeof(float));
    if (!v) {
        fprintf(stderr, "Memory allocation failed for vector v.\n");
        free(mean_points);
        return;
    }

    // Compute the norm of the vector
    float v_norm = 0;
    int tmp = 2;

    while (v_norm < __ZERO__ && tmp < 5) {
        // Initialize midpoints
        for (int i = 0; i < 3 * d; i++) {
            mean_points[i] = 0;
        }

        // Compute mean of the first half of the dataset
        for (int i = 0; i < dataset_length / tmp; i++) {
            for (int j = 0; j < d; j++) {
                mean_points[j] += dataset[i * d + j];
            }
        }
        for (int i = 0; i < d; i++) {
            mean_points[i] /= (dataset_length / tmp);  // Normalize
        }

        // Compute mean of the second half of the dataset
        for (int i = dataset_length / 2; i < dataset_length / 2 + dataset_length / tmp; i++) {
            for (int j = 0; j < d; j++) {
                mean_points[d + j] += dataset[i * d + j];
            }
        }
        for (int i = 0; i < d; i++) {
            mean_points[d + i] /= (dataset_length / tmp);  // Normalize
        }

        // Compute the actual midpoint
        for (int i = 0; i < d; i++) {
            mean_points[2 * d + i] = (mean_points[i] + mean_points[d + i]) / 2;
        }

        for (int i = 0; i < d; i++) {
            v[i] = mean_points[d + i] - mean_points[i];
        }

        for (int i = 0; i < d; i++) {
            v_norm += v[i] * v[i];
        }
        v_norm = sqrt(v_norm);

    }

    // Compute the distance of each point from the hyperplane
    for (int i = 0; i < dataset_length; i++) {
        float dot_product = 0;
        for (int j = 0; j < d; j++) {
            dot_product += v[j] * (dataset[i * d + j] - mean_points[2 * d + j]);
        }
        distances_from_hyperplane[i] = dot_product / v_norm;  // Store distance (If dot_product < 0 -> sub-space 1, dot_product > 0 -> sub-space 2)
    }

    // Cleanup
    free(mean_points);
    free(v);
}


void knn_approx_serial(const float* dataset, int k, int* indices, float* distances, 
                       int dataset_length, int d, int num_of_threads, int accuracy) {
    float* distances_from_hyperplane = (float*)malloc(dataset_length * sizeof(float));
    if (!distances_from_hyperplane) {
        fprintf(stderr, "Memory allocation failed for distances_from_hyperplane.\n");
        return;
    }

    // Step 2: Calculate n_norm (distance of helper midpoints)
    float n_norm = 0;

    // Step 1: Split dataset into three parts using `split_dataset`
    split_dataset(dataset, distances_from_hyperplane, dataset_length, d, num_of_threads, accuracy, &n_norm);

    // Step 3: Partition dataset based on the distances from the hyperplane
    int* part1_indices = (int*)malloc(dataset_length * sizeof(int));
    int* part2_indices = (int*)malloc(dataset_length * sizeof(int));
    int* part3_indices = (int*)malloc(dataset_length * sizeof(int));
    if (!part1_indices || !part2_indices || !part3_indices) {
        fprintf(stderr, "Memory allocation failed for subsets.\n");
        free(distances_from_hyperplane);
        free(part1_indices);
        free(part2_indices);
        free(part3_indices);
        return;
    }

    int part1_count = 0, part2_count = 0, part3_count = 0;
    for (int i = 0; i < dataset_length; i++) {
        if (distances_from_hyperplane[i] < -n_norm) {
            part1_indices[part1_count++] = i;
        }
        if (distances_from_hyperplane[i] > n_norm) {
            part2_indices[part2_count++] = i;
        }
        if (fabs(distances_from_hyperplane[i]) <= n_norm) {
            part3_indices[part3_count++] = i;
        }
    }

    // Create reduced datasets for each part
    float* part1_data = (float*)malloc(part1_count * d * sizeof(float));
    float* part2_data = (float*)malloc(part2_count * d * sizeof(float));
    float* part3_data = (float*)malloc(part3_count * d * sizeof(float));

    if (!part1_data || !part2_data || !part3_data) {
        fprintf(stderr, "Memory allocation failed for reduced datasets.\n");
        free(distances_from_hyperplane);
        free(part1_indices);
        free(part2_indices);
        free(part3_indices);
        free(part1_data);
        free(part2_data);
        free(part3_data);
        return;
    }

    for (int i = 0; i < part1_count; i++) {
        memcpy(&part1_data[i * d], &dataset[part1_indices[i] * d], d * sizeof(float));
    }

    for (int i = 0; i < part2_count; i++) {
        memcpy(&part2_data[i * d], &dataset[part2_indices[i] * d], d * sizeof(float));
    }

    for (int i = 0; i < part3_count; i++) {
        memcpy(&part3_data[i * d], &dataset[part3_indices[i] * d], d * sizeof(float));
    }

    // Step 4: Process each subset using exact k-NN
    int* part1_knn_indices = (int*)malloc(part1_count * k * sizeof(int));
    float* part1_knn_distances = (float*)malloc(part1_count * k * sizeof(float));
    knn_exact_serial(part1_data, part1_data, k, part1_knn_indices, part1_knn_distances, 
                     part1_count, part1_count, d, num_of_threads);

    int* part2_knn_indices = (int*)malloc(part2_count * k * sizeof(int));
    float* part2_knn_distances = (float*)malloc(part2_count * k * sizeof(float));
    knn_exact_serial(part2_data, part2_data, k, part2_knn_indices, part2_knn_distances, 
                     part2_count, part2_count, d, num_of_threads);

    int* part3_knn_indices = (int*)malloc(part3_count * k * sizeof(int));
    float* part3_knn_distances = (float*)malloc(part3_count * k * sizeof(float));
    knn_exact_serial(part3_data, part3_data, k, part3_knn_indices, part3_knn_distances, 
                     part3_count, part3_count, d, num_of_threads);

    // Step 5: Assign results directly from the subsets

    // Assign results for Part 1
    for (int i = 0; i < part1_count; i++) {
        int original_idx = part1_indices[i]; // Map back to the original dataset index
        for (int j = 0; j < k; j++) {
            indices[original_idx * k + j] = part1_knn_indices[i * k + j];
            distances[original_idx * k + j] = part1_knn_distances[i * k + j];
        }
    }

    // Assign results for Part 2
    for (int i = 0; i < part2_count; i++) {
        int original_idx = part2_indices[i]; // Map back to the original dataset index
        for (int j = 0; j < k; j++) {
            indices[original_idx * k + j] = part2_knn_indices[i * k + j];
            distances[original_idx * k + j] = part2_knn_distances[i * k + j];
        }
    }

    // Assign results for Part 3
    for (int i = 0; i < part3_count; i++) {
        int original_idx = part3_indices[i]; // Map back to the original dataset index
        for (int j = 0; j < k; j++) {
            indices[original_idx * k + j] = part3_knn_indices[i * k + j];
            distances[original_idx * k + j] = part3_knn_distances[i * k + j];
        }
    }


    // Step 6: Cleanup
    free(distances_from_hyperplane);
    free(part1_indices);
    free(part2_indices);
    free(part3_indices);
    free(part1_data);
    free(part2_data);
    free(part3_data);
    free(part1_knn_indices);
    free(part1_knn_distances);
    free(part2_knn_indices);
    free(part2_knn_distances);
    free(part3_knn_indices);
    free(part3_knn_distances);
}
