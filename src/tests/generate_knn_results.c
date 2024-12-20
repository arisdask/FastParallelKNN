#include "../../include/tests/tests.h"

int generate_knn_exact_results(knn_exact_t knnsearch, const char* data_path, const char* corpus_name, const char* query_name, int k, int num_of_threads, int id) {
    int corpus_length, query_length, d;

    // Load the corpus (training) set
    float* corpus = load_hdf5(data_path, corpus_name, &corpus_length, &d);
    if (corpus == NULL) {
        fprintf(stderr, "generate_knn_exact_results: Failed to load the %s data of %s.\n", corpus_name, data_path);
        return -1;
    }

    // Load the query (test) set
    float* query = load_hdf5(data_path, query_name, &query_length, &d);
    if (query == NULL) {
        fprintf(stderr, "generate_knn_exact_results: Failed to load the %s data of %s.\n", query_name, data_path);
        free(corpus);
        return -1;
    }

    // Allocate memory for the k-NN results (indices and distances)
    int* idx = (int*)malloc(query_length * k * sizeof(int));
    float* dst = (float*)malloc(query_length * k * sizeof(float));
    if (idx == NULL || dst == NULL) {
        fprintf(stderr, "generate_knn_exact_results: Memory allocation failed for k-NN results.\n");
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
    printf("Running time: %lf seconds, Queries per second: %lf\n ", time_taken, (query_length / time_taken));

    // Cleanup test and train datasets
    free(corpus);
    free(query);

    
    // Save the results:
    if (id == 1) {
        save_int_hdf5("results/data_knn/knn_exact_serial.hdf5", "neighbors", idx, query_length, k);
        save_float_hdf5("results/data_knn/knn_exact_serial.hdf5", "distances", dst, query_length, k);
    } else
    if (id == 2) {
        save_int_hdf5("results/data_knn/knn_exact_pthread.hdf5", "neighbors", idx, query_length, k);
        save_float_hdf5("results/data_knn/knn_exact_pthread.hdf5", "distances", dst, query_length, k);
    } else
    if (id == 3) {
        save_int_hdf5("results/data_knn/knn_exact_openmp.hdf5", "neighbors", idx, query_length, k);
        save_float_hdf5("results/data_knn/knn_exact_openmp.hdf5", "distances", dst, query_length, k);
    } else
    if (id == 4) {
        save_int_hdf5("results/data_knn/knn_exact_opencilk.hdf5", "neighbors", idx, query_length, k);
        save_float_hdf5("results/data_knn/knn_exact_opencilk.hdf5", "distances", dst, query_length, k);
    }
    

    // Cleanup
    free(idx);
    free(dst);

    return 0;
}

int generate_knn_approx_results(knn_approx_t knnsearch, const char* data_path, const char* dataset_name, int k, int num_of_threads, int accuracy, int id) {
    int dataset_length, d;

    // Load the corpus (training) set
    float* dataset = load_hdf5(data_path, dataset_name, &dataset_length, &d);
    if (dataset == NULL) {
        fprintf(stderr, "generate_knn_approx_results: Failed to load the %s data of %s.\n", dataset_name, data_path);
        return -1;
    }

    // Allocate memory for the k-NN results (indices and distances)
    int* idx = (int*)malloc(dataset_length * k * sizeof(int));
    float* dst = (float*)malloc(dataset_length * k * sizeof(float));
    if (idx == NULL || dst == NULL) {
        fprintf(stderr, "generate_knn_approx_results: Memory allocation failed for k-NN results.\n");
        free(dataset);
        return -1;
    }

    // Timing the k-NN function
    struct timeval start, end;
    gettimeofday(&start, NULL);
    knnsearch(dataset, k, idx, dst, dataset_length, d, num_of_threads, accuracy);
    gettimeofday(&end, NULL);

    // Calculate elapsed time in seconds
    double time_taken = (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) / 1e6);
    printf("Running time: %lf seconds, Queries per second: %lf\n ", time_taken, (dataset_length / time_taken));

    // Cleanup test and train datasets
    free(dataset);

    
    // Save the results: 
    if (id == 5) {
        save_int_hdf5("results/data_knn/knn_approx_serial.hdf5", "neighbors", idx, dataset_length, k);
        save_float_hdf5("results/data_knn/knn_approx_serial.hdf5", "distances", dst, dataset_length, k);
    } else 
    if (id == 6) {
        save_int_hdf5("results/data_knn/knn_approx_pthread.hdf5", "neighbors", idx, dataset_length, k);
        save_float_hdf5("results/data_knn/knn_approx_pthread.hdf5", "distances", dst, dataset_length, k);
    } else 
    if (id == 7) {
        save_int_hdf5("results/data_knn/knn_approx_openmp.hdf5", "neighbors", idx, dataset_length, k);
        save_float_hdf5("results/data_knn/knn_approx_openmp.hdf5", "distances", dst, dataset_length, k);
    } else 
    if (id == 8) {
        save_int_hdf5("results/data_knn/knn_approx_opencilk.hdf5", "neighbors", idx, dataset_length, k);
        save_float_hdf5("results/data_knn/knn_approx_opencilk.hdf5", "distances", dst, dataset_length, k);
    }
    

    // Cleanup
    free(idx);
    free(dst);

    return 0;
}