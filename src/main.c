#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../include/utils/data_io.h"
#include "../include/exact/knn_exact_serial.h"
#include "../include/exact/knn_exact_pthread.h"
#include "../include/exact/knn_exact_openmp.h"
#include "../include/approximate/knn_approx_serial.h"
#include "../include/approximate/knn_approx_pthread.h"
#include "../include/tests/tests.h"

int main(int argc, char* argv[]) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s [method] [num_of_threads] [data_path] [corpus_name] [query_name] [k] [compare_results (optional)] [neighbors (optional)] [distances (optional)]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Parse command line arguments
    int             method          = atoi(argv[1]);                    // ex. 2 ( -> this means use the knn_exact_pthread)
    int             num_of_threads  = atoi(argv[2]);                    // ex. 4
    const char*     data_path       = argv[3];                          // ex. data/sift-128-euclidean.hdf5
    const char*     corpus_name     = argv[4];                          // ex. train
    const char*     query_name      = argv[5];                          // ex. test (or again train for C == Q)
    int             k               = atoi(argv[6]);                    // ex. 100
    const char*     compare_results = (argc > 7) ? argv[7] : NULL;      // ex. results/data_knn/knn_exact_serial.hdf5
    const char*     neighbors       = (argc > 7) ? argv[8] : NULL;      // ex. neighbors
    const char*     distances       = (argc > 7) ? argv[9] : NULL;      // ex. distances

    // Perform operation based on the method
    switch (method) {
        case 0:    // Runs all the exact knn functions and evaluates/compares the results based on a given dataset
            printf("Running knn_exact_serial:\n");
            generate_knn_exact_results(knn_exact_serial, data_path, corpus_name, query_name, k, 1, 1);
            printf("\n");

            printf("Running knn_exact_pthread with %d threads:\n", num_of_threads);
            generate_knn_exact_results(knn_exact_pthread, data_path, corpus_name, query_name, k, num_of_threads, 2);
            printf("\n");

            printf("Running knn_exact_openmp with %d threads:\n", num_of_threads);
            generate_knn_exact_results(knn_exact_openmp, data_path, corpus_name, query_name, k, num_of_threads, 3);
            printf("\n");
            printf("\n");

            // The results of serial knn have already been tested, using the julia algorithm or via MATLABS knnsearch
            printf("Compare knn_exact_serial results with expected:\n");
            compare_knn_exact_results("results/data_knn/knn_exact_serial.hdf5", "neighbors", "distances", 
                                      compare_results, neighbors, distances);

            // The results of serial knn have already been tested, using the julia algorithm or via MATLABS knnsearch
            printf("Compare knn_exact_pthread results with expected:\n");
            compare_knn_exact_results(compare_results, neighbors, distances,
                                      "results/data_knn/knn_exact_pthread.hdf5", "neighbors", "distances");

            printf("Compare knn_exact_openmp results with expected:\n");
            compare_knn_exact_results(compare_results, neighbors, distances,
                                      "results/data_knn/knn_exact_openmp.hdf5", "neighbors", "distances");

            // printf("\n");
            // printf("Compare knn_exact_serial results with knn_approx_serial:\n");
            // // This means that we should also compare our results with the given dataset.
            // compare_knn_approx_results("results/data_knn/knn_exact_serial.hdf5", "neighbors", "distances",
            //                            "results/data_knn/knn_approx_serial.hdf5", "neighbors", "distances");
            // compare_knn_exact_results("results/data_knn/knn_exact_serial.hdf5", "neighbors", "distances",
            //                            "results/data_knn/knn_approx_serial.hdf5", "neighbors", "distances");


            break;

        case 1:  // knn_exact_serial
            printf("Running knn_exact_serial:\n");
            generate_knn_exact_results(knn_exact_serial, data_path, corpus_name, query_name, k, 1, 1);
            printf("\n");

            if (argc > 7) {
                // This means that we should also compare our results with the given dataset.
                compare_knn_exact_results("results/data_knn/knn_exact_serial.hdf5", "neighbors", "distances", 
                                      compare_results, neighbors, distances);

            }
            break;

        case 2:  // knn_exact_pthread with num_of_threads threads
            printf("Running knn_exact_pthread with %d threads:\n", num_of_threads);
            generate_knn_exact_results(knn_exact_pthread, data_path, corpus_name, query_name, k, num_of_threads, 2);
            printf("\n");

            if (argc > 7) {
                // This means that we should also compare our results with the given dataset.
                compare_knn_exact_results("results/data_knn/knn_exact_pthread.hdf5", "neighbors", "distances", 
                                      compare_results, neighbors, distances);

            }
            break;

        case 3:  // knn_exact_openmp with num_of_threads threads
            printf("Running knn_exact_openmp with %d threads:\n", num_of_threads);
            generate_knn_exact_results(knn_exact_openmp, data_path, corpus_name, query_name, k, num_of_threads, 3);
            printf("\n");

            if (argc > 7) {
                // This means that we should also compare our results with the given dataset.
                compare_knn_exact_results("results/data_knn/knn_exact_openmp.hdf5", "neighbors", "distances", 
                                      compare_results, neighbors, distances);

            }
            break;

        case 4:
            break;

        case 5:  // knn_approx_serial
            printf("Running knn_approx_serial:\n");
            generate_knn_approx_results(knn_approx_serial, data_path, corpus_name, k, 1, 100, 5);
            printf("\n");

            if (argc > 7) {
                // This means that we should also compare our results with the given dataset.
                compare_knn_approx_results(compare_results, neighbors, distances,
                    "results/data_knn/knn_approx_serial.hdf5", "neighbors", "distances");

            }
            break;


        case 6:  // knn_approx_pthread
            printf("Running knn_approx_pthread:\n");
            generate_knn_approx_results(knn_approx_pthread, data_path, corpus_name, k, 3, 0, 6);
            printf("\n");

            if (argc > 7) {
                // This means that we should also compare our results with the given dataset.
                compare_knn_approx_results(compare_results, neighbors, distances,
                    "results/data_knn/knn_approx_pthread.hdf5", "neighbors", "distances");

            }
            break;



        case 9:  // Rundom data test for knn_approx_pthread (playground)
            srand(time(NULL));
            int data_length = 100000 + rand() % 50000;
            int dim = 150 + rand() % 50;
            printf("data_length = %d, ", data_length);
            printf("dim = %d\n", dim);
            
            float* M = (float*)malloc(data_length * dim * sizeof(float));
            for (int i = 0; i < data_length; i++) {
                for (int j = 0; j < dim; j++) {
                    M[i * dim + j] = 100 + rand() % 300;
                }
            }

            save_float_hdf5("data/tmp/test_9_corpus.hdf5", "test", M, data_length, dim);

            printf("Running knn_exact_pthread with %d threads:\n", num_of_threads);
            generate_knn_exact_results(knn_exact_pthread, "data/tmp/test_9_corpus.hdf5", "test", "test", k, num_of_threads, 2);
            printf("\n");

            printf("Running knn_approx_pthread:\n");
            generate_knn_approx_results(knn_approx_pthread, "data/tmp/test_9_corpus.hdf5", "test", k, 2, 0, 6);
            compare_knn_approx_results("results/data_knn/knn_exact_pthread.hdf5", "neighbors", "distances",
                    "results/data_knn/knn_approx_pthread.hdf5", "neighbors", "distances");
            printf("\n");
            
            break;


        default:
            fprintf(stderr, "Unknown method: %d\n", method);
            return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
