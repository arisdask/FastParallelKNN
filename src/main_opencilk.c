#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../include/exact/knn_exact_opencilk.h"
#include "../include/exact/knn_exact_serial.h"
#include "../include/tests/tests.h"

int main(int argc, char* argv[]) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s [method] [num_of_threads] [data_path] [corpus_name] [query_name] [k] [compare_results (optional)] [neighbors (optional)] [distances (optional)]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Parse command line arguments
    int             method          = atoi(argv[1]);                    // ex. 4 ( -> this means use the knn_exact_opencilk)
    int             num_of_threads  = atoi(argv[2]);                    // ex. 4
    const char*     data_path       = argv[3];                          // ex. data/sift-128-euclidean.hdf5
    const char*     corpus_name     = argv[4];                          // ex. train
    const char*     query_name      = argv[5];                          // ex. test (or again train for C == Q)
    int             k               = atoi(argv[6]);                    // ex. 100
    const char*     compare_results = (argc > 7) ? argv[7] : NULL;      // ex. results/data_knn/knn_exact_pthread.hdf5
    const char*     neighbors       = (argc > 7) ? argv[8] : NULL;      // ex. neighbors
    const char*     distances       = (argc > 7) ? argv[9] : NULL;      // ex. distances

    // Perform operation based on the method
    switch (method) {
        case 0:
            printf("Running knn_exact_serial (for OpenCilk Project):\n");
            generate_knn_exact_results(knn_exact_serial, data_path, corpus_name, query_name, k, 1, 1);
            printf("\n");

            printf("Running knn_exact_opencilk with %d threads:\n", num_of_threads);
            generate_knn_exact_results(knn_exact_opencilk, data_path, corpus_name, query_name, k, num_of_threads, 4);
            printf("\n");
            printf("\n");

            // The results of serial knn have already been tested, using the julia algorithm or via MATLABS knnsearch
            printf("Compare knn_exact_serial results with knn_exact_pthread:\n");
            compare_knn_exact_results(compare_results, neighbors, distances,
                                      "results/data_knn/knn_exact_opencilk.hdf5", "neighbors", "distances");

            break;

        case 4:  // knn_exact_opencilk with num_of_threads threads
            printf("Running knn_exact_opencilk with %d threads:\n", num_of_threads);
            generate_knn_exact_results(knn_exact_opencilk, data_path, corpus_name, query_name, k, num_of_threads, 4);
            printf("\n");

            if (argc > 7) {
                // This means that we should also compare our results with the given dataset.
                compare_knn_exact_results("results/data_knn/knn_exact_opencilk.hdf5", "neighbors", "distances", 
                                      compare_results, neighbors, distances);

            }
            break;

        case 8:

            break;

        
        case 9:

            break;

        default:
            fprintf(stderr, "Unknown method: %d\n", method);
            return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
