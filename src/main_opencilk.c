#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../include/exact/knn_exact_opencilk.h"
#include "../include/approximate/knn_approx_opencilk.h"
#include "../include/tests/tests.h"

int main(int argc, char* argv[]) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s [method] [num_of_threads] [data_path] [corpus_name] [query_name] [k] [compare_results (optional)] [neighbors (optional)] [distances (optional)]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Parse command line arguments
    int             method          = atoi(argv[1]);
    int             num_of_threads  = atoi(argv[2]);
    const char*     data_path       = argv[3];
    const char*     corpus_name     = argv[4];
    const char*     query_name      = argv[5];
    int             k               = atoi(argv[6]);
    const char*     compare_results = (argc > 7) ? argv[7] : NULL;
    const char*     neighbors       = (argc > 7) ? argv[8] : NULL;
    const char*     distances       = (argc > 7) ? argv[9] : NULL;

    // Perform operation based on the method
    switch (method) {
        case 0:
            printf("Running knn_exact_opencilk with %d threads:\n", num_of_threads);
            generate_knn_exact_results(knn_exact_opencilk, data_path, corpus_name, query_name, k, num_of_threads, 4);
            printf("\n");
            printf("\n");

            printf("Compare knn_exact_opencilk results with expected:\n");
            compare_knn_exact_results(compare_results, neighbors, distances,
                    "results/data_knn/knn_exact_opencilk.hdf5", "neighbors", "distances");

            break;

        case 1:
            printf("Running knn_approx_opencilk:\n");
            generate_knn_approx_results(knn_approx_opencilk, "data/random_dataset/test_corpus.hdf5", "test", k, num_of_threads, 0, 8);
            printf("\n");

            printf("Compare knn_approx_opencilk results with expected:\n");
            compare_knn_approx_results("results/data_knn/knn_exact_pthread.hdf5", "neighbors", "distances",
                    "results/data_knn/knn_approx_opencilk.hdf5", "neighbors", "distances");

            break;


        default:
            printf("Unknown method for main_opencilk.c: %d\n", method);
    }

    return EXIT_SUCCESS;
}
