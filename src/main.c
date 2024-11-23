#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../include/utils/data_io.h"
#include "../include/exact/knn_exact_serial.h"
#include "../include/exact/knn_exact_pthread.h"
#include "../include/exact/knn_exact_openmp.h"
#include "../include/approximate/knn_approx_serial.h"
#include "../include/approximate/knn_approx_pthread.h"
#include "../include/approximate/knn_approx_openmp.h"
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

    int     data_length = 0;
    int     dim         = 0;
    float*  M           = NULL;

    
    // 0 - Run all the *exact* knn functions and evaluate/compare the results based on the given dataset
    // 1 - Run all the *approx* knn functions and evaluate/compare the results (based on the exact results of an exact knn)
    //     Keep in mind that the approximate solutions solve only the all-to-all k-NN problem in which C == Q
    // 2 - Random Data Test for knn_approx_pthread (Playground)
    // 3 - You can add your own custom tests here!
    switch (method) {
        case 0:    // Runs all the exact knn functions and evaluates/compares the results based on a given dataset
            // To run the exact methods you can set the `USABLE_MEM_PREDICTION` inside the mem_info.h up to
            // 6000000 for a system that has 16GB of total memory (see README for a more detailed analysis).
            // In case the program crashes the `USABLE_MEM_PREDICTION` should become even lower. 

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

            // The results of knn_exact_serial have also been tested, using the julia algorithm or via MATLABS knnsearch
            printf("Compare knn_exact_serial results with expected:\n");
            compare_knn_exact_results("results/data_knn/knn_exact_serial.hdf5", "neighbors", "distances", 
                                      compare_results, neighbors, distances);

            printf("Compare knn_exact_pthread results with expected:\n");
            compare_knn_exact_results(compare_results, neighbors, distances,
                                      "results/data_knn/knn_exact_pthread.hdf5", "neighbors", "distances");

            printf("Compare knn_exact_openmp results with expected:\n");
            compare_knn_exact_results(compare_results, neighbors, distances,
                                      "results/data_knn/knn_exact_openmp.hdf5", "neighbors", "distances");
            printf("\n");

            break;

        case 1:  // Runs all the approx knn functions (keep in mind that the approximate solutions solve only the all-to-all k-NN problem in which C == Q) 
                 // and evaluates/compares the results based on the results of the knn_exact_pthread

            // In case the program crashes the `USABLE_MEM_PREDICTION` inside the mem_info.h should become even lower. 
            srand(time(NULL));
            data_length = 100000 + rand() % 50000;
            dim = 150 + rand() % 50;
            M = (float*)malloc(data_length * dim * sizeof(float));
            
            printf("data_length = %d, ", data_length);
            printf("dim = %d\n\n", dim);
            for (int i = 0; i < data_length; i++) {
                for (int j = 0; j < dim; j++) {
                    M[i * dim + j] = 100 + rand() % 300;
                }
            }
            save_float_hdf5("data/random_dataset/test_corpus.hdf5", "test", M, data_length, dim);

            printf("Running knn_exact_pthread with %d threads:\n", num_of_threads);
            generate_knn_exact_results(knn_exact_pthread, "data/random_dataset/test_corpus.hdf5", "test", "test", k, num_of_threads, 2);
            printf("\n");

            printf("Running knn_approx_serial:\n");
            generate_knn_approx_results(knn_approx_serial, "data/random_dataset/test_corpus.hdf5", "test", k, num_of_threads, 0, 5);
            printf("\n");

            printf("Running knn_approx_pthread with %d threads:\n", num_of_threads);
            generate_knn_approx_results(knn_approx_pthread, "data/random_dataset/test_corpus.hdf5", "test", k, num_of_threads, 0, 6);
            printf("\n");

            printf("Running knn_approx_openmp with %d threads:\n", num_of_threads);
            generate_knn_approx_results(knn_approx_openmp, "data/random_dataset/test_corpus.hdf5", "test", k, num_of_threads, 0, 7);
            printf("\n");
            printf("\n");

            // We already have test that the knn_exact_pthread gives correct results:
            printf("Compare knn_approx_serial results with expected:\n");
            compare_knn_approx_results("results/data_knn/knn_exact_pthread.hdf5", "neighbors", "distances",
                    "results/data_knn/knn_approx_serial.hdf5", "neighbors", "distances");

            printf("Compare knn_approx_pthread results with expected:\n");
            compare_knn_approx_results("results/data_knn/knn_exact_pthread.hdf5", "neighbors", "distances",
                    "results/data_knn/knn_approx_pthread.hdf5", "neighbors", "distances");

            printf("Compare knn_approx_openmp results with expected:\n");
            compare_knn_approx_results("results/data_knn/knn_exact_pthread.hdf5", "neighbors", "distances",
                    "results/data_knn/knn_approx_openmp.hdf5", "neighbors", "distances");
            printf("\n");

            break;

        case 2:  // Random Data Test for knn_approx_pthread (Playground)

            // In case the program crashes the `USABLE_MEM_PREDICTION` inside the mem_info.h should become even lower.
            srand(time(NULL));
            data_length = 100000 + rand() % 50000;
            dim = 150 + rand() % 50;
            M = (float*)malloc(data_length * dim * sizeof(float));
            
            printf("data_length = %d, ", data_length);
            printf("dim = %d\n\n", dim);
            for (int i = 0; i < data_length; i++) {
                for (int j = 0; j < dim; j++) {
                    M[i * dim + j] = 100 + rand() % 300;
                }
            }
            save_float_hdf5("data/random_dataset/test_corpus.hdf5", "test", M, data_length, dim);

            printf("Running knn_exact_pthread with %d threads:\n", num_of_threads);
            generate_knn_exact_results(knn_exact_pthread, "data/random_dataset/test_corpus.hdf5", "test", "test", k, num_of_threads, 2);
            printf("\n");

            printf("Running knn_approx_pthread with %d threads:\n", num_of_threads);
            generate_knn_approx_results(knn_approx_pthread, "data/random_dataset/test_corpus.hdf5", "test", k, num_of_threads, 0, 6);
            printf("\n");
            printf("\n");

            // We already have test that the knn_exact_pthread gives correct results.
            compare_knn_approx_results("results/data_knn/knn_exact_pthread.hdf5", "neighbors", "distances",
                    "results/data_knn/knn_approx_pthread.hdf5", "neighbors", "distances");
            printf("\n");
            
            break;

        case 3:
            // Download `sift-128-euclidean.hdf5` in `data/` folder to run the approximate knn for the train (or the test) dataset.
            // Make sure to correctly change the datapaths or dataset names in case you want to run a test for a different dataset.

            // For the approximate tests below, the `USABLE_MEM_PREDICTION` inside the mem_info.h should be <=3000000 (kByte) 
            // for a system that has 16GB of total memory (for more than 8 threads this value should become even lower - see README).
            // In case the program crashes the `USABLE_MEM_PREDICTION` should become even lower. 

            printf("Running knn_exact_pthread with %d threads:\n", num_of_threads);
            generate_knn_exact_results(knn_exact_pthread, "data/sift-128-euclidean.hdf5", "train", "train", k, num_of_threads, 2);
            printf("\n");

            printf("Running knn_approx_serial:\n");
            generate_knn_approx_results(knn_approx_serial, "data/sift-128-euclidean.hdf5", "train", k, num_of_threads, 0, 5);
            printf("\n");

            printf("Running knn_approx_pthread with %d threads:\n", num_of_threads);
            generate_knn_approx_results(knn_approx_pthread, "data/sift-128-euclidean.hdf5", "train", k, num_of_threads, 0, 6);
            printf("\n");

            printf("Running knn_approx_openmp with %d threads:\n", num_of_threads);
            generate_knn_approx_results(knn_approx_openmp, "data/sift-128-euclidean.hdf5", "train", k, num_of_threads, 0, 7);
            printf("\n");
            printf("\n");

            // We already have test that the knn_exact_pthread gives correct results.
            printf("Compare knn_approx_serial results with expected:\n");
            compare_knn_approx_results("results/data_knn/knn_exact_pthread.hdf5", "neighbors", "distances",
                    "results/data_knn/knn_approx_serial.hdf5", "neighbors", "distances");

            printf("Compare knn_approx_pthread results with expected:\n");
            compare_knn_approx_results("results/data_knn/knn_exact_pthread.hdf5", "neighbors", "distances",
                    "results/data_knn/knn_approx_pthread.hdf5", "neighbors", "distances");

            printf("Compare knn_approx_openmp results with expected:\n");
            compare_knn_approx_results("results/data_knn/knn_exact_pthread.hdf5", "neighbors", "distances",
                    "results/data_knn/knn_approx_openmp.hdf5", "neighbors", "distances");
            printf("\n");
            
            break;


        default:
            printf("Unknown method for main.c: %d\n", method);
    }

    return EXIT_SUCCESS;
}
