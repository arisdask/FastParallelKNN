#include <stdlib.h>
#include <stdio.h>
#include "../include/exact/knn_exact_serial.h"
#include "../include/exact/knn_exact_pthread.h"
#include "../include/exact/knn_exact_openmp.h"

#include "../include/tests/verify_knn_results_test.h"

int main(int argc, char* argv[]) {
    // int num_of_threads = (argc > 1) ? atoi(argv[1]) : 1;

    printf("knn_exact_serial:\n");
    if (verify_knn_results_test(knn_exact_serial, "data/sift-128-euclidean.hdf5", "train", "test", "neighbors", "distances", 1) != 0) {
        fprintf(stderr, "main-> knn_exact_serial-> verify_knn_results_test failed to test data.\n");
    }

    for (int i = 1; i <= 8; i++) {
        printf("knn_exact_pthread, num_of_threads = %d:\n", i);
        if (verify_knn_results_test(knn_exact_pthread, "data/sift-128-euclidean.hdf5", "train", "test", "neighbors", "distances", i) != 0) {
            fprintf(stderr, "main-> knn_exact_pthread-> verify_knn_results_test failed to test data.\n");
        }
    }

    for (int i = 1; i <= 8; i++) {
        printf("\nknn_exact_openmp, num_of_threads = %d:\n", i);
        if (verify_knn_results_test(knn_exact_openmp, "data/sift-128-euclidean.hdf5", "train", "test", "neighbors", "distances", i) != 0) {
            fprintf(stderr, "main-> knn_exact_openmp-> verify_knn_results_test failed to test data.\n");
        }
    }


    return 0;
}
