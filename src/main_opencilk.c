#include <stdlib.h>
#include <stdio.h>
#include "../include/exact/knn_exact_serial.h"
#include "../include/exact/knn_exact_opencilk.h"

#include "../include/tests/verify_knn_results_test.h"

int main(int argc, char* argv[]) {
    int num_of_threads = (argc > 1) ? atoi(argv[1]) : 1;

    // printf("knn_exact_serial: verify_knn_results_test:\n");
    // if (verify_knn_results_test(knn_exact_serial, "data/sift-128-euclidean.hdf5", "train", "test", "neighbors", "distances", 1) != 0) {
    //     fprintf(stderr, "main-> knn_exact_serial-> verify_knn_results_test failed to test data.\n");
    // }

    printf("knn_exact_opencilk, num_of_threads = %d:\n", num_of_threads);
    if (verify_knn_results_test(knn_exact_opencilk, "data/sift-128-euclidean.hdf5", "train", "test", "neighbors", "distances", num_of_threads) != 0) {
        fprintf(stderr, "main-> knn_exact_opencilk-> verify_knn_results_test failed to test data.\n");
    }


    return 0;
}
