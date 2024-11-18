#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "../include/exact/knn_exact_serial.h"
#include "../include/utils/data_io.h"

#define ZERO 0.001

int main(int argc, char* argv[]) {
	// atoi(argv[1]);
	int 		k 				= 100;

    int 		corpus_length 	= 0;
	int 		query_length 	= 0;
	int			d 				= 0;
    const char* data_path 		= "data/sift-128-euclidean.hdf5"; 

	// Open train (corpus) set:
    char* dataset_name = "train";
    float* corpus = load_hdf5(data_path, dataset_name, &corpus_length, &d);
    if (corpus == NULL) {
        fprintf(stderr, "Failed to load the %s data.\n", dataset_name);
        exit(EXIT_FAILURE);
    }

	// Open test (query) set:
    dataset_name = "test";
    float* query = load_hdf5(data_path, dataset_name, &query_length, &d);
    if (query == NULL) {
        fprintf(stderr, "Failed to load the %s data.\n", dataset_name);
        exit(EXIT_FAILURE);
    }

	// Allocate memory for indices and distances results.
    int* 	idx = (int*)	malloc(query_length * k * sizeof(int));
    float* 	dst = (float*)	malloc(query_length * k * sizeof(float));

	/////////////////////////////  knn_exact_serial - result test  /////////////////////////////
	// ====================================================================================== //
    struct timeval start, end;
    gettimeofday(&start, NULL);
    knn_exact_serial(corpus, query, k, idx, dst, corpus_length, query_length, d, -1);
    gettimeofday(&end, NULL);

    // Calculate elapsed time in seconds:
    double time_taken = (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) / 1e6);
    printf("The \"knn_exact_serial\" function elapsed time is %lf seconds\n", time_taken);


	// Open neighbors set:
    dataset_name = "neighbors";
    float* neighbors = load_hdf5(data_path, dataset_name, &query_length, &k);
    if (neighbors == NULL) {
        fprintf(stderr, "Failed to load the %s data.\n", dataset_name);
        exit(EXIT_FAILURE);
    }
	int err_cnt = 0;
    for (int i = 0; i < query_length; i++) {
        for (int j = 0; j < k; j++) {
            if ( abs(idx[i * k + j] - neighbors[i * k + j]) > ZERO ) {
				err_cnt++;
				printf("Error at neighbors: %d, %d, idx: %d, neighbors: %f\n", i, j, idx[i * k + j], neighbors[i * k + j]);
			}
        }
    }
	printf("Neighbors errors: %f %%\n", 100 * err_cnt / (float)(query_length*k));
	free(neighbors);

	// Open distances set:
    dataset_name = "distances";
    float* distances = load_hdf5(data_path, dataset_name, &query_length, &k);
    if (distances == NULL) {
        fprintf(stderr, "Failed to load the %s data.\n", dataset_name);
        exit(EXIT_FAILURE);
    }
	err_cnt = 0;
    for (int i = 0; i < query_length; i++) {
        for (int j = 0; j < k; j++) {
			if ( abs(dst[i * k + j] - distances[i * k + j]) > ZERO ) {
				err_cnt++;
				printf("Error at distances: %d, %d, dst: %f, distances: %f\n", i, j, dst[i * k + j], distances[i * k + j]);
			}
        }
    }
	printf("Distances errors: %f %%", 100 * err_cnt / (float)(query_length*k));
	free(distances);

    free(idx);
    free(dst);

    return 0;
}
