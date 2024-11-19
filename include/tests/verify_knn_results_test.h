#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include "../../include/utils/data_io.h"

// Define the tolerance for floating-point comparison
#define ZERO 0.01

// Generic function pointer type for k-NN exact search
typedef void (*knn_func_t)(const float* corpus, const float* query, int k, int* indices, float* distances, int corpus_length, int query_length, int d, int num_of_threads);


/**
 * Function to verify k-NN results using a specified k-NN search function.
 *
 * @param knnsearch         Function pointer to the k-NN search implementation to be tested.
 * @param data_path         Path to the HDF5 file containing the dataset.
 * @param corpus_name       Dataset name for the corpus (training set).
 * @param query_name        Dataset name for the query (test set).
 * @param neighbors_name    Dataset name for the ground truth neighbors.
 * @param distances_name    Dataset name for the ground truth distances.
 * @param num_of_threads    Number of threads to use in the k-NN search function.
 *
 * @return                  -1 if there's an error in loading data or memory allocation, 0 otherwise
 */
int verify_knn_results_test(knn_func_t knnsearch, const char* data_path, const char* corpus_name, const char* query_name, const char* neighbors_name, const char* distances_name, int num_of_threads);