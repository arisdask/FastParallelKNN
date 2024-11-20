#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include "../../include/utils/data_io.h"

// Define the tolerance for comparison
#define ZERO 0.01

// Generic function pointer type for k-NN exact search
typedef void (*knn_exact_t)(const float* corpus, const float* query, int k, int* indices, float* distances, int corpus_length, int query_length, int d, int num_of_threads);


/**
 * Function to exact (one by one) compare k-NN results of 2 datasets.
 *
 * @param data_path*         Path to the HDF5 file containing the dataset.
 * @param neighbors_name*    Dataset name for the neighbors.
 * @param distances_name*    Dataset name for the distances.
 *
 * @return                  -1 if there's an error in loading data or memory allocation, 0 otherwise
 */
int compare_knn_exact_results(const char* data_path1, const char* neighbors_name1, const char* distances_name1, 
                            const char* data_path2, const char* neighbors_name2, const char* distances_name2);


/**
 * Function to calculate k-NN results and store them in data_knn using a specified k-NN search function.
 *
 * @param knnsearch         Function pointer to the k-NN search implementation to be tested.
 * @param data_path         Path to the HDF5 file containing the dataset.
 * @param corpus_name       Dataset name for the corpus (training set).
 * @param query_name        Dataset name for the query (test set).
 * @param k                 Evaluate k - NN
 * @param num_of_threads    Number of threads to use in the k-NN search function.
 * @param id                Identifier to the `knnsearch`:
 *                           1 -> knn_exact_serial
 *                           2 -> knn_exact_pthread
 *                           3 -> knn_exact_openmp
 *                           4 -> knn_exact_opencilk
 *                           5 -> knn_approx_serial
 *                           6 -> knn_approx_pthread
 *                           7 -> knn_approx_openmp
 *                           8 -> knn_approx_opencilk
 *
 * @return                  -1 if there's an error in loading data or memory allocation, 0 otherwise
 */
int generate_knn_exact_results(knn_exact_t knnsearch, const char* data_path, const char* corpus_name, const char* query_name, int k, int num_of_threads, int id);