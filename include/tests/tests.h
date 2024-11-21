#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include "../../include/utils/data_io.h"

// Define the tolerance for comparison
#define ZERO 0.01

// Generic function pointer type for k-NN exact search
typedef void (*knn_exact_t)(const float* corpus, const float* query, int k, int* indices, float* distances, int corpus_length, int query_length, int d, int num_of_threads);
typedef void (*knn_approx_t)(const float* dataset, int k, int* indices, float* distances, int dataset_length, int d, int num_of_threads, int accuracy);

/**
 * Function to exact (one by one) compare k-NN results of 2 datasets.
 *
 * @param data_path*         Path to the .hdf5 file containing the dataset.
 * @param neighbors_name*    Dataset name for the neighbors.
 * @param distances_name*    Dataset name for the distances.
 *
 * @return                  -1 if there's an error in loading data or memory allocation, 0 otherwise
 */
int compare_knn_exact_results(const char* data_path1, const char* neighbors_name1, const char* distances_name1, 
                            const char* data_path2, const char* neighbors_name2, const char* distances_name2);

/**
 * Function to approximate compare k-NN results of 2 datasets.
 * Instead of comparing the values one by one, it checks how many of data_results1 are in
 * data_results2. We assume that data_results1 are the ground truth values.
 *
 * @param data_path1         Path to the .hdf5 file containing the ground truth dataset.
 * @param neighbors_name1    Dataset name for the ground truth neighbors.
 * @param distances_name1    Dataset name for the ground truth distances.
 * @param data_path2         Path to the .hdf5 file containing the approximate results dataset.
 * @param neighbors_name2    Dataset name for the approximate neighbors.
 * @param distances_name2    Dataset name for the approximate distances.
 *
 * @return                   -1 if there's an error in loading data or memory allocation, 0 otherwise
 */
int compare_knn_approx_results(const char* data_path1, const char* neighbors_name1, const char* distances_name1, 
                                const char* data_path2, const char* neighbors_name2, const char* distances_name2);


/**
 * Function to calculate exact k-NN results and store them in data_knn using a specified k-NN search function.
 *
 * @param knnsearch         Function pointer to the k-NN search implementation to be tested.
 * @param data_path         Path to the .hdf5 file containing the dataset.
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

/**
 * Function to calculate approximate k-NN results and store them in data_knn using a specified k-NN search function.
 *
 * @param knnsearch         Function pointer to the k-NN search implementation to be tested.
 * @param data_path         Path to the .hdf5 file containing the dataset.
 * @param corpus_name       Dataset name for the corpus (training set).
 * @param query_name        Dataset name for the query (test set).
 * @param k                 Evaluate k - NN
 * @param num_of_threads    Number of threads to use in the k-NN search function.
 * @param accuracy          The degrees of freedom we let the `knnsearch` have.
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
int generate_knn_approx_results(knn_approx_t knnsearch, const char* data_path, const char* dataset_name, int k, int num_of_threads, int accuracy, int id);