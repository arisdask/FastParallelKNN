#ifndef KNN_EXACT_SERIAL_H
#define KNN_EXACT_SERIAL_H

#include "../../include/utils/distance.h"
#include "../../include/utils/mem_info.h"
#include <float.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_sort_float.h>  // sudo apt-get install libgsl-dev

/**
 * Compute the k-nearest neighbors using a brute-force method between corpus and query data points.
 * 
 * @param corpus        Pointer to the corpus matrix (reference data points)
 * @param query         Pointer to the query matrix (data points to compare)
 * @param k             Number of nearest neighbors to find
 * @param indices       Pre-allocated array to store indices of the k-nearest neighbors for each query (length `query_length x k`)
 * @param distances     Pre-allocated array to store the Euclidean distances to the k-nearest neighbors (length `query_length x k`)
 * @param corpus_length Number of rows (data points) in the corpus
 * @param query_length  Number of rows (data points) in the query
 * @param d             Dimensionality of each data point (number of columns in corpus/query)
 * 
 * @return              None (results are stored in the pre-allocated arrays indices and distances)
 */
void knn_exact_serial_core(const float* corpus, const float* query, int k, int* indices, float* distances, int corpus_length, int query_length, int d);


/**
 * Wrapper function to perform k-nearest neighbor search using a serial, brute-force approach,
 * handling memory constraints by processing in blocks (if necessary).
 * 
 * @param corpus        Pointer to the corpus matrix (reference data points)
 * @param query         Pointer to the query matrix (data points to compare)
 * @param k             Number of nearest neighbors to find
 * @param indices       Pre-allocated array to store indices of the k-nearest neighbors for each query (length `query_length x k`)
 * @param distances     Pre-allocated array to store the Euclidean distances to the k-nearest neighbors (length `query_length x k`)
 * @param corpus_length Number of rows (data points) in the corpus
 * @param query_length  Number of rows (data points) in the query
 * @param d             Dimensionality of each data point (number of columns in corpus/query)
 * @param num_of_blocks Number of blocks to split the query data into (for memory efficiency).
 *                      If this value is invalid, the minimum possible number of blocks is used
 *                      based on the available usable memory.
 * 
 * @return              None (results are stored in the pre-allocated arrays `indices` and `distances`)
 */
void knn_exact_serial(const float* corpus, const float* query, int k, int* indices, float* distances, int corpus_length, int query_length, int d, int num_of_blocks);

#endif // KNN_EXACT_SERIAL_H
