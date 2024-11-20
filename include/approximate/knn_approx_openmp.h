#ifndef KNN_APPROX_OPENMP_H
#define KNN_APPROX_OPENMP_H

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../../include/approximate/knn_approx_serial.h"
#include "../exact/knn_exact_serial.h"

#include "../../include/utils/mem_info.h"

/**
 * Function to perform k-NN search using OpenMP for parallel computation.
 * 
 * @param dataset           Pointer to the corpus == query matrix (reference data points)
 * @param k                 Number of nearest neighbors to find
 * @param indices           Pre-allocated array to store indices of the k-nearest neighbors for each query (length `query_length x k`)
 * @param distances         Pre-allocated array to store the Euclidean distances to the k-nearest neighbors (length `query_length x k`)
 * @param dataset_length    Number of rows (data points) in the corpus
 * @param d                 Dimensionality of each data point (number of columns in corpus/query)
 * @param num_of_threads    Number of threads to be used for parallelization
 * @param accuracy          Tolerance or the number of subproblems to split the dataset
 * 
 * @return                  None (results are stored in the pre-allocated arrays `indices` and `distances`)
 */
void knn_approx_openmp(const float* dataset, int k, int* indices, float* distances,
                       int dataset_length, int d, int num_of_threads, int accuracy);

#endif // KNN_APPROX_OPENMP_H
